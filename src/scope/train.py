"""Train the main SCOPE model: 10 teacher epochs followed by 500 student epochs."""

import argparse
from collections import defaultdict
import copy
import gc
import json
from pathlib import Path
import signal
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from filelock import FileLock
from . import checkpoint, probe
from .config import base_config, load_config, science_identity, stage_config
from .data import verify_cache
from .dataset import HomogeneousDataset, assert_batch
from .io import append_json, code_hashes, read_json, sha256, write_json
from .model import ModelSpec, SCOPE
from .observations import TASKS, FAMILIES, seed_for
from .optimization import make_optimizer, one_update
from .runtime import initialize, state_digest

STOP = {"signal": None}


def make_model(c, device):
    if c["stage"] == "probe":
        model, _ = probe.build(c, device)
        return model
    model = SCOPE(ModelSpec(**c["model_spec"])).to(device)
    if c["stage"] == "teacher_pretrain":
        model.predictor.requires_grad_(False)
        model.observation_condition.requires_grad_(False)
    return model


def expected_trainable(c):
    if c["stage"] == "probe":
        return probe.DECLARED[c["probe_size"]]
    return 38328962 if c["stage"] == "teacher_pretrain" else 40040834


def initialize_stage(c, output, device, resume=False):
    initialize(c["seed"])
    model = make_model(c, device)
    info = {"random_state_sha256": state_digest(model.state_dict())}
    if c["stage"] == "main" and not resume and c.get("pretrained_initialization", True):
        teacher_dir = Path(output) / "teacher_pretrain" / "checkpoints"
        path, record = checkpoint.current(teacher_dir)
        p = checkpoint.read_payload(path)
        expected = copy.deepcopy(c)
        expected.update(
            stage="teacher_pretrain",
            epochs=c["pretrain_epochs"],
            use_jepa=False,
            total_draws=c["pretrain_epochs"] * c["train_records"],
        )
        expected["optimizer"]["warmup_epochs"] = 3
        if p["science_sha256"] != science_identity(expected) or p["draws"] != expected["total_draws"]:
            raise ValueError("Completed matching teacher pretraining is required")
        predictor_hash = state_digest(model.predictor.state_dict())
        condition_hash = state_digest(model.observation_condition.state_dict())
        model.load_state_dict(p["model"], strict=True)
        if predictor_hash != state_digest(model.predictor.state_dict()) or condition_hash != state_digest(
            model.observation_condition.state_dict()
        ):
            raise ValueError("Predictor/observation condition unexpectedly changed during pretraining")
        model.target_encoder.load_state_dict(model.encoder.state_dict(), strict=True)
        model.ema_updates.zero_()
        info["transfer"] = {
            "teacher_checkpoint_sha256": record["sha256"],
            "encoder_decoder_transferred": True,
            "predictor_condition_unchanged": True,
            "optimizer_reset": True,
            "ema_reset_from_online": True,
        }
    return model, make_optimizer(model, c, device), info


def run_stage(base, stage, data_root, output, device, workers=4, resume=False, max_updates=None):
    c = stage_config(base, stage, data_root)
    out = Path(output) / stage
    out.mkdir(parents=True, exist_ok=True)
    cp_dir = out / "checkpoints"
    existing = (cp_dir / "index.json").exists()
    if existing and not resume:
        raise FileExistsError("This run has a checkpoint. Use --resume; nothing was overwritten.")
    model, opt, init = initialize_stage(c, output, device, resume=existing)
    if (
        not base.get("smoke_test")
        and sum(p.numel() for p in model.parameters() if p.requires_grad) != expected_trainable(c)
    ):
        raise ValueError("Trainable parameter count mismatch")
    fixture = HomogeneousDataset(c, stop=c["batch_size"])
    sentinel_input = torch.stack([fixture[i][0] for i in range(2)])
    del fixture
    draws = 0
    if existing:
        path, record = checkpoint.current(cp_dir)
        p = checkpoint.load(path, model, opt, c, device)
        draws = p["draws"]
        del p
    else:
        write_json(out / "initialization.json", init)
    if draws == c["total_draws"]:
        result = {
            "status": "complete",
            "stage": stage,
            "epoch": c["epochs"],
            "draws": draws,
            "strict_reload": True,
            "checkpoint": record,
        }
        write_json(out / "result.json", result)
        return result
    generator = torch.Generator().manual_seed(seed_for(c["seed"], "loader", stage))
    dataset = HomogeneousDataset(c, start=draws)
    kw = dict(
        batch_size=c["batch_size"],
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=generator,
    )
    if workers:
        kw.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(dataset, **kw)
    t0 = time.monotonic()
    initial_draws = draws
    windows, bins = defaultdict(float), {}
    updates = 0
    window_n = 0
    window_start = draws
    record = None
    for observed, target, meta, target_norms, zero_targets in loader:
        if STOP["signal"]:
            break
        assert_batch(meta, c["batch_size"])
        if int(meta[0, 0]) != draws:
            raise ValueError("Sample position mismatch")
        previous_epoch = draws // c["train_records"]
        values, fields, latent = one_update(
            model, opt, observed, target, c, draws, device, target_norms, zero_targets
        )
        draws += c["batch_size"]
        updates += 1
        for key, value in values.items():
            windows[key] += value
        window_n += 1
        task, budget, family = map(int, meta[0, [2, 3, 4]])
        key = (task, family, budget)
        entry = bins.setdefault(key, [0, 0.0, 0.0, 0.0])
        entry[0] += len(fields)
        entry[1] += float(fields[:, 0].double().sum())
        entry[2] += float(fields[:, 1].double().sum())
        entry[3] += float(latent.double().sum())
        crossed = draws // c["train_records"] > previous_epoch
        finished = draws == c["total_draws"]
        paused = STOP["signal"] or (max_updates is not None and updates >= max_updates)
        if window_n == 100 or crossed or finished or paused:
            rate = (draws - initial_draws) / (time.monotonic() - t0)
            row = {
                "stage": stage,
                "epoch": draws / c["train_records"],
                "epoch_from": window_start / c["train_records"],
                "draws": draws,
                "metrics": {k: v / window_n for k, v in windows.items()},
                "by_condition": [
                    {
                        "task": TASKS[t],
                        "family": FAMILIES[f],
                        "nominal_points": b,
                        "nominal_visible_percent": 100 * b / 16384,
                        "records": s[0],
                        "field_a": s[1] / s[0],
                        "field_u": s[2] / s[0],
                        "jepa_latent_mse": s[3] / s[0],
                    }
                    for (t, f, b), s in sorted(bins.items())
                ],
                "samples_per_second": rate,
                "seconds_per_epoch_this_attempt": c["train_records"] / rate,
                "field_units": "fraction, not percent; NS zero targets use the documented finite training extension",
            }
            append_json(out / "metrics.jsonl", row)
            write_json(out / "status.json", {"status": "running", **row})
            print(json.dumps({k: v for k, v in row.items() if k != "by_condition"}), flush=True)
            windows.clear()
            bins.clear()
            window_n = 0
            window_start = draws
        if crossed or finished or paused:
            milestone = finished or (
                crossed and draws // c["train_records"] % c["milestone_every_epochs"] == 0
            )
            record = checkpoint.save(cp_dir, model, opt, c, draws, sentinel_input, milestone=milestone)
        if paused:
            break
    if record is None or record["draws"] != draws:
        record = checkpoint.save(cp_dir, model, opt, c, draws, sentinel_input)
    path, _ = checkpoint.current(cp_dir)
    p = checkpoint.load(path, model, opt, c, device)
    if p["draws"] != draws:
        raise ValueError("Final reload position mismatch")
    complete = draws == c["total_draws"]
    result = {
        "status": "complete" if complete else "paused",
        "stage": stage,
        "epoch": draws / c["train_records"],
        "draws": draws,
        "strict_reload": True,
        "checkpoint": record,
        "test_data_used": False,
        "smoke_test": base.get("smoke_test", False),
    }
    write_json(out / ("result.json" if complete else "paused.json"), result)
    write_json(out / "status.json", result)
    del p, loader, dataset, model, opt
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def smoke_data(data_root):
    """Synthetic software check only; never a scientific experiment or benchmark score."""
    c = base_config("poisson")
    c.update(
        train_records=32, test_records=8, batch_size=4, pretrain_epochs=1, main_epochs=2, smoke_test=True
    )
    c["model_spec"].update(
        encoder_width=16,
        encoder_depth=1,
        encoder_heads=2,
        predictor_width=16,
        predictor_depth=1,
        predictor_heads=2,
        anchor_channels=8,
        decoder_width=8,
    )
    folder = Path(data_root) / "poisson"
    folder.mkdir(parents=True, exist_ok=True)
    for split, size in (("train", 32), ("test", 8)):
        rng = np.random.default_rng(71 if split == "train" else 72)
        fields = rng.normal(0, 0.5, (size, 2, 128, 128)).astype(np.float32)
        path = folder / f"{split}-f32.npy"
        if path.exists():
            if not np.array_equal(np.load(path, allow_pickle=False), fields):
                raise ValueError("Refusing to overwrite existing smoke data")
        else:
            np.save(path, fields, allow_pickle=False)
            physical = (
                fields.astype(np.float64) * 2 * np.array(c["normalization"]["std"])[None, :, None, None]
                + np.array(c["normalization"]["mean"])[None, :, None, None]
            )
            np.save(
                folder / f"{split}-physical-norms.npy",
                np.linalg.norm(physical.reshape(size, 2, -1), axis=2),
                allow_pickle=False,
            )
            np.save(folder / f"{split}-zero-targets.npy", np.zeros((size, 2), np.bool_), allow_pickle=False)
    c["train_sha256"] = sha256(folder / "train-f32.npy")
    return c


def stages_for(base):
    """`fo` has no teacher and no pretraining stage; a probe has only its own."""
    if base.get("variant") == "probe":
        return ("probe",)
    if base.get("pretrain_epochs", 10) == 0:
        return ("main",)
    return ("teacher_pretrain", "main")


def run(base, data_root, output, device, workers=4, resume=False, max_updates=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".run.lock"), timeout=1):
        contract = {"config": base, "code_hashes": code_hashes()}
        if (output / "run.json").exists():
            if not resume:
                raise FileExistsError("Run exists: pass --resume; existing artifacts preserved")
            if read_json(output / "run.json") != contract:
                raise ValueError("Run identity mismatch")
        else:
            write_json(output / "run.json", contract)
        for stage in stages_for(base):
            result = run_stage(base, stage, data_root, output, device, workers, resume, max_updates)
            if result["status"] != "complete":
                return result
        return result


def run_probe(c, data_root, output, device, workers=4, resume=False, max_updates=None):
    """One probe run: a frozen base and a single 100-epoch decoder stage."""
    if c.get("variant") != "probe":
        raise ValueError("Not a probe configuration")
    return run(c, data_root, output, device, workers, resume, max_updates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/poisson.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-updates",
        type=int,
        help="Pause safely after this many updates per stage; keep the full LR horizon",
    )
    parser.add_argument("--smoke", action="store_true", help="Tiny synthetic CPU test, not a research run")
    args = parser.parse_args()
    if args.workers < 0 or (args.max_updates is not None and args.max_updates < 1):
        parser.error("workers must be nonnegative and max-updates positive")
    if args.smoke:
        args.output = args.output or Path("runs/smoke")
        args.data_root = args.output / "synthetic-data"
        base = smoke_data(args.data_root)
        args.device, args.workers = "cpu", 0
    else:
        base = load_config(args.config)
        if (
            not args.device.startswith("cuda")
            or not torch.cuda.is_available()
            or not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError(
                "Main training requires a CUDA GPU with BF16 support. Use --smoke for a CPU software test."
            )
        if torch.__version__.split("+")[0] != "2.12.1":
            raise RuntimeError("Use the pinned PyTorch 2.12.1 training environment")
        verify_cache(base["pde"], "train", args.data_root)
        variant = base.get("variant", "full")
        name = base["pde"] if variant == "full" else f"{variant}-{base['pde']}"
        args.output = args.output or Path("runs") / name
    STOP["signal"] = None
    for name in ("SIGTERM", "SIGINT", "SIGUSR1"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), lambda number, frame: STOP.update(signal=number))
    try:
        result = run(
            base,
            args.data_root,
            args.output,
            torch.device(args.device),
            args.workers,
            args.resume,
            args.max_updates,
        )
    except Exception as exc:
        write_json(
            args.output / "failure.json",
            {"error_type": type(exc).__name__, "error": str(exc), "prior_checkpoints_preserved": True},
        )
        raise
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
