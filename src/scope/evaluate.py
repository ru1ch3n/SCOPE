"""Evaluate fixed checkpoints without test-based selection or training side effects."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from filelock import FileLock
from . import checkpoint
from .config import ASSETS, stage_config, science_identity
from .data import verify_cache
from .io import code_hashes, object_hash, read_json, sha256, write_json
from .losses import physical_relative
from .metrics import darcy_errors, metric_arrays, summarize_arrays
from .observations import observation_view, observation_counts, evaluation_conditions, seed_for
from .runtime import initialize, autocast, sentinel, state_digest
from .train import make_model


def conditions(pde, protocol):
    if protocol == "full-observation":
        # Both channels fully visible: every variant and every probe answers the
        # same question, so the numbers are comparable across them.
        return [("joint", "full", 16384)]
    if protocol == "uniform3":
        return [(t, "uniform", 500) for t in ("forward", "inverse")]
    if protocol == "benchmark" and pde == "ns-bounded":
        return [(t, "benchmark_uniform", 100) for t in ("forward", "inverse")]
    if protocol == "benchmark":
        return conditions(pde, "uniform3")
    result = evaluation_conditions()
    if pde == "ns-bounded":
        result += conditions(pde, "benchmark")
    return result


@torch.no_grad()
def evaluate(run, data_root, device, protocol="uniform3", batch_size=32, allow_partial=False):
    run = Path(run)
    manifest = read_json(run / "run.json")
    if manifest["code_hashes"] != code_hashes():
        raise ValueError("Run code identity differs from this release")
    base = manifest["config"]
    if base.get("smoke_test"):
        data_root = run / "synthetic-data"
    stage = "probe" if base.get("variant") == "probe" else "main"
    c = stage_config(base, stage, data_root)
    if not base.get("smoke_test"):
        verify_cache(c["pde"], "test", data_root)
    path, cp = checkpoint.current(run / stage / "checkpoints")
    p = checkpoint.read_payload(path)
    if p["science_sha256"] != science_identity(c):
        raise ValueError("Checkpoint configuration mismatch")
    if p["draws"] != c["total_draws"] and not allow_partial:
        raise ValueError(
            "Training is incomplete. Pass --allow-partial to label an intermediate assessment explicitly."
        )
    initialize(c["seed"])
    model = make_model(c, device)
    model.load_state_dict(p["model"], strict=True)
    model.eval()
    same_runtime = p["runtime"] == {"torch": torch.__version__, "device_type": device.type}
    if same_runtime and not torch.equal(sentinel(model, p["sentinel_input"], device), p["sentinel_output"]):
        raise ValueError("Evaluation reload sentinel failed")
    del p
    weights_hash = state_digest(model.state_dict())
    folder = Path(data_root) / c["pde"]
    fields = np.load(folder / "test-f32.npy", mmap_mode="r", allow_pickle=False)
    if fields.shape != (c["test_records"], 2, 128, 128):
        raise ValueError("Test shape mismatch")
    boundary = c["pde"] == "ns-bounded"
    if boundary:
        norms = np.load(folder / "test-physical-norms.npy", mmap_mode="r", allow_pickle=False)
        zeros = np.load(folder / "test-zero-targets.npy", allow_pickle=False)
        geometry = np.load(ASSETS / "test-geometry.npy", allow_pickle=False)
    data_hash = sha256(folder / "test-f32.npy")
    out = run / "evaluation" / protocol / f"draws-{cp['draws']:09d}"
    out.mkdir(parents=True, exist_ok=True)
    identity = {
        "checkpoint_sha256": cp["sha256"],
        "science_sha256": science_identity(c),
        "test_sha256": data_hash,
        "code_hashes": code_hashes(),
        "protocol": protocol,
        "evaluation_batch_size": batch_size,
        "torch": torch.__version__,
        "device_type": device.type,
    }
    rows = []
    with FileLock(str(out / ".evaluation.lock"), timeout=1):
        for task, family, budget in conditions(c["pde"], protocol):
            stem = f"{task}-{family}-{budget}"
            json_path, npz_path = out / (stem + ".json"), out / (stem + ".npz")
            if json_path.exists():
                saved = read_json(json_path)
                if saved["identity"] != identity or sha256(npz_path) != saved["arrays_sha256"]:
                    raise ValueError("Cached evaluation identity mismatch")
                rows.append(saved["summary"])
                continue
            error_chunks, ber_chunks, counts, parts = [], [], [], {}
            mask_hash = hashlib.sha256()
            for begin in range(0, len(fields), batch_size):
                target = np.array(fields[begin : begin + batch_size], copy=True)
                if family == "full":
                    ones = np.ones((len(target), 128, 128), np.float32)
                    observed = np.stack((target[:, 0], ones, target[:, 1], ones), axis=1)
                else:
                    observed = np.stack(
                        [
                            observation_view(
                                x,
                                task,
                                budget,
                                family,
                                seed_for(c["evaluation"]["mask_seed"], "test", begin + j),
                                geometry[begin + j] if boundary else None,
                            )
                            for j, x in enumerate(target)
                        ]
                    )
                mask_hash.update(observed[:, [1, 3]].astype(np.uint8).tobytes())
                truth = torch.from_numpy(target).to(device)
                with autocast(device):
                    predicted = model(torch.from_numpy(observed).to(device))["fields"]
                if not bool(torch.isfinite(predicted).all()):
                    raise FloatingPointError("Nonfinite prediction")
                if boundary:
                    arrays = metric_arrays(
                        predicted,
                        truth,
                        c["normalization"],
                        norms[begin : begin + len(target)],
                        zeros[begin : begin + len(target)],
                        geometry[begin : begin + len(target)],
                    )
                    for key, value in arrays.items():
                        parts.setdefault(key, []).append(value)
                    counts.extend(observation_counts(x, geometry[begin + j]) for j, x in enumerate(observed))
                else:
                    values = physical_relative(predicted, truth, c["normalization"]).cpu().numpy()
                    if not np.isfinite(values).all():
                        raise FloatingPointError("Undefined relative error")
                    error_chunks.append(values)
                    expected = (
                        (budget, 0)
                        if task == "forward"
                        else ((0, budget) if task == "inverse" else (budget, budget))
                    )
                    if family != "full" and not np.all(
                        observed[:, [1, 3]].sum((2, 3)) == np.array(expected)
                    ):
                        raise ValueError("Observation count mismatch")
                    if c["pde"] == "darcy":
                        ber_chunks.append(darcy_errors(predicted, truth, c["normalization"]).cpu().numpy())
            arrays = {"record_ids": np.arange(len(fields))}
            row = {
                "pde": c["pde"],
                "epoch": cp["epoch"],
                "task": task,
                "family": family,
                "test_records": len(fields),
                "mask_sha256": mask_hash.hexdigest(),
                "variant": base.get("variant", "full"),
                "probe_size": base.get("probe_size"),
                "probe_observation": base.get("probe_observation"),
                "nominal_points": budget if family != "benchmark_uniform" else None,
                "nominal_visible_percent": 100 * budget / 16384 if family != "benchmark_uniform" else 1.0,
                "target_channel": "u" if task == "forward" else ("a" if task == "inverse" else "both"),
                "exact_official_mask_files": False,
            }
            if boundary:
                arrays.update({k: np.concatenate(v) for k, v in parts.items()})
                row["regions"] = summarize_arrays(arrays, zeros)
                arrays["source_zero_targets"] = zeros
                arrays["extra_fluid_points_a_u"] = np.array([x["extra_fluid_points_a_u"] for x in counts])
                arrays["total_unique_points_a_u"] = np.array([x["total_unique_points_a_u"] for x in counts])
                row["mean_extra_fluid_points_a_u"] = arrays["extra_fluid_points_a_u"].mean(0).tolist()
                row["mean_total_unique_points_a_u"] = arrays["total_unique_points_a_u"].mean(0).tolist()
                for channel in ("a", "u"):
                    row[f"relative_l2_{channel}_defined_target_mean_percent"] = row["regions"]["full"][
                        channel
                    ]["relative_l2_defined_target_mean_percent"]
            else:
                values = np.concatenate(error_chunks)
                arrays["per_record_relative_l2"] = values
                row.update(
                    relative_l2_a_percent=float(values[:, 0].astype(np.float64).mean() * 100),
                    relative_l2_u_percent=float(values[:, 1].astype(np.float64).mean() * 100),
                )
                if ber_chunks:
                    ber = np.concatenate(ber_chunks)
                    arrays["per_record_binary_error_rate_a"] = ber
                    row["ber_a_percent"] = float(ber.astype(np.float64).mean() * 100)
                    row["pixel_accuracy_a_percent"] = 100 - row["ber_a_percent"]
            with npz_path.with_suffix(".npz.partial").open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            npz_path.with_suffix(".npz.partial").replace(npz_path)
            write_json(json_path, {"summary": row, "identity": identity, "arrays_sha256": sha256(npz_path)})
            rows.append(row)
            print(json.dumps(row), flush=True)
        if state_digest(model.state_dict()) != weights_hash:
            raise RuntimeError("Evaluation mutated model state")
        report = {
            "status": "evaluated",
            "smoke_test": base.get("smoke_test", False),
            "training_complete": cp["draws"] == c["total_draws"],
            "epoch": cp["epoch"],
            "identity": identity,
            "strict_same_runtime_sentinel": same_runtime,
            "rows": rows,
            "test_used_for_checkpoint_selection": False,
            "comparison": "Released benchmark data; predeclared SCOPE masks, not bitwise-identical baseline masks",
        }
        write_json(out / "results.json", report)
    return out / "results.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--protocol", choices=("uniform3", "benchmark", "full", "full-observation"), default="uniform3"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    report = evaluate(
        args.run,
        args.data_root,
        torch.device(args.device),
        args.protocol,
        args.batch_size,
        args.allow_partial,
    )
    print(json.dumps({"results": str(report)}), flush=True)


if __name__ == "__main__":
    main()
