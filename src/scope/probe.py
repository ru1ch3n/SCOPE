"""Decoder probes: freeze a trained SCOPE model and retrain only a larger decoder.

A probe answers one question: how much field information survives in a frozen
latent, independently of the decoder that was trained beside it. Encoder,
predictor, observation condition and EMA teacher are loaded from a completed run
and frozen (78,027,776 parameters); `model.decoder` is replaced by a fresh
`ProbeDecoder` of the requested size, which is the only module that receives a
gradient. The topology is the release decoder's, widened: per-token MLP to 8x8
field tiles, then a residual convolutional refinement. Only Linear/Conv2d/GELU
are used, so a probe stays inside the deterministic-algorithm setting.

Two observation settings, deliberately distinct experiments:

`full`        the decoder is trained on `decoder(encoder(full_pair_view(y)))`.
              This is the path the grounding term trains in the `full` variant.
`uniform500`  the decoder is trained on the real sparse inference path,
              `decoder(predictor(encoder(observation_view(y, task, 500,
              "uniform", seed))))`, with the latent detached, forward and
              inverse alternating by batch index.

A `full` probe may still be *evaluated* sparsely (`--protocol uniform3`). That is
an out-of-distribution use of that decoder and is reported as such; it is not the
same experiment as a `uniform500` probe.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn

from . import checkpoint
from .config import science_identity, stage_config
from .io import read_json
from .losses import field_objective
from .model import ModelSpec, SCOPE

# Fixed tables, never searched at run time, so a probe's configuration - and
# therefore its identity hash - is a pure function of the size name.
ORIGINAL = {"hidden": 256, "mlp_layers": 3, "conv_width": 64, "conv_layers": 3}
ORIGINAL_PARAMETERS = 171_010
SIZES = {
    "orig": dict(ORIGINAL),
    "1m": {"hidden": 512, "mlp_layers": 4, "conv_width": 112, "conv_layers": 5},
    "5m": {"hidden": 1248, "mlp_layers": 4, "conv_width": 240, "conv_layers": 5},
    "10m": {"hidden": 1760, "mlp_layers": 4, "conv_width": 352, "conv_layers": 5},
    "15m": {"hidden": 2168, "mlp_layers": 4, "conv_width": 432, "conv_layers": 5},
}
DECLARED = {"orig": 171_010, "1m": 1_000_194, "5m": 5_003_170, "10m": 10_010_658, "15m": 15_018_218}
REPORTED_SIZES = ("5m", "10m", "15m")
OBSERVATIONS = ("full", "uniform500")
TRAINABLE_PREFIX = "decoder."
FROZEN_PREFIXES = ("encoder.", "predictor.", "observation_condition.", "target_encoder.")
FROZEN_PARTS = {
    "encoder": 38_157_952,
    "predictor": 1_711_488,
    "observation_condition": 384,
    "target_encoder": 38_157_952,
}
FROZEN_PARAMETERS = sum(FROZEN_PARTS.values())  # 78,027,776
PROBE_EPOCHS = 100
PROBE_WARMUP_EPOCHS = 5


def parameter_count(anchor_channels, patch, hidden, mlp_layers, conv_width, conv_layers, output_channels=2):
    """Closed-form trainable-parameter count of ProbeDecoder, bias terms included."""
    if mlp_layers < 2 or conv_layers < 2:
        raise ValueError("at least an input and an output layer are required")
    tile = output_channels * patch * patch
    mlp = (
        (anchor_channels * hidden + hidden)
        + (mlp_layers - 2) * (hidden * hidden + hidden)
        + (hidden * tile + tile)
    )
    conv = (
        (output_channels * conv_width * 9 + conv_width)
        + (conv_layers - 2) * (conv_width * conv_width * 9 + conv_width)
        + (conv_width * output_channels * 9 + output_channels)
    )
    return mlp + conv


def release_decoder_table(spec):
    """`NonlinearPatchDecoder` expressed in the same table as the probe sizes."""
    return {
        "hidden": max(2 * spec.anchor_channels, 32),
        "mlp_layers": 3,
        "conv_width": spec.decoder_width,
        "conv_layers": 3,
    }


def declared_count(spec, size):
    if size not in SIZES:
        raise ValueError(f"unknown probe size {size!r}")
    patch = spec.resolution // spec.anchor_grid
    n = parameter_count(
        spec.anchor_channels, patch, output_channels=spec.decoder_output_channels, **SIZES[size]
    )
    if spec == ModelSpec() and n != DECLARED[size]:
        raise AssertionError((size, n, DECLARED[size]))
    return n


class ProbeDecoder(nn.Module):
    """The release decoder's mechanism at a larger width; no observed-field skip."""

    def __init__(self, spec, size):
        super().__init__()
        if size not in SIZES:
            raise ValueError(f"unknown probe size {size!r}")
        table = SIZES[size]
        self.spec = spec
        self.size = size
        self.patch = spec.resolution // spec.anchor_grid
        out = spec.decoder_output_channels
        tile = out * self.patch**2
        hidden, width = table["hidden"], table["conv_width"]
        layers = [nn.Linear(spec.anchor_channels, hidden), nn.GELU()]
        for _ in range(table["mlp_layers"] - 2):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers.append(nn.Linear(hidden, tile))
        self.patch_head = nn.Sequential(*layers)
        convs = [nn.Conv2d(out, width, 3, padding=1), nn.GELU()]
        for _ in range(table["conv_layers"] - 2):
            convs += [nn.Conv2d(width, width, 3, padding=1), nn.GELU()]
        convs.append(nn.Conv2d(width, out, 3, padding=1))
        self.refinement = nn.Sequential(*convs)
        expected = parameter_count(spec.anchor_channels, self.patch, output_channels=out, **table)
        actual = sum(p.numel() for p in self.parameters())
        if actual != expected:
            raise AssertionError((actual, expected))
        self.parameter_total = actual

    def forward(self, latent):
        grid, patch = self.spec.anchor_grid, self.patch
        if latent.ndim != 3 or latent.shape[1:] != (grid**2, self.spec.anchor_channels):
            raise ValueError("Unexpected predicted spatial latent shape")
        out = self.spec.decoder_output_channels
        tiles = self.patch_head(latent).reshape(len(latent), grid, grid, out, patch, patch)
        field = tiles.permute(0, 3, 1, 4, 2, 5).reshape(
            len(latent), out, self.spec.resolution, self.spec.resolution
        )
        return field + self.refinement(field)


def attach(model, size):
    """Replace `model.decoder` by a fresh ProbeDecoder and freeze everything else."""
    spec = model.spec
    replaced = sum(p.numel() for p in model.decoder.parameters())
    original = parameter_count(
        spec.anchor_channels,
        spec.resolution // spec.anchor_grid,
        output_channels=spec.decoder_output_channels,
        **release_decoder_table(spec),
    )
    if replaced != original:
        raise AssertionError(f"the base decoder is not the release topology: {replaced} != {original}")
    if spec == ModelSpec() and replaced != ORIGINAL_PARAMETERS:
        raise AssertionError(f"the base decoder is not the release decoder: {replaced}")
    model.decoder = ProbeDecoder(spec, size)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(TRAINABLE_PREFIX))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    if trainable != sum(p.numel() for p in model.decoder.parameters()):
        raise AssertionError("a non-decoder parameter is still trainable")
    if spec == ModelSpec() and frozen != FROZEN_PARAMETERS:
        raise AssertionError(("frozen parameter count is not the release architecture", frozen))
    return {
        "size": size,
        "decoder": dict(SIZES[size]),
        "trainable_parameters": trainable,
        "replaced_decoder_parameters": replaced,
        "frozen_parameters": frozen,
    }


def assert_frozen_base(model):
    """The freeze is the experiment. Check it directly, not through the graph."""
    for prefix in FROZEN_PREFIXES:
        if any(p.requires_grad for n, p in model.named_parameters() if n.startswith(prefix)):
            raise AssertionError(f"the base must stay frozen: {prefix}")


def probe_objective(model, observed, target, c, target_norms=None, zero_targets=None):
    """Field reconstruction from a frozen latent; the decoder is the only trainee.

    `full`: the latent is the online encoder on the fully visible pair, so a
    thawed base would also show up as `latent.requires_grad`. `uniform500`: the
    latent comes from the sparse inference path under `no_grad`, which removes
    that signal, so the freeze is asserted explicitly instead.
    """
    assert_frozen_base(model)
    if any(v != 0.0 for v in c["loss"].values()):
        raise ValueError("A probe carries no auxiliary objective")
    if c["probe_observation"] == "full":
        latent = model.encoder(model.full_pair_view(target))
        if latent.requires_grad:
            raise AssertionError("the frozen base produced a differentiable latent")
    else:
        with torch.no_grad():
            latent = model(observed)["predicted_latent"]
        latent = latent.detach()
    predicted = model.decoder(latent)
    per_field = field_objective(predicted, target, c, target_norms, zero_targets)
    raw = {
        "field": per_field.mean(),
        "field_a_objective": per_field[:, 0].mean(),
        "field_u_objective": per_field[:, 1].mean(),
        "probe_latent_std": latent.float().std(correction=0),
    }
    return raw, {}, per_field.detach(), per_field.new_zeros(len(target))


def base_record(base_run, data_root):
    """Identity of the completed run a probe freezes; nothing about it is changed."""
    base_run = Path(base_run)
    manifest = read_json(base_run / "run.json")
    base = manifest["config"]
    c = stage_config(base, "main", data_root)
    path, record = checkpoint.current(base_run / "main" / "checkpoints")
    payload = checkpoint.read_payload(path)
    if payload["science_sha256"] != science_identity(c):
        raise ValueError("Base checkpoint configuration mismatch")
    if payload["draws"] != c["total_draws"]:
        raise ValueError("A probe freezes a completed base run only")
    identity = {
        "run": str(base_run),
        "variant": base.get("variant", "full"),
        "pde": base["pde"],
        "epoch": record["epoch"],
        "checkpoint_sha256": record["sha256"],
        "science_sha256": payload["science_sha256"],
    }
    del payload
    return identity, base


def probe_config(base, record, size, observation):
    """A probe is bound to one base checkpoint, one size and one observation setting."""
    if size not in SIZES:
        raise ValueError(f"unknown probe size {size!r}")
    if observation not in OBSERVATIONS:
        raise ValueError(f"unknown probe observation setting {observation!r}")
    c = copy.deepcopy(base)
    c.update(
        schema="scope-decoder-probe/v1",
        variant="probe",
        base=record,
        probe_size=size,
        probe_observation=observation,
        probe_epochs=PROBE_EPOCHS,
        pretrain_epochs=0,
        main_epochs=PROBE_EPOCHS,
        loss={"jepa_weight": 0.0, "grounding_weight": 0.0, "variance_weight": 0.0},
        minimum_field_gradient_share=1.0,
        ema_start=0.0,
        ema_end=0.0,
        pretrained_initialization=False,
        ablation={
            "trainable": "decoder only",
            "frozen": list(FROZEN_PREFIXES),
            "objective": (
                "field_on_full_observation_latent"
                if observation == "full"
                else "field_on_uniform500_predictor_latent"
            ),
        },
    )
    c["optimizer"] = dict(c["optimizer"], warmup_epochs=PROBE_WARMUP_EPOCHS)
    c["evaluation"] = dict(c["evaluation"], default_protocol="full-observation" if observation == "full" else "uniform3")
    return c


def build(c, device):
    """A frozen base from its own committed checkpoint, plus a fresh decoder."""
    spec = ModelSpec(**c["model_spec"])
    model = SCOPE(spec)
    path, record = checkpoint.current(Path(c["base"]["run"]) / "main" / "checkpoints")
    payload = checkpoint.read_payload(path)
    if payload["science_sha256"] != c["base"]["science_sha256"]:
        raise ValueError("Base checkpoint identity changed since this probe was configured")
    if record["sha256"] != c["base"]["checkpoint_sha256"]:
        raise ValueError("Base checkpoint is not the one this probe was pinned to")
    model.load_state_dict(payload["model"], strict=True)
    report = attach(model, c["probe_size"])
    del payload
    if spec == ModelSpec() and report["trainable_parameters"] != DECLARED[c["probe_size"]]:
        raise AssertionError(report)
    return model.to(device), report


def main():
    parser = argparse.ArgumentParser(description="Train one decoder probe on a completed run")
    parser.add_argument("--base", type=Path, required=True, help="a completed run directory")
    parser.add_argument("--size", choices=tuple(SIZES), required=True)
    parser.add_argument("--observation", choices=OBSERVATIONS, default="full")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-updates", type=int)
    args = parser.parse_args()
    from .train import run_probe  # local import: train imports this module

    record, base = base_record(args.base, args.data_root)
    c = probe_config(base, record, args.size, args.observation)
    output = args.output or Path("runs") / "probes" / (
        f"{args.size}-{args.observation}-{record['variant']}-{record['pde']}"
    )
    result = run_probe(
        c, args.data_root, output, torch.device(args.device), args.workers, args.resume, args.max_updates
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
