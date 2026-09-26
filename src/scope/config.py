"""Frozen scientific defaults. Runtime paths and worker count are separate."""

import copy
from dataclasses import asdict
from pathlib import Path
from .io import read_json, object_hash
from .model import ModelSpec

ASSETS = Path(__file__).parent / "assets"
MANIFEST = read_json(ASSETS / "data-manifest.json")
PDES = tuple(MANIFEST["datasets"])

# The objective ablations. Everything outside this table - data, model, seed,
# optimizer, schedule, batch size, observation sampling, evaluation contract -
# is the same in every variant, and `test_variants.py` asserts that. `full` is
# returned byte-identical to `base_config`, so its configuration identity, and
# any run already started from it, are unaffected by this table existing.
VARIANTS = {
    "full": {
        "loss": {"jepa_weight": 1.0, "grounding_weight": 0.25, "variance_weight": 0.01},
        "pretrain_epochs": 10,
        "grounding_gradient": "encoder_and_decoder",
        "minimum_field_gradient_share": 0.5,
        "ema": (0.996, 0.9999),
        "pretrained_initialization": True,
        "summary": "field + JEPA + grounding + variance; the reference recipe",
    },
    "fo": {
        "loss": {"jepa_weight": 0.0, "grounding_weight": 0.0, "variance_weight": 0.0},
        "pretrain_epochs": 0,
        "grounding_gradient": None,
        "minimum_field_gradient_share": 1.0,
        "ema": (0.0, 0.0),
        "pretrained_initialization": False,
        "summary": "field only, from random initialization; no teacher, no pretraining stage",
    },
    "fjv": {
        "loss": {"jepa_weight": 1.0, "grounding_weight": 0.0, "variance_weight": 0.01},
        "pretrain_epochs": 10,
        "grounding_gradient": None,
        "minimum_field_gradient_share": 0.5,
        "ema": (0.996, 0.9999),
        "pretrained_initialization": True,
        "summary": "full minus grounding",
    },
    "fg": {
        "loss": {"jepa_weight": 0.0, "grounding_weight": 0.25, "variance_weight": 0.0},
        "pretrain_epochs": 10,
        "grounding_gradient": "encoder_and_decoder",
        "minimum_field_gradient_share": 0.5,
        "ema": (0.996, 0.9999),
        "pretrained_initialization": True,
        "summary": "full minus JEPA minus variance; the teacher is never read in main training",
    },
    "fjvg": {
        "loss": {"jepa_weight": 1.0, "grounding_weight": 0.25, "variance_weight": 0.01},
        "pretrain_epochs": 10,
        "grounding_gradient": "decoder_only",
        "minimum_field_gradient_share": 0.5,
        "ema": (0.996, 0.9999),
        "pretrained_initialization": True,
        "summary": "the full weights, with grounding decoding a stop-gradient latent",
    },
}
STAGES = ("teacher_pretrain", "main", "probe")


def base_config(pde):
    if pde not in PDES:
        raise ValueError(f"Unknown PDE: {pde}")
    source = MANIFEST["datasets"][pde]
    c = {
        "schema": "scope-main/v1",
        "pde": pde,
        "seed": 20260913,
        "resolution": 128,
        "train_records": source["train_records"],
        "test_records": source["test_records"],
        "validation_records": 0,
        "model_spec": asdict(ModelSpec()),
        "batch_size": 32,
        "precision": "bfloat16",
        "pretrain_epochs": 10,
        "main_epochs": 500,
        "normalization": source["normalization"],
        "data_source": {"repo": MANIFEST["repo"], "revision": MANIFEST["revision"]},
        "train_sha256": source["train_cache_sha256"],
        "optimizer": {
            "lr": 0.000125,
            "start_lr": 1e-6,
            "min_lr": 1e-6,
            "weight_decay": 0.05,
            "betas": [0.9, 0.999],
            "grad_clip": 1.0,
            "warmup_epochs": 5,
        },
        "loss": {"jepa_weight": 1.0, "grounding_weight": 0.25, "variance_weight": 0.01},
        "minimum_field_gradient_share": 0.5,
        "ema_start": 0.996,
        "ema_end": 0.9999,
        "checkpoint_every_epochs": 1,
        "milestone_every_epochs": 10,
        "evaluation": {
            "mask_seed": 20261013,
            "mask_replicates": 1,
            "default_protocol": "uniform3",
            "automatic_mid_training_test": False,
        },
        "task_probabilities": [0.4, 0.4, 0.2],
        "families": ["uniform", "low_resolution", "random_lines", "cluster", "block"],
        "family_probabilities": [0.2] * 5,
        "regular_budgets": [500, 1024, 2048, 4096, 8192, 12288, 16384],
        "regular_budget_probabilities": [0.25, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
        "block_and_lines_budgets": [4915, 9830],
    }
    if pde == "ns-bounded":
        c["zero_target_policy"] = {"train_rms_a_u": source["audited_splits"]["train"]["physical_rms_a_u"]}
    return c


def variant_config(pde, variant="full"):
    """One published configuration per (PDE, objective variant).

    Only the loss weights, the grounding gradient path, the pretraining stage,
    the gradient-share floor and the EMA constants differ between variants.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    c = base_config(pde)
    if variant == "full":
        return c
    v = VARIANTS[variant]
    c.update(
        schema="scope-variant/v1",
        variant=variant,
        loss=dict(v["loss"]),
        pretrain_epochs=v["pretrain_epochs"],
        minimum_field_gradient_share=v["minimum_field_gradient_share"],
        ema_start=v["ema"][0],
        ema_end=v["ema"][1],
        pretrained_initialization=v["pretrained_initialization"],
        grounding_gradient=v["grounding_gradient"],
        ablation={"variant": variant, "summary": v["summary"]},
    )
    return c


def load_config(path):
    c = read_json(path)
    expected = variant_config(c.get("pde"), c.get("variant", "full"))
    if c != expected:
        raise ValueError(
            "This release expects an unmodified published configuration. "
            "Add an entry to VARIANTS and document it for a new scientific experiment."
        )
    return c


def science_identity(c):
    # Allow relocation, not changed data, model, schedule or objective.
    return object_hash({k: v for k, v in c.items() if not k.endswith("_path")})


def stage_config(base, stage, data_root):
    if stage not in STAGES:
        raise ValueError(stage)
    c = copy.deepcopy(base)
    folder = Path(data_root).resolve() / c["pde"]
    epochs = {
        "teacher_pretrain": c["pretrain_epochs"],
        "main": c["main_epochs"],
        "probe": c.get("probe_epochs"),
    }[stage]
    if stage == "probe" and not epochs:
        raise ValueError("A probe stage needs probe_epochs")
    c.update(
        stage=stage,
        epochs=epochs,
        train_path=str(folder / "train-f32.npy"),
        train_norms_path=str(folder / "train-physical-norms.npy"),
        train_zero_targets_path=str(folder / "train-zero-targets.npy"),
    )
    if c["pde"] == "ns-bounded":
        c["train_geometry_path"] = str(ASSETS / "train-geometry.npy")
    c["total_draws"] = c["epochs"] * c["train_records"]
    # A variant whose JEPA weight is zero performs no teacher forward at all.
    c["use_jepa"] = stage == "main" and c["loss"]["jepa_weight"] > 0
    c["optimizer"]["warmup_epochs"] = 3 if stage == "teacher_pretrain" else 5
    if c["total_draws"] % c["batch_size"]:
        raise ValueError("Stage must end at a full optimizer batch")
    return c
