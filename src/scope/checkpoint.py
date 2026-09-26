"""Transactional checkpoints with full optimizer/EMA/RNG state and reload sentinels."""

import os
from pathlib import Path
import time
import torch
from .config import science_identity
from .io import code_hashes, read_json, sha256, write_json
from .runtime import atomic_link, rng_state, restore_rng, sentinel


def save(directory, model, optimizer, c, draws, sentinel_input, milestone=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if draws % c["batch_size"] or not 0 <= draws <= c["total_draws"]:
        raise ValueError("Invalid checkpoint position")
    for name, value in model.state_dict().items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"Invalid parameter {name}; previous checkpoint preserved")
    payload = {
        "schema": "scope-checkpoint/v1",
        "config": c,
        "science_sha256": science_identity(c),
        "code_hashes": code_hashes(),
        "draws": draws,
        "epoch": draws / c["train_records"],
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": rng_state(),
        "sentinel_input": sentinel_input.cpu(),
        "sentinel_output": sentinel(model, sentinel_input, next(model.parameters()).device),
        "runtime": {"torch": torch.__version__, "device_type": next(model.parameters()).device.type},
    }
    blob = directory / f"state-{draws:09d}-{time.time_ns()}.pt"
    tmp = blob.with_suffix(".partial")
    with tmp.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, blob)
    record = {
        "file": blob.name,
        "sha256": sha256(blob),
        "bytes": blob.stat().st_size,
        "draws": draws,
        "epoch": payload["epoch"],
        "milestone": bool(milestone),
    }
    previous_index = (
        read_json(directory / "index.json") if (directory / "index.json").exists() else {"history": []}
    )
    history = previous_index["history"] + [record]
    # The atomic pointer is the authoritative commit; aliases are only conveniences.
    # An interrupted alias update cannot make the next resume use a partial file.
    write_json(directory / "index.json", {"current": record, "history": history})
    for alias in ("last.pt", "resume.pt"):
        atomic_link(blob, directory / alias)
        if sha256(directory / alias) != record["sha256"]:
            raise RuntimeError("Checkpoint alias verification failed")
    if milestone:
        atomic_link(blob, directory / f"epoch-{int(record['epoch']):03d}.pt")
    # Retain every 10-epoch milestone, current and previous rolling checkpoint.
    # Only files created and named in this exact run's ledger are candidates.
    retained = {h["file"] for h in history if h["milestone"]} | {h["file"] for h in history[-2:]}
    for item in history[:-2]:
        name = item["file"]
        if name not in retained and Path(name).name == name:
            path = directory / name
            if path.exists():
                path.unlink()
    return record


def current(directory):
    directory = Path(directory)
    record = read_json(directory / "index.json")["current"]
    if Path(record["file"]).name != record["file"]:
        raise ValueError("Unsafe checkpoint filename")
    path = directory / record["file"]
    if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
        raise ValueError("Committed checkpoint checksum mismatch")
    return path, record


def read_payload(path):
    # Checkpoints contain Python/NumPy RNG state; only load files from a trusted run.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "scope-checkpoint/v1":
        raise ValueError("Unsupported checkpoint format; not an automatic legacy importer")
    if payload["code_hashes"] != code_hashes():
        raise ValueError("Checkpoint code identity differs from this release")
    return payload


def load(path, model, optimizer, c, device):
    p = read_payload(path)
    if p["science_sha256"] != science_identity(c):
        raise ValueError("Resume configuration mismatch; refusing a changed experiment")
    if p["draws"] % c["batch_size"] or not 0 <= p["draws"] <= c["total_draws"]:
        raise ValueError("Invalid saved sample position")
    if p["runtime"] != {"torch": torch.__version__, "device_type": device.type}:
        raise ValueError("Strict resume requires the same PyTorch build and device type")
    model.load_state_dict(p["model"], strict=True)
    optimizer.load_state_dict(p["optimizer"])
    result = sentinel(model, p["sentinel_input"], device)
    if not torch.equal(result, p["sentinel_output"]):
        raise ValueError("Reload sentinel mismatch; no optimizer step was taken")
    restore_rng(p["rng"])
    return p
