"""Selectively download pinned public data and build audited memory-mapped caches."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import numpy as np
from filelock import FileLock
from .config import ASSETS, MANIFEST, PDES
from .io import sha256, object_hash, read_json, write_json


def verify_source(path, item):
    path = Path(path)
    if path.stat().st_size != item["bytes"]:
        raise ValueError(f"Wrong source size: {path.name}")
    if "sha256" in item:
        actual = sha256(path)
        expected = item["sha256"]
    else:
        value = path.read_bytes()  # Only small Git-managed JSON files use SHA-1.
        actual = hashlib.sha1(f"blob {len(value)}\0".encode() + value).hexdigest()
        expected = item["git_blob_sha1"]
    if actual != expected:
        raise ValueError(f"Source checksum mismatch: {path.name}; existing file preserved")


def arrow_batches(folder):
    import pyarrow as pa
    import pyarrow.compute as pc

    state = read_json(Path(folder) / "state.json")
    for item in state["_data_files"]:
        name = item["filename"]
        if PurePosixPath(name).name != name or not name.endswith(".arrow"):
            raise ValueError("Unsafe Arrow filename in state metadata")
        with pa.memory_map(str(Path(folder) / name), "r") as stream:
            for batch in pa.ipc.open_stream(stream):
                ids = batch.column("id").to_numpy()
                col = batch.column("data")
                if isinstance(col, pa.ExtensionArray):
                    col = col.storage
                for size in (2, 128, 128):
                    if col.null_count or not np.all(pc.list_value_length(col).to_numpy() == size):
                        raise ValueError("Unexpected source shape or missing value")
                    col = pc.list_flatten(col)
                fields = col.to_numpy().reshape(len(batch), 2, 128, 128)
                if not np.isfinite(fields).all():
                    raise ValueError("Nonfinite public source data")
                yield ids, fields


def build_cache(pde, split, source_folder, folder):
    source = MANIFEST["datasets"][pde]
    n = source[split + "_records"]
    folder = Path(folder)
    receipt_path = folder / (split + "-receipt.json")
    if receipt_path.exists():
        return verify_cache(pde, split, folder.parent)
    final = folder / (split + "-f32.npy")
    if final.exists():
        raise FileExistsError(f"Unreceipted cache preserved: {final}. Inspect before moving it aside.")
    # Incomplete caches have no receipt and are never consumed by training.
    temporary = final.with_suffix(".npy.partial")
    array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(n, 2, 128, 128))
    norms = np.zeros((n, 2), np.float64)
    zeros = np.zeros((n, 2), np.bool_)
    energy = np.zeros(2, np.float64)
    norm = source["normalization"]
    mean = np.asarray(norm["mean"], np.float64)[None, :, None, None]
    scale = 2 * np.asarray(norm["std"], np.float64)[None, :, None, None]
    cursor = 0
    for ids, fields in arrow_batches(source_folder):
        if cursor + len(ids) > n or not np.array_equal(ids, np.arange(cursor, cursor + len(ids))):
            raise ValueError("Official sample count/order mismatch")
        physical = fields.astype(np.float64, copy=False) * scale + mean
        norms[cursor : cursor + len(ids)] = np.linalg.norm(physical.reshape(len(ids), 2, -1), axis=2)
        zeros[cursor : cursor + len(ids)] = np.abs(physical).max((2, 3)) <= 1e-10
        energy += np.square(physical).sum((0, 2, 3))
        array[cursor : cursor + len(ids)] = fields
        cursor += len(ids)
    if cursor != n:
        raise ValueError(f"Expected {n} records, found {cursor}")
    array.flush()
    del array
    expected = source.get("audited_splits", {}).get(split, {})
    expected_cache = source["train_cache_sha256"] if split == "train" else expected.get("cache_sha256")
    cache_hash = sha256(temporary)
    if expected_cache and cache_hash != expected_cache:
        raise ValueError("Prepared cache differs from the original experiment's data")
    zero_ids = [np.flatnonzero(zeros[:, ch]).tolist() for ch in range(2)]
    if pde == "ns-bounded":
        if zero_ids != expected["zero_target_rows_a_u"]:
            raise ValueError("Source zero-target audit changed")
        geometry = ASSETS / (split + "-geometry.npy")
        if sha256(geometry) != expected["geometry_sha256"]:
            raise ValueError("Known-region geometry checksum mismatch")
    elif zeros.any():
        raise ValueError("Unexpected zero target: no silent omission or epsilon substitution")
    files = {"fields": {"name": final.name, "sha256": cache_hash}}
    for label, suffix, value in (("norms", "physical-norms", norms), ("zeros", "zero-targets", zeros)):
        path = folder / f"{split}-{suffix}.npy"
        with path.with_suffix(".npy.partial").open("wb") as stream:
            np.save(stream, value, allow_pickle=False)
        path.with_suffix(".npy.partial").replace(path)
        digest = sha256(path)
        expected_hash = expected.get("norms_sha256" if label == "norms" else "zero_targets_sha256")
        if expected_hash and digest != expected_hash:
            raise ValueError(f"Source-derived {label} differs from the audited data")
        files[label] = {"name": path.name, "sha256": digest}
    temporary.replace(final)
    receipt = {
        "pde": pde,
        "split": split,
        "records": n,
        "files": files,
        "manifest_sha256": object_hash(MANIFEST),
        "repo": MANIFEST["repo"],
        "revision": MANIFEST["revision"],
        "zero_target_ids_a_u": zero_ids,
        "physical_rms_a_u": np.sqrt(energy / (n * 128**2)).tolist(),
    }
    write_json(receipt_path, receipt)
    return receipt


def verify_cache(pde, split, data_root):
    folder = Path(data_root) / pde
    receipt = read_json(folder / f"{split}-receipt.json")
    if (receipt["pde"], receipt["split"], receipt["manifest_sha256"]) != (pde, split, object_hash(MANIFEST)):
        raise ValueError("Data receipt identity mismatch")
    if receipt["records"] != MANIFEST["datasets"][pde][split + "_records"]:
        raise ValueError("Data receipt count mismatch")
    for item in receipt["files"].values():
        if Path(item["name"]).name != item["name"] or sha256(folder / item["name"]) != item["sha256"]:
            raise ValueError("Local cache checksum mismatch")
    if (
        split == "train"
        and receipt["files"]["fields"]["sha256"] != MANIFEST["datasets"][pde]["train_cache_sha256"]
    ):
        raise ValueError("Training split differs from the released experiment")
    return receipt


def download(pde, split, data_root):
    from huggingface_hub import hf_hub_download

    folder = Path(data_root).resolve() / pde
    folder.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder / ".data.lock"), timeout=1):
        if (folder / f"{split}-receipt.json").exists():
            return verify_cache(pde, split, folder.parent)
        subdir = pde + ("_hf" if split == "train" else "_test_hf")
        selected = [f for f in MANIFEST["datasets"][pde]["files"] if f["path"].split("/")[0] == subdir]
        raw = folder / "source"
        missing = sum(f["bytes"] for f in selected if not (raw / f["path"]).exists())
        reserve = MANIFEST["datasets"][pde][split + "_records"] * 2 * 128**2 * 4 + 2 * 1024**3
        if shutil.disk_usage(folder).free < missing + reserve:
            raise OSError(
                f"Insufficient free storage: need approximately {(missing + reserve) / 1024**3:.1f} GiB"
            )
        for item in selected:
            path = raw / item["path"]
            if not path.exists():
                path = Path(
                    hf_hub_download(
                        repo_id=MANIFEST["repo"],
                        repo_type="dataset",
                        revision=MANIFEST["revision"],
                        filename=item["path"],
                        local_dir=raw,
                        token=False,
                    )
                )
            verify_source(path, item)
            print(json.dumps({"event": "source_verified", "file": item["path"]}), flush=True)
        return build_cache(pde, split, raw / subdir, folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pde", required=True, choices=PDES + ("all",))
    parser.add_argument("--split", choices=("train", "test", "both"), default="both")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    for pde in PDES if args.pde == "all" else (args.pde,):
        for split in ("train", "test") if args.split == "both" else (args.split,):
            receipt = (verify_cache if args.verify_only else download)(pde, split, args.data_root)
            print(
                json.dumps(
                    {"event": "data_ready", "pde": pde, "split": split, "records": receipt["records"]}
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
