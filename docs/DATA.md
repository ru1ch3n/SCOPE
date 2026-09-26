# Data source and storage

The sole field-data source is the public **FunDPS processed DiffusionPDE** release:

- Repository: `jcy20/DiffusionPDE-normalized`
- Revision: `3ededc4f2d8a1592a52a7a865cb903dab7647820`
- [Dataset card and license](https://huggingface.co/datasets/jcy20/DiffusionPDE-normalized)

Do not download the automatically converted default dataset-viewer split. We read
the original `<pde>_hf` and `<pde>_test_hf` Arrow directories at the pinned revision,
in their `state.json` order, and check record IDs. Download manifests in
`src/scope/assets/` contain LFS SHA-256 or Git-blob SHA-1 identities for every source
file. Float32 training caches must also match the original experiment's SHA-256.
No training/test repartitioning, fresh simulation, or normalization fit is done.

Download more data into an existing environment:

```bash
.venv/bin/python -m scope.data --pde darcy --split both
```

Verify existing caches without downloading:

```bash
.venv/bin/python -m scope.data --pde darcy --verify-only
```

Use `--data-root /path/to/storage` consistently in download, train, and evaluate.
Only `--pde all` downloads all five PDEs. Downloading a test split does not cause
the trainer to read it. The downloader audits sources/caches in bounded Arrow
batches instead of materializing all 50,000 records in memory.

For a 50,000-record PDE, the float32 training cache alone is **6.10 GiB** and the
upstream float64 Arrow data is roughly twice that. Test caches add about 0.13 GiB
(1,000 records) or 1.22 GiB (10,000 records), plus their sources. Allow **25 GiB for
one static PDE's sources and caches**, and additional checkpoint space. A full
500-epoch run with 10-epoch full-state milestones needs roughly **32–36 GiB of
checkpoint space**. Start with at least **65 GiB free per static PDE** for data,
checkpoints, environment, logs, and working space; this is a conservative estimate,
not a measured peak for every platform. The 14,000-record cylinder problem is
smaller. The setup does not automatically delete data to recover space.

## Cylinder metadata

The small `train-geometry.npy` and `test-geometry.npy` files contain only public
`(cx,cy,r)` coordinates, not field tensors. They preserve the benchmark's original
integer-grid axis convention. Their checksums and original public metadata-file
identifiers/source order are included. This avoids reconstructing geometry from
hidden target fields or requiring a private file server.

The existing geometry audit established pinned source order and all-row
stencil-core consistency. Original full raw-file comparisons covered only part
of the data because the source host imposed download quotas; we do not claim a
full raw-file equality audit for every source group. The exact processed field
files used for training are independently checksum-verified. Attribution and
terms for these metadata are in [THIRD_PARTY.md](../THIRD_PARTY.md).

Existing files failing checksum checks are preserved and cause a clear error.
Inspect them before moving them aside and re-downloading. Hugging Face resumes
partial downloads; only a complete cache with an atomic receipt is accepted.
