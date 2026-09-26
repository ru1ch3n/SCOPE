# Changelog

## 0.2.0 — 2026-09-24

The snapshot now covers the whole experimental program, not the main model alone.

### Added

- **Five objective variants** (`full`, `fo`, `fjv`, `fg`, `fjvg`) as a table in
  `config.py`, with one published configuration per (variant, PDE) under
  `configs/variants/`. `losses.objective` serves all of them from one body:
  a term is present exactly when its published weight is nonzero.
  [docs/VARIANTS.md](docs/VARIANTS.md).
- **Decoder probes**, `src/scope/probe.py` and `python -m scope.probe`: freeze a
  completed run's 78,027,776 base parameters, attach a fresh 5M/10M/15M-parameter
  decoder, train only that decoder for 100 epochs, in either the `full` or the
  `uniform500` observation setting. [docs/PROBES.md](docs/PROBES.md).
- **`--protocol full-observation`**: a single condition with both channels fully
  visible, the one on which variants and probes are directly comparable.
- **Measured results** with provenance and explicit gaps:
  [docs/RESULTS.md](docs/RESULTS.md).
- **Deterministic operator baselines**, specified but not bundled:
  [docs/BASELINES.md](docs/BASELINES.md).
- **[docs/OPERATIONS.md](docs/OPERATIONS.md)**: every experiment as a command,
  plus the operating invariants that silently invalidate results when broken.
- `tests/test_variants.py` and `tests/test_probe.py`: 60 new checks covering the
  variant table, the objective dispatch, the grounding gradient path, the probe
  parameter counts, the freeze, and the sparse probe's training condition.

### Changed

- `stage_config` gained a `probe` stage, and `use_jepa` now follows the declared
  JEPA weight rather than the stage alone — so a variant with no JEPA weight
  performs no teacher forward at all.
- `balanced_gradients` takes the field-share floor from the configuration
  (`fo` declares 1.0 and passes no auxiliaries).
- `train.run` derives its stage list: `fo` has no teacher-pretraining stage, a
  probe has only its own.
- Variant runs default to `runs/<variant>-<pde>/`; the reference recipe still
  writes `runs/<pde>/`.
- Evaluation rows carry `variant`, `probe_size` and `probe_observation`.

### Unchanged on purpose

- `configs/<pde>.json` and `variant_config(pde, "full")` are byte-identical to
  the previous `base_config(pde)`. The reference objective is computed term by
  term in the same order as before, and a test asserts it against an explicit
  recomputation.
- The observation sampler, the schedule contract, the gradient guard, the
  checkpoint format, the evaluation contract and the mask seed are untouched.

### Note

This release changes the code identity hash, so runs started under 0.1.0 do not
resume under it. See "Changing this code invalidates existing results" in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## 0.1.0 — 2026-09-20

Initial review snapshot: the SCOPE main model, the five observation families, the
training and evaluation entrypoints, and the protocol/data/reproducibility notes.
