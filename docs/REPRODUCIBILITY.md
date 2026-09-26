# Reproduction and verification scope

The scientific model, observation sampler, objectives, gradient guard, optimizer,
learning-rate schedule, and EMA calculations were extracted from the main training
implementation. Private paths and provider-specific launch/backup tools were
replaced by portable command-line entrypoints. The unrelated affine/projector
prototype and the legacy MSE convenience objective are absent.

## What the commands reproduce

The twenty-five configurations (five PDEs x five objective variants) prescribe a
new 10+500-epoch run each, and `scope.probe` prescribes a 100-epoch decoder run on
a completed one. They do not imply that executing a command instantly reproduces a
numerical table. Final pretrained-weight availability and schema compatibility
are tracked in [MODELS.md](MODELS.md); large weights are not Git objects.
Comparison caveats are in [RESULTS.md](RESULTS.md). Test-set masks are the documented SCOPE masks, not
copies of every baseline's official mask file. Literature values must not be
described as scores obtained by these scripts. The deterministic operator
baselines in that file were produced by third-party architectures that are
specified in [BASELINES.md](BASELINES.md) but are not bundled here, so those rows
are not reproducible from this repository alone.

Training uses PyTorch 2.12.1, one GPU, BF16, deterministic algorithms, fixed random
seeds, fused CUDA AdamW, and full batches. A GPU with **48 GiB VRAM** is a practical
reference class; actual memory/time depend on device/runtime and must be measured.
Changing batch size to fit a smaller GPU is not automatically equivalent, because
the latent variance regularizer and homogeneous condition schedule depend on the
batch. No distributed or gradient-accumulation substitute is silently enabled.

## Checkpoints and interruption

- Each completed physical-epoch boundary commits a rolling full-state checkpoint.
  A batch may cross an epoch boundary, so a rolling checkpoint can be slightly
  beyond an integer epoch; 10-epoch milestones and the final budget are exact.
- Every 10-epoch milestone, plus the current and previous rolling checkpoints, is
  retained. Stale nonmilestone blobs created by this run are pruned only after a
  successful commit. No external checkpoint/data directory is cleaned.
- The atomic `index.json` pointer and blob checksum are authoritative. `last.pt`
  and `resume.pt` are verified hard-link/copy conveniences. Interrupted alias
  creation does not invalidate a previously committed state.
- Reload checks code/config/data identity, model keys/shapes, the exact sample
  cursor and a deterministic inference sentinel, then restores optimizer and RNG.
  The full original stage horizon is retained, not restarted from a short schedule.
- Strict training resume requires the same PyTorch build and device type; CUDA
  hardware/backend changes can still fail the bitwise sentinel and should be
  audited, not bypassed. Evaluation on another device records that it was not a
  same-runtime bitwise-sentinel verification.
- This release's checkpoints have their own schema. Private legacy checkpoint
  import is **not** automatic. Do not load an arbitrary downloaded pickle; full
  checkpoints contain Python/NumPy RNG state and must come from a trusted run.

## Checks available to reviewers

`pytest` checks visible-point semantics, exact mask counts and balanced conditions,
no hidden-target leakage into the observation tensor, model size/shape, physical
loss/BER, zero-target handling, field-gradient share, full-state resume, and a tiny
synthetic end-to-end train/evaluate path. The smoke test uses a reduced model and
synthetic fields and is expressly not a PDE benchmark experiment.

For the variants and probes specifically, the suite asserts: that every published
variant configuration differs from the reference only in the declared keys, for
every PDE; that the auxiliaries present in a stage are exactly the terms with a
nonzero weight; that a variant with no JEPA weight never calls the teacher
encoder; that `fjvg` reports the same grounding value as `full` while its
gradient reaches the decoder and neither the encoder nor the predictor; that the
variance term is measured on the online full-view latent and carries no gradient
through the predictor; that `fo` never touches the fully visible pair; that the
reference objective is still computed exactly as it was before the variant
machinery existed; that every probe size has its declared parameter count and
that attaching one freezes all 78,027,776 base parameters; that a probe's
gradient reaches only the decoder and that the objective refuses an unfrozen
base in **both** observation settings; and that a sparse probe's training
condition is uniform / 500 with forward and inverse alternating by batch index.

The release is additionally checked against the original scientific functions
before publishing. Local CPU tests do not establish full 500-epoch CUDA execution
or convergence, and those are not asserted here. The production GPU training
jobs are independent of preparing this portable review snapshot.

Release preparation checks passed under PyTorch 2.12.1 CPU / NumPy 2.2.6:
23 automated tests; the full 40,040,834-parameter model's forward shape/finiteness;
225 task/family/budget/seed observation comparisons (both plain and cylinder
variants) and 750 batch-condition comparisons against the original implementation;
and bitwise-equal initialization, loss, optimizer and EMA next states for all five
PDEs in both stages using a reduced-width CPU fixture. The complete public
1,000-record cylinder test split was also downloaded, converted, and verified
against its field/norm/zero-flag checksums. The large 50,000-record training
downloads and full CUDA training were not rerun as part of this packaging audit.

## Changing this code invalidates existing results

Both the configuration identity hash and the hash of every `.py`, `.json` and
`.npy` file under `src/scope/` are written into each checkpoint and re-checked on
load, and a completed run's `run.json` records the same code hashes. Editing any
of those files therefore invalidates every checkpoint **and every completed
result that referenced them**, retroactively. To change behaviour, start a new
run directory with the new code; never patch code under a run that is in progress
or already finished. Adding `probe.py` and extending `config.py`, `losses.py`,
`dataset.py`, `train.py`, `evaluate.py` and `optimization.py` in this release
changed the code identity, so runs started under the previous snapshot do not
resume under this one. See [OPERATIONS.md](OPERATIONS.md) for the rest of the
operating invariants.

## Separate public and anonymous distributions

This repository is public and identifies the authors. It is not the anonymous
review link. The private review backing repository and its anonymous mirror are
preserved separately. Do not put this public author-identifying URL into a
double-blind submission when the venue requires anonymous code links.
Third-party dataset/software attribution remains intact in both distributions.

No telemetry, credentials, external experiment tracker, cloud provisioning, or
account authentication is required by the training/evaluation code.
