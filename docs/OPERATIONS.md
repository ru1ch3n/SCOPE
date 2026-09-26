# Operations

Every experiment in the paper, as the exact command that produces it, in the
order the dependencies require. Nothing here needs a scheduler, an account or a
site-specific path: one command is one GPU job.

## 0. Once per PDE

```bash
python3 tools/setup.py --pde poisson
```

Creates `.venv`, installs the pinned dependencies, downloads the released files
at a fixed revision, verifies checksums and builds the memory-mapped caches. See
[DATA.md](DATA.md) for sizes. `--env-only` skips the download; `--cpu --env-only`
builds a CPU-only environment for the software checks.

## 1. Train a variant

```bash
.venv/bin/python -m scope.train --config configs/poisson.json                  # full
.venv/bin/python -m scope.train --config configs/variants/fjv-poisson.json     # an ablation
```

10 teacher-pretraining epochs then 500 main epochs, one GPU, no intermediate
test evaluation. `fo` skips the pretraining stage and runs the 500 main epochs
from random initialization. Output: `runs/poisson/`, `runs/fjv-poisson/`, and so
on. The five variants are documented in [VARIANTS.md](VARIANTS.md).

Resume after any interruption, keeping the declared learning-rate horizon:

```bash
.venv/bin/python -m scope.train --config configs/variants/fjv-poisson.json --resume
```

## 2. Evaluate a variant

```bash
# the two primary sparse rows: 500 / 16384 = 3.0518 % visible
.venv/bin/python -m scope.evaluate --run runs/poisson --protocol uniform3

# all 75 training-supported conditions (77 on the cylinder problem)
.venv/bin/python -m scope.evaluate --run runs/poisson --protocol full

# every grid point of both channels visible: the cross-variant comparable number
.venv/bin/python -m scope.evaluate --run runs/poisson --protocol full-observation

# cylinder problem only: known disk + 1 % Bernoulli fluid interior
.venv/bin/python -m scope.evaluate --run runs/ns-bounded --protocol benchmark
```

Evaluation reads the planned last main checkpoint. There is no validation split,
no test-based early stopping and no best-checkpoint selection; an intermediate
checkpoint requires `--allow-partial` and is labelled incomplete.

## 3. Train a decoder probe

A probe needs a **completed** base run.

```bash
# how much the frozen encoder latent holds
.venv/bin/python -m scope.probe --base runs/fjv-poisson --size 10m --observation full

# the same question on the real sparse path
.venv/bin/python -m scope.probe --base runs/fjv-poisson --size 10m --observation uniform500
```

100 epochs, decoder only, sizes `5m` / `10m` / `15m`. Output:
`runs/probes/<size>-<observation>-<base variant>-<pde>/`. See
[PROBES.md](PROBES.md).

## 4. Evaluate a probe

```bash
.venv/bin/python -m scope.evaluate --run runs/probes/10m-full-fjv-poisson --protocol full-observation
.venv/bin/python -m scope.evaluate --run runs/probes/10m-full-fjv-poisson --protocol uniform3
```

Running a `full` probe under `uniform3` is a deliberate out-of-distribution
transfer test and must be labelled as one.

## 5. Deterministic operator baselines

Not bundled; specified completely in [BASELINES.md](BASELINES.md), including the
batch-125 requirement, the CNO reload tolerance, the DeepONet loader finding and
the Transolver memory limit.

## Software checks, no data and no GPU

```bash
python3 tools/setup.py --cpu --env-only
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m scope.train --smoke
.venv/bin/python -m scope.evaluate --run runs/smoke --device cpu
```

The smoke run is a tiny synthetic end-to-end exercise of the same code path. It
is a software check, never a research result.

---

# Operating invariants

These are properties of this code, learned by running it. They are stated here
because breaking one of them silently invalidates results rather than failing
loudly.

**One run, one GPU, one job.** Nothing in this repository coordinates multiple
processes over one run directory beyond a single advisory lock. A run directory
must have exactly one live writer.

**A checkpoint is bound to its GPU type.** Strict resume replays a sentinel
forward pass and requires bitwise-identical output, so a checkpoint written on
one GPU generation will not resume on another. A run must spend its whole life
on one kind of card. Moving a partly trained run to a different GPU means
restarting it, not resuming it.

**A checkpoint is bound to its configuration and to this code.** Both the
configuration identity hash and the hash of every `.py`/`.json`/`.npy` file under
`src/scope/` are recorded in the checkpoint and re-checked on load. Editing any
of those files invalidates every checkpoint and every completed result that
referenced them — including results already written. If you need to change
behaviour, start a new run directory with the new code; do not patch code under a
run that is in progress or already finished.

**A run directory is not a scratch space.** Renaming, moving or deleting a run
that something else may still be writing to loses that work. Check for a live
writer first.

**Resume, do not restart, after an interruption.** The checkpoint carries the
model, the EMA teacher, the optimizer, every RNG stream and the exact sample
cursor, and the learning-rate horizon stays at its declared 10/500 epochs. A
clean interruption saves at an optimizer boundary; a forced kill resumes from the
last committed checkpoint.

**Read final numbers from `epoch-metrics.jsonl`, not from the progress stream.**
The progress rows are optimizer-batch windows and can differ from the full
physical epoch by around 1 %. The last row of the epoch file is the reported
number.

**Do not compare across tables.** Probe training loss is full-observation
reconstruction; variant training loss is sparse-observation reconstruction over
the masked conditions; the deterministic baselines are a third task setup. The
only cross-experiment comparable numbers are the test-time rows under a named
protocol.

**Never present a partial-epoch number as a result.** An unfinished run has no
row in [RESULTS.md](RESULTS.md); it is listed as running.
