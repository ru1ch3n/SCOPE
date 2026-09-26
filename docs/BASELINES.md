# Deterministic operator baselines

Four deterministic neural-operator baselines are trained on the **same masked
observation task** as SCOPE: same released splits, same normalization, same
five observation families and budget schedule, same physical relative-L2 field
objective, same zero-target handling on the cylinder problem. They have no
teacher, no latent objective and no auxiliary terms — they map the four-channel
observation directly to the two physical fields.

**The baseline implementations are not bundled in this repository.** They are
third-party architectures used unmodified apart from the input/output adapter
described below, and each carries its own upstream license. This document is the
complete specification of how they were configured and trained, so a reviewer can
reproduce them from the upstream sources; the SCOPE-side pieces they consume —
the dataset, the observation families, the field objective, the evaluation
contract — are all in this repository and are the same objects SCOPE uses.

| model | trainable parameters |
|---|---:|
| FNO | 5,029,946 |
| DeepONet | 4,977,354 |
| CNO | 4,774,776 |
| Transolver | 4,792,482 |

All four are sized to approximately 5M parameters, against SCOPE's 40,040,834.
That is a deliberate asymmetry and the paper must state it: the baselines are
the standard ~5M configurations of their families, not parameter-matched to
SCOPE.

## Training recipe

| | value |
|---|---|
| Epochs | 100 full physical epochs |
| Batch size | **125** |
| Optimizer | AdamW, the same settings as SCOPE, cosine horizon set to 100 epochs |
| Precision | BF16 autocast |
| Input | `[a_visible, mask_a, u_visible, mask_u]`, 128×128 |
| Output | both physical fields, 128×128 |
| Objective | the same per-record unsquared physical relative L2 used by SCOPE |
| Checkpointing | one per physical epoch, resume from the last committed one |

Batch 125 is not a rounding of 128. The training splits have 50,000 records and
the milestone accounting uses 500,000 draws; 125 is the largest divisor of both
below 128, so every epoch ends on a full optimizer batch and no partial tail is
dropped or oversampled. Using 128 would break that contract.

## Two model-specific findings worth recording

**CNO cannot pass a bitwise reload gate on its optimizer state.** The training
loop checks reproducibility by saving, redoing the same update and comparing.
CNO's activation calls `F.interpolate(mode="bicubic", antialias=True)` twice; its
CUDA backward accumulates with `atomicAdd`, for which PyTorch has no
deterministic implementation, so `torch.use_deterministic_algorithms(True)`
raises rather than fixing it. Measured twice independently: model state agrees to
1.52e-6 and 1.53e-6 (inside a 2e-6 tolerance), optimizer state to 1.44e-5 and
1.32e-5 (outside it). The large relative deviations are all on near-zero
`exp_avg` entries with absolute values around 1e-7 to 1e-6, and the worst tensor
differs between runs — the signature of near-cancellation, not of a bug. The
optimizer-state tolerance was therefore relaxed to 1e-4 for CNO only; the model
state and the next-update metric comparison stay at 2e-6. Transolver, FNO and
DeepONet reproduce bitwise and keep the original tolerance.

**DeepONet is loader-bound, not compute-bound.** Measured GPU utilization 3.8–5.2 %
with a median of 0, peak memory 430 MB, 36 s per epoch against roughly 5 s of
actual GPU work. Raising the batch size makes this *worse*, not better: the
number of records per epoch is unchanged, so the loader cost is unchanged, while
larger kernels finish sooner and the card idles longer. Raising loader workers
from 4 to 10 (and CPU allocation accordingly) halved the epoch time to 23.5 s and
lifted utilization to 12.5 %. The loader is a fixed-order sampler with a fixed
generator, so the batch contents and order do not depend on the worker count.
Running several DeepONet tasks on one card is what actually raises utilization.

**Transolver does not fit a 96 GB card at batch 125.** Its memory scales with
batch × tokens (125 × 16384) and is independent of the dataset size, so this
holds on every PDE. It needs a card with more memory; a smaller batch would
change the optimization trajectory and break the epoch/milestone contract above.

## Hardware is mixed, and the tables say so

The baseline grid was filled on two GPU generations as capacity allowed. Every
run records its hardware in its own schema string, so no cell is ambiguous, and
[RESULTS.md](RESULTS.md) tags each cell. **The baseline table is not
single-hardware**; the paper must state that, and cells must not be compared
across hardware as if the difference were purely the model. Hardware affects
throughput, not the objective, the data or the metric — but it is recorded rather
than assumed.
