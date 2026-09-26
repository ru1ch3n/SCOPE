# Exact main-model protocol

## Observations, not a second hiding mask

`mask=1` means visible. Unobserved normalized values are exactly zero, while the
mask channel distinguishes a hidden entry from a visible zero. Only the visible
values and their two masks enter inference. The full pair enters the training
target/grounding paths, never the sparse inference signature.

Forward observes `a`; inverse observes `u`; joint observes both with separate
per-channel seeds. Both `a` and `u` are supervised on the complete grid in all
three tasks. There is no observed-value copying after decoding.

Every five batches contain one of each family. For each family, five occurrences
contain two forward, two inverse, and one joint batch. The point counts for
uniform/grid/cluster are `[500, 1024, 2048, 4096, 8192, 12288, 16384]`; the 500-point
slot occurs twice per eight-slot shuffled cycle, each other slot once. Block and
lines alternate shuffled 4915/9830-point slots (rounded 30%/60%). Conditions are
homogeneous within a batch; the physical samples and actual point locations are
different. Seeds depend on the original draw and record indices, including after
resume. An epoch is a full permutation of the training records. Batches may cross
a physical-epoch boundary; there is no dropped tail or oversampled tail.

Low resolution samples original pixels on Cartesian grids of shapes
20×25, 32×32, 32×64, 64×64, 64×128, 96×128, and 128×128, with an axis swap and
subcell phase. It is not interpolation of a dense hidden field. Clusters use
2–4 equally weighted broad Gaussian centers, separated by at least 0.30 domain
width, with widths 0.12–0.20. Lines are random whole rows/columns plus at most one
partial line for the exact count. Blocks use a rectangular distance ranking.
Grids and clusters are not guaranteed to be nested across budgets.

## Physical loss

The released arrays obey `physical = normalized * (2 * std) + mean` using the
fixed upstream metadata. For a nonzero physical target `y`, the per-field error is

```text
relative_L2 = ||predicted_physical - y||_2 / ||y||_2
L_field = mean over records and the two physical channels of relative_L2
```

This is unsquared relative L2, not normalized MSE or a root of a dataset-averaged
MSE. Training logs use fractions; evaluation summaries multiply by 100.

Pretraining optimizes the full-pair encoder/decoder using `L_field + 0.01 L_var`.
The predictor and observation-condition linear layer are frozen at their seeded
random initialization. Main training starts from the pretrained encoder/decoder;
the predictor/condition are unchanged. The optimizer is reset, and the EMA encoder
is reset from the pretrained online encoder. It is not a second scratch student
and is not a frozen-teacher method.

The main objective before the dynamic gradient adjustment is

```text
L_field + 1.0 L_JEPA + 0.25 L_grounding + 0.01 L_var
```

- `L_JEPA`: mean squared difference of the predicted 256×128 latent and the
  stop-gradient full-field EMA-encoder latent.
- `L_grounding`: the same physical-field loss after decoding the online encoder's
  full-field latent; it grounds the latent coordinates in field recovery.
- `L_var`: mean of `relu(0.1 - std(full_latent, batch, correction=0))^2`, with
  latent tokens/channels flattened into coordinates.

Let `g_f` be the gradient of `L_field`, and `g_i` the gradient of each already
weighted auxiliary. Let the subscript `enc` restrict a vector to online-encoder
parameters. The implementation chooses

```text
alpha = min(1, ||g_f|| / sum_i ||g_i||,
               ||g_f,enc|| / sum_i ||g_i,enc||)
gradient = g_f + alpha * sum_i g_i
```

Zero denominators impose no cap; a small roundoff margin is applied only when
reducing `alpha`. Each component is measured separately, so cancellation between
auxiliary gradients cannot inflate the reported field share. The guarantee is
about these **pre-AdamW Euclidean norm sums**, not scalar-loss proportions, the
norm of the summed gradient, or final parameter displacement. Global gradient
clipping uses norm 1.0; AdamW then steps; the EMA target then updates.

AdamW uses `(beta1, beta2)=(0.9,0.999)`, epsilon `1e-8`. Biases, position embeddings,
and one-dimensional parameters have no weight decay. Warmup goes from `1e-6` to
`1.25e-4`, followed by cosine decay to `1e-6`; the declared horizon remains
10/500 epochs when a run is interrupted. EMA momentum follows cosine from 0.996
to 0.9999 over each stage.

## Internal-cylinder NS

`ns-bounded` uses the released per-record closed disk, not the exterior border of
the square. Both channels are known in that disk. Values come from the released
fields; they are **not forcibly set to zero**. The active task channel has its
additional sparse mask unioned with the disk. Grid points overlapping the disk
are not replaced by off-grid samples. Therefore nominal whole-grid budget and
actual extra-fluid point count differ; evaluation records both.

76 training response fields and test response fields with IDs **39, 163, 710**
are zero in the float64 source audit. They are all retained. For these training
targets only, the loss uses physical absolute RMS divided by the training-split
channel RMS. The fixed training RMS values are 2.070225306780491 (`a`) and
3.3495987840120813 (`u`). Nonzero targets use source-audited physical L2 norms.
The zero flag never depends on a prediction.

For testing, relative L2 is undefined for a zero target. Every record has absolute
RMSE; relative-error means include only explicitly marked defined targets and
report the undefined IDs/count. The full-domain `u` mean is thus labelled as a
997-defined-target mean, not an unqualified 1000-record relative error. Separate
fluid, obstacle-band, and fluid-interior metrics are also returned.

## Evaluation contract

No validation split, test-based early stopping, or best-test checkpoint selection
is implemented. The planned last main checkpoint is used. Intermediate evaluation
requires `--allow-partial` and remains labelled incomplete. `uniform3` uses
500 uniform points; `benchmark` switches the cylinder case to known disk + 1%
Bernoulli fluid sampling. `full` uses 75 task/family/budget conditions (77 for the
cylinder problem including its two benchmark conditions). Mask seed: 20261013.

Errors are per-record first, then averaged. Darcy inverse classifies physical
coefficients above 7.5 as 12, otherwise 3; BER is the fraction of misclassified
pixels. Pixel accuracy is `100% - BER%`, not whole-sample success rate. Continuous
Darcy coefficient relative L2 is retained as a separate diagnostic, never renamed
BER. Test masks and per-record errors are fingerprinted, and model state is checked
for changes across evaluation. One released split and one mask realization are not
a multi-seed uncertainty estimate.
