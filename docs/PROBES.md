# Decoder probes

A probe asks how much field information a frozen latent still carries, using a
decoder that was never trained beside it. Encoder, predictor, observation
condition and EMA teacher are loaded from a completed run and frozen —
**78,027,776 parameters**, none of which receives a gradient — and `model.decoder`
is replaced by a fresh `ProbeDecoder`, which is the only module that is trained.

The probe decoder is the release decoder's mechanism at a larger width: a
per-token MLP producing 8×8 field tiles, reassembled into the field, plus a
residual 3×3 convolutional refinement. There is no observed-field skip
connection, so everything the probe outputs passed through the frozen latent.
Only `Linear`, `Conv2d` and `GELU` are used, so a probe stays inside
`torch.use_deterministic_algorithms(True)`.

| size | hidden | MLP layers | conv width | conv layers | trainable parameters |
|---|---:|---:|---:|---:|---:|
| `orig` | 256 | 3 | 64 | 3 | 171,010 (the release decoder's own topology) |
| `1m` | 512 | 4 | 112 | 5 | 1,000,194 |
| `5m` | 1248 | 4 | 240 | 5 | 5,003,170 |
| `10m` | 1760 | 4 | 352 | 5 | 10,010,658 |
| `15m` | 2168 | 4 | 432 | 5 | 15,018,218 |

The tables are fixed constants, never searched at run time, so a probe's
configuration — and therefore its identity hash — is a pure function of the size
name. Each reported size is within 0.2 % of its nominal count; the counts are
pinned by a closed-form formula and by `tests/test_probe.py`. Sizes `5m`, `10m`
and `15m` are the reported ones.

## Two observation settings, two different experiments

| `--observation` | what the decoder is trained on | what it answers |
|---|---|---|
| `full` | `decoder(encoder(full_pair_view(y)))` | how much of the field survives in the **encoder** latent of a fully visible pair — the path the grounding term trains |
| `uniform500` | `decoder(predictor(encoder(observation_view(y, task, 500, "uniform", seed))))`, latent detached | how much survives in the **predictor** latent on the task the model is actually for |

Both train for **100 epochs**, 5 warmup epochs, with the same optimizer, batch
size, seed and data as the base run. Neither carries any auxiliary term: the loss
is the base run's own field objective and nothing else, `minimum_field_gradient_share`
is 1.0, and the EMA teacher is never updated.

A `uniform500` probe trains on one fixed condition — uniform family, 500 visible
points, forward and inverse alternating by optimizer batch. The condition is a
pure function of the batch index, so a resumed probe repeats the same schedule
exactly. `joint` is never trained because it is never evaluated.

## Evaluating a probe

```bash
# how much the frozen latent holds, all grid points of both channels visible
.venv/bin/python -m scope.evaluate --run runs/probes/10m-full-fjv-poisson --protocol full-observation

# the same probe on the real sparse task
.venv/bin/python -m scope.evaluate --run runs/probes/10m-full-fjv-poisson --protocol uniform3
```

`full-observation` is a single condition, `("joint", "full", 16384)`: both
channels fully visible, so every variant and every probe answers the same
question and the numbers are comparable across them. The base run itself can be
evaluated on it too, which is how `forward` (the run's own encoder → predictor →
decoder path) and `dec(enc)` (decoder of the encoder latent, the path a probe
replaces) are obtained.

**Evaluating a `full` probe sparsely is out-of-distribution by construction.**
That decoder only ever saw encoder latents of fully visible pairs; feeding it
predictor output on a 3 % observation is a deliberate transfer test, not the
setting it was trained for. It is worth running — the result is one of the
sharper findings in [RESULTS.md](RESULTS.md) — but it must be labelled, and it is
not the same experiment as a `uniform500` probe.

## Safety properties the code enforces

- The base is checked frozen before every update, by inspecting
  `requires_grad` on `encoder.`, `predictor.`, `observation_condition.` and
  `target_encoder.` directly. For the `full` setting the latent is built inside
  the graph, so a thawed base would additionally show up as
  `latent.requires_grad`; for `uniform500` the latent is built under `no_grad`,
  which removes that signal, so the explicit check is what carries the
  guarantee.
- A probe run is pinned to one base checkpoint by sha256 and to that base's
  science identity. If either changes, the probe refuses to start.
- The probe refuses to run if any loss weight is nonzero.
- The trainable parameter count must equal the declared count for the size.
- A base run must be **complete** (all 500 main epochs) before it can be probed.

## Running one

```bash
.venv/bin/python -m scope.probe --base runs/fjv-poisson --size 10m --observation full
.venv/bin/python -m scope.probe --base runs/poisson --size 15m --observation uniform500
```

Output defaults to `runs/probes/<size>-<observation>-<base variant>-<pde>/`. The
run directory has a single `probe/` stage; `--resume` behaves exactly as it does
for training. A probe checkpoint stores the frozen base as well as the trained
decoder, so it is roughly the size of a full model checkpoint.
