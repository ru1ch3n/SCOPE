# Objective variants

Five training recipes share one model, one dataset, one optimizer, one schedule
and one evaluation contract. They differ only in which terms of the objective are
present and, in one case, in where the grounding gradient is allowed to flow.
`configs/<pde>.json` is the reference recipe; `configs/variants/<variant>-<pde>.json`
are the four ablations. `tests/test_variants.py` asserts, for every PDE, that a
variant configuration differs from the reference only in the keys listed below.

| id | teacher-pretraining objective (10 epochs) | main objective (500 epochs) | grounding decodes | teacher read in main | EMA updated |
|---|---|---|---|---|---|
| `full` | `L_F(dec(z_F))` + 0.01·`L_V(z_F)` | `L_F` + 1.0·`L_J` + 0.25·`L_G` + 0.01·`L_V` | `z_F`; gradient to encoder and decoder | yes | both stages |
| `fjv` | `L_F` + 0.01·`L_V` | `L_F` + 1.0·`L_J` + 0.01·`L_V` | — | yes | both stages |
| `fg` | `L_F` only | `L_F` + 0.25·`L_G` | `z_F`; gradient to encoder and decoder | **no** | pretraining only, then frozen and never read |
| `fjvg` | `L_F` + 0.01·`L_V` | `L_F` + 1.0·`L_J` + 0.25·`L_G` + 0.01·`L_V` | `z_F.detach()`; gradient to the **decoder only** | yes | both stages |
| `fo` | *no such stage* | `L_F` only, from random initialization | — | no teacher is ever read | never |

`z_F = encoder(full_pair_view(target))` is the online encoder on the fully visible
pair. `L_V` always acts on `z_F`, never on the predicted latent `z_M`; `L_J`
compares `z_M` with the stop-gradient EMA-teacher latent on the same full view.
The definitions themselves are in [PROTOCOL.md](PROTOCOL.md) and are unchanged.

## What each variant isolates

- `fjv` removes grounding and nothing else, so `full` − `fjv` is the effect of
  grounding.
- `fjvg` keeps the full objective, the full weights and the reported grounding
  value byte for byte, and changes only the gradient path: grounding decodes a
  stop-gradient copy of `z_F`, so it trains the decoder but never pulls the
  encoder. `full` − `fjvg` is therefore the effect of grounding *on the encoder*,
  with the decoder's grounding signal held fixed.
- `fg` removes the JEPA and variance terms, so no teacher forward happens at all
  during main training. Its pretraining stage is also the plain field
  autoencoder with no variance term — the one place `fg` differs from `full`
  outside main training.
- `fo` is deliberately **not** a single-term ablation. It removes the teacher,
  the teacher-pretraining stage and every auxiliary term together, and trains the
  500 main epochs from a fresh random initialization. Do not describe it as
  "SCOPE without the JEPA loss".

## The keys a variant is allowed to change

`loss`, `grounding_gradient`, `pretrain_epochs`, `minimum_field_gradient_share`,
`ema_start`, `ema_end`, `pretrained_initialization`, plus the naming keys
`schema`, `variant` and `ablation`. Everything else — seed 20260913, `model_spec`,
AdamW settings, batch 32, 500 main epochs, normalization, the dataset identity,
the five observation families, the task and budget schedule, the evaluation mask
seed and condition list — is equal to the reference configuration, and the test
suite fails if it is not.

## The gradient guard still applies

Every auxiliary is multiplied by the per-step factor `alpha <= 1` described in
[PROTOCOL.md](PROTOCOL.md) before it reaches any parameter, so the weights above
are **upper bounds on the effective contribution**, not realized ratios. The
realized factor is logged every step as `aux_scale`, with a separate
`grad_norm_jepa` / `grad_norm_grounding` / `grad_norm_variance`. `fo` declares
`minimum_field_gradient_share = 1.0` and passes no auxiliaries at all, so the
guard is a no-op for it.

## Running one

```bash
.venv/bin/python -m scope.train --config configs/variants/fjv-poisson.json
```

Output goes to `runs/fjv-poisson/` (the reference recipe still writes
`runs/poisson/`). Evaluation is identical for every variant:

```bash
.venv/bin/python -m scope.evaluate --run runs/fjv-poisson --protocol uniform3
```

`fo` has no `teacher_pretrain/` directory because it has no such stage; its run
directory contains `main/` only. See [OPERATIONS.md](OPERATIONS.md) for the full
sequence including probes.
