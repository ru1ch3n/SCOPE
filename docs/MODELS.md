# Final trained models

Final-weight packaging is in progress. No downloadable asset is claimed until
its checksum, completion metadata and loader verification are recorded.

Scope: every **completed** Full, ablation, SP/DP and operator model in this
campaign. Partial runs, historical snapshots, datasets, optimizer states, RNG
histories and account configuration are excluded. Mirrored copies of one
checkpoint are not separate models.

Weights will be release assets, not Git commits. The manifest identifies PDE,
objective, head size, training view, final epoch, checksum and base dependency.
SP and DP names must not be interchanged.

Private campaign checkpoints and this portable training package have different
schemas. Final-weight exports support inference, not bitwise training resume.
Do not feed an arbitrary legacy pickle to `scope.train --resume`.

Dataset and third-party conditions are not superseded by public repository
visibility. See [THIRD_PARTY.md](../THIRD_PARTY.md).
