# Results and comparison scope

The manuscript distinguishes primary recovery (Full and SP), backbone-objective
ablations, and decoder-transfer diagnostics. The public release does not carry
forward outdated training-status tables or mix operator training losses with
test errors. Use the accompanying manuscript for current numerical tables.

- Forward's primary target is a dense `u`; inverse's is `a`.
- Relative L2 is the per-record **unsquared** norm ratio in physical units,
  averaged over the specified population and expressed in percent.
- Darcy inverse uses pixelwise BER, not continuous relative L2.
- `uniform3` is 500 of 16,384 points (3.0518%). Cylinder and zero-target
  conventions must be stated explicitly.
- Per-record SD is not uncertainty across training seeds.
- Literature values come from the cited papers, not matched reruns by this code.
- Native-to-SP adds capacity and training. Matched-size DP-to-SP is a distinct,
  more controlled comparison.
- FO omits pretraining as well as auxiliaries, so it is not a single-term ablation.

Decoder transfer supports readout compatibility under the measured protocol;
it does not prove encoder/predictor distribution equality, nor identify capacity
or information content as the sole bottleneck.

Only verified final checkpoints are eligible for release. Partial FJVG and
withdrawn probe runs are excluded. The preserved scientific code and recipes
have not been changed to improve a score. See [MODELS.md](MODELS.md).
