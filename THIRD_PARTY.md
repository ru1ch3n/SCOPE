# Third-party acknowledgements and terms

## Benchmark fields and cylinder metadata

Source: the **DiffusionPDE** benchmark and its **FunDPS** processed release,
[jcy20/DiffusionPDE-normalized](https://huggingface.co/datasets/jcy20/DiffusionPDE-normalized),
revision `3ededc4f2d8a1592a52a7a865cb903dab7647820`.

The dataset card attributes the underlying fields to DiffusionPDE and the
normalization/processing to FunDPS. These are external works, not anonymous
contributions of this repository. Please credit both works when using the data.

The dataset card specifies **Creative Commons Attribution–NonCommercial–ShareAlike
4.0 International**:
[license summary](https://creativecommons.org/licenses/by-nc-sa/4.0/) ·
[legal text](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en).

The field tensors are downloaded from their public source, not redistributed in
Git. The included small cylinder metadata arrays are derived from the original
public `cx`, `cy`, and `r` arrays by concatenating them in the processed dataset's
row order. Those derived arrays retain the same CC BY-NC-SA 4.0 terms. Their
public source identifiers, checksums, and concatenation order are in
`src/scope/assets/geometry-provenance.json`; this conversion does not alter the
coordinates. No endorsement by the original authors is implied.

## Software dependencies

PyTorch, NumPy, PyArrow, Hugging Face Hub, FileLock, pytest and Ruff are installed
as dependencies, not vendored. Their own license terms continue to apply. SCOPE's
model implementation uses PyTorch modules and our own encoder/predictor/decoder
code; this repository does not include DiffusionPDE or FunDPS model implementations.
