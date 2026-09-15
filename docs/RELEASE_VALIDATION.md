# Release validation — 2026-09-15

Completed locally on Windows with Python 3.11.14, PyTorch 2.9.1+cu130, Rasterio 1.4.4, NumPy 2.2.6, and an NVIDIA RTX 3070 Laptop GPU. This is implementation validation, not a new scientific experiment.

| Check | Result |
|---|---|
| Six checkpoint SHA-256 checksums | Passed |
| Strict model-state loading and finite 128 × 128 CPU forward for all six variants | Passed |
| Exact model-tensor equality when exporting compact checkpoints | Passed |
| Production checkpoint preserved byte-for-byte | Passed |
| Complete tile coverage and anchor reconstruction on 17 × 25, 128 × 128, 129 × 173 and 257 × 301 grids | Passed |
| Preprocessing/inference normalization agreement and input NoData footprint | Passed |
| Diagnostic plot elevations in metres, reference alignment and preservation of valid zero elevation | Passed |
| Composite loss with finite, nonzero gradients | Passed |
| Synthetic 256 × 256 production-model GPU inference and GeoTIFF evaluation | Passed |
| Output CRS, transform, dimensions and NoData restoration | Passed |
| CNNLite and UNetLite synthetic GPU inference/evaluation through portable baseline runner | Passed |
| Synthetic preprocessing into nine 128 × 128 patches | Passed |
| One-epoch limited-batch Swin quick-profile CPU training, validation and checkpoint writing | Passed |
| One-epoch limited-batch CNNLite CPU training, validation and checkpoint writing | Passed |
| Python source compilation and local documentation links | Passed |

All synthetic artifacts and smoke-training outputs remain in ignored directories. Reported manuscript benchmark files were copied from completed historical runs and were not recomputed here. Full training, full-city benchmark reproduction, fresh-environment installation, the Conda reference environment and external data reconstruction were not tested.

An initial GPU quick-training run completed with an AMP optimizer/scheduler ordering warning; the final CPU smoke check completed without that warning. Console encoding failures found on Windows were fixed before the final CLI checks.
