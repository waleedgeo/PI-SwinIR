# Installation

Use Python 3.10 or 3.11 in a fresh environment. Run from the repository root and use module invocation (`python -m src.train`) for package imports.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/verify_release.py
```

For CUDA, install the appropriate GPU PyTorch distribution before the other requirements. An optional L4-oriented Conda reference is provided:

```bash
conda env create -f environment.yml
conda activate pi-swinir
```

Release verification used Python 3.11, PyTorch 2.9.1+cu130, Rasterio 1.4.4 and NumPy 2.2.6. The reference Conda environment and every dependency combination were not independently tested. CPU supports verification and small examples. TensorBoard is optional (`pip install tensorboard`); local execution needs no cloud credentials.

## Paths and cloud use

Defaults resolve relative to the repository. `DEM_DATA_ROOT` overrides the local root containing `data/`, `checkpoints/`, `logs/` and `results/`:

```bash
export DEM_DATA_ROOT=/absolute/path/to/run
# PowerShell:
# $env:DEM_DATA_ROOT = 'D:\datasets\pi_swinir_run'
```

This accepts a local filesystem path, not a `gs://` URI. Explicit CLI paths usually resolve from the current working directory; baseline outputs use `--output-dir`.

Cloud uploads are disabled by default. Set `DEM_GCS_BUCKET` to a bucket you control and configure `gsutil` only if uploads are wanted. `--upload-processed` additionally enables array uploads.

Full-city inference holds the feature stack in host RAM; evaluation reads the native reference raster into RAM. Large rasters need several gigabytes or more. `--batch-size 1` reduces GPU memory use; crop study areas if host RAM is insufficient.
