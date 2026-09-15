<div align="center"><img src="img/PI-SwinIR-Logo.png" alt="PI-SwinIR" width="65%"></div>

# PI-SwinIR

**A Physics-Informed Swin Transformer for Hydrology-Aware Vertical Refinement of Global DEMs from Multimodal Earth Observation Data**

This repository provides the core computational materials for a manuscript being prepared for submission to *Computers & Geosciences*: model architecture, physics-informed losses, preprocessing, training, tiled inference, evaluation, six trained checkpoints, and CNN/U-Net baselines.

PI-SwinIR refines elevations on a **10 m grid** using FABDEM resampled to that grid and multimodal Earth observation inputs. Its output retains the input grid dimensions.

## Start here

| Guide | Contents |
|---|---|
| [Installation](docs/INSTALLATION.md) | Environment and runtime checks |
| [Data](docs/DATA.md) | Bands, units, normalization, boundaries and availability |
| [Training](docs/TRAINING.md) | Fresh training, resume and ablations |
| [Inference](docs/INFERENCE.md) | Study cities and your own study area |
| [Weights](checkpoints/README.md) | Six trained models and checksummed provenance |
| [Reproducibility](docs/REPRODUCIBILITY.md) | Reviewer checks and limitations |
| [Benchmarks](benchmarks/README.md) | Saved city metrics and training logs |
| [Baselines](baselines/README.md) | CNNLite and UNetLite |

## Quick start

Run commands from the repository root with Python 3.10 or 3.11. For GPU use, install a PyTorch build compatible with your GPU environment first.

```bash
python -m pip install -r requirements.txt
python scripts/verify_release.py
python scripts/make_demo_data.py --output-dir data/demo
python -m src.inference --features data/demo/Houston_Features_10m.tif --checkpoint checkpoints/best_l4_full.pt --output results/demo/refined.tif --batch-size 1 --no-figures
python -m src.evaluate --pred results/demo/refined.tif --gt data/demo/Houston_GroundTruth_10m.tif --features data/demo/Houston_Features_10m.tif --output-dir results/demo/evaluation
```

The example uses **synthetic terrain** to check software operation. Its metrics have no scientific interpretation. Full study rasters and prepared training arrays are **not bundled**; see the [data availability statement](docs/DATA.md#availability).

## Method

- Input: 11 normalized channels — VV, VH, Red, Green, Blue, NIR, FABDEM, HAND, Roads, NDVI and NDWI.
- Backbone: 6 residual Swin blocks, 4 Swin layers per block, embedding 96, heads 4, window 8.
- Prediction: `gate * FABDEM_normalized + residual`, converted back to metres.
- Objective: elevation L1 plus slope, curvature and local differentiable eight-neighbor flow regularization.
- Inference: overlapping 128 × 128 tiles, stride 64 and cosine blending.

The flow regularizer does not implement a catchment-scale hydrological solver or guarantee drainage correctness.

<div align="center"><img src="img/PI-SwinIR Graphical Abstract - 1920rescale.png" alt="Graphical abstract" width="100%"></div>

## Evaluation context

The default training setup uses seven development cities and excludes **San Francisco**. Houston is a development city. Validation uses a random 85:15 split of overlapping patches within development cities, which limits generalization claims.

Saved San Francisco full-city RMSE is **4.181 m** for PI-SwinIR versus **4.508 m** for FABDEM (7.25% reduction). CNNLite reaches **3.935 m** and UNetLite **4.263 m**. PI-SwinIR does not outperform CNNLite on this holdout elevation metric. See [benchmark context](benchmarks/README.md) for capacity, loss and pixel-mask differences.

## Layout

```text
src/             Core model, losses, data pipeline, training and evaluation
baselines/       Learned CNN/U-Net models and trainer
checkpoints/     Six trained checkpoints and SHA-256 manifest
benchmarks/      Saved metrics and training logs
data/regions/    Study-area GeoJSON boundaries
scripts/         Synthetic example, verification and baseline evaluation
docs/            Installation, data, usage and reviewer guides
```

## Citation and contact

Publication metadata and DOI are pending. Cite this software using [CITATION.cff](CITATION.cff) and record the Git commit used.

**Mirza Waleed**, Department of Geography, Hong Kong Baptist University.
[waleedgeo@outlook.com](mailto:waleedgeo@outlook.com) · [Website](https://waleedgeo.com) · [GitHub](https://github.com/waleedgeo)

Code uses the existing [MIT license](LICENSE). Third-party datasets retain their own terms; this code license does not grant dataset redistribution rights.
