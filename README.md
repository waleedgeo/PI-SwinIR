<div align="center"><img src="img/PI-SwinIR-Logo.png" alt="PI-SwinIR" width="65%"></div>

# PI-SwinIR

**A Physics-Informed Swin Transformer for Hydrology-Aware Vertical Refinement of Digital Elevation Models from Multimodal Earth Observation Data**

This repository provides the computational materials for the PI-SwinIR manuscript submitted to *Computers & Geosciences*: architecture, terrain objectives, preprocessing, training, tiled inference, evaluation and trained checkpoints. It also includes the matched Sydney U-Net versus Swin V2 experiment added during revision.

PI-SwinIR refines elevations on a **10 m grid** using FABDEM resampled to that grid and multimodal Earth observation inputs. Its output retains the input grid dimensions.

## Start here

| Guide | Contents |
|---|---|
| [Installation](docs/INSTALLATION.md) | Environment and runtime checks |
| [Data](docs/DATA.md) | Bands, units, normalization, boundaries and availability |
| [Training](docs/TRAINING.md) | Fresh training, resume and ablations |
| [Inference](docs/INFERENCE.md) | Study cities and your own study area |
| [Weights](checkpoints/README.md) | Six trained models and checksummed provenance |
| [Reproducibility](docs/REPRODUCIBILITY.md) | Experiment protocols and reproduction checks |
| [Sydney backbone comparison](experiments/sydney_backbone/README.md) | Matched protocol, runnable code and two-model CSV |
| [Historical benchmark records](benchmarks/README.md) | Original city metrics, supporting controls and training qualifications |

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

The flow regularizer matches local terrain structure over two propagation passes. Catchment-scale routing lies outside the definition of this local training objective.

<div align="center"><img src="img/PI-SwinIR Graphical Abstract - 1920rescale.png" alt="Graphical abstract" width="100%"></div>

## Manuscript results and Sydney backbone comparison

The historical training setup uses seven development cities, including Houston, with a random 85:15 split of overlapping development patches. **San Francisco** is held out throughout for geographic evaluation. The separate Sydney experiment uses buffered training, validation and test partitions.

The historical manuscript evaluation reports San Francisco RMSE of **4.181 m** for PI-SwinIR versus **4.508 m** for its paired FABDEM comparator (7.25% reduction). The separate revision experiment compares Swin V2 and U-Net inside the same framework on buffered Sydney test patches:

| Backbone | MAE (m) | RMSE (m) |
|---|---:|---:|
| U-Net | 2.446 | 3.595 |
| Swin V2 | 2.116 | 3.092 |

Swin V2 had **13.99% lower RMSE** under this matched Sydney protocol. Both models used the same eleven inputs, gated head, corrected terrain objective, frozen splits, 50-epoch budget and validation-only checkpoint selection. See the [experiment code and protocol](experiments/sydney_backbone/README.md) and [two-model CSV](experiments/sydney_backbone/results/comparison.csv).

Sydney was selected after exploratory screening and this follow-up uses one seed, so the result supports a local backbone comparison. Historical learned controls use different objectives, capacities, checkpoint budgets and model-specific evaluation support. Their saved scores and provenance are retained in the [historical benchmark record](benchmarks/README.md); they are separate from the matched Sydney comparison.

## Layout

```text
src/             Core model, losses, data pipeline, training and evaluation
experiments/     Matched Sydney backbone comparison
baselines/       Historical L1 learned controls and trainer
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
