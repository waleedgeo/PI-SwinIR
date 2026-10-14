# Sydney U-Net versus Swin V2 backbone comparison

This directory reports a matched local backbone experiment within the PI-SwinIR framework. The deep feature extractor is either the paper-profile Swin V2 or a compact U-Net; both retain the eleven input channels, shallow embedding, feature fusion, gated residual elevation head and identical terrain objective.

| Backbone | Parameters | Selected epoch | MAE (m) | RMSE (m) | Bias (m) |
|---|---:|---:|---:|---:|---:|
| U-Net | 2,125,474 | 36 | 2.445692 | 3.594613 | -0.711112 |
| Swin V2 | 2,527,842 | 45 | 2.116429 | 3.091639 | -0.689882 |

Swin had 13.99% lower RMSE and 13.46% lower MAE on 356,605 common valid test cells. Both models completed 50 epochs and 1,600 optimizer updates. The pair took about 63 minutes on an RTX 3070 Laptop GPU. Checkpoint re-inference and source/data hashes were independently verified. The [two-model results CSV](results/comparison.csv) and runnable code are provided. Detailed experiment records remain local.

## Scope

Sydney was selected after exploratory five-epoch screening of seven development cities, which favored U-Net in six cities. Other completed London/San Francisco RMSE comparisons also favored U-Net. The Sydney follow-up uses one seed and an already observed holdout, so it supports a local result rather than general backbone superiority. Full experiment records remain retained by the author. This focused release presents the two learned backbones; it does not remove or replace any repository results. No statistical significance or repeated-seed uncertainty is claimed.

The experiment uses the corrected mask-aware objective `cageo-r1-metric-downslope-mfd-p1-v2`. It compares backbones under a fixed full objective; loss-term interactions, HAND contribution and stream-network recovery are separate experimental questions. The flow component uses metric downslope eight-neighbor fractions, power 1, no uphill routing, and two local passes. This finite local stencil defines a supervised structural objective. L1/Sobel/Laplacian terms remain on affine-normalized elevations. Weights are 1/0.1/0.05/0.01.

## Data and design

- 256 training, 64 validation and 64 test windows, each 128 × 128 on the 10 m grid; no augmentation.
- Frozen north–south geometry with 5.12 and 6.40 km context gaps; validation/test windows do not overlap.
- Eleven channels: VV, VH, R, G, B, NIR, FABDEM, HAND, ROADS, NDVI, NDWI. FABDEM remains an input required by both models' elevation head.
- The aligned 5 m reference is averaged over 2 × 2 cells. The mask requires finite/valid inputs and complete reference coverage. Zero and negative elevations are retained; missing channels are filled only after mask construction.
- The local data manifest and coordinates record exact support, grids, coordinates and array hashes. Raw rasters, prepared arrays, coordinates, detailed manifests, training records and weights are not distributed in this focused release. They must be supplied to reproduce the original training/inference; this is not a complete data deposit. Preparation generates new manifest and coordinate files from the declared geometry and seed.
- Source lineage, acquisition dates, vertical-reference compatibility and actual LiDAR land/coverage have not been independently certified.

## Architecture and optimization

Swin: 96 embedding features, six residual Swin blocks with four layers each, four attention heads, 8 × 8 windows, MLP ratio 2 and stochastic depth 0.1. U-Net: encoder widths 32/64/128/256, three pooling stages, bilinear upsampling, skip concatenation and a projection to 96 features. Model capacities differ and are reported above; the comparison uses one fixed configuration per backbone, not an exhaustive architecture search.

Both models were trained freshly with seed 42, AdamW (weight decay 0.0001), two warmup epochs and cosine learning rate 0.0002 to 0.000001 over a fixed 50-epoch schedule. FP32, TF32 disabled, microbatch 1 × accumulation 8, global L2 clipping threshold 1 with float64 norm arithmetic. FP32 and overflow-safe clipping were required after nonfinite gradients in earlier mixed-precision attempts; all attempts are retained. Individual nonfinite gradients are rejected, not hidden. Validation minimum support-pooled weighted loss selects the checkpoint; test scores do not select epochs, stopping or extensions.

## Running

Use Python 3.11 with the versions in `requirements.txt`; the recorded CUDA build is PyTorch 2.9.1+cu130. Install the appropriate official PyTorch build for your GPU separately. Execute from this directory:

```text
python run_sydney.py prepare --root PATH_TO_PROJECT --output PATH_TO_NEW_PATCH_FOLDER
python run_sydney.py train --data PATH_TO_PATCH_FOLDER --output PATH_TO_NEW_RUN
python run_sydney.py verify --data PATH_TO_ORIGINAL_PATCH_FOLDER --checkpoints PATH_TO_COMPLETED_RUN
```

Preparation expects `data/raw/Sydney_Features_10m.tif` (nine physical bands with explicit descriptions) and a unique `data/raw/Sydney_GroundTruth_5m.tif`. Header/nesting/scale/mask checks are mandatory. Existing output folders are refused. `verify` compares exact original array/coordinate hashes and independently recomputes selected-checkpoint metrics. It requires the original prepared arrays and checkpoints, not provided here. Floating-point/GPU differences can affect training trajectories; the one-seed result is not guaranteed on every environment.

The original Swin, kernel, mask and corrected objective source files are included byte-for-byte. The release changes paths/imports, limits preparation to Sydney and omits baseline evaluation; learned-model training and scoring mathematics are unchanged. The MIT license at the repository root applies.
