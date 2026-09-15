# Inference and evaluation

Use physical-unit rasters matching the [data contract](DATA.md), from the repository root:

```bash
python -m src.inference --city San_Francisco --checkpoint checkpoints/best_l4_full.pt --output results/pi_swinir/San_Francisco.tif --batch-size 1
python -m src.evaluate --city San_Francisco --pred results/pi_swinir/San_Francisco.tif --output-dir results/pi_swinir/San_Francisco
```

`--city` resolves files under `data/raw/`. Explicit `--features`, `--gt`, `--output` and `--output-dir` override defaults. Sydney's reference glob resolves `_5m.tif`.

## Your own area

```bash
python -m src.inference --features /path/to/Features_10m.tif --checkpoint checkpoints/best_l4_full.pt --output results/my_area/refined.tif --batch-size 1 --no-figures
python -m src.evaluate --pred results/my_area/refined.tif --gt /path/to/reference_metres.tif --features /path/to/Features_10m.tif --output-dir results/my_area/evaluation
```

Use `--device cpu` or `--device cuda` to force execution device. Model/profile metadata select the correct architecture. All six released files work with the same inference command.

Inference blends 128 × 128 tiles with stride 64, pads small/irregular areas, and crops to the input extent. Output is float32 elevation in metres on the feature CRS/transform/grid, with restored NoData. Extreme predictions are clamped using configured bounds/margins. Diagnostic plots are beside outputs under `figures/<output-stem>/`; disable with `--no-figures`.

Evaluation writes `metrics.json` and figures: MAE, RMSE, bias, R², SSIM, slope RMSE, and FABDEM comparison when supplied. Reference terrain is averaged/aligned to the prediction grid. This module does not compute catchment hydrology, flood simulation or manuscript stream-network F1. Extent, masks and resampling affect metrics.

Large full-city rasters can exceed host RAM, especially native-reference evaluation. Crop before use if needed; inference batch size only reduces device memory.

## Learned baselines

```bash
python scripts/evaluate_baselines.py --cities San_Francisco --models cnn_lite unet_lite --batch-size 1
```

This runner uses your active Python environment and requires prepared rasters. See [baseline guide](../baselines/README.md).
