# Saved benchmarks

These are completed historical research outputs, not new evaluations from release assembly.

- [pi_swinir_metrics.json](pi_swinir_metrics.json): consolidated PI-SwinIR/FABDEM metrics and available embedded comparison/hydrology records.
- [paper_table.csv](paper_table.csv): saved eight-city summary.
- [model_comparison.csv](model_comparison.csv): 32 rows, four models × eight cities, with default city roles.
- `cnn_lite/`, `unet_lite/`: full-raster per-city metrics; city names filled from source directories.
- `training/`: original production/combined CSVs and baseline configs/loss curves. The combined/full logs overlap; they are not independent runs.

Seven cities are development; **San Francisco is unseen**. Saved San Francisco results:

| Model | RMSE (m) | MAE (m) |
|---|---:|---:|
| FABDEM (PI-SwinIR mask) | 4.508 | 2.406 |
| PI-SwinIR | 4.181 | 2.267 |
| CNNLite | 3.935 | 2.158 |
| UNetLite | 4.263 | 2.242 |

CNNLite outperforms PI-SwinIR on this holdout elevation metric. Loss/capacity differ; minor valid-pixel-count differences exist across saved model evaluations. This table is a historical comparison, not a new common-mask experiment. Elevation errors/bias are in metres; implementation is in `src.evaluate`.

Development metrics do not establish unseen-region performance. Consolidated prior hydrology statistics do not imply that their external extraction workflow is included: the core evaluation module does not reproduce all manuscript hydrology figures or revision analyses. See [reproducibility](../docs/REPRODUCIBILITY.md).
