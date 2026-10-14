# Historical manuscript and learned-control records

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

These saved evaluations have different model capacities, objectives, selected checkpoint budgets and model-specific valid support. The numerical ordering is descriptive; it does not isolate the backbone under a controlled protocol. Elevation errors and bias are in metres; implementation is in `src.evaluate`. Complete historical outcomes are retained here and in the manuscript's Supplementary Table S5.

The revision's [Sydney backbone comparison](../experiments/sydney_backbone/README.md) is a separate matched experiment with common inputs, objective, splits, budget and evaluation mask. Its two-model [results CSV](../experiments/sydney_backbone/results/comparison.csv) should be used for that specific backbone comparison. It does not replace these San Francisco or eight-city records.

Development metrics describe the stated development footprints; geographic evaluation is reported for withheld San Francisco. Consolidated hydrology statistics record outputs from the external extraction workflow. The core evaluation module covers its documented metrics; the original figure-generation and wider hydrology workflows are separate materials. See [reproducibility](../docs/REPRODUCIBILITY.md).
