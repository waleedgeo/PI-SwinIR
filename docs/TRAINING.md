# Training

Place prepared physical-unit rasters in `data/raw/` (see [data guide](DATA.md)):

```bash
python -m src.preprocess
# Or one city:
python -m src.preprocess --city Houston
```

Outputs are `{City}_X_patches.npy` `(N, 11, 128, 128)` and `{City}_Y_patches.npy` `(N, 1, 128, 128)` with stride 64. The loader concatenates available development-city arrays, splits patches 85:15 with a seeded shuffle, and uses city-balanced sampling. San Francisco is excluded by default. A combined-array fallback exists: ensure those arrays exclude your intended holdout.

## Fresh runs

```bash
# Small debugging architecture, not paper accuracy:
python -m src.train --profile quick --epochs 1 --max-train-batches 2 --max-val-batches 1
# Production architecture:
python -m src.train --profile l4_full
# Same architecture, smaller physical batches for an 8 GB GPU:
python -m src.train --profile rtx3070_full
```

`l4_full`: 15 configured epochs, batch 8, AdamW lr 2e-4, weight decay 1e-4, 2 warmup epochs, cosine decay, AMP, gradient clipping, seed 42. Loss weights: L1 1.0, slope 0.1, curvature 0.05, flow 0.01, with 2 flow iterations. `rtx3070_full` uses batch 2 × accumulation 4. The released production checkpoint records best epoch 12.

Use a **separate `DEM_DATA_ROOT`** for retraining to keep new outputs separate from released weights; place arrays under its `data/processed/`. CPU works for small checks, but is impractical for the full campaign.

Both trainers accept `--device cpu` or `--device cuda`; otherwise they select automatically.

The historical workflow describes a 3-epoch `l4_probe` followed by full training. Probe weights are not included; fresh profile runs cannot establish bit-for-bit reproduction of that trajectory. No full training campaign was rerun during release preparation.

## Resume

```bash
python -m src.train --profile l4_full --resume /path/to/full_checkpoint.pt
python -m src.train --profile l4_full --resume auto
```

The original `best_l4_full.pt` includes optimizer/scheduler/scaler state. Compact ablation/baseline exports do not support full-state resume. `auto` searches latest/best files in the active output root. Architecture must match. Resume restores scheduler state; resuming a probe is not equivalent to restarting a fresh full-run learning-rate schedule.

## Ablations and city exclusion

```bash
python -m src.train --profile ablation_l1_only
python -m src.train --profile ablation_no_flow
python -m src.train --profile ablation_no_curv
```

These retain the production architecture and disable selected losses. No retrained No-HAND, No-road or No-gate weight is included.

`--holdout-city Houston` changes city exclusion for a **new** run, not the provenance of released models. Houston is a default development city. Overlapping-patch random validation monitors optimization but does not establish spatial independence. Geographically separated splits/additional excluded cities require new experiments.
