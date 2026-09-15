# Learned baselines

CNNLite: full-resolution residual CNN, 8 residual blocks, **597,825 parameters**. UNetLite: four-level encoder/decoder, base width 32, **7,765,345 parameters**. Both predict a FABDEM-anchored residual using the same 11-channel contract and plain L1 loss.

```bash
python -m baselines.scripts.train_learned_baseline --model cnn_lite --epochs 15 --batch-size 16 --lr 2e-4 --seed 42 --output-dir baselines/outputs/cnn_lite
python -m baselines.scripts.train_learned_baseline --model unet_lite --epochs 15 --batch-size 16 --lr 2e-4 --seed 42 --output-dir baselines/outputs/unet_lite
python scripts/evaluate_baselines.py --cities San_Francisco --models cnn_lite unet_lite
```

Prepared arrays must exist under `data/processed/` or the `DEM_DATA_ROOT` override. The shared loader uses the default San Francisco exclusion, seeded patch split and city-balanced sampler. `--quick` checks a small batch subset. New training saves `best.pt` under the specified output directory, usable with `src.inference`.

Released exports are `checkpoints/cnn_lite.pt` and `checkpoints/unet_lite.pt`. Historical configs/logs are in [benchmarks/training](../benchmarks/training/); both completed 15 epochs.

Baselines differ in capacity from PI-SwinIR and were not hyperparameter-tuned. The Swin L1-only variant controls architecture when comparing physics losses. Keep patch-validation and full-raster metrics distinct, and separate development cities from unseen evaluation.
