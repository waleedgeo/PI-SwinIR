# Trained weights

Six trained variants are included directly in Git. [manifest.json](manifest.json) records filenames, bytes, source/release SHA-256 checksums, epoch and metadata.

| File | Model/loss | State |
|---|---|---|
| `best_l4_full.pt` | PI-SwinIR: L1 + slope + curvature + flow | Original full training state |
| `best_ablation_l1_only.pt` | Swin: L1 only | Inference export |
| `best_ablation_no_flow.pt` | Swin: L1 + slope + curvature | Inference export |
| `best_ablation_no_curv.pt` | Swin: L1 + slope + flow | Inference export |
| `cnn_lite.pt` | CNNLite: L1 | Inference export |
| `unet_lite.pt` | UNetLite: L1 | Inference export |

PI-SwinIR uses the production architecture: 6 × 4 Swin layers, embedding 96, heads 4, window 8, MLP ratio 2. The full checkpoint records epoch 12 and validation loss 0.001532538827228502. Baseline parameters: CNNLite 597,825; UNetLite 7,765,345.

The production file is copied unchanged. Compact exports retain **every model-state tensor exactly**, plus model/profile/epoch metadata; optimizer state and local path-bearing configuration are omitted. Tensor equality was checked during export. Manifest source paths are research provenance, not required reviewer paths. Only the original production file supports full-state training resume.

```bash
python scripts/verify_release.py
```

The available configuration/project audit identify San Francisco as excluded for these runs. Historical Swin checkpoints store the profile name rather than a complete resolved split manifest. These are single-run resources, not multi-seed ensembles. No retrained No-HAND, No-road or No-gate variants are included.
