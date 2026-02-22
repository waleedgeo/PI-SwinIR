# Physics-Informed SwinIR (PI-SwinIR)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg)](https://pytorch.org/)

## Overview

**PI-SwinIR** (Physics-Informed Swin Image Restoration) integrates physics-based constraints into the SwinIR transformer architecture to enhance image super-resolution and restoration. By incorporating physical priors — such as gradient consistency, spectral fidelity, and total variation regularization — the model produces visually sharper and physically consistent outputs, making it especially suitable for remote sensing and scientific imaging applications.

## Architecture

PI-SwinIR extends the original SwinIR model with:

1. **SwinIR Backbone** — Hierarchical Swin Transformer blocks with shifted window self-attention for efficient high-resolution feature extraction.
2. **Physics-Informed Loss** — A composite loss combining:
   - **Pixel Loss** (L1/L2): Pixel-wise reconstruction fidelity.
   - **Gradient Consistency Loss**: Enforces edge and structural consistency via Sobel-filtered gradients.
   - **Total Variation (TV) Loss**: Encourages spatial smoothness while preserving edges.
   - **Spectral Loss**: Ensures frequency-domain consistency between the prediction and the ground truth.

## Repository Structure

```
PI-SwinIR/
├── models/
│   ├── __init__.py
│   ├── network_swinir.py       # SwinIR base model
│   └── network_pi_swinir.py    # Physics-Informed SwinIR wrapper
├── losses/
│   ├── __init__.py
│   └── physics_loss.py         # Physics-informed loss functions
├── data/
│   ├── __init__.py
│   └── dataset.py              # Dataset utilities
├── utils/
│   ├── __init__.py
│   └── utils.py                # Metrics and helper functions
├── train.py                    # Training script
├── test.py                     # Inference/evaluation script
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/waleedgeo/PI-SwinIR.git
cd PI-SwinIR
pip install -r requirements.txt
```

## Dataset Preparation

Organize your dataset as follows:

```
data/
├── train/
│   ├── HR/        # High-resolution ground-truth images
│   └── LR/        # Low-resolution input images (optional; generated on-the-fly if absent)
└── val/
    ├── HR/
    └── LR/
```

If `LR/` images are absent, they are generated on-the-fly by bicubic downsampling the `HR` images.

## Training

```bash
python train.py \
  --data_dir ./data \
  --save_dir ./experiments/pi_swinir_x4 \
  --scale 4 \
  --img_size 64 \
  --batch_size 16 \
  --epochs 500 \
  --lr 2e-4 \
  --lambda_grad 0.1 \
  --lambda_tv 0.01 \
  --lambda_spectral 0.05
```

### Key Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--scale` | `4` | Super-resolution upscale factor (2, 3, or 4) |
| `--img_size` | `64` | LR patch size during training |
| `--lambda_grad` | `0.1` | Weight for gradient consistency loss |
| `--lambda_tv` | `0.01` | Weight for total variation loss |
| `--lambda_spectral` | `0.05` | Weight for spectral (FFT) loss |
| `--loss_type` | `l1` | Pixel loss type (`l1` or `l2`) |
| `--num_blocks` | `6` | Number of Residual Swin Transformer Blocks |
| `--embed_dim` | `60` | Embedding dimension |

## Inference

```bash
python test.py \
  --checkpoint ./experiments/pi_swinir_x4/best_model.pth \
  --input_dir ./data/test/LR \
  --output_dir ./results \
  --scale 4
```

## Physics-Informed Losses

### Gradient Consistency Loss
Computes the L1 distance between the Sobel-filtered SR output and the HR ground truth, penalising incorrect edges and structures:

```
L_grad = || grad_x(SR) - grad_x(HR) ||_1 + || grad_y(SR) - grad_y(HR) ||_1
```

### Total Variation Loss
Promotes spatial smoothness while preserving sharp edges by minimising the total variation of the SR output:

```
L_tv = sum( |SR_{i+1,j} - SR_{i,j}| + |SR_{i,j+1} - SR_{i,j}| )
```

### Spectral Loss
Enforces frequency-domain fidelity using the 2D Fast Fourier Transform:

```
L_spectral = || |FFT(SR)| - |FFT(HR)| ||_1
```

## Citation

If you use PI-SwinIR in your research, please cite:

```bibtex
@misc{piswinir2026,
  title     = {PI-SwinIR: Physics-Informed Swin Image Restoration},
  author    = {Waleed Ejaz},
  year      = {2026},
  url       = {https://github.com/waleedgeo/PI-SwinIR}
}
```

Original SwinIR paper:

```bibtex
@inproceedings{liang2021swinir,
  title     = {SwinIR: Image Restoration Using Swin Transformer},
  author    = {Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops},
  year      = {2021}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
