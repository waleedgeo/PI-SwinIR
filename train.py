"""
Training script for PI-SwinIR.

Example usage::

    python train.py \\
        --data_dir ./data \\
        --save_dir ./experiments/pi_swinir_x4 \\
        --scale 4 \\
        --img_size 64 \\
        --batch_size 16 \\
        --epochs 500 \\
        --lr 2e-4 \\
        --lambda_grad 0.1 \\
        --lambda_tv 0.01 \\
        --lambda_spectral 0.05
"""

import argparse
import os
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.dataset import SRDataset
from losses.physics_loss import PhysicsInformedLoss
from models.network_pi_swinir import PISwinIR
from utils.utils import (
    AverageMeter,
    calculate_psnr,
    calculate_ssim,
    load_checkpoint,
    save_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PI-SwinIR")

    # Data
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Root data directory containing train/ and val/ sub-dirs.")
    parser.add_argument("--save_dir", type=str, default="./experiments/pi_swinir",
                        help="Directory to save checkpoints and logs.")

    # Model
    parser.add_argument("--scale", type=int, default=4, choices=[2, 3, 4],
                        help="Super-resolution upscale factor.")
    parser.add_argument("--img_size", type=int, default=64,
                        help="LR patch size for training crops.")
    parser.add_argument("--embed_dim", type=int, default=60,
                        help="Embedding dimension.")
    parser.add_argument("--num_blocks", type=int, default=6,
                        help="Number of SwinTransformer layers per RSTB block.")
    parser.add_argument("--num_feat", type=int, default=64,
                        help="Number of intermediate feature channels.")
    parser.add_argument("--window_size", type=int, default=8,
                        help="Swin Transformer window size.")
    parser.add_argument("--drop_path_rate", type=float, default=0.1,
                        help="Stochastic depth drop path rate.")

    # Training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate.")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from.")

    # Loss weights
    parser.add_argument("--loss_type", type=str, default="l1", choices=["l1", "l2"],
                        help="Pixel-level loss type.")
    parser.add_argument("--lambda_grad", type=float, default=0.1,
                        help="Weight for gradient consistency loss.")
    parser.add_argument("--lambda_tv", type=float, default=0.01,
                        help="Weight for total variation loss.")
    parser.add_argument("--lambda_spectral", type=float, default=0.05,
                        help="Weight for spectral (FFT) loss.")

    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, writer):
    model.train()
    meters = {k: AverageMeter(k) for k in
              ["loss_total", "loss_pixel", "loss_grad", "loss_tv", "loss_spectral"]}

    for i, (lr, hr) in enumerate(loader):
        lr = lr.to(device)
        hr = hr.to(device)

        sr = model(lr)
        loss, components = criterion(sr, hr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for k, v in components.items():
            meters[k].update(v, lr.size(0))

        if (i + 1) % 50 == 0:
            print(
                f"  Epoch [{epoch}] Step [{i+1}/{len(loader)}] "
                + " | ".join(f"{k}: {v.avg:.4f}" for k, v in meters.items())
            )

    # TensorBoard logging
    for k, v in meters.items():
        writer.add_scalar(f"train/{k}", v.avg, epoch)

    return meters["loss_total"].avg


@torch.no_grad()
def validate(model, loader, device, epoch, writer):
    model.eval()
    psnr_meter = AverageMeter("PSNR")
    ssim_meter = AverageMeter("SSIM")

    for lr, hr in loader:
        lr = lr.to(device)
        hr = hr.to(device)
        sr = model(lr).clamp(0.0, 1.0)

        for b in range(sr.size(0)):
            psnr_meter.update(calculate_psnr(sr[b], hr[b], crop_border=4))
            ssim_meter.update(calculate_ssim(sr[b], hr[b], crop_border=4))

    writer.add_scalar("val/PSNR", psnr_meter.avg, epoch)
    writer.add_scalar("val/SSIM", ssim_meter.avg, epoch)
    print(f"  Val PSNR: {psnr_meter.avg:.2f} dB | SSIM: {ssim_meter.avg:.4f}")
    return psnr_meter.avg


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets
    train_dataset = SRDataset(
        hr_dir=os.path.join(args.data_dir, "train", "HR"),
        lr_dir=os.path.join(args.data_dir, "train", "LR"),
        scale=args.scale,
        patch_size=args.img_size,
        augment=True,
    )
    val_dataset = SRDataset(
        hr_dir=os.path.join(args.data_dir, "val", "HR"),
        lr_dir=os.path.join(args.data_dir, "val", "LR"),
        scale=args.scale,
        patch_size=0,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model
    depth_per_block = args.num_blocks
    model = PISwinIR(
        img_size=args.img_size,
        embed_dim=args.embed_dim,
        depths=(depth_per_block,) * 4,
        num_heads=(6,) * 4,
        window_size=args.window_size,
        upscale=args.scale,
        num_feat=args.num_feat,
        drop_path_rate=args.drop_path_rate,
    ).to(device)

    # Loss
    criterion = PhysicsInformedLoss(
        loss_type=args.loss_type,
        lambda_grad=args.lambda_grad,
        lambda_tv=args.lambda_tv,
        lambda_spectral=args.lambda_spectral,
    ).to(device)

    # Optimiser & scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )

    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "logs"))

    start_epoch = 1
    best_psnr = 0.0

    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_psnr = ckpt.get("best_psnr", 0.0)
        print(f"Resumed from epoch {start_epoch - 1}, best PSNR: {best_psnr:.2f} dB")

    print(f"Training PI-SwinIR x{args.scale} for {args.epochs} epochs ...")
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        scheduler.step()
        elapsed = time.time() - t0
        print(f"Epoch [{epoch}/{args.epochs}] loss={train_loss:.4f} ({elapsed:.1f}s)")

        if epoch % 5 == 0 or epoch == args.epochs:
            psnr = validate(model, val_loader, device, epoch, writer)
            is_best = psnr > best_psnr
            if is_best:
                best_psnr = psnr

            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_psnr": best_psnr,
                },
                filepath=os.path.join(args.save_dir, f"checkpoint_epoch{epoch:04d}.pth"),
                is_best=is_best,
                best_filepath=os.path.join(args.save_dir, "best_model.pth"),
            )

    writer.close()
    print(f"Training complete. Best PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    main()
