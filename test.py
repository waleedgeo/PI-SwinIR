"""
Inference/evaluation script for PI-SwinIR.

Example usage::

    python test.py \\
        --checkpoint ./experiments/pi_swinir_x4/best_model.pth \\
        --input_dir  ./data/test/LR \\
        --output_dir ./results \\
        --scale 4

When ``--hr_dir`` is also specified, PSNR and SSIM metrics are computed
and printed after processing all images.
"""

import argparse
import os
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

from models.network_pi_swinir import PISwinIR
from utils.utils import AverageMeter, calculate_psnr, calculate_ssim, load_checkpoint


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="PI-SwinIR Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing LR input images.")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save SR output images.")
    parser.add_argument("--hr_dir", type=str, default=None,
                        help="(Optional) HR ground-truth directory for metric evaluation.")
    parser.add_argument("--scale", type=int, default=4, choices=[2, 3, 4],
                        help="Super-resolution upscale factor.")
    parser.add_argument("--embed_dim", type=int, default=60)
    parser.add_argument("--num_blocks", type=int, default=6)
    parser.add_argument("--num_feat", type=int, default=64)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--tile", type=int, default=None,
                        help="Tile size for large-image inference. None = full image.")
    parser.add_argument("--tile_overlap", type=int, default=32,
                        help="Overlap between tiles (pixels in LR space).")
    parser.add_argument("--save_images", action="store_true", default=True,
                        help="Save SR images to --output_dir.")
    return parser.parse_args()


def tiled_forward(model, lr, tile, overlap, scale, device):
    """Run model inference tile-by-tile for large images.

    Args:
        model: PI-SwinIR model (eval mode).
        lr (Tensor): LR input (1, C, H, W).
        tile (int): tile size in LR pixels.
        overlap (int): overlap between tiles in LR pixels.
        scale (int): upscale factor.
        device: torch device.

    Returns:
        Tensor: SR output (1, C, H*scale, W*scale).
    """
    B, C, h, w = lr.shape
    stride = tile - overlap
    sr = torch.zeros(B, C, h * scale, w * scale, device=device)
    weight = torch.zeros(B, C, h * scale, w * scale, device=device)

    for top in range(0, h, stride):
        for left in range(0, w, stride):
            bottom = min(top + tile, h)
            right = min(left + tile, w)
            lr_patch = lr[:, :, top:bottom, left:right]
            sr_patch = model(lr_patch.to(device))
            sr[:, :, top * scale : bottom * scale, left * scale : right * scale] += sr_patch
            weight[:, :, top * scale : bottom * scale, left * scale : right * scale] += 1.0

    sr = sr / weight.clamp(min=1.0)
    return sr


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build model with same config as training
    depth_per_block = args.num_blocks
    model = PISwinIR(
        embed_dim=args.embed_dim,
        depths=(depth_per_block,) * 4,
        num_heads=(6,) * 4,
        window_size=args.window_size,
        upscale=args.scale,
        num_feat=args.num_feat,
    ).to(device)

    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    input_paths = sorted(
        p for p in Path(args.input_dir).iterdir()
        if p.suffix.lower() in IMG_EXTENSIONS
    )
    if not input_paths:
        raise FileNotFoundError(f"No images found in: {args.input_dir}")

    psnr_meter = AverageMeter("PSNR")
    ssim_meter = AverageMeter("SSIM")

    with torch.no_grad():
        for img_path in tqdm(input_paths, desc="Processing"):
            lr_img = Image.open(str(img_path)).convert("RGB")
            lr = to_tensor(lr_img).unsqueeze(0).to(device)

            t0 = time.time()
            if args.tile is not None:
                sr = tiled_forward(model, lr, args.tile, args.tile_overlap, args.scale, device)
            else:
                sr = model(lr)
            elapsed = time.time() - t0

            sr = sr.clamp(0.0, 1.0).squeeze(0).cpu()

            if args.save_images:
                out_path = Path(args.output_dir) / img_path.name
                to_pil(sr).save(str(out_path))

            # Compute metrics if HR ground truth is available
            if args.hr_dir:
                hr_path = Path(args.hr_dir) / img_path.name
                if hr_path.exists():
                    hr = to_tensor(Image.open(str(hr_path)).convert("RGB"))
                    psnr = calculate_psnr(sr, hr, crop_border=args.scale)
                    ssim = calculate_ssim(sr, hr, crop_border=args.scale)
                    psnr_meter.update(psnr)
                    ssim_meter.update(ssim)
                    tqdm.write(
                        f"{img_path.name}: PSNR={psnr:.2f} dB | SSIM={ssim:.4f}"
                        f" | {elapsed*1000:.1f}ms"
                    )
                else:
                    tqdm.write(f"{img_path.name}: processed in {elapsed*1000:.1f}ms")
            else:
                tqdm.write(f"{img_path.name}: processed in {elapsed*1000:.1f}ms")

    if args.hr_dir and psnr_meter.count > 0:
        print(
            f"\nAverage PSNR: {psnr_meter.avg:.2f} dB | "
            f"Average SSIM: {ssim_meter.avg:.4f}"
        )
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
