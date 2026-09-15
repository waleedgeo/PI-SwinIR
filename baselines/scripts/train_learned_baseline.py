#!/usr/bin/env python
"""Train compact U-Net/CNN learned baseline.

Run from the repository root with `python -m baselines.scripts.train_learned_baseline`. The script refuses to train if the
processed patch arrays are absent and writes a failure note instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from baselines.scripts._revision_utils import append_tracker, write_json


def fail_note(reason: str, output_dir: Path) -> None:
    note = ROOT / "baselines" / "notes" / "learned_baseline_failure.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        "# Learned Baseline Failure\n\n"
        f"Attempt time: {datetime.now().isoformat(timespec='seconds')}\n\n"
        f"Reason: {reason}\n\n"
        "No learned-baseline result is claimed. The available SwinIR L1-only "
        "ablation remains a related internal learned comparison, but it is not "
        "a compact external baseline.\n",
        encoding="utf-8",
    )
    append_tracker({
        "experiment_id": "learned_baseline",
        "task": "Learned baseline",
        "model": "unet_lite/cnn_lite",
        "input_channels": 11,
        "loss": "L1",
        "status": "failed",
        "end_time": datetime.now().isoformat(timespec="seconds"),
        "metrics_path": str(note),
        "notes": reason,
    })
    print(f"Wrote failure note: {note}")


def main() -> None:
    from src.cli import configure_console
    configure_console()
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--model", choices=["unet_lite", "cnn_lite"], default="unet_lite")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    args = parser.parse_args()

    from src.config import PROCESSED_DIR
    processed = PROCESSED_DIR
    if not processed.exists():
        fail_note("`data/processed` does not exist in this local checkout.", args.output_dir)
        raise SystemExit(2)

    import numpy as np
    import torch
    from torch import nn
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    from src.config import ELEV_MIN, ELEV_MAX, get_profile
    from src.dataset import build_dataloaders
    from baselines.models.unet_lite import UNetLite, count_parameters as count_unet
    from baselines.models.cnn_lite import CNNLite, count_parameters as count_cnn

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = get_profile("quick" if args.quick else "dev")
    cfg.epochs = 3 if args.quick else args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.num_workers = 0
    cfg.max_train_batches = 20 if args.quick else None
    cfg.max_val_batches = 10 if args.quick else None
    if args.max_train_batches is not None:
        cfg.max_train_batches = args.max_train_batches
    if args.max_val_batches is not None:
        cfg.max_val_batches = args.max_val_batches

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_loader, val_loader = build_dataloaders(cfg, processed_dir=processed)
    if args.model == "unet_lite":
        model = UNetLite().to(device)
        n_params = count_unet(model)
    else:
        model = CNNLite().to(device)
        n_params = count_cnn(model)
    print(f"model={args.model} n_params={n_params} ({n_params/1e6:.2f}M)")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.L1Loss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    elev_range = ELEV_MAX - ELEV_MIN

    log_path = args.output_dir / "train_val_loss.csv"
    best_path = args.output_dir / "best.pt"
    best_val = float("inf")

    start = datetime.now().isoformat(timespec="seconds")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_l1", "val_l1", "val_mae_m", "val_rmse_m", "time_s", "peak_vram_gb"])
        writer.writeheader()
        for epoch in range(cfg.epochs):
            t0 = time.time()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            model.train()
            tr = []
            for i, (x, y) in enumerate(train_loader):
                if cfg.max_train_batches is not None and i >= cfg.max_train_batches:
                    break
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    pred = model(x)
                    loss = criterion(pred, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                tr.append(float(loss.detach().cpu()))

            model.eval()
            vals, mae_m, mse_m = [], [], []
            with torch.no_grad():
                for i, (x, y) in enumerate(val_loader):
                    if cfg.max_val_batches is not None and i >= cfg.max_val_batches:
                        break
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    loss = criterion(pred, y)
                    vals.append(float(loss.detach().cpu()))
                    diff_m = (pred.float() - y.float()) * elev_range
                    mae_m.append(float(diff_m.abs().mean().cpu()))
                    mse_m.append(float((diff_m ** 2).mean().cpu()))
            val = sum(vals) / max(1, len(vals))
            peak_vram_gb = (
                torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device.type == "cuda" else 0.0
            )
            row = {
                "epoch": epoch + 1,
                "train_l1": sum(tr) / max(1, len(tr)),
                "val_l1": val,
                "val_mae_m": sum(mae_m) / max(1, len(mae_m)),
                "val_rmse_m": (sum(mse_m) / max(1, len(mse_m))) ** 0.5,
                "time_s": time.time() - t0,
                "peak_vram_gb": peak_vram_gb,
            }
            writer.writerow(row)
            f.flush()
            print(row)
            if val < best_val:
                best_val = val
                torch.save({
                    "model": args.model,
                    "model_state": model.state_dict(),
                    "n_params": n_params,
                    "epoch": epoch + 1,
                    "config": vars(args),
                }, best_path)

    config = {
        "model": args.model,
        "parameters": n_params,
        "epochs": cfg.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "quick": args.quick,
        "seed": args.seed,
        "loss": "L1",
    }
    write_json(args.output_dir / "config.json", config)
    append_tracker({
        "experiment_id": f"{args.model}_l1",
        "task": "Learned baseline",
        "model": args.model,
        "input_channels": 11,
        "loss": "L1",
        "train_cities": ",".join(cfg.train_cities),
        "test_cities": cfg.holdout_city,
        "status": "completed_quick" if args.quick else "completed",
        "start_time": start,
        "end_time": datetime.now().isoformat(timespec="seconds"),
        "checkpoint_path": str(best_path),
        "metrics_path": str(log_path),
        "notes": f"parameters={n_params}",
    })


if __name__ == "__main__":
    main()
