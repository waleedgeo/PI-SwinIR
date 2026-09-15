"""
=============================================================================
train.py — Stage 4: Training Loop
=============================================================================
Full training pipeline for PI-SwinIR DEM refinement.

Profiles:
  • quick      — Rapid sanity check (~10-20 min)
  • dev        — Development iteration (~1-2 hr)
  • local      — Overnight training (~8-12 hr on RTX 3070)
  • full_local — Full local convergence (~2-4 days)
  • gcp        — Cloud training (Vertex AI T4/A100)

Usage:
  python -m src.train --profile quick
  python -m src.train --profile local --epochs 50
  python -m src.train --profile local --holdout-city New_Orleans
  python -m src.train --profile local --resume checkpoints/best_local.pt
=============================================================================
"""

import argparse
import csv
import os
import random
import time
import warnings
from datetime import datetime
from pathlib import Path
import subprocess

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# ── Project imports ──
from src.config import (
    PROCESSED_DIR, CHECKPOINT_DIR, LOG_DIR, VIS_DIR,
    IN_CHANNELS, FABDEM_CHANNEL_IDX, PATCH_SIZE,
    ELEV_MIN, ELEV_MAX, GCS_BUCKET,
    get_profile,
)
from src.model import SwinIRDEM, print_model_stats
from src.losses import PhysicsInformedLoss
from src.dataset import build_dataloaders


# ═══════════════════════════════════════════════════════════════════════════
#  Utility
# ═══════════════════════════════════════════════════════════════════════════

def _log(msg, end="\n"):
    print(msg, end=end, flush=True)


def _fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def set_seed(seed):
    """Set random seed for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Note: full determinism requires torch.use_deterministic_algorithms(True)
    # but that can break some CUDA ops. We trade off minor non-determinism
    # for practical training speed.
    torch.backends.cudnn.benchmark = True


# ═══════════════════════════════════════════════════════════════════════════
#  Training step
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    grad_accum_steps, use_amp, grad_clip, max_batches=None,
                    epoch_num=0, total_epochs=1):
    """
    Train for one epoch with AMP and gradient accumulation.

    Returns
    -------
    avg_loss : float
    avg_components : dict — mean of each loss component
    """
    model.train()
    running_loss = 0.0
    running_comps = {}
    n_batches = 0
    total_batches = len(loader) if max_batches is None else min(max_batches, len(loader))
    log_interval = max(total_batches // 10, 50)  # log ~10 times per epoch

    optimizer.zero_grad()

    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            pred = model(x)
            loss, comps = criterion(pred, y)
            loss = loss / grad_accum_steps    # scale for accumulation

        scaler.scale(loss).backward()

        # Accumulate component stats
        for k, v in comps.items():
            running_comps[k] = running_comps.get(k, 0.0) + v
        running_loss += comps["total"]
        n_batches += 1

        # Step optimiser every grad_accum_steps
        if (i + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # ── Progress logging (keeps user informed on long epochs) ──
        if (i + 1) % log_interval == 0:
            avg_so_far = running_loss / n_batches
            pct = 100 * (i + 1) / total_batches
            _log(f"    [{epoch_num+1}/{total_epochs}] "
                 f"batch {i+1}/{total_batches} ({pct:.0f}%) "
                 f"loss={avg_so_far:.6f}", end="\r")

    # Clear progress line
    if total_batches > log_interval:
        _log(" " * 80, end="\r")

    # ── Flush leftover gradients ──
    if n_batches % grad_accum_steps != 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    avg_loss = running_loss / max(n_batches, 1)
    avg_comps = {k: v / max(n_batches, 1) for k, v in running_comps.items()}
    return avg_loss, avg_comps


# ═══════════════════════════════════════════════════════════════════════════
#  Validation step (with denormalised metrics)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(model, loader, criterion, device, use_amp, max_batches=None):
    """
    Run one pass over the validation set.

    Returns
    -------
    avg_loss : float
    avg_components : dict — includes denormalised MAE/RMSE in metres
    """
    model.eval()
    running_loss = 0.0
    running_comps = {}
    running_mae_m = 0.0     # denormalised MAE in metres
    running_mse_m = 0.0     # denormalised MSE in metres²
    n_batches = 0
    pred_min_global = float("inf")
    pred_max_global = float("-inf")
    pred_mean_sum = 0.0

    elev_range = ELEV_MAX - ELEV_MIN

    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            pred = model(x)
            loss, comps = criterion(pred, y)

        running_loss += comps["total"]
        for k, v in comps.items():
            running_comps[k] = running_comps.get(k, 0.0) + v
        n_batches += 1

        # Track model output range (normalised space)
        pred_f = pred.float()
        pred_min_global = min(pred_min_global, pred_f.min().item())
        pred_max_global = max(pred_max_global, pred_f.max().item())
        pred_mean_sum += pred_f.mean().item()

        # Denormalised error metrics (in metres)
        pred_m = pred_f * elev_range + ELEV_MIN
        y_m    = y.float() * elev_range + ELEV_MIN
        diff   = pred_m - y_m
        running_mae_m += diff.abs().mean().item()
        running_mse_m += (diff ** 2).mean().item()

    avg_loss = running_loss / max(n_batches, 1)
    avg_comps = {k: v / max(n_batches, 1) for k, v in running_comps.items()}

    # Add denormalised metrics
    avg_comps["mae_m"] = running_mae_m / max(n_batches, 1)
    avg_comps["rmse_m"] = (running_mse_m / max(n_batches, 1)) ** 0.5

    # Add output range monitoring (normalised space, for divergence detection)
    avg_comps["pred_min"] = pred_min_global
    avg_comps["pred_max"] = pred_max_global
    avg_comps["pred_mean"] = pred_mean_sum / max(n_batches, 1)

    return avg_loss, avg_comps


# ═══════════════════════════════════════════════════════════════════════════
#  Checkpoint I/O
# ═══════════════════════════════════════════════════════════════════════════

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_loss,
                    path, profile_name="local", patience_counter=0,
                    best_val_loss=float("inf"), run_id=None):
    """Save full training state including resume metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state":    scaler.state_dict(),
        "val_loss":        val_loss,
        "best_val_loss":   best_val_loss,
        "patience_counter": patience_counter,
        "profile":         profile_name,
        "run_id":          run_id,
    }, path)


def upload_to_gcs(local_path, gcs_folder, bucket=GCS_BUCKET):
    """Upload a local file or directory to GCS. Silently skips on failure."""
    if not bucket:
        return
    local_path = Path(local_path)
    if not local_path.exists():
        return
    dst = f"{bucket}/{gcs_folder}/{local_path.name}"
    try:
        if local_path.is_dir():
            cmd = ["gsutil", "-m", "cp", "-r", str(local_path), f"{bucket}/{gcs_folder}/"]
        else:
            cmd = ["gsutil", "cp", str(local_path), dst]
        subprocess.run(cmd, capture_output=True, timeout=600)
        _log(f"  ☁ Uploaded → {dst}")
    except Exception as e:
        _log(f"  ⚠ GCS upload skipped: {e}")


def find_latest_checkpoint(profile_name):
    """Find the latest checkpoint for auto-resume.
    Priority: latest_{profile}.pt > best_{profile}.pt
    """
    latest = CHECKPOINT_DIR / f"latest_{profile_name}.pt"
    if latest.exists():
        return latest
    best = CHECKPOINT_DIR / f"best_{profile_name}.pt"
    if best.exists():
        return best
    return None


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    """Load training state from checkpoint. Returns (epoch, val_loss, extras)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    extras = {
        "best_val_loss":    ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf"))),
        "patience_counter": ckpt.get("patience_counter", 0),
        "run_id":           ckpt.get("run_id", None),
    }
    return ckpt["epoch"], ckpt.get("val_loss", float("inf")), extras


# ═══════════════════════════════════════════════════════════════════════════
#  CSV Logger
# ═══════════════════════════════════════════════════════════════════════════

class CSVLogger:
    """Append-mode CSV logger for training metrics."""

    def __init__(self, path, fieldnames):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.path.exists()
        self._file = open(self.path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        if not file_exists:
            self._writer.writeheader()

    def log(self, row: dict):
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        self._file.close()


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    from src.cli import configure_console
    configure_console()
    parser = argparse.ArgumentParser(description="Train PI-SwinIR DEM model")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None,
                        help="Execution device (default: automatically selected)")
    parser.add_argument("--profile", type=str, default="quick",
                        help="Training profile (default: quick). "
                             "Available: quick, dev, test, local, full_local, gcp, sanity, "
                             "l4_probe, l4_full, rtx3070_full (l4_full arch resized for 8GB VRAM)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override profile epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override profile batch size")
    parser.add_argument("--max-train-batches", type=int, default=None,
                        help="Override profile max_train_batches (for smoke tests / benchmarking)")
    parser.add_argument("--max-val-batches", type=int, default=None,
                        help="Override profile max_val_batches (for smoke tests / benchmarking)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override profile learning rate")
    parser.add_argument("--holdout-city", type=str, default=None,
                        help="Override holdout city for cross-validation")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    parser.add_argument("--upload-processed", action="store_true", default=False,
                        help="Upload processed .npy data to GCS after training "
                             "(default: skip to avoid ~49 GB redundant uploads)")
    args = parser.parse_args()

    # ── Load profile ──
    cfg = get_profile(args.profile)

    # ── CLI overrides ──
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.max_train_batches is not None:
        cfg.max_train_batches = args.max_train_batches
    if args.max_val_batches is not None:
        cfg.max_val_batches = args.max_val_batches
    if args.lr is not None:
        cfg.lr = args.lr
    if args.holdout_city is not None:
        cfg.holdout_city = args.holdout_city
    if args.seed is not None:
        cfg.seed = args.seed

    # ── Reproducibility ──
    set_seed(cfg.seed)

    # ── Device ──
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    _log(f"\n{'='*65}")
    _log(f"  PI-SwinIR DEM Training")
    _log(f"{'='*65}")
    _log(f"  Profile     : {cfg.name}")
    _log(f"  {cfg.summary()}")
    _log(f"  Device      : {device}")
    if device.type == "cuda":
        _log(f"  GPU         : {torch.cuda.get_device_name(0)}")
        _log(f"  VRAM        : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    _log(f"  AMP         : {cfg.use_amp and device.type == 'cuda'}")
    _log(f"  Batch size  : {cfg.batch_size} × {cfg.grad_accum_steps} accum = "
         f"{cfg.effective_batch} effective")
    _log(f"  Epochs      : {cfg.epochs}")
    _log(f"  LR          : {cfg.lr} → {cfg.lr_min} (warmup {cfg.warmup_epochs}ep + cosine)")
    _log(f"  Grad clip   : {cfg.grad_clip}")
    _log(f"  Seed        : {cfg.seed}")
    _log(f"  Holdout     : {cfg.holdout_city}")
    _log(f"  Train cities: {', '.join(cfg.train_cities)}")
    if cfg.max_train_batches is not None:
        _log(f"  Batch limit : train={cfg.max_train_batches}, val={cfg.max_val_batches}")

    # ── Data ──
    _log(f"\n  Loading data ...")
    train_loader, val_loader = build_dataloaders(cfg)

    # ── Model ──
    _log(f"\n  Building model ...")
    model = SwinIRDEM(
        in_channels=IN_CHANNELS,
        embed_dim=cfg.embed_dim,
        num_rstb=cfg.num_rstb,
        num_stl=cfg.num_stl,
        num_heads=cfg.num_heads,
        window_size=cfg.window_size,
        mlp_ratio=cfg.mlp_ratio,
        fabdem_channel_idx=FABDEM_CHANNEL_IDX,
        img_size=PATCH_SIZE,
        drop_path_rate=cfg.drop_path,
    ).to(device)
    print_model_stats(model)

    # ── Loss, Optimiser, Scheduler ──
    criterion = PhysicsInformedLoss(
        lambda_l1=cfg.lambda_l1,
        lambda_slope=cfg.lambda_slope,
        lambda_curvature=cfg.lambda_curvature,
        lambda_flow=cfg.lambda_flow,
        flow_iters=cfg.flow_iters,
    ).to(device)
    _log(f"  Flow iters  : {cfg.flow_iters}")

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # LR schedule: optional linear warmup → cosine decay
    # Guard: ensure T_max ≥ 1 to avoid ZeroDivisionError
    warmup_ep = min(cfg.warmup_epochs, cfg.epochs - 1)
    cosine_T = max(cfg.epochs - warmup_ep, 1)

    if warmup_ep > 0:
        warmup_sched = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_ep)
        cosine_sched = CosineAnnealingLR(optimizer, T_max=cosine_T, eta_min=cfg.lr_min)
        # SequentialLR.__init__ internally calls sub-scheduler .step() which
        # triggers a harmless ordering warning — suppress it.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            scheduler = SequentialLR(optimizer,
                                     schedulers=[warmup_sched, cosine_sched],
                                     milestones=[warmup_ep])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=cosine_T, eta_min=cfg.lr_min)

    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp and device.type == "cuda")

    # ── Resume ──
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    resumed_run_id = None

    resume_path = None
    if args.resume == "auto":
        resume_path = find_latest_checkpoint(cfg.name)
        if resume_path:
            _log(f"\n  Auto-resume: found {resume_path}")
        else:
            _log(f"\n  Auto-resume: no checkpoint found for '{cfg.name}', starting fresh")
    elif args.resume:
        resume_path = Path(args.resume)

    if resume_path and resume_path.exists():
        _log(f"\n  Resuming from: {resume_path}")
        start_epoch, _, extras = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, device
        )
        best_val_loss = extras["best_val_loss"]
        patience_counter = extras["patience_counter"]
        resumed_run_id = extras.get("run_id")
        _log(f"  Resumed at epoch {start_epoch}, best val loss = {best_val_loss:.6f}, "
             f"patience = {patience_counter}")

    # ── Logger (reuse run_id on resume for continuous logs) ──
    run_id = resumed_run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"train_{cfg.name}_{run_id}.csv"
    logger = CSVLogger(log_path, fieldnames=[
        "epoch", "lr", "train_total", "train_l1", "train_slope",
        "train_curvature", "train_flow", "val_total", "val_l1",
        "val_slope", "val_curvature", "val_flow",
        "val_mae_m", "val_rmse_m", "time_s",
    ])
    _log(f"  Logging to  : {log_path}")

    # TensorBoard logger (for Vertex AI Training dashboard)
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = LOG_DIR / "tb" / f"{cfg.name}_{run_id}"
        tb_writer = SummaryWriter(log_dir=str(tb_dir))
        _log(f"  TensorBoard : {tb_dir}")
    except ImportError:
        tb_writer = None
        _log("  TensorBoard : not available (install tensorboard)")

    # ── Training Loop ──
    total_t0 = time.time()

    _log(f"\n{'='*65}")
    _log(f"  {'Epoch':>5s}  {'LR':>9s}  {'Train':>9s}  {'Val':>9s}  "
         f"{'MAE(m)':>8s}  {'RMSE(m)':>8s}  {'Time':>8s}")
    _log(f"  {'─'*5}  {'─'*9}  {'─'*9}  {'─'*9}  "
         f"{'─'*8}  {'─'*8}  {'─'*8}")

    for epoch in range(start_epoch, cfg.epochs):
        epoch_t0 = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # ── Train ──
        train_loss, train_comps = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            cfg.grad_accum_steps, cfg.use_amp, cfg.grad_clip,
            max_batches=cfg.max_train_batches,
            epoch_num=epoch, total_epochs=cfg.epochs,
        )

        # ── Validate ──
        val_loss, val_comps = validate(
            model, val_loader, criterion, device, cfg.use_amp,
            max_batches=cfg.max_val_batches,
        )

        # ── Scheduler step ──
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_t0

        # ── Console log ──
        is_best = val_loss < best_val_loss
        marker = " ★" if is_best else ""
        mae_m  = val_comps.get("mae_m", 0)
        rmse_m = val_comps.get("rmse_m", 0)
        _log(f"  {epoch+1:5d}  {current_lr:9.2e}  {train_loss:9.6f}  "
             f"{val_loss:9.6f}  {mae_m:8.3f}  {rmse_m:8.3f}  "
             f"{_fmt_time(epoch_time)}{marker}")
        if device.type == "cuda":
            peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
            _log(f"         peak VRAM this epoch: {peak_gb:.2f} GB "
                 f"(of {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB)")

        # ── CSV log ──
        logger.log({
            "epoch": epoch + 1,
            "lr": f"{current_lr:.2e}",
            "train_total": f"{train_loss:.6f}",
            "train_l1": f"{train_comps.get('l1', 0):.6f}",
            "train_slope": f"{train_comps.get('slope', 0):.6f}",
            "train_curvature": f"{train_comps.get('curvature', 0):.6f}",
            "train_flow": f"{train_comps.get('flow', 0):.6f}",
            "val_total": f"{val_loss:.6f}",
            "val_l1": f"{val_comps.get('l1', 0):.6f}",
            "val_slope": f"{val_comps.get('slope', 0):.6f}",
            "val_curvature": f"{val_comps.get('curvature', 0):.6f}",
            "val_flow": f"{val_comps.get('flow', 0):.6f}",
            "val_mae_m": f"{mae_m:.4f}",
            "val_rmse_m": f"{rmse_m:.4f}",
            "time_s": f"{epoch_time:.1f}",
        })

        # ── TensorBoard log ──
        if tb_writer is not None:
            step = epoch + 1
            tb_writer.add_scalars("loss/total",
                {"train": train_loss, "val": val_loss}, step)
            tb_writer.add_scalar("lr", current_lr, step)
            for comp_name in ["l1", "slope", "curvature", "flow"]:
                tb_writer.add_scalars(f"loss/{comp_name}", {
                    "train": train_comps.get(comp_name, 0),
                    "val":   val_comps.get(comp_name, 0),
                }, step)
            tb_writer.add_scalar("val/mae_m", mae_m, step)
            tb_writer.add_scalar("val/rmse_m", rmse_m, step)

        # ── Checkpoint ──
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch + 1, val_loss,
                CHECKPOINT_DIR / f"best_{cfg.name}.pt",
                profile_name=cfg.name,
                patience_counter=patience_counter,
                best_val_loss=best_val_loss,
                run_id=run_id,
            )
        else:
            patience_counter += 1

        # Always save latest for auto-resume
        save_checkpoint(
            model, optimizer, scheduler, scaler,
            epoch + 1, val_loss,
            CHECKPOINT_DIR / f"latest_{cfg.name}.pt",
            profile_name=cfg.name,
            patience_counter=patience_counter,
            best_val_loss=best_val_loss,
            run_id=run_id,
        )

        if (epoch + 1) % cfg.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch + 1, val_loss,
                CHECKPOINT_DIR / f"epoch_{epoch+1:04d}_{cfg.name}.pt",
                profile_name=cfg.name,
                patience_counter=patience_counter,
                best_val_loss=best_val_loss,
                run_id=run_id,
            )

        # ── Early stopping ──
        if patience_counter >= cfg.early_stop_pat:
            _log(f"\n  ⚠ Early stopping at epoch {epoch+1} "
                 f"(no improvement for {cfg.early_stop_pat} epochs)")
            break

    # ── Finish ──
    total_time = time.time() - total_t0
    logger.close()
    if tb_writer is not None:
        tb_writer.close()

    _log(f"\n{'='*65}")
    _log(f"  Training complete  [{cfg.name} profile]")
    _log(f"  Total time   : {_fmt_time(total_time)}")
    _log(f"  Best val loss: {best_val_loss:.6f}")
    _log(f"  Checkpoints  : {CHECKPOINT_DIR}")
    _log(f"  Log file     : {log_path}")
    _log(f"{'='*65}\n")

    # ── Auto-generate training visualisations ──
    try:
        from src.visualize import plot_training_dashboard, plot_model_diagram
        fig_dir = VIS_DIR / f"train_{cfg.name}"
        _log(f"  Generating figures → {fig_dir}")
        plot_training_dashboard(str(log_path), output_dir=str(fig_dir))
        plot_model_diagram(cfg, output_dir=str(fig_dir))
        _log(f"  ✓ Figures saved.")
    except Exception as e:
        _log(f"  ⚠ Visualisation skipped: {e}")

    # ── Auto-upload best model & processed data to GCS ──
    if GCS_BUCKET:
        _log(f"\n  Uploading to configured GCS bucket ...")
    best_ckpt = CHECKPOINT_DIR / f"best_{cfg.name}.pt"
    upload_to_gcs(best_ckpt, "checkpoints")
    # Also upload a timestamped/versioned copy for history
    versioned_name = f"best_{cfg.name}_{run_id}.pt"
    versioned_ckpt = CHECKPOINT_DIR / versioned_name
    if best_ckpt.exists() and not versioned_ckpt.exists():
        import shutil
        shutil.copy2(best_ckpt, versioned_ckpt)
    upload_to_gcs(versioned_ckpt, "checkpoints")
    upload_to_gcs(log_path, "logs")
    # Upload processed data only if explicitly requested (avoids ~49 GB redundant uploads)
    if getattr(args, 'upload_processed', False) and PROCESSED_DIR.exists():
        _log("  Uploading processed data (--upload-processed) ...")
        for npy_file in PROCESSED_DIR.glob("*.npy"):
            upload_to_gcs(npy_file, "processed")
    else:
        _log("  Skipping processed data upload (use --upload-processed to enable)")


if __name__ == "__main__":
    main()
