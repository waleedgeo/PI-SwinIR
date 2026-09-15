"""
=============================================================================
run_report.py — Comprehensive Training Run Report Generator
=============================================================================
Generates a detailed JSON + Markdown report for a completed training run.

Reads: CSV training log, checkpoint metadata, system info
Outputs: results/reports/{profile}_{run_id}/
    ├── report.json    (machine-readable)
    └── report.md      (human-readable)

Usage:
  python -m src.run_report --profile l4_probe
  python -m src.run_report --profile l4_probe --csv logs/train_l4_probe_*.csv
  python -m src.run_report --profile l4_full --checkpoint checkpoints/best_l4_full.pt
=============================================================================
"""

import argparse
import csv
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.config import (
    CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR,
    get_profile,
)


def _get_system_info():
    """Gather hardware and environment info."""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        info["cuda_version"] = torch.version.cuda
    try:
        import psutil
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
        info["cpu_count"] = psutil.cpu_count(logical=True)
    except ImportError:
        import os
        info["cpu_count"] = os.cpu_count()
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        info["ram_total_gb"] = round(kb / 1e6, 1)
                        break
        except Exception:
            pass
    return info


def _read_csv_log(csv_path):
    """Read training CSV log into a list of dicts."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        return list(reader)


def _read_checkpoint_meta(ckpt_path):
    """Read metadata from checkpoint (without loading model weights)."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return {
            "epoch": ckpt.get("epoch"),
            "val_loss": ckpt.get("val_loss"),
            "best_val_loss": ckpt.get("best_val_loss"),
            "patience_counter": ckpt.get("patience_counter"),
            "run_id": ckpt.get("run_id"),
            "profile": ckpt.get("profile"),
        }
    except Exception as e:
        return {"error": str(e)}


def generate_report(profile_name, csv_path=None, checkpoint_path=None):
    """Generate a comprehensive training report."""
    cfg = get_profile(profile_name)

    # ── Auto-find CSV if not provided ──
    if csv_path is None:
        csvs = sorted(LOG_DIR.glob(f"train_{profile_name}_*.csv"))
        if csvs:
            csv_path = csvs[-1]
        else:
            print(f"  ⚠ No CSV log found for profile '{profile_name}'")
            return None

    csv_path = Path(csv_path)
    print(f"  Reading CSV: {csv_path}")

    # ── Auto-find checkpoint ──
    if checkpoint_path is None:
        best = CHECKPOINT_DIR / f"best_{profile_name}.pt"
        if best.exists():
            checkpoint_path = best

    # ── Read data ──
    rows = _read_csv_log(csv_path)
    if not rows:
        print("  ⚠ Empty CSV log")
        return None

    ckpt_meta = {}
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"  Reading checkpoint: {checkpoint_path}")
        ckpt_meta = _read_checkpoint_meta(checkpoint_path)

    sys_info = _get_system_info()
    run_id = ckpt_meta.get("run_id") or csv_path.stem.split("_")[-2] + "_" + csv_path.stem.split("_")[-1]

    # ── Parse epoch data ──
    epochs = []
    for r in rows:
        epoch_data = {
            "epoch": int(r["epoch"]),
            "lr": float(r["lr"]),
            "train_total": float(r["train_total"]),
            "val_total": float(r["val_total"]),
            "val_mae_m": float(r["val_mae_m"]),
            "val_rmse_m": float(r["val_rmse_m"]),
            "time_s": float(r["time_s"]),
        }
        # Component losses
        for comp in ["l1", "slope", "curvature", "flow"]:
            for split in ["train", "val"]:
                key = f"{split}_{comp}"
                if key in r:
                    epoch_data[key] = float(r[key])
        epochs.append(epoch_data)

    # ── Compute summary stats ──
    best_epoch = min(epochs, key=lambda e: e["val_total"])
    first_epoch = epochs[0]
    last_epoch = epochs[-1]
    total_time_s = sum(e["time_s"] for e in epochs)
    avg_epoch_time_s = total_time_s / len(epochs)

    mae_improvement = (1 - last_epoch["val_mae_m"] / first_epoch["val_mae_m"]) * 100
    rmse_improvement = (1 - last_epoch["val_rmse_m"] / first_epoch["val_rmse_m"]) * 100

    # ── Profile config ──
    profile_config = {
        "name": cfg.name,
        "embed_dim": cfg.embed_dim,
        "num_rstb": cfg.num_rstb,
        "num_stl": cfg.num_stl,
        "num_heads": cfg.num_heads,
        "window_size": cfg.window_size,
        "mlp_ratio": cfg.mlp_ratio,
        "batch_size": cfg.batch_size,
        "grad_accum_steps": cfg.grad_accum_steps,
        "effective_batch": cfg.effective_batch,
        "epochs_configured": cfg.epochs,
        "epochs_completed": len(epochs),
        "lr": cfg.lr,
        "lr_min": cfg.lr_min,
        "warmup_epochs": cfg.warmup_epochs,
        "weight_decay": cfg.weight_decay,
        "grad_clip": cfg.grad_clip,
        "use_amp": cfg.use_amp,
        "num_workers": cfg.num_workers,
        "flow_iters": cfg.flow_iters,
        "val_split": cfg.val_split,
        "early_stop_patience": cfg.early_stop_pat,
        "holdout_city": cfg.holdout_city,
    }

    # ── Assemble report ──
    report = {
        "report_generated": datetime.now().isoformat(),
        "run_id": run_id,
        "profile": profile_config,
        "system": sys_info,
        "training": {
            "epochs_completed": len(epochs),
            "total_time_s": round(total_time_s, 1),
            "total_time_human": _format_time(total_time_s),
            "avg_epoch_time_s": round(avg_epoch_time_s, 1),
            "avg_epoch_time_human": _format_time(avg_epoch_time_s),
        },
        "best_epoch": {
            "epoch": best_epoch["epoch"],
            "val_loss": best_epoch["val_total"],
            "val_mae_m": best_epoch["val_mae_m"],
            "val_rmse_m": best_epoch["val_rmse_m"],
        },
        "final_epoch": {
            "epoch": last_epoch["epoch"],
            "val_loss": last_epoch["val_total"],
            "val_mae_m": last_epoch["val_mae_m"],
            "val_rmse_m": last_epoch["val_rmse_m"],
        },
        "improvement": {
            "mae_pct": round(mae_improvement, 1),
            "rmse_pct": round(rmse_improvement, 1),
        },
        "checkpoint": ckpt_meta,
        "epoch_details": epochs,
    }

    # ── Save ──
    out_dir = RESULTS_DIR / "reports" / f"{profile_name}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = out_dir / "report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  ✓ Saved: {json_path}")

    # Markdown
    md_path = out_dir / "report.md"
    _write_markdown_report(report, md_path)
    print(f"  ✓ Saved: {md_path}")

    return report


def _format_time(seconds):
    """Format seconds to human-readable string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def _write_markdown_report(report, path):
    """Write a formatted markdown report."""
    r = report
    p = r["profile"]
    s = r["system"]
    t = r["training"]
    b = r["best_epoch"]
    f = r["final_epoch"]
    imp = r["improvement"]
    epochs = r["epoch_details"]

    lines = []
    lines.append(f"# Training Report: `{p['name']}` profile")
    lines.append(f"")
    lines.append(f"**Run ID**: `{r['run_id']}`  ")
    lines.append(f"**Generated**: {r['report_generated']}  ")
    lines.append(f"")

    # System
    lines.append(f"## System")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| **GPU** | {s.get('gpu_name', 'N/A')} |")
    lines.append(f"| **VRAM** | {s.get('gpu_vram_gb', 'N/A')} GB |")
    lines.append(f"| **RAM** | {s.get('ram_total_gb', 'N/A')} GB |")
    lines.append(f"| **CPUs** | {s.get('cpu_count', 'N/A')} |")
    lines.append(f"| **CUDA** | {s.get('cuda_version', 'N/A')} |")
    lines.append(f"| **PyTorch** | {s.get('pytorch_version', 'N/A')} |")
    lines.append(f"")

    # Model Config
    lines.append(f"## Model Configuration")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| **Architecture** | SwinIR-DEM ({p['num_rstb']} RSTB × {p['num_stl']} STL) |")
    lines.append(f"| **Embed dim** | {p['embed_dim']} |")
    lines.append(f"| **Heads** | {p['num_heads']} |")
    lines.append(f"| **MLP ratio** | {p['mlp_ratio']} |")
    lines.append(f"| **Window size** | {p['window_size']} |")
    lines.append(f"| **Batch size** | {p['batch_size']} × {p['grad_accum_steps']} = {p['effective_batch']} |")
    lines.append(f"| **Epochs** | {p['epochs_completed']} / {p['epochs_configured']} |")
    lines.append(f"| **LR** | {p['lr']} → {p['lr_min']} |")
    lines.append(f"| **Warmup** | {p['warmup_epochs']} epochs |")
    lines.append(f"| **AMP** | {p['use_amp']} |")
    lines.append(f"| **Flow iters** | {p['flow_iters']} |")
    lines.append(f"| **Workers** | {p['num_workers']} |")
    lines.append(f"| **Holdout** | {p['holdout_city']} |")
    lines.append(f"")

    # Summary
    lines.append(f"## Results Summary")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| **Total time** | {t['total_time_human']} |")
    lines.append(f"| **Avg epoch** | {t['avg_epoch_time_human']} |")
    lines.append(f"| **Best epoch** | {b['epoch']} |")
    lines.append(f"| **Best MAE** | {b['val_mae_m']:.4f} m |")
    lines.append(f"| **Best RMSE** | {b['val_rmse_m']:.4f} m |")
    lines.append(f"| **MAE improvement** | {imp['mae_pct']:+.1f}% |")
    lines.append(f"| **RMSE improvement** | {imp['rmse_pct']:+.1f}% |")
    lines.append(f"")

    # Epoch table
    lines.append(f"## Epoch Details")
    lines.append(f"| Epoch | LR | Train Loss | Val Loss | MAE (m) | RMSE (m) | Time |")
    lines.append(f"|-------|----|-----------|----------|---------|----------|------|")
    for e in epochs:
        marker = " ★" if e["epoch"] == b["epoch"] else ""
        lines.append(
            f"| {e['epoch']} | {e['lr']:.2e} | {e['train_total']:.6f} | "
            f"{e['val_total']:.6f} | {e['val_mae_m']:.4f} | "
            f"{e['val_rmse_m']:.4f} | {_format_time(e['time_s'])}{marker} |"
        )
    lines.append(f"")

    # Component losses
    lines.append(f"## Loss Components (Validation)")
    lines.append(f"| Epoch | L1 | Slope | Curvature | Flow |")
    lines.append(f"|-------|----|-------|-----------|------|")
    for e in epochs:
        lines.append(
            f"| {e['epoch']} | {e.get('val_l1', 0):.6f} | "
            f"{e.get('val_slope', 0):.6f} | "
            f"{e.get('val_curvature', 0):.6f} | "
            f"{e.get('val_flow', 0):.6f} |"
        )
    lines.append(f"")

    # Convergence analysis
    lines.append(f"## Convergence Analysis")
    if len(epochs) >= 2:
        last_drop = (1 - epochs[-1]["val_mae_m"] / epochs[-2]["val_mae_m"]) * 100
        lines.append(f"- Last epoch MAE change: **{last_drop:+.1f}%**")
        if abs(last_drop) < 2:
            lines.append(f"- ⚠ Model appears to be **plateauing** — minimal improvement in last epoch")
        elif last_drop > 5:
            lines.append(f"- ✅ Model is **still converging** — further training recommended")
        else:
            lines.append(f"- Model is **nearing convergence** — 1-2 more epochs may help")
    lines.append(f"")

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    from src.cli import configure_console
    configure_console()
    parser = argparse.ArgumentParser(description="Generate training run report")
    parser.add_argument("--profile", type=str, required=True,
                        help="Training profile name")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to training CSV log (auto-detected if omitted)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (auto-detected if omitted)")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  PI-SwinIR Training Report Generator")
    print(f"{'='*65}\n")

    report = generate_report(args.profile, args.csv, args.checkpoint)

    if report:
        print(f"\n{'='*65}")
        print(f"  Report complete!")
        print(f"  Best MAE : {report['best_epoch']['val_mae_m']:.4f} m")
        print(f"  Best RMSE: {report['best_epoch']['val_rmse_m']:.4f} m")
        print(f"  Total    : {report['training']['total_time_human']}")
        print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
