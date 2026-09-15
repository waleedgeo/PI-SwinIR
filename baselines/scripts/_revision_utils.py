"""Shared helpers for emergency revision scripts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / "baselines"


def ensure_output_dirs(output_dir: Path) -> None:
    for sub in ["tables", "figures", "metrics", "predictions", "logs"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)
    (REV / "notes").mkdir(parents=True, exist_ok=True)


def json_safe(value):
    try:
        import numpy as np

        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=json_safe)


def write_csv(path: Path, rows: Sequence[Mapping], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_markdown_table(path: Path, rows: Sequence[Mapping], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            vals = []
            for key in fieldnames:
                val = row.get(key, "")
                if isinstance(val, float):
                    vals.append(f"{val:.6g}")
                else:
                    vals.append(str(val))
            f.write("| " + " | ".join(vals) + " |\n")


def append_tracker(row: Mapping) -> None:
    path = REV / "outputs" / "experiment_tracker.csv"
    fieldnames = [
        "experiment_id", "task", "model", "input_channels", "loss",
        "train_cities", "test_cities", "status", "start_time", "end_time",
        "checkpoint_path", "metrics_path", "notes",
    ]
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def read_csv_dicts(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean_ignore_nan(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v not in ("", None) and not math.isnan(float(v))]
    return sum(vals) / len(vals) if vals else float("nan")
