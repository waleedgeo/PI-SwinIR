"""Small, frozen and mask-aware London pilot preparation; never edits sources."""
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset


CHANNELS = ("VV", "VH", "R", "G", "B", "NIR", "FABDEM", "HAND", "ROADS", "NDVI", "NDWI")
NORMALIZATION = "r1c1-sensor-clipped-elevation-affine-unclipped-v1"
PATCH = 128


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_target(target, supported, factor=10):
    """Return supported cell means and genuine fine-cell coverage, not fill masks."""
    if target.ndim != 2 or supported.shape != target.shape or supported.dtype != np.bool_:
        raise ValueError("Require a 2D target and matching boolean support")
    if target.shape[0] % factor or target.shape[1] % factor:
        raise ValueError("Fine-grid dimensions must be multiples of aggregation factor")
    if not np.isfinite(target[supported]).all():
        raise ValueError("Nonfinite supported reference elevations")
    h, w = target.shape[0] // factor, target.shape[1] // factor
    blocks = supported.reshape(h, factor, w, factor)
    count = blocks.sum(axis=(1, 3))
    sums = np.where(supported, target, 0).reshape(h, factor, w, factor).sum(axis=(1, 3), dtype=np.float64)
    mean = np.divide(sums, count, out=np.zeros_like(sums), where=count > 0).astype(np.float32)
    coverage = (count / float(factor * factor)).astype(np.float32)
    return mean[None], coverage[None], (count > 0)[None]


def normalize_features(features, feature_valid):
    """Historical sensor clipping; elevation affine is deliberately unclipped."""
    if features.ndim != 3 or features.shape[0] != 9 or feature_valid.shape != features.shape:
        raise ValueError("Require nine physical bands and aligned per-band masks")
    if feature_valid.dtype != np.bool_ or not np.isfinite(features[feature_valid]).all():
        raise ValueError("Require boolean masks and finite supported features")
    roads = features[8][feature_valid[8]]
    if not np.logical_or(np.isclose(roads, 0, atol=1e-6, rtol=0),
                         np.isclose(roads, 1, atol=1e-6, rtol=0)).all():
        raise ValueError("Supported road values are not binary")
    # Mask first; fill only after constructing the eleven-channel masks.
    f = np.where(feature_valid, features, 0).astype(np.float32)
    f[2:6] = np.clip(f[2:6], 0, 1)
    x = np.empty((11, *f.shape[1:]), dtype=np.float32)
    x[:2] = np.clip((f[:2] + 32) / 76, 0, 1)
    x[2:6] = f[2:6]
    x[6] = (f[6] + 150) / 650
    x[7] = np.clip(f[7] / 500, 0, 1)
    x[8] = np.round(f[8])
    x[9] = np.clip(((f[5] - f[2]) / (f[5] + f[2] + 1e-10) + 1) / 2, 0, 1)
    x[10] = np.clip(((f[3] - f[5]) / (f[3] + f[5] + 1e-10) + 1) / 2, 0, 1)
    derived = np.concatenate((feature_valid, (feature_valid[2] & feature_valid[5])[None],
                              (feature_valid[3] & feature_valid[5])[None]))
    x = np.where(derived, x, 0).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite normalized features")
    return x


class PatchDataset(Dataset):
    """Frozen no-augmentation patches, preserving common support for every model."""
    def __init__(self, folder, role):
        self.folder = Path(folder)
        self.manifest = json.loads((self.folder / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema") not in ("r1c1-london-frozen-patches-v1", "r1c1-city-frozen-patches-v1") or tuple(self.manifest["channels"]) != CHANNELS:
            raise ValueError("Unknown preparation schema or channel order")
        if self.manifest["normalization"] != NORMALIZATION:
            raise ValueError("Normalization version mismatch")
        spec = self.manifest["roles"][role]
        self.samples = spec["samples"]
        self.arrays = {key: np.load(self.folder / spec["arrays"][key]["path"], mmap_mode="r", allow_pickle=False)
                       for key in ("x", "y", "valid")}
        if any(len(a) != len(self.samples) for a in self.arrays.values()):
            raise ValueError("Patch identity/array lengths mismatch")
        expected={"x":(len(self.samples),11,PATCH,PATCH),"y":(len(self.samples),1,PATCH,PATCH),
                  "valid":(len(self.samples),1,PATCH,PATCH)}
        if any(a.shape!=expected[key] for key,a in self.arrays.items()):
            raise ValueError("Patch shapes do not match the eleven-input same-grid contract")
        if self.arrays['x'].dtype!=np.float32 or self.arrays['y'].dtype!=np.float32 or self.arrays['valid'].dtype!=np.bool_:
            raise ValueError("Unexpected patch dtypes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        result = {k: torch.from_numpy(np.array(v[index], copy=True)) for k, v in self.arrays.items()}
        result["sample_id"] = self.samples[index]["sample_id"]
        return result
