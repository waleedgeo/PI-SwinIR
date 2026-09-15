"""Verify shipped states and meaningful core invariants without study data."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
from unittest.mock import patch
import numpy as np
import rasterio
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import IN_CHANNELS, FABDEM_CHANNEL_IDX, PATCH_SIZE, get_profile
from src.model import SwinIRDEM
from src.losses import PhysicsInformedLoss
from src.inference import tiled_inference, load_and_normalise
from src.preprocess import load_features, compute_spectral_indices, normalise_and_stack
from baselines.models.cnn_lite import CNNLite
from baselines.models.unet_lite import UNetLite


def main():
    from src.cli import configure_console
    configure_console()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    manifest = json.loads((ROOT / 'checkpoints/manifest.json').read_text())
    for entry in manifest:
        path = ROOT / entry['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], path
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint.get('model') in ('cnn_lite', 'unet_lite'):
            cls = CNNLite if checkpoint['model'] == 'cnn_lite' else UNetLite
            model = cls()
        else:
            cfg = get_profile(checkpoint['profile'])
            model = SwinIRDEM(in_channels=IN_CHANNELS, embed_dim=cfg.embed_dim,
                num_rstb=cfg.num_rstb, num_stl=cfg.num_stl, num_heads=cfg.num_heads,
                window_size=cfg.window_size, mlp_ratio=cfg.mlp_ratio,
                fabdem_channel_idx=FABDEM_CHANNEL_IDX, img_size=PATCH_SIZE, drop_path_rate=0)
        model.load_state_dict(checkpoint['model_state'], strict=True)
        model.eval()
        with torch.no_grad():
            output = model(torch.rand(1, 11, 128, 128))
        assert output.shape == (1, 1, 128, 128) and torch.isfinite(output).all()
        print(f'PASS checksum, strict load and finite forward: {entry["file"]} ({sum(p.numel() for p in model.parameters()):,} params)')

    # An identity-to-FABDEM model must retain arbitrary surfaces across edges,
    # blending boundaries, small areas and incomplete tiles.
    class Anchor(torch.nn.Module):
        def forward(self, x):
            return x[:, 6:7]
    rng = np.random.default_rng(42)
    for height, width in [(17, 25), (128, 128), (129, 173), (257, 301)]:
        x = rng.random((11, height, width), dtype=np.float32)
        result = tiled_inference(Anchor(), x, torch.device('cpu'), batch_size=2, use_amp=False)
        np.testing.assert_allclose(result, x[6], atol=2e-7)
    print('PASS complete tile coverage and exact anchor blending on four grid shapes')

    with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
        path = Path(tmp) / 'features.tif'
        x = np.empty((9, 16, 24), dtype=np.float32)
        for index, value in enumerate([-15, -22, .2, .25, .15, .4, 23, 9, 1]):
            x[index] = value
        x[:, 0, 0] = -9999
        with rasterio.open(path, 'w', driver='GTiff', width=24, height=16,
             count=9, dtype='float32', crs='EPSG:32615', nodata=-9999,
             transform=rasterio.transform.from_origin(300000, 3300000, 10, 10)) as dst:
            dst.write(x)
        actual, _, mask = load_and_normalise(path)
        features, _, _ = load_features(path)
        ndvi, ndwi = compute_spectral_indices(features)
        expected, _ = normalise_and_stack(features, ndvi, ndwi)
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        assert mask.sum() == 1 and mask[0, 0]
        assert np.isfinite(actual).all()
        # Plotting must use the same metre-based contract and retain sea level.
        from src.visualize import plot_inference_result, _load_gt_for_viz
        import matplotlib.pyplot as plt
        pred_path = Path(tmp) / 'prediction.tif'
        gt_path = Path(tmp) / 'reference.tif'
        elevation = np.full((16, 24), 23, dtype=np.float32)
        elevation[0, 0] = np.nan
        elevation[1, 1] = 0
        with rasterio.open(path) as src:
            profile = src.profile.copy()
        profile.update(count=1, nodata=np.nan)
        for output_path in [pred_path, gt_path]:
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(elevation, 1)
        aligned = _load_gt_for_viz(gt_path, profile)
        np.testing.assert_allclose(aligned, elevation, atol=1e-6, equal_nan=True)
        def inspect_figure(fig, output_path, label):
            panels = {ax.get_title(loc='left'): ax for ax in fig.axes if ax.images}
            pred_panel = panels['(a) Predicted DEM (10 m)'].images[0].get_array()
            fab_panel = panels['(d) FABDEM Input (10 m grid)'].images[0].get_array()
            gt_panel = panels['(e) Ground Truth (LiDAR)'].images[0].get_array()
            assert not np.ma.getmaskarray(pred_panel)[1, 1] and pred_panel[1, 1] == 0
            assert fab_panel[2, 2] == 23 and gt_panel[2, 2] == 23
            assert np.ma.getmaskarray(pred_panel)[0, 0]
            plt.close(fig)
        with patch('src.visualize._save_fig', side_effect=inspect_figure) as saver:
            plot_inference_result(pred_path, path, gt_path, city='Synthetic', output_dir=tmp)
            assert saver.call_count == 1
    print('PASS matching training/inference normalization and NoData footprint')
    print('PASS plotting in metres, reference alignment and preservation of zero elevation')

    pred = torch.rand(1, 1, 32, 32, requires_grad=True)
    target = torch.rand_like(pred)
    loss, components = PhysicsInformedLoss(lambda_l1=1, lambda_slope=.1,
         lambda_curvature=.05, lambda_flow=.01, flow_iters=2)(pred, target)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0
    assert all(np.isfinite(v) for v in components.values())
    print('PASS composite loss and finite nonzero gradients')
    print('Release verification passed. Full study results require original data.')


if __name__ == '__main__':
    main()
