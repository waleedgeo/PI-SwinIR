"""Create synthetic physical-unit inputs; these are not study data."""
import argparse
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('data/demo'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    row, col = np.mgrid[:256, :256].astype(np.float32)
    truth = 15 + row * .02 + col * .01 + 2 * np.sin(col / 25)
    features = np.stack([
        -15 + np.sin(row / 30), -22 + np.cos(col / 40),
        np.full_like(row, .18), np.full_like(row, .22),
        np.full_like(row, .12), np.full_like(row, .4),
        truth + 1.5 + .5 * np.sin(row / 18),
        np.maximum(truth - 14, 0), ((col % 40) < 2).astype(np.float32),
    ]).astype(np.float32)
    # A modest corner has all-band NoData; remaining zero GT is not NoData.
    features[:, :8, :8] = -9999
    truth[:8, :8] = -9999
    profile = dict(driver='GTiff', width=256, height=256, dtype='float32',
                   crs='EPSG:32615', transform=from_origin(300000, 3300000, 10, 10),
                   nodata=-9999, compress='deflate')
    for name, data in [('Houston_Features_10m.tif', features),
                       ('Houston_GroundTruth_10m.tif', truth[None])]:
        with rasterio.open(args.output_dir / name, 'w', count=data.shape[0], **profile) as dst:
            dst.write(data)
    print(f'Synthetic inputs written to {args.output_dir}. Not suitable for scientific benchmarking.')


if __name__ == '__main__':
    main()
