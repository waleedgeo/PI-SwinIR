"""Run released learned baselines with the active Python environment."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import CITIES, RAW_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cities', nargs='+', choices=list(CITIES), default=list(CITIES))
    parser.add_argument('--models', nargs='+', choices=['cnn_lite', 'unet_lite'], default=['cnn_lite', 'unet_lite'])
    parser.add_argument('--raw-dir', type=Path, default=RAW_DIR)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/baselines_learned')
    parser.add_argument('--checkpoint-dir', type=Path, default=ROOT / 'checkpoints')
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()
    for model in args.models:
        checkpoint = args.checkpoint_dir / f'{model}.pt'
        if not checkpoint.is_file():
            parser.error(f'Missing checkpoint: {checkpoint}')
        for city in args.cities:
            features = args.raw_dir / f'{city}_Features_10m.tif'
            refs = sorted(args.raw_dir.glob(f'{city}_GroundTruth_*.tif'))
            if not features.is_file() or not refs:
                parser.error(f'Missing prepared feature/reference rasters for {city} in {args.raw_dir}')
            output = args.output_dir / model / f'{city}.tif'
            subprocess.run([sys.executable, '-m', 'src.inference', '--features', str(features.resolve()),
                '--checkpoint', str(checkpoint.resolve()), '--output', str(output.resolve()),
                '--batch-size', str(args.batch_size), '--no-figures'], cwd=ROOT, check=True)
            subprocess.run([sys.executable, '-m', 'src.evaluate', '--city', city,
                '--pred', str(output.resolve()), '--gt', str(refs[0].resolve()),
                '--features', str(features.resolve()), '--output-dir', str((args.output_dir/model/city).resolve())],
                cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
