"""Download frozen public release assets and verify SHA256 before making them available."""
import argparse
import hashlib
import os
from pathlib import Path
import urllib.request

BASE = 'https://github.com/Cucom/UAVCrack-R10/releases/download/v1.0.0/'
ASSETS = {
    'weights': ('uavcrack_r10_weights.zip', 'd59fc31d69bc3151205bd78286a439dcba1db2e9a2e10c1c374423907cf1da62'),
    'onnx': ('uavcrack_r10_onnx.zip', '63ea8f36ad6e74200009c7a598c15917b0f22ed02ee26cc93ce4921abe773f9b'),
}

def download(url, destination, expected):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.part')
    digest = hashlib.sha256()
    created = False
    try:
        with partial.open('xb') as output:
            created = True
            with urllib.request.urlopen(url, timeout=60) as response:
                while True:
                    block = response.read(4 << 20)
                    if not block:
                        break
                    digest.update(block)
                    output.write(block)
        if digest.hexdigest() != expected:
            raise ValueError('SHA256 mismatch: ' + destination.name)
        # Refuse overwriting an output created by another process during download.
        os.link(partial, destination)
    finally:
        if created and partial.exists():
            partial.unlink()
    return destination

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--format', choices=['weights', 'onnx', 'both'], default='weights')
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts'))
    args = parser.parse_args()
    for kind in (ASSETS if args.format == 'both' else [args.format]):
        name, digest = ASSETS[kind]
        path = download(BASE + name, args.output_dir / name, digest)
        print(f'Verified {path}')

if __name__ == '__main__':
    main()
