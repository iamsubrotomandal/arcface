import os
import hashlib
import urllib.request
from pathlib import Path

"""ArcFace weight downloader.

Usage:
    python scripts/download_arcface_weights.py --model iresnet100

NOTE: Official InsightFace ArcFace (IR-100) weights are typically distributed
as MXNet (.params) or in onnx/glint variants. For PyTorch, community mirrors exist.
Because direct redistribution links can change and licensing may apply, this script
ships with commented reference URLs; you should verify the source before enabling.

Suggested sources (uncomment one and add expected md5):
    - InsightFace GitHub release assets (convert if needed)
    - https://github.com/deepinsight/insightface (original repo)
    - Community-converted PyTorch checkpoints (verify integrity!)

Set an environment variable ARCFACE_WEIGHTS_URL to override.
"""

URLS = {
        # 'iresnet100': 'https://your_verified_host/path/to/arcface_iresnet100.pth'
}

# Optional expected md5 hashes for integrity checking
EXPECTED_MD5 = {
        # 'iresnet100': 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
}

CHUNK = 1<<14

def md5sum(path: Path, chunk: int = CHUNK) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def download(name: str, url: str, target_dir: str = 'weights'):
    os.makedirs(target_dir, exist_ok=True)
    dest = Path(target_dir) / f'{name}.pth'
    if dest.exists():
        print(f'[skip] {dest} already exists')
        return dest
    print(f'[download] {url} -> {dest}')
    with urllib.request.urlopen(url) as r, open(dest, 'wb') as f:
        while True:
            b = r.read(CHUNK)
            if not b:
                break
            f.write(b)
    print('Download complete.')
    if name in EXPECTED_MD5:
        got = md5sum(dest)
        exp = EXPECTED_MD5[name]
        if got != exp:
            print(f'[warn] md5 mismatch for {name}: expected {exp}, got {got}')
        else:
            print('[ok] md5 verified')
    return dest

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='iresnet100', help='Model key to download')
    parser.add_argument('--url', default=None, help='Override URL (or set ARCFACE_WEIGHTS_URL env)')
    parser.add_argument('--out', default='weights', help='Target directory')
    args = parser.parse_args()

    model = args.model
    url = args.url or os.environ.get('ARCFACE_WEIGHTS_URL') or URLS.get(model)
    if not url:
        print(f'No URL configured for {model}. Edit URLS dict or pass --url.')
    else:
        download(model, url, target_dir=args.out)
