"""Automatically fetch ArcFace backbone weights via the insightface package.

This uses insightface's model zoo to download a pretrained face recognition model
and extracts / saves the PyTorch state_dict compatible with our ArcFaceRecognizer
(backbone iresnet100) when possible.

If the exact architecture mismatch occurs, the script will attempt partial key mapping.

Usage:
  python scripts/auto_download_arcface.py --model antelopev2 --out weights/arcface_iresnet100_auto.pth

List available models:
  python scripts/auto_download_arcface.py --list

Note:
  The insightface package may provide models in ONNX or internal format; this script
  focuses on those exposing a .pth or internal torch module. For ONNX, conversion would
  require graph parsing (not implemented here).
"""
import argparse
import os
from pathlib import Path
import torch

try:
    import insightface  # type: ignore
except ImportError:
    insightface = None

from models.arcface import ArcFaceRecognizer


def list_models():
    if insightface is None:
        print('insightface not installed.')
        return
    # Model zoo listing (simple heuristic; API may change)
    print('Common model names (check insightface docs for updates):')
    print('- antelopev2 (includes recognition, detection, gender-age)')
    print('- glintr100')
    print('- w600k_r50')
    print('- w600k_r100')


def fetch_model(name: str):
    if insightface is None:
        raise SystemExit('insightface not installed. Install it first.')
    store = insightface.model_zoo.get_model(name)
    store.prepare(ctx_id=-1)  # CPU
    return store


def extract_state(store, target: ArcFaceRecognizer):
    target_state = target.backbone.state_dict()
    mapped = {}
    src_items = []
    if hasattr(store, 'model'):
        # some models wrap underlying torch model
        src_items = list(store.model.state_dict().items())
    else:
        # fallback: direct state_dict if available
        if hasattr(store, 'state_dict'):
            src_items = list(store.state_dict().items())
    src_dict = dict(src_items)
    matched = 0
    for k_t, w_t in target_state.items():
        # direct match
        if k_t in src_dict and src_dict[k_t].shape == w_t.shape:
            mapped[k_t] = src_dict[k_t]
            matched += 1
            continue
        # attempt stripped prefixes
        short = k_t.replace('backbone.', '') if k_t.startswith('backbone.') else k_t
        if short in src_dict and src_dict[short].shape == w_t.shape:
            mapped[k_t] = src_dict[short]
            matched += 1
    print(f'Matched {matched}/{len(target_state)} parameters ({matched/len(target_state):.2%}).')
    return mapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='glintr100', help='insightface model name')
    parser.add_argument('--out', default='weights/arcface_iresnet100_auto.pth')
    parser.add_argument('--list', action='store_true', help='List common models')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    store = fetch_model(args.model)
    target = ArcFaceRecognizer()
    mapped = extract_state(store, target)
    if args.dry_run:
        print('Dry run complete (no file written).')
        return
    state = target.backbone.state_dict()
    state.update(mapped)
    target.backbone.load_state_dict(state, strict=False)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(target.backbone.state_dict(), out_path)
    print(f'Saved mapped weights to {out_path}')

if __name__ == '__main__':
    main()
