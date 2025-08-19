"""Convert MXNet ArcFace .params/.json pair to PyTorch state_dict for iresnet100.

Prerequisites:
  pip install mxnet==1.9.1  (or matching version used to export)
  The MXNet model typically comes as:
    model-symbol.json
    model-0000.params  (filename pattern may vary)

Usage example:
  python scripts/convert_mxnet_arcface.py \
      --mxnet-params /path/model-0000.params \
      --mxnet-symbol /path/model-symbol.json \
      --output weights/arcface_iresnet100_converted.pth

Notes:
  This script maps only matching parameter names where shapes agree.
  Unmatched MXNet parameters are reported; remaining PyTorch layers retain init weights.
  Fine-tuning or additional adaptation may be required for exact parity.
"""
import argparse
import os
from pathlib import Path
import torch

try:
    import mxnet as mx  # type: ignore
except ImportError:
    mx = None

from models.arcface import ArcFaceRecognizer


def load_mxnet_params(param_path: str):
    if mx is None:
        raise RuntimeError('mxnet not available')
    save_dict = mx.nd.load(param_path)
    params = {}
    for k, v in save_dict.items():
        # k format: 'arg:XXXX' or 'aux:XXXX'
        tp, name = k.split(':', 1)
        params[name] = v.asnumpy()
    return params


def map_params(mx_params, model: ArcFaceRecognizer):
    pt_state = model.backbone.state_dict()
    mapped = {}
    unused = []
    matched = 0
    for k_pt, tensor in pt_state.items():
        # Heuristic: direct name match or with common replacements
        candidates = [k_pt,
                      k_pt.replace('backbone.', ''),
                      k_pt.replace('stage', 'layer'),
                      k_pt.replace('embedding.', 'fc.')]
        found = None
        for cand in candidates:
            if cand in mx_params and mx_params[cand].shape == tuple(tensor.shape):
                found = cand
                break
        if found is not None:
            mapped[k_pt] = torch.from_numpy(mx_params[found])
            matched += 1
        else:
            unused.append(k_pt)
    return mapped, unused, matched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mxnet-params', required=True)
    parser.add_argument('--mxnet-symbol', required=False, help='Symbol JSON (not strictly required for raw params)')
    parser.add_argument('--output', required=True)
    parser.add_argument('--dry-run', action='store_true', help='Only report mapping stats')
    args = parser.parse_args()

    if mx is None:
        raise SystemExit('mxnet not installed. Install mxnet to use this converter.')

    mx_params = load_mxnet_params(args.mxnet_params)
    model = ArcFaceRecognizer()
    mapped, unused, matched = map_params(mx_params, model)
    ratio = matched / len(model.backbone.state_dict())
    print(f'Matched {matched} / {len(model.backbone.state_dict())} tensors ({ratio:.2%}).')
    if unused:
        print(f'Unmatched PyTorch tensors retained with random init: {len(unused)}')
    if args.dry_run:
        return
    # Load mapped subset
    current = model.backbone.state_dict()
    current.update(mapped)
    model.backbone.load_state_dict(current, strict=False)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.backbone.state_dict(), out_path)
    print(f'Saved converted weights to {out_path}')

if __name__ == '__main__':
    main()
