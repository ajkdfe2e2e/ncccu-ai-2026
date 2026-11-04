"""Check exact epoch-boundary recovery with 4 train/valid studies; no test use."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import runpy
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def invoke(args):
    before = sys.argv
    try:
        sys.argv = [str(ROOT / 'train.py'), *map(str, args)]
        runpy.run_path(str(ROOT / 'train.py'), run_name='__main__')
    finally:
        sys.argv = before


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    directory = ROOT / 'outputs' / ('resume-verification-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    directory.mkdir(parents=True)
    common = ['--data-root', ROOT/'cxr_first', '--epochs', 2, '--batch-size', 2,
              '--image-size', 64, '--smoke-studies', 4, '--device', a.device,
              '--lr', .0001, '--scheduler', 'cosine', '--min-lr', .000001]
    invoke([*common, '--from-scratch', '--output', directory/'continuous'])
    invoke([*common, '--from-scratch', '--stop-after-epochs', 1, '--output', directory/'interrupted'])
    invoke([*common, '--resume', directory/'interrupted'/'last-training.pt', '--output', directory/'resumed'])
    continuous = torch.load(directory/'continuous'/'last-training.pt', map_location='cpu', weights_only=True)
    resumed = torch.load(directory/'resumed'/'last-training.pt', map_location='cpu', weights_only=True)
    def identical(left, right, path='state'):
        if isinstance(left, torch.Tensor):
            if not isinstance(right, torch.Tensor) or not torch.equal(left, right):
                raise AssertionError(f'Tensor differs: {path}')
        elif isinstance(left, dict):
            if left.keys() != right.keys():
                raise AssertionError(f'Keys differ: {path}')
            for key in left:
                identical(left[key], right[key], f'{path}.{key}')
        elif isinstance(left, (list, tuple)):
            if len(left) != len(right):
                raise AssertionError(f'Length differs: {path}')
            for i, (one, two) in enumerate(zip(left, right)):
                identical(one, two, f'{path}[{i}]')
        elif left != right:
            raise AssertionError(f'Value differs: {path}: {left} != {right}')
    # Compare all training state except provenance paths/wall-clock fields; verify after schema review.
    for key in ('model', 'optimizer', 'scaler', 'scheduler'):
        if key not in continuous or key not in resumed:
            raise AssertionError(f'State format missing {key}')
        identical(continuous[key], resumed[key], key)
    baseline = torch.load(directory/'continuous'/'best.pt', map_location='cpu', weights_only=True)
    candidate = torch.load(directory/'resumed'/'best.pt', map_location='cpu', weights_only=True)
    identical(baseline, candidate, 'best_inference_payload')
    report = dict(passed=True, device=a.device, directory=str(directory),
                  checked=['model','optimizer','scaler','scheduler','best_inference_payload'],
                  scope='Two epochs on four train/valid studies, same machine. Not cross-hardware reproducibility.')
    (directory/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
