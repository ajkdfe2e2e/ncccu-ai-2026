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
    p.add_argument('--data-root', type=Path, default=ROOT / 'cxr_first')
    p.add_argument('--architecture', choices=('resnet18', 'convnext_tiny'), default='resnet18')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--image-size', type=int, default=64)
    p.add_argument('--accumulation-steps', type=int, default=1)
    p.add_argument('--ema-decay', type=float, default=0.)
    p.add_argument('--ema-device', choices=('model', 'cpu'), default='model')
    p.add_argument('--warmup-epochs', type=int, default=0)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--smoke-studies', type=int, default=4)
    p.add_argument('--gradient-checkpointing', action='store_true')
    p.add_argument('--pretrained', type=Path)
    p.add_argument('--image-cache', type=Path)
    a = p.parse_args()
    directory = ROOT / 'outputs' / ('resume-verification-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    directory.mkdir(parents=True)
    common = ['--data-root', a.data_root, '--epochs', a.epochs, '--batch-size', a.batch_size,
              '--architecture', a.architecture, '--accumulation-steps', a.accumulation_steps,
              '--ema-decay', a.ema_decay, '--ema-device', a.ema_device, '--warmup-epochs', a.warmup_epochs,
              '--image-size', a.image_size, '--smoke-studies', a.smoke_studies, '--device', a.device,
              '--lr', .0001, '--scheduler', 'cosine', '--min-lr', .000001]
    if a.gradient_checkpointing:
        common.append('--gradient-checkpointing')
    if a.image_cache is not None:
        common.extend(('--image-cache', a.image_cache))
    initialization = ['--pretrained', a.pretrained] if a.pretrained else ['--from-scratch']
    invoke([*common, *initialization, '--output', directory/'continuous'])
    invoke([*common, *initialization, '--stop-after-epochs', 1, '--output', directory/'interrupted'])
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
    checked = ['model', 'optimizer', 'scaler', 'scheduler', 'ema', 'rng', 'best_score', 'completed_epochs']
    for key in checked:
        if key not in continuous or key not in resumed:
            raise AssertionError(f'State format missing {key}')
        identical(continuous[key], resumed[key], key)
    baseline = torch.load(directory/'continuous'/'best.pt', map_location='cpu', weights_only=True)
    candidate = torch.load(directory/'resumed'/'best.pt', map_location='cpu', weights_only=True)
    identical(baseline, candidate, 'best_inference_payload')
    report = dict(passed=True, device=a.device, directory=str(directory), architecture=a.architecture,
                  checked=[*checked, 'best_inference_payload'],
                  scope=f'{a.epochs} epochs on {a.smoke_studies} train/valid studies, same machine. Not cross-hardware reproducibility.')
    (directory/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
