"""Record local readiness and train-only decoding overhead; never opens test data."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
import time
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cxr.data import read_records
from cxr.safety import safe_output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=ROOT / 'cxr_first')
    p.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'preflight.json')
    p.add_argument('--samples', type=int, default=32)
    a = p.parse_args()
    if a.samples < 1:
        p.error('--samples must be positive')
    output = safe_output(a.output, a.data_root)
    records = read_records(a.data_root / 'train.csv', a.data_root / 'train')[:a.samples]
    if not records:
        raise ValueError('No training records')
    elapsed = []
    for record in records:
        start = time.perf_counter()
        with Image.open(record['path']) as im:
            ImageOps.pad(im.convert('RGB'), (224, 224), method=Image.Resampling.BILINEAR, color=0).load()
        elapsed.append((time.perf_counter() - start) * 1000)
    packages = {}
    for name in ('torch', 'torchvision', 'numpy', 'Pillow'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    weight = ROOT / 'weights' / 'resnet18-f37072fd.pth'
    digest = hashlib.sha256(weight.read_bytes()).hexdigest() if weight.exists() else None
    expected = 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
    report = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                  packages=packages, free_disk_bytes=shutil.disk_usage(ROOT).free,
                  train_decode_sample_count=len(records), train_decode_mean_ms=float(np.mean(elapsed)),
                  train_decode_p95_ms=float(np.percentile(elapsed, 95)),
                  decode_note='First training images, PIL RGB decode+224 pad only; not full-model or official benchmark',
                  pretrained_sha256=digest, pretrained_hash_verified=digest == expected,
                  pretrained_bytes=weight.stat().st_size if weight.exists() else None)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
