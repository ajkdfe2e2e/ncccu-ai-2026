"""Fully decode train/valid images only, without changing them or reading test images."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cxr.data import read_records
from cxr.safety import safe_output


def check(record):
    try:
        with Image.open(record['path']) as image:
            image.load()
            if min(image.size) < 1:
                raise ValueError('Empty image')
        return None
    except Exception as exc:
        return dict(dicom_id=record['dicom_id'], error=f'{type(exc).__name__}: {exc}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=ROOT / 'cxr_first')
    p.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'image_integrity.json')
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    if not 1 <= a.workers <= 8:
        p.error('--workers must be 1..8 to bound image memory')
    output = safe_output(a.output, a.data_root)
    report = dict(test_images_opened=0, splits={})
    start = time.perf_counter()
    for split in ('train', 'valid'):
        records = read_records(a.data_root / f'{split}.csv', a.data_root / split)
        failures = []
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            for i, result in enumerate(pool.map(check, records), 1):
                if result:
                    failures.append(result)
                if i % 2000 == 0:
                    print(f'{split}: decoded {i}/{len(records)}, errors={len(failures)}', flush=True)
        report['splits'][split] = dict(checked=len(records), failures=failures)
    report['elapsed_seconds'] = time.perf_counter() - start
    report['passed'] = not any(x['failures'] for x in report['splits'].values())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
