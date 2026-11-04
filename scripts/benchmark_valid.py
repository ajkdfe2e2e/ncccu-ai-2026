"""Run 100-image local timing and consistency on validation only (never test)."""
import argparse
import runpy
import shutil
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cxr.data import read_records
from cxr.safety import fresh_run_directory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=ROOT / 'cxr_first')
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    records = read_records(a.data_root / 'valid.csv', a.data_root / 'valid')
    if len(records) < 105:
        raise ValueError('Need 105 distinct validation images')
    directory = fresh_run_directory(a.run / 'benchmark-valid', a.data_root)
    samples, warmup = directory / 'samples', directory / 'warmup'
    samples.mkdir()
    warmup.mkdir()
    for i, record in enumerate(records[:105]):
        source = Path(record['path'])
        shutil.copyfile(source, (warmup if i < 5 else samples) / source.name)
    sys.argv = [str(ROOT / 'cost.py'), str(a.run / 'valid.csv'), str(samples), str(warmup),
                '--checkpoint', str(a.run / 'best.pt'), '--metadata', str(a.data_root / 'valid.csv'),
                '--image-root', str(a.data_root / 'valid'), '--device', a.device,
                '--output', str(directory / 'cost.json')]
    runpy.run_path(str(ROOT / 'cost.py'), run_name='__main__')


if __name__ == '__main__':
    main()
