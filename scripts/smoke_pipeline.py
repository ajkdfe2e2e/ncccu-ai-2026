"""Exercise train -> checkpoint -> validation CSV -> consistency on real validation only.
Creates a fresh run under outputs/ and uses 8 complete train/valid studies.
No test image is opened. This is an engineering check, not a quality estimate.
"""
import argparse
import json
import runpy
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def invoke(script, arguments):
    old = sys.argv
    try:
        sys.argv = [str(ROOT / script), *[str(x) for x in arguments]]
        runpy.run_path(str(ROOT / script), run_name='__main__')
    finally:
        sys.argv = old


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'cxr_first')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--pretrained', type=Path)
    args = parser.parse_args()
    from cxr.data import read_records
    import torch
    output = ROOT / 'outputs' / ('smoke-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    training = ['--data-root', args.data_root, '--output', output, '--epochs', 1,
                '--batch-size', 2, '--image-size', 64, '--smoke-studies', 8, '--device', args.device]
    training += ['--pretrained', args.pretrained] if args.pretrained else ['--from-scratch']
    invoke('train.py', training)
    checkpoint = output / 'best.pt'
    invoke('predict.py', ['--data-root', args.data_root, '--split', 'valid', '--limit-studies', 8,
                          '--checkpoint', checkpoint, '--output', output / 'valid.csv', '--device', args.device])
    # Dedicated validation subset manifest and image folders for an honest small benchmark.
    import csv
    records = read_records(args.data_root / 'valid.csv', args.data_root / 'valid')
    ids = set(list(dict.fromkeys(r['study_id'] for r in records))[:8])
    records = [r for r in records if r['study_id'] in ids]
    metadata = output / 'valid.csv.metadata.csv'
    with metadata.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Dicom_id', 'Subject_id', 'Study_id', 'Predict_class'])
        for record in records:
            writer.writerow([record['dicom_id'], record['subject_id'], record['study_id'],
                             ';'.join(str(i+1) for i, value in enumerate(record['labels']) if value)])
    measured, warmup = output / 'measured', output / 'warmup'
    measured.mkdir()
    warmup.mkdir()
    # Two images from disjoint IDs; the checker completes multi-view studies from original valid root.
    for record, dest in zip(records[:2], (warmup, measured)):
        source = Path(record['path'])
        shutil.copyfile(source, dest / source.name)
    invoke('cost.py', [output / 'valid.csv', measured, warmup, '--checkpoint', checkpoint,
                      '--metadata', metadata, '--image-root', args.data_root / 'valid',
                      '--device', args.device, '--local-smoke', '--output', output / 'cost.json'])
    summary = dict(status='passed', engineering_only=True, output=str(output),
                   torch=str(torch.__version__), cuda_available=torch.cuda.is_available(),
                   checkpoint_bytes=checkpoint.stat().st_size)
    (output / 'smoke_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
