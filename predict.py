"""Deterministic image inference, study mean aggregation and strict CSV validation."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from cxr.data import read_records
from cxr.engine import Predictor, seed_everything, sha256
from cxr.metrics import aggregate_studies, evaluate
from cxr.submission import write_submission, validate_submission
from cxr.safety import safe_output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('cxr_first'))
    p.add_argument('--split', choices=['valid', 'test'], default='valid')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='auto')
    p.add_argument('--limit-studies', type=int, default=0, help='Smoke on validation only; never partial test submission')
    a = p.parse_args()
    if a.limit_studies < 0 or (a.limit_studies and a.split == 'test'):
        p.error('--limit-studies is nonnegative and allowed only on validation')
    a.output = safe_output(a.output, a.data_root, inputs=[a.checkpoint])
    safe_output(a.output.with_suffix('.report.json'), a.data_root, inputs=[a.checkpoint])
    seed_everything(2026)
    torch.set_num_threads(4)
    records = read_records(a.data_root / f'{a.split}.csv', a.data_root / a.split)
    if a.limit_studies:
        ids = set(list(dict.fromkeys(r['study_id'] for r in records))[:a.limit_studies])
        records = [r for r in records if r['study_id'] in ids]
    predictor = Predictor(a.checkpoint, a.device)
    if a.split == 'test' and predictor.smoke:
        raise ValueError('Refusing test prediction from an engineering-only smoke checkpoint')
    study_records, probabilities, targets = aggregate_studies(records, predictor.predict_records(records))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_submission(a.output, study_records, probabilities, predictor.thresholds)
    report = validate_submission(a.output, records)
    report.update(checkpoint_sha256=sha256(a.checkpoint), split=a.split,
                  aggregation='study_mean', inference_precision=predictor.precision, engineering_only=predictor.smoke)
    if targets is not None:
        report['metrics'] = evaluate(targets, probabilities, predictor.thresholds)
    a.output.with_suffix('.report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
