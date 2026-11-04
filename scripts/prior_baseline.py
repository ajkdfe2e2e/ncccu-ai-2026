"""Training-study label prevalence baseline, validation ONLY, no image decoding.
A constant score gives AUC .5 for every defined class; useful metric sanity check,
NOT a useful submission model and never claimed to be a trained image classifier.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cxr.data import read_records
from cxr.metrics import aggregate_studies, evaluate
from cxr.safety import fresh_run_directory
from cxr.submission import write_submission, validate_submission


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=ROOT / 'cxr_first')
    p.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'prior-baseline')
    a = p.parse_args()
    a.output = fresh_run_directory(a.output, a.data_root)
    train = read_records(a.data_root / 'train.csv', a.data_root / 'train')
    valid = read_records(a.data_root / 'valid.csv', a.data_root / 'valid')
    train_studies = {r['study_id']: r for r in train}
    priors = np.mean([r['labels'] for r in train_studies.values()], axis=0)
    records, probs, targets = aggregate_studies(valid, np.tile(priors, (len(valid), 1)))
    metrics = evaluate(targets, probs)
    if any(x['auc'] not in (.5, None) for x in metrics['per_class']):
        raise AssertionError('Constant baseline AUC should be 0.5')
    output = a.output / 'valid.csv'
    write_submission(output, records, probs)
    summary = dict(kind='constant_train_study_prevalence', image_model=False,
                   train_studies=len(train_studies), valid_studies=len(records),
                   class_priors=priors.tolist(), metrics=metrics,
                   validation_csv=validate_submission(output, valid))
    (a.output / 'report.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
