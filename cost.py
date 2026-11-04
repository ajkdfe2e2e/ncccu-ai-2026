"""Local batch=1 benchmark; not an official T4 certification.
Study-level CSV consistency requires ALL views of each checked study. Missing
views cause an explicit error, never a false 'zero mismatch' conclusion.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from cxr.data import read_records
from cxr.engine import Predictor, seed_everything, sha256
from cxr.metrics import aggregate_studies
from cxr.consistency import compare_predictions
from cxr.submission import validate_submission
from cxr.safety import safe_output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('submission', type=Path)
    p.add_argument('pictures_100', type=Path)
    p.add_argument('pictures_5', type=Path)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--image-root', type=Path, required=True, help='Full split root, for study consistency checking')
    p.add_argument('--device', default='auto')
    p.add_argument('--output', type=Path, default=Path('outputs/cost.json'))
    p.add_argument('--local-smoke', action='store_true', help='Permit fewer than 100/5 files; not official benchmark')
    a = p.parse_args()
    a.output = safe_output(a.output, a.image_root.parent, inputs=[a.submission, a.checkpoint, a.metadata])
    seed_everything(2026)
    torch.set_num_threads(4)
    def paths(root):
        return sorted(x for x in root.rglob('*') if x.is_file() and x.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    samples, warmup = paths(a.pictures_100), paths(a.pictures_5)
    if not samples or not warmup:
        raise ValueError('Both benchmark and warmup folders must contain images')
    if not a.local_smoke and (len(samples) != 100 or len(warmup) != 5):
        raise ValueError('Official protocol requires exactly 100 sample images and 5 warmup images')
    if set(samples) & set(warmup):
        raise ValueError('Warmup and measured images must be separate')
    records = read_records(a.metadata, a.image_root)
    validate_submission(a.submission, records)
    lookup = {r['dicom_id']: r for r in records}
    for group in (samples, warmup):
        if len(set(x.stem for x in group)) != len(group) or any(x.stem not in lookup for x in group):
            raise ValueError('Sample and warmup IDs must be unique and present in metadata')
    if {x.stem for x in samples} & {x.stem for x in warmup}:
        raise ValueError('Warmup and sample image IDs overlap')
    for path in warmup:
        if sha256(path) != sha256(lookup[path.stem]['path']):
            raise ValueError('Warmup bytes do not match source')
    checkpoint_hash = sha256(a.checkpoint)
    submission_hash = sha256(a.submission)
    metadata_hash = sha256(a.metadata)
    predictor = Predictor(a.checkpoint, a.device)
    for path in warmup:
        predictor.predict_image(path)
    def sync():
        if predictor.device.type == 'cuda':
            torch.cuda.synchronize(predictor.device)
    timings, rounds = [], []
    for _ in range(3):
        elapsed, predictions = [], []
        for path in samples:
            sync()
            start = time.perf_counter()
            pred = predictor.predict_image(path)
            sync()
            elapsed.append((time.perf_counter() - start) * 1000)
            predictions.append(pred)
        timings.append(elapsed)
        rounds.append(np.stack(predictions))
    if any(not np.array_equal(rounds[0], x) for x in rounds[1:]):
        raise ValueError('Repeated fixed-parameter inference is not deterministic')
    # Verify sampled bytes match source; do this outside measured interval.
    for path in samples:
        if sha256(path) != sha256(lookup[path.stem]['path']):
            raise ValueError(f'Sample does not match full-source image: {path.name}')
    selected_studies = {lookup[x.stem]['study_id'] for x in samples}
    all_views = [r for r in records if r['study_id'] in selected_studies]
    sampled = {path.stem: rounds[0][i] for i, path in enumerate(samples)}
    predictions = np.stack([sampled[r['dicom_id']] if r['dicom_id'] in sampled else predictor.predict_image(r['path']) for r in all_views])
    study_records, study_probs, _ = aggregate_studies(all_views, predictions)
    with a.submission.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    expected = {(r['Study_id'], int(r['Predict_class'])): (float(r['Probability']), float(r['Gate'] or .5)) for r in rows}
    originals, gate_rows = [], []
    for record in study_records:
        pairs = [expected[(record['study_id'], cls)] for cls in range(1, 11)]
        originals.append([x[0] for x in pairs])
        gate_rows.append([x[1] for x in pairs])
    originals, gate_rows = np.asarray(originals), np.asarray(gate_rows)
    if not np.all(gate_rows == gate_rows[0]):
        raise ValueError('This baseline requires fixed per-class thresholds across studies')
    consistency = compare_predictions(study_probs, originals, predictor.thresholds, gate_rows[0])
    mismatches = consistency['inconsistent_studies']
    max_error = consistency['max_probability_error']
    mean_ms = float(np.asarray(timings).mean(axis=0).mean())
    if (sha256(a.checkpoint) != checkpoint_hash or sha256(a.submission) != submission_hash
            or sha256(a.metadata) != metadata_hash):
        raise ValueError('Input checkpoint/CSV changed during benchmark')
    report = dict(checkpoint_sha256=checkpoint_hash, submission_sha256=submission_hash,
                  metadata_sha256=metadata_hash,
                  local_only=True, local_smoke=a.local_smoke, device=str(predictor.device),
                  device_name=torch.cuda.get_device_name(predictor.device) if predictor.device.type == 'cuda' else 'CPU',
                  samples=len(samples), warmup=len(warmup), rounds=3, batch_size=1,
                  mean_ms=mean_ms, checked_studies=len(study_records), inconsistent_studies=mismatches,
                  max_probability_error=max_error, ranking_changed=consistency['ranking_changed'], precision=predictor.precision,
                  note='Study consistency uses all views. Confirm official partial-view benchmark policy with organizer.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('[CONCLUSION] ' + json.dumps(report))
    if mismatches:
        raise SystemExit('Prediction consistency failed')


if __name__ == '__main__':
    main()
