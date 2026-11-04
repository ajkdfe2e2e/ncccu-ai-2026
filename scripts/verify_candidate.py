"""Cross-check completed optimization artifacts without opening test images or loading torch.
This checks local evidence consistency, not official compliance or unbiased generalization.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cxr.safety import safe_output


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify(run, baseline):
    run, baseline = Path(run).resolve(), Path(baseline).resolve()
    report = read_json(run / 'valid.report.json')
    old = read_json(baseline / 'valid.report.json')
    comparison = read_json(run / 'comparison.json')
    cost = read_json(run / 'benchmark-valid' / 'cost.json')
    manifest = read_json(run / 'inference_manifest.json')
    config = read_json(run / 'run.json')
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(run != baseline, 'Candidate must not overwrite baseline')
    for folder, item in ((run, report), (baseline, old)):
        require(item['valid'] and item['split'] == 'valid' and not item['engineering_only'], 'Not a full validation run')
        require(item['studies'] == 859 and item['rows'] == 8590, 'Unexpected competition validation coverage')
        require(item['metrics']['macro_auc_defined_classes'] == 10, 'All ten AUC classes must be defined')
        require(digest(folder/'best.pt') == item['checkpoint_sha256'], 'Checkpoint hash differs from prediction report')
    for key, folder in (('baseline', baseline), ('candidate', run)):
        require(comparison['inputs'][key]['sha256'] == digest(folder/'valid.csv'), f'{key} CSV hash mismatch')
    require(comparison['studies'] == 859 and comparison['patients'] == 228, 'Comparison cohort mismatch')
    for name, metric in (('auc', 'macro_auc'), ('f1_fixed_0_5', 'macro_f1')):
        values = comparison['macro'][name]
        require(math.isclose(values['baseline'], old['metrics'][metric], abs_tol=1e-12), 'Baseline metric mismatch')
        require(math.isclose(values['candidate'], report['metrics'][metric], abs_tol=1e-12), 'Candidate metric mismatch')
        require(math.isclose(values['delta'], values['candidate']-values['baseline'], abs_tol=1e-12), 'Delta mismatch')
    require(config['initialization_lineage']['sha256'] == old['checkpoint_sha256'], 'Unexpected initialization lineage')
    require(cost['checkpoint_sha256'] == report['checkpoint_sha256'], 'Timing checkpoint mismatch')
    require(cost['submission_sha256'] == comparison['inputs']['candidate']['sha256'], 'Timing CSV mismatch')
    require(cost['metadata_sha256'] == comparison['inputs']['metadata']['sha256'], 'Timing metadata mismatch')
    require(cost['samples'] == 100 and cost['warmup'] == 5 and cost['rounds'] == 3 and cost['batch_size'] == 1,
            'Unexpected benchmark protocol')
    require(cost['local_only'] and not cost['local_smoke'], 'Benchmark scope marker mismatch')
    require(cost['inconsistent_studies'] == 0 and not cost['ranking_changed'] and cost['max_probability_error'] <= .001,
            'Prediction consistency failed')
    require(math.isfinite(cost['mean_ms']) and cost['mean_ms'] > 0, 'Invalid timing')
    require(manifest['within_size_limit'] and manifest['total_bytes'] <= 500_000_000, 'Oversized inference artifacts')
    require(len(manifest['files']) == 1 and manifest['files'][0]['sha256'] == report['checkpoint_sha256'], 'Model manifest mismatch')
    require(manifest['total_bytes'] == (run/'best.pt').stat().st_size, 'Manifest byte count mismatch')
    return dict(local_evidence_verified=True, candidate=str(run), baseline=str(baseline),
                recommended_by_validation_auc='candidate' if comparison['macro']['auc']['delta'] > 0 else 'baseline',
                metrics=comparison['macro'], local_mean_ms=cost['mean_ms'], checkpoint_bytes=manifest['total_bytes'],
                official_t4_verified=False, caveat=comparison['caveat'],
                note='Timing and multi-view aggregation remain local auxiliary evidence, not organizer certification.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    a = p.parse_args()
    result = verify(a.run, a.baseline)
    output = safe_output(a.run/'acceptance.json')
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
