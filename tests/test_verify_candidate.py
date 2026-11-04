import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from scripts.verify_candidate import verify


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.run = Path('outputs/synthetic-candidate').resolve()
        self.base = Path('outputs/synthetic-baseline').resolve()
        report = dict(valid=True, split='valid', engineering_only=False, studies=859, rows=8590,
                      metrics=dict(macro_auc_defined_classes=10, macro_auc=.75, macro_f1=.4), checkpoint_sha256='new')
        old = copy.deepcopy(report)
        old.update(checkpoint_sha256='old')
        old['metrics'].update(macro_auc=.7, macro_f1=.3)
        comparison = dict(studies=859, patients=228, inputs={k:dict(sha256=k) for k in ('baseline','candidate')},
                          macro={'auc':dict(baseline=.7,candidate=.75,delta=.05),
                                 'f1_fixed_0_5':dict(baseline=.3,candidate=.4,delta=.1)}, caveat='descriptive')
        comparison['inputs']['metadata'] = dict(sha256='metadata')
        self.items = {
            self.run/'valid.report.json': report, self.base/'valid.report.json': old,
            self.run/'comparison.json': comparison,
            self.run/'run.json': dict(initialization_lineage=dict(sha256='old')),
            self.run/'benchmark-valid'/'cost.json': dict(samples=100,warmup=5,rounds=3,batch_size=1,
                checkpoint_sha256='new',submission_sha256='candidate',metadata_sha256='metadata',
                local_only=True,local_smoke=False,inconsistent_studies=0,ranking_changed=False,max_probability_error=0.,mean_ms=60.),
            self.run/'inference_manifest.json': dict(within_size_limit=True,total_bytes=100,files=[dict(sha256='new')])}
        self.hashes = {self.run/'best.pt':'new',self.base/'best.pt':'old',self.run/'valid.csv':'candidate',self.base/'valid.csv':'baseline'}

    def check(self):
        with patch('scripts.verify_candidate.read_json',side_effect=lambda p:self.items[Path(p)]), \
             patch('scripts.verify_candidate.digest',side_effect=lambda p:self.hashes[Path(p)]), \
             patch('pathlib.Path.stat',return_value=SimpleNamespace(st_size=100)):
            return verify(self.run,self.base)

    def test_valid(self):
        self.assertEqual(self.check()['recommended_by_validation_auc'],'candidate')
        self.assertFalse(self.check()['official_t4_verified'])

    def test_checkpoint_mismatch(self):
        self.hashes[self.run/'best.pt']='changed'
        with self.assertRaisesRegex(ValueError,'Checkpoint hash'):
            self.check()

    def test_csv_mismatch(self):
        self.hashes[self.run/'valid.csv']='changed'
        with self.assertRaisesRegex(ValueError,'CSV hash'):
            self.check()

    def test_metric_mismatch(self):
        self.items[self.run/'comparison.json']['macro']['auc']['candidate']=.9
        with self.assertRaisesRegex(ValueError,'metric mismatch'):
            self.check()

    def test_consistency_failure(self):
        self.items[self.run/'benchmark-valid'/'cost.json']['inconsistent_studies']=1
        with self.assertRaisesRegex(ValueError,'consistency failed'):
            self.check()

    def test_stale_timing_checkpoint(self):
        self.items[self.run/'benchmark-valid'/'cost.json']['checkpoint_sha256']='old'
        with self.assertRaisesRegex(ValueError,'Timing checkpoint'):
            self.check()

    def test_stale_timing_csv(self):
        self.items[self.run/'benchmark-valid'/'cost.json']['submission_sha256']='old'
        with self.assertRaisesRegex(ValueError,'Timing CSV'):
            self.check()

    def test_bad_size(self):
        self.items[self.run/'inference_manifest.json']['total_bytes']=500_000_001
        with self.assertRaisesRegex(ValueError,'Oversized'):
            self.check()

    def test_baseline_retained_if_candidate_worse(self):
        self.items[self.run/'valid.report.json']['metrics']['macro_auc']=.69
        self.items[self.run/'comparison.json']['macro']['auc'].update(candidate=.69,delta=-.01)
        self.assertEqual(self.check()['recommended_by_validation_auc'],'baseline')


if __name__ == '__main__':
    unittest.main()
