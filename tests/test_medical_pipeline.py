import unittest

import numpy as np
import torch
from torch import nn

from scripts.run_medical_transfer import (
    align_and_fuse,
    configure_stage,
    make_optimizer,
    zero_shot_report,
)


class TinyMedicalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4))
        self.classifier = nn.Linear(4, 10)


class MedicalPipelineTests(unittest.TestCase):
    def test_frozen_stage_keeps_feature_batchnorm_eval_and_head_trainable(self):
        model = TinyMedicalModel()

        configure_stage(model, frozen=True)

        self.assertTrue(model.training)
        self.assertFalse(model.features.training)
        self.assertFalse(model.features[1].training)
        self.assertTrue(model.classifier.training)
        self.assertTrue(all(not p.requires_grad for p in model.features.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.classifier.parameters()))

    def test_optimizer_uses_requested_lrs_for_both_stages(self):
        model = TinyMedicalModel()
        configure_stage(model, frozen=True)
        frozen = make_optimizer(model, frozen=True)
        self.assertEqual([(group["name"], group["lr"]) for group in frozen.param_groups],
                         [("classifier", 1e-3)])

        configure_stage(model, frozen=False)
        finetune = make_optimizer(model, frozen=False)
        self.assertEqual([(group["name"], group["lr"]) for group in finetune.param_groups],
                         [("features", 1e-5), ("classifier", 1e-4)])
        self.assertTrue(all(p.requires_grad for p in model.features.parameters()))

    def test_fusion_aligns_study_ids_before_fixed_mean(self):
        medical_studies = [
            {"subject_id": "p2", "study_id": "s2"},
            {"subject_id": "p1", "study_id": "s1"},
        ]
        baseline_studies = list(reversed(medical_studies))
        medical = np.full((2, 10), 0.8)
        baseline = np.stack([np.full(10, 0.2), np.full(10, 0.4)])

        fused = align_and_fuse(medical_studies, baseline_studies, medical, baseline)

        np.testing.assert_allclose(fused[0], 0.6)
        np.testing.assert_allclose(fused[1], 0.5)
        with self.assertRaisesRegex(ValueError, "identical unique IDs"):
            align_and_fuse(medical_studies[:1], baseline_studies, medical[:1], baseline)

    def test_zero_shot_report_contains_nine_classes_and_no_ten_class_macro(self):
        targets = np.zeros((4, 10), dtype=np.float64)
        targets[2:, :9] = 1
        zero_shot = np.tile(np.linspace(0.1, 0.9, 10), (4, 1))
        baseline = zero_shot[:, ::-1].copy()

        report = zero_shot_report(targets, zero_shot, baseline)

        self.assertEqual(report["classes_evaluated"], 9)
        self.assertEqual(len(report["per_class"]), 9)
        self.assertEqual(report["excluded_class"], "No Finding")
        self.assertNotIn("macro_auc_all_ten_classes", report)
        self.assertNotIn("No Finding", [row["name"] for row in report["per_class"]])
        self.assertEqual(report["macro_auc_defined_classes"], 9)


if __name__ == "__main__":
    unittest.main()
