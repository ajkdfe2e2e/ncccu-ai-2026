import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from scripts.train_resnet_grouped import (
    EPOCHS,
    FINETUNE_LR,
    INITIAL_LR,
    MIN_LR,
    phase_spec,
    select_records,
    smoke_subset,
    study_weights,
    _optimizer,
    optimizer_step,
    _restore_resume,
    _save_resume,
)


def row(dicom, patient, study, positive):
    labels = [0.0] * 10
    labels[0] = float(positive)
    return {"dicom_id": dicom, "subject_id": patient, "study_id": study,
            "labels": labels, "path": f"/{dicom}.jpg"}


class ResnetGroupedTests(unittest.TestCase):
    def setUp(self):
        self.records = [row("a", "p1", "s1", 1), row("b", "p1", "s1", 1),
                        row("c", "p2", "s2", 0), row("d", "p3", "s3", 1)]
        self.partition = {"folds": 2, "subject_partition": {"p1": 0, "p2": 1, "p3": "audit"}}

    def test_cv_and_audit_are_patient_disjoint_and_exclude_other_buckets(self):
        train, held = select_records(self.records, self.partition, "cv", 0)
        self.assertEqual({r["subject_id"] for r in train}, {"p2"})
        self.assertEqual({r["subject_id"] for r in held}, {"p1"})
        train, held = select_records(self.records, self.partition, "audit", None)
        self.assertEqual({r["subject_id"] for r in train}, {"p1", "p2"})
        self.assertEqual({r["subject_id"] for r in held}, {"p3"})

    def test_partition_must_cover_records_and_fold_is_required_only_for_cv(self):
        with self.assertRaises(ValueError):
            select_records(self.records, {"folds": 2, "subject_partition": {"p1": 0}}, "cv", 0)
        with self.assertRaises(ValueError):
            select_records(self.records, self.partition, "audit", 0)
        with self.assertRaises(ValueError):
            select_records(self.records, self.partition, "cv", None)

    def test_smoke_caps_images_without_changing_patient_scope(self):
        result = smoke_subset(self.records, 2)
        self.assertEqual([row["dicom_id"] for row in result], ["a", "b"])
        self.assertEqual(smoke_subset(self.records, 0), self.records)

    def test_study_balanced_weights_and_clipped_positive_weights(self):
        weights, pos_weight = study_weights(self.records)
        torch.testing.assert_close(weights, torch.tensor([0.5, 0.5, 1.0, 1.0]))
        torch.testing.assert_close(pos_weight[:2], torch.tensor([1.0, 3.0]))
        self.assertEqual(tuple(pos_weight.shape), (10,))

    def test_fixed_one_plus_four_optimizer_schedule(self):
        model = nn.Linear(1, 1)
        phase, optimizer, scheduler = _optimizer(model, 0)
        self.assertEqual((phase, optimizer.param_groups[0]["lr"], scheduler),
                         ("initial", INITIAL_LR, None))
        phase, optimizer, scheduler = _optimizer(model, 1)
        self.assertEqual(phase, "finetune")
        self.assertEqual(optimizer.param_groups[0]["lr"], FINETUNE_LR)
        self.assertEqual(scheduler.T_max, 4)
        self.assertEqual(scheduler.eta_min, MIN_LR)
        self.assertEqual(EPOCHS, 5)
        self.assertEqual([phase_spec(i)[0] for i in range(EPOCHS)],
                         ["initial", "finetune", "finetune", "finetune", "finetune"])

    def test_finetune_epoch_learning_rates_follow_cosine_schedule(self):
        model = nn.Linear(1, 1)
        _, optimizer, scheduler = _optimizer(model, 1)
        logged_lrs = []
        for _ in range(4):
            logged_lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(logged_lrs[0], FINETUNE_LR)
        self.assertTrue(all(left > right for left, right in zip(logged_lrs, logged_lrs[1:])))
        self.assertAlmostEqual(logged_lrs[2], (FINETUNE_LR + MIN_LR) / 2, places=9)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], MIN_LR, places=9)

    def test_epoch_boundary_resume_restores_phase_and_scheduler(self):
        config = {"run": "test"}
        model = nn.Linear(2, 1)
        _, optimizer, scheduler = _optimizer(model, 1)
        generator = torch.Generator().manual_seed(2026)
        history = [{"epoch": 2}]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resume.pt"
            with patch("torch.cuda.is_available", return_value=False):
                _save_resume(path, config, 1, model, optimizer, scheduler,
                             torch.amp.GradScaler("cuda", enabled=False), generator,
                             history, 4, 0)
                restored = nn.Linear(2, 1)
                _, restored_optimizer, restored_scheduler = _optimizer(restored, 1)
                result = _restore_resume(
                    path, config, restored, restored_optimizer, restored_scheduler,
                    torch.amp.GradScaler("cuda", enabled=False), torch.Generator(),
                )
        completed, _, loaded_scheduler, loaded_history, updates, skipped = result
        self.assertEqual(completed, 2)
        self.assertEqual(loaded_scheduler.T_max, 4)
        self.assertEqual(loaded_history, history)
        self.assertEqual((updates, skipped), (4, 0))

    def test_each_batch_has_its_own_optimizer_update(self):
        model = nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        initial = model.weight.detach().clone()
        successful_updates = 0
        for value in (1.0, 2.0, 3.0):
            optimizer.zero_grad(set_to_none=True)
            model(torch.tensor([[value]])).sum().backward()
            successful_updates += optimizer_step(model, optimizer, scaler)
            self.assertIsNone(model.weight.grad)
        self.assertEqual(successful_updates, 3)
        self.assertFalse(torch.equal(model.weight.detach(), initial))


if __name__ == "__main__":
    unittest.main()
