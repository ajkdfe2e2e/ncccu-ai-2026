import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts.predict_rad_dino_oof_tta import (
    LOCAL_LABELS, align_and_compare_original, predict_image_views, select_smoke_records,
    validate_source_provenance,
)


class RadDinoOOFTTATests(unittest.TestCase):
    def _provenance_fixture(self):
        train = [{"subject_id": "p1"}, {"subject_id": "p2"}]
        held = [{"subject_id": "p3"}]
        payload = {"labels": LOCAL_LABELS, "epoch": 3, "variant": "raw", "provenance": {
            "phase": "cv", "fold": 1, "head_lr": 1e-3, "epochs": 3,
            "checkpoint_policy": "fixed_final_epoch", "train_metadata_sha256": "train",
            "partition_sha256": "partition", "public_checkpoint_sha256": "public",
            "train_images": 2, "evaluation_images": 1, "train_patients": 2,
            "evaluation_patients": 1,
        }}
        report = {"status": "complete", "phase": "cv", "fold": 1, "head_lr": 1e-3,
                  "epochs": 3, "checkpoint_policy": "fixed_final_epoch",
                  "train_metadata_sha256": "train", "partition_sha256": "partition",
                  "public_checkpoint_sha256": "public", "train_images": 2,
                  "train_patients": 2, "evaluation_images": 1, "evaluation_patients": 1,
                  "variants": {"raw": {"checkpoint_sha256": "checkpoint"}}}
        return train, held, payload, report

    def test_validates_source_split_and_checkpoint_provenance(self):
        train, held, payload, report = self._provenance_fixture()
        validate_source_provenance(Path("cv_A_fold1"), report, payload, phase="cv", fold=1,
            train_records=train, held_records=held, train_metadata_sha256="train",
            partition_sha256="partition", public_checkpoint_sha256="public",
            checkpoint_sha256="checkpoint")
        report["fold"] = 0
        with self.assertRaisesRegex(ValueError, "phase/fold"):
            validate_source_provenance(Path("cv_A_fold1"), report, payload, phase="cv", fold=1,
                train_records=train, held_records=held, train_metadata_sha256="train",
                partition_sha256="partition", public_checkpoint_sha256="public",
                checkpoint_sha256="checkpoint")

    def test_rejects_checkpoint_trained_on_different_partition(self):
        train, held, payload, report = self._provenance_fixture()
        payload["provenance"]["partition_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "checkpoint training provenance"):
            validate_source_provenance(Path("cv_A_fold1"), report, payload, phase="cv", fold=1,
                train_records=train, held_records=held, train_metadata_sha256="train",
                partition_sha256="partition", public_checkpoint_sha256="public",
                checkpoint_sha256="checkpoint")

    def test_predicts_original_and_float32_flip_average_and_reports_progress(self):
        class Model:
            def eval(self):
                return self

            def __call__(self, pixel_values):
                value = pixel_values[:, 0, 0, 0]
                return value[:, None].expand(-1, 10)

        records = [{}, {}, {}]
        batches = [
            (torch.tensor([[[[0., 1.]]], [[[1., 1.]]]]), None),
            (torch.tensor([[[[0., 0.]]]]), None),
        ]
        updates = []
        with patch("scripts.predict_rad_dino_oof_tta.ImageRecords", return_value=object()), \
             patch("scripts.predict_rad_dino_oof_tta.DataLoader", return_value=batches):
            original, tta = predict_image_views(Model(), records, None, torch.device("cpu"), 2,
                                                progress_callback=lambda done, total: updates.append((done, total)))
        self.assertEqual(original.dtype, np.float32)
        self.assertEqual(tta.dtype, np.float32)
        np.testing.assert_allclose(original[:, 0], torch.sigmoid(torch.tensor([0., 1., 0.])).numpy())
        expected_tta_first = (torch.sigmoid(torch.tensor(0.)).item() +
                              torch.sigmoid(torch.tensor(1.)).item()) / 2
        np.testing.assert_allclose(tta[:, 0], [expected_tta_first,
                                               torch.sigmoid(torch.tensor(1.)).item(), 0.5])
        self.assertEqual(updates, [(3, 3)])

    def test_smoke_selection_keeps_complete_studies(self):
        records = [
            {"subject_id": "p1", "study_id": "s1", "image": 1},
            {"subject_id": "p1", "study_id": "s1", "image": 2},
            {"subject_id": "p2", "study_id": "s2", "image": 3},
            {"subject_id": "p3", "study_id": "s3", "image": 4},
            {"subject_id": "p4", "study_id": "s4", "image": 5},
            {"subject_id": "p4", "study_id": "s4", "image": 6},
        ]
        selected = select_smoke_records(records, 3, batch_size=2)
        self.assertEqual([row["image"] for row in selected], [1, 2, 3, 4])
        self.assertEqual(len(selected) % 2, 0)
        self.assertEqual(select_smoke_records(records, 0), records)

    def test_aligns_saved_raw_predictions_by_study_id(self):
        studies = [{"subject_id": "p2", "study_id": "s2"}, {"subject_id": "p1", "study_id": "s1"}]
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "study_ids.json").write_text('[ ["p1", "s1"], ["p2", "s2"] ]', encoding="utf-8")
            np.save(folder / "study_probabilities.npy", np.array([[.2] * 10, [.8] * 10]))
            self.assertEqual(align_and_compare_original(folder, studies, np.array([[.8] * 10, [.2] * 10])), 0.0)


if __name__ == "__main__":
    unittest.main()
