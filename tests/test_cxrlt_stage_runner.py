import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.run_cxrlt_stage import batch_reader, fold_indices, study_layout, predict, verify_plan
from scripts.run_frozen_study_probe import atomic_json, sha256


def record(subject, study, dicom, label=0):
    return {"subject_id": subject, "study_id": study, "dicom_id": dicom, "labels": [label] * 10}


class StageRunnerTests(unittest.TestCase):
    def test_study_order_and_all_image_batch_mapping(self):
        records = [record("p1", "s1", "a"), record("p2", "s2", "b", 1), record("p1", "s1", "c")]
        studies, groups, labels = study_layout(records)
        self.assertEqual([r["study_id"] for r in studies], ["s1", "s2"])
        self.assertEqual(groups, [[0, 2], [1]])
        values = np.arange(3 * 8, dtype=np.float16).reshape(3, 2, 2, 2)
        batch = batch_reader(values, groups, labels)([1, 0])
        batch.validate([1, 0])
        self.assertTrue(torch.equal(batch.image_to_study, torch.tensor([0, 1, 1])))
        np.testing.assert_array_equal(batch.prefix_features.numpy(), values[[1, 0, 2]])
        np.testing.assert_array_equal(batch.targets.numpy(), labels[[1, 0]])

    def test_conflicting_study_patients_and_labels_rejected(self):
        for records in ([record("p1", "s", "a"), record("p2", "s", "b")],
                        [record("p1", "s", "a"), record("p1", "s", "b", 1)]):
            with self.assertRaises(ValueError):
                study_layout(records)

    def test_fixed_patient_fold_isolation(self):
        studies = [record("p1", "s1", "a"), record("p1", "s2", "b"), record("p2", "s3", "c"), record("p3", "s4", "d")]
        # Use the source partition's exact contract through a controlled mapping.
        with patch("scripts.run_cxrlt_stage._fold_memberships", return_value=np.asarray([0, 0, 1, "audit"], dtype=object)):
            train, held = fold_indices(studies, {}, 0)
        np.testing.assert_array_equal(train, [2, 3])
        np.testing.assert_array_equal(held, [0, 1])
        with patch("scripts.run_cxrlt_stage._fold_memberships", return_value=np.asarray([0, 1, 1, "audit"], dtype=object)):
            with self.assertRaisesRegex(ValueError, "patients overlap"):
                fold_indices(studies, {}, 0)

    def test_plan_and_source_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source/cxr/file.py"
            source.parent.mkdir(parents=True)
            source.write_text("fixed")
            original = root / "input.json"
            original.write_text("input")
            plan = {"environment": {"test": True}, "input_locks": {str(original): sha256(original)},
                    "source_locks": {"cxr/file.py": sha256(source)}}
            atomic_json(root / "plan.json", plan)
            atomic_json(root / "plan_lock.json", {"sha256": sha256(root / "plan.json")})
            with patch("scripts.run_cxrlt_stage.environment", return_value={"test": True}):
                verify_plan(root, plan)
                source.write_text("changed")
                with self.assertRaisesRegex(ValueError, "source snapshot"):
                    verify_plan(root, plan)


if __name__ == "__main__":
    unittest.main()
