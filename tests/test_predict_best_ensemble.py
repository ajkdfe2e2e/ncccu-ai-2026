"""Submission replay must preserve coefficients and reject wrong split IDs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cxr.submission import validate_submission, write_submission
from scripts.predict_best_ensemble import load_scores, replay_recipe


class FrozenSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.recipe = {
            "baseline_per_class_selection": [
                {"class": j, "kind": "full_model_blend", "left": "a", "right": "b", "alpha_left": 0.25}
                for j in range(1, 11)
            ],
            "classes": [{"class": j, "alpha_candidate": 0.0, "source": "immutable_saved_ensemble"}
                        for j in range(1, 11)],
        }
        self.sources = {"a": np.full((2, 10), 0.2), "b": np.full((2, 10), 0.6),
                        "rad_dino_tta": np.full((2, 10), 0.9)}

    def test_replay_uses_recorded_weights_without_changing_inputs(self):
        recipe_before = copy.deepcopy(self.recipe)
        self.recipe["classes"][0] = {
            "class": 1, "alpha_candidate": 0.5,
            "source": "yaLLMan/rad-dino-chexpert fine-tuned on the local train split",
        }
        baseline, result = replay_recipe(self.sources, self.recipe)
        np.testing.assert_allclose(baseline, 0.5)
        np.testing.assert_allclose(result[:, 0], 0.7)
        np.testing.assert_allclose(result[:, 1:], 0.5)
        np.testing.assert_array_equal(self.sources["a"], np.full((2, 10), 0.2))
        self.assertEqual(self.recipe["baseline_per_class_selection"], recipe_before["baseline_per_class_selection"])

    def test_missing_class_and_invalid_coefficients_fail_closed(self):
        self.sources["b"][0, 4] = np.nan
        with self.assertRaisesRegex(ValueError, "no valid prediction for class 5"):
            replay_recipe(self.sources, self.recipe)
        self.sources["b"][0, 4] = 0.6
        for alpha in (float("nan"), -0.1, 1.1):
            with self.subTest(alpha=alpha):
                self.recipe["baseline_per_class_selection"][0]["alpha_left"] = alpha
                with self.assertRaisesRegex(ValueError, "coefficient"):
                    replay_recipe(self.sources, self.recipe)

    def test_duplicate_or_missing_recipe_classes_are_rejected(self):
        self.recipe["classes"][0]["class"] = 2
        with self.assertRaisesRegex(ValueError, "exactly one entry"):
            replay_recipe(self.sources, self.recipe)

    def test_component_rows_are_aligned_by_ids_and_wrong_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            studies = [{"subject_id": "p1", "study_id": "test1"},
                       {"subject_id": "p2", "study_id": "test2"}]
            (folder / "study_ids.json").write_text(json.dumps([["p2", "test2"], ["p1", "test1"]]))
            (folder / "report.json").write_text(json.dumps({"local_label_indices": [4, 8]}))
            np.save(folder / "study_probabilities.npy", np.array([[0.8, 0.9], [0.2, 0.3]]))
            scores = load_scores(folder, studies)
            np.testing.assert_array_equal(scores[:, [4, 8]], [[0.2, 0.3], [0.8, 0.9]])
            self.assertTrue(np.isnan(scores[:, 0]).all())
            (folder / "study_ids.json").write_text(json.dumps([["p2", "valid2"], ["p1", "valid1"]]))
            with self.assertRaisesRegex(ValueError, "requested split"):
                load_scores(folder, studies)

    def test_validation_csv_cannot_pass_test_manifest_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "submission.csv"
            valid = [{"subject_id": "p1", "study_id": "valid1"}]
            test = [{"subject_id": "p2", "study_id": "test1"}]
            write_submission(path, valid, np.full((1, 10), 0.5))
            with self.assertRaisesRegex(ValueError, "unexpected subject/study"):
                validate_submission(path, test)


if __name__ == "__main__":
    unittest.main()
