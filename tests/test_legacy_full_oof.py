import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.run_legacy_full_oof import (
    MODEL_ORDER,
    apply_public_weights,
    aligned_component,
    gate_passed,
    locked_recipe,
    sha256,
    stage_folder,
    verify_local_report,
    weight_delta,
    write_json,
)


class LegacyFullOofTests(unittest.TestCase):
    def _component(self, folder, ids, targets):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "study_ids.json").write_text(json.dumps(ids), encoding="utf-8")
        np.save(folder / "study_targets.npy", np.asarray(targets, dtype=np.float64))
        np.save(folder / "study_probabilities.npy", np.full((len(ids), 10), 0.4))

    def test_alignment_rejects_duplicate_and_missing_studies(self):
        studies = [
            {"subject_id": "p1", "study_id": "s1", "labels": [0] * 10},
            {"subject_id": "p2", "study_id": "s2", "labels": [1] * 10},
        ]
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self._component(folder, [["p1", "s1"], ["p1", "s1"]], [[0] * 10] * 2)
            with self.assertRaises(ValueError):
                aligned_component(folder, studies)
            self._component(folder, [["p1", "s1"]], [[0] * 10])
            with self.assertRaises(ValueError):
                aligned_component(folder, studies)

    def test_alignment_rejects_target_mismatch(self):
        studies = [{"subject_id": "p1", "study_id": "s1", "labels": [0] * 10}]
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self._component(folder, [["p1", "s1"]], [[1] + [0] * 9])
            with self.assertRaisesRegex(ValueError, "labels differ"):
                aligned_component(folder, studies)

    def test_zero_delta_preserves_anchor_exactly(self):
        rng = np.random.default_rng(17)
        components = rng.random((7, len(MODEL_ORDER), 10))
        prior = np.full((len(MODEL_ORDER), 10), 1 / len(MODEL_ORDER))
        anchor = apply_public_weights(components, prior)
        result = weight_delta(anchor, components, prior, prior.copy())
        np.testing.assert_array_equal(result, anchor)

    def test_full_eight_source_delta_is_convex_and_keeps_unavailable_zero_heads(self):
        rng = np.random.default_rng(23)
        components = rng.random((9, len(MODEL_ORDER), 10))
        unavailable = 7
        components[:, unavailable, 3] = np.nan
        prior = np.zeros((len(MODEL_ORDER), 10))
        prior[0] = 1.0
        trained = prior.copy()
        trained[0] = 0.8
        trained[1] = 0.2
        trained[:, 3] = prior[:, 3]
        anchor = apply_public_weights(components, prior)
        result = weight_delta(anchor, components, prior, trained)
        expected = apply_public_weights(components, trained)
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(np.all((result >= 0) & (result <= 1)))
        np.testing.assert_array_equal(result[:, 3], anchor[:, 3])
        np.testing.assert_allclose(trained.sum(axis=0), 1.0, rtol=0, atol=0)
        self.assertEqual(trained[unavailable, 3], 0.0)

    def test_guardrail_requires_all_classes_gain_ci_and_enough_repeats(self):
        def comparison(defined=10, delta=0.002, lower=0.0001, repeats=1000, interval=True):
            return {
                "macro_auc_defined_classes": defined,
                "macro": {"auc": {
                    "delta": delta,
                    "ci95_percentile": [lower, 0.01] if interval else None,
                    "defined_replicates": repeats,
                }},
            }

        self.assertTrue(gate_passed(comparison()))
        self.assertFalse(gate_passed(comparison(repeats=949)))
        self.assertFalse(gate_passed(comparison(lower=0.0)))
        self.assertFalse(gate_passed(comparison(defined=9)))
        self.assertFalse(gate_passed(comparison(delta=0.0009)))
        self.assertFalse(gate_passed(comparison(interval=False)))

    def test_provenance_rejects_smoke_and_train_evaluation_overlap(self):
        records = [
            {"subject_id": subject, "study_id": f"s-{subject}", "dicom_id": f"d-{subject}", "labels": [0] * 10}
            for subject in ("p1", "p2", "p3", "p4")
        ]
        partition = {"folds": 3, "subject_partition": {"p1": 0, "p2": 1, "p3": 2, "p4": "audit"}}
        good = {
            "status": "complete", "smoke": False, "phase": "cv", "fold": 0,
            "train_metadata_sha256": "train", "partition_sha256": "partition",
            "train_subjects": ["p2", "p3"], "held_subjects": ["p1"],
        }
        self.assertEqual(len(verify_local_report(good, records, partition, "cv", 0, "train", "partition")), 1)
        smoke = dict(good, smoke=True)
        with self.assertRaisesRegex(ValueError, "provenance"):
            verify_local_report(smoke, records, partition, "cv", 0, "train", "partition")
        overlap = dict(good, train_subjects=["p1", "p2", "p3"])
        with self.assertRaisesRegex(ValueError, "provenance"):
            verify_local_report(overlap, records, partition, "cv", 0, "train", "partition")

    def test_pending_recipe_recovers_before_state_hash_registration(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            pending = output / "trained_recipe.pending.json"
            fitted = {"plan_sha256": "plan", "official_valid_labels_read": False, "weights": []}
            write_json(pending, fitted)
            state = {}

            self.assertEqual(locked_recipe(output, "plan", state), fitted)
            self.assertEqual(state["recipe_sha256"], sha256(output / "trained_recipe.json"))
            self.assertTrue((output / "trained_recipe.json").is_file())
            self.assertFalse(pending.exists())

    def test_pending_recipe_recovers_after_state_hash_registration(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            pending = output / "trained_recipe.pending.json"
            fitted = {"plan_sha256": "plan", "official_valid_labels_read": False, "weights": []}
            write_json(pending, fitted)
            registered_sha = sha256(pending)
            state = {"recipe_sha256": registered_sha}

            self.assertEqual(locked_recipe(output, "plan", state), fitted)
            self.assertEqual(state["recipe_sha256"], registered_sha)
            self.assertTrue((output / "trained_recipe.json").is_file())
            self.assertFalse(pending.exists())

    def test_locked_recipe_rejects_plan_or_fingerprint_mismatch_and_missing_file(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            final = output / "trained_recipe.json"
            fitted = {"plan_sha256": "different-plan", "official_valid_labels_read": False, "weights": []}
            write_json(final, fitted)
            with self.assertRaisesRegex(ValueError, "differs"):
                locked_recipe(output, "plan", {"recipe_sha256": sha256(final)})

            fitted["plan_sha256"] = "plan"
            write_json(final, fitted)
            with self.assertRaisesRegex(ValueError, "differs"):
                locked_recipe(output, "plan", {"recipe_sha256": "stale-hash"})
            final.unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                locked_recipe(output, "plan", {"recipe_sha256": "registered-hash"})

    def test_stage_folder_requires_direct_child_of_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            runs = output / "runs"
            runs.mkdir()
            accepted = stage_folder(output, {"stages": {"resnet_x": {"output": str(runs / "resnet_x")}}},
                                    "resnet_x")
            self.assertEqual(accepted, runs / "resnet_x")
            with self.assertRaisesRegex(ValueError, "outside"):
                stage_folder(output, {"stages": {"resnet_x": {"output": str(output / "elsewhere")}}},
                             "resnet_x")


if __name__ == "__main__":
    unittest.main()
