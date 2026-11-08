import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

import numpy as np

from cxr.metrics import _rank_auc
from scripts.run_oof_fusion import (
    aligned_component, bootstrap_family, load_locked, register_plan,
    require_unopened_audit, run_stage,
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


class OOFPipelineTests(unittest.TestCase):
    def test_registration_rechecks_audit_after_plan_is_persisted(self):
        for opened in (False, True):
            with self.subTest(opened=opened), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)

                def audit_boundary(source):
                    self.assertTrue((output / "plan.json").exists())
                    self.assertEqual(json.loads((output / "experiment.json").read_text())["status"], "registering")
                    if opened:
                        raise ValueError("audit opened during registration")
                    return {"audit_opened": False}

                with patch("scripts.run_oof_fusion.require_unopened_audit", side_effect=audit_boundary):
                    if opened:
                        with self.assertRaisesRegex(ValueError, "audit opened"):
                            register_plan(output, output, {"fixed": "hypothesis"})
                    else:
                        register_plan(output, output, {"fixed": "hypothesis"})
                state = json.loads((output / "experiment.json").read_text())
                self.assertEqual(state["source_audit_unopened_at_registration"], not opened)
                self.assertEqual(state["status"], "registration_rejected" if opened else "prepared")
                if opened:
                    with self.assertRaisesRegex(ValueError, "registration did not complete"):
                        load_locked(output)

    def test_registration_refuses_an_opened_or_scored_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            write_json(source / "experiment.json", {"status": "running", "audit_opened": False})
            self.assertFalse(require_unopened_audit(source)["audit_opened"])
            write_json(source / "experiment.json", {"audit_opened": True})
            with self.assertRaisesRegex(ValueError, "before any source audit"):
                require_unopened_audit(source)
            write_json(source / "experiment.json", {"audit_opened": False})
            write_json(source / "audit_comparison.json", {})
            with self.assertRaises(ValueError):
                require_unopened_audit(source)

    def test_partial_head_alignment_preserves_ids_labels_and_absence(self):
        rows = [{"subject_id": "p1", "study_id": "s1", "labels": [0] * 10},
                {"subject_id": "p2", "study_id": "s2", "labels": [1] * 10}]
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            write_json(folder / "study_ids.json", [["p2", "s2"], ["p1", "s1"]])
            write_json(folder / "report.json", {"local_label_indices": [8, 2]})
            np.save(folder / "study_probabilities.npy", [[.9, .8], [.2, .1]])
            np.save(folder / "study_targets.npy", [[1] * 10, [0] * 10])
            keys, targets, probabilities = aligned_component(folder, rows, [8, 2])
            self.assertEqual(keys, [("p1", "s1"), ("p2", "s2")])
            np.testing.assert_array_equal(targets, [[0] * 10, [1] * 10])
            np.testing.assert_allclose(probabilities[:, [8, 2]], [[.2, .1], [.9, .8]])
            self.assertTrue(np.isnan(probabilities[:, [0, 1, 3, 4, 5, 6, 7, 9]]).all())
            with self.assertRaisesRegex(ValueError, "class mapping"):
                aligned_component(folder, rows, [2, 8])

    def test_prediction_alignment_fails_for_wrong_patient_targets_or_missing_study(self):
        rows = [{"subject_id": "p1", "study_id": "s1", "labels": [0] * 10},
                {"subject_id": "p2", "study_id": "s2", "labels": [1] * 10}]
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            write_json(folder / "study_ids.json", [["p1", "s1"], ["p2", "s2"]])
            write_json(folder / "report.json", {})
            np.save(folder / "study_probabilities.npy", [[.2] * 10, [.8] * 10])
            np.save(folder / "study_targets.npy", [[1] * 10, [0] * 10])
            with self.assertRaisesRegex(ValueError, "targets disagree"):
                aligned_component(folder, rows, list(range(10)))
            np.save(folder / "study_targets.npy", [[0] * 10, [1] * 10])
            write_json(folder / "study_ids.json", [["p1", "s1"], ["p9", "s2"]])
            with self.assertRaisesRegex(ValueError, "coverage"):
                aligned_component(folder, rows, list(range(10)))

    def test_patient_bootstrap_matches_manual_cluster_reference(self):
        labels = np.tile(np.array([0, 1, 0, 1, 0, 1, 0, 1])[:, None], (1, 10))
        first = np.tile(np.array([.1, .7, .8, .6, .4, .5, .2, .9])[:, None], (1, 10))
        second = np.tile(np.array([.1, .8, .3, .7, .4, .6, .2, .9])[:, None], (1, 10))
        patients = ["a", "a", "a", "b", "c", "c", "c", "c"]
        result = bootstrap_family(patients, labels, first, {"fusion": second},
                                  repeats=100, seed=42, confidence=.975, minimum_defined=1)
        rng = np.random.default_rng(42)
        groups = [np.array([0, 1, 2]), np.array([3]), np.array([4, 5, 6, 7])]
        manual = []
        for _ in range(100):
            indexes = np.concatenate([groups[i] for i in rng.integers(3, size=3)])
            y = labels[indexes, 0]
            baseline_auc = _rank_auc(y, first[indexes, 0])
            if baseline_auc is not None:
                manual.append(_rank_auc(y, second[indexes, 0]) - baseline_auc)
        comparison = result["comparisons"]["fusion"]
        self.assertEqual(comparison["defined_replicates"], len(manual))
        np.testing.assert_allclose(comparison["paired_patient_interval"], np.percentile(manual, [1.25, 98.75]))
        self.assertAlmostEqual(comparison["delta"], _rank_auc(labels[:, 0], second[:, 0]) - _rank_auc(labels[:, 0], first[:, 0]))
        self.assertEqual(result["bootstrap"]["patients"], 3)
        self.assertEqual(result["bootstrap"]["family_size"], 2)

    def test_interval_unavailable_when_too_few_full_ten_class_draws(self):
        labels = np.tile(np.array([0, 1])[:, None], (1, 10))
        scores = np.tile(np.array([.2, .8])[:, None], (1, 10))
        result = bootstrap_family(["p1", "p2"], labels, scores, {"same": scores},
                                  repeats=20, seed=1, minimum_defined=20)
        self.assertIsNone(result["comparisons"]["same"]["paired_patient_interval"])
        self.assertLess(result["comparisons"]["same"]["defined_replicates"], 20)

    def test_stage_reuses_verified_completion_and_rejects_changed_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "source").mkdir()
            (output / "logs").mkdir()
            stage = output / "components/example"
            state = {"stages": {}}
            code = "from pathlib import Path; import sys; p=Path(sys.argv[1]); p.mkdir(); (p/'report.json').write_text('{}')"
            command = [sys.executable, "-c", code, str(stage)]
            with patch("scripts.run_oof_fusion.load_locked", return_value=({}, state)):
                run_stage(output, state, "example", command, stage, ["report.json"])
                self.assertEqual(state["stages"]["example"]["status"], "complete")
                first_time = (stage / "report.json").stat().st_mtime_ns
                run_stage(output, state, "example", command, stage, ["report.json"])
                self.assertEqual((stage / "report.json").stat().st_mtime_ns, first_time)
                (stage / "report.json").write_text('{"changed":true}')
                with self.assertRaisesRegex(ValueError, "cached stage differs"):
                    run_stage(output, state, "example", command, stage, ["report.json"])

    def test_incomplete_stage_preserved_and_never_silently_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            folder = output / "components/partial"
            folder.mkdir(parents=True)
            marker = folder / "partial.npy"
            marker.write_bytes(b"incomplete data")
            state = {"stages": {}}
            with patch("scripts.run_oof_fusion.load_locked", return_value=({}, state)):
                with self.assertRaisesRegex(ValueError, "incomplete output"):
                    run_stage(output, state, "partial", [sys.executable], folder, [])
            self.assertEqual(marker.read_bytes(), b"incomplete data")


if __name__ == "__main__":
    unittest.main()
