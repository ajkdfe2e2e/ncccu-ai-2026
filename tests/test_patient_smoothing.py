import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cxr.patient_smoothing import smooth_patient_probabilities
from scripts import run_patient_smoothing as runner


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_manifest(path, rows, *, labeled=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["Dicom_id", "Subject_id", "Study_id"] + (["Predict_class"] if labeled else [])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class PatientSmoothingTests(unittest.TestCase):
    def test_singleton_is_exact_and_smoothing_is_deterministic(self):
        p = np.arange(30, dtype=np.float64).reshape(3, 10) / 30
        ids = (["a", "a", "b"], ["s1", "s2", "s3"])
        got = smooth_patient_probabilities(p, *ids)
        np.testing.assert_array_equal(got[2], p[2])
        np.testing.assert_allclose(got[:2], 0.75 * p[:2] + 0.25 * p[:2].mean(axis=0))
        np.testing.assert_array_equal(got, smooth_patient_probabilities(p, *ids))

    def test_rejects_invalid_shape_ids_probabilities_and_coefficient(self):
        p = np.full((2, 10), 0.4)
        invalid = [np.ones((2, 9)), np.full((2, 10), np.nan), np.full((2, 10), 1.1)]
        for values in invalid:
            with self.subTest(values=values.shape):
                with self.assertRaises(ValueError):
                    smooth_patient_probabilities(values, ["a", "b"], ["x", "y"])
        for subjects, studies in ((["a"], ["x", "y"]), (["a", "a"], ["x", "x"]),
                                  (["a", "b"], ["x", "x"]), ([1, "b"], ["x", "y"])):
            with self.subTest(subjects=subjects, studies=studies):
                with self.assertRaises(ValueError):
                    smooth_patient_probabilities(p, subjects, studies)
        with self.assertRaises(ValueError):
            smooth_patient_probabilities(p, ["a", "b"], ["x", "y"], alpha=0.2)

    def test_alignment_requires_exact_string_ids_and_labels(self):
        expected = [["p1", "s1"], ["p2", "s2"]]
        targets = np.eye(2, 10)
        np.testing.assert_array_equal(
            runner._align(expected[::-1], targets[::-1], expected, targets, "fixture"), [1, 0])
        with self.assertRaisesRegex(ValueError, "ID set"):
            runner._align(expected[:1], targets[:1], expected, targets, "fixture")
        with self.assertRaisesRegex(ValueError, "labels"):
            runner._align(expected, targets[::-1], expected, targets, "fixture")
        with self.assertRaisesRegex(ValueError, "nonempty strings"):
            runner._align([[1, "s1"], expected[1]], targets, expected, targets, "fixture")

    def test_legacy_target_evidence_rejects_conflicts_and_missing_coverage(self):
        expected_ids = [("p1", "s1"), ("p2", "s2")]
        targets = np.eye(2, 10)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            ids_path = output / "inputs/legacy_targets/00/study_ids.json"
            values_path = output / "inputs/legacy_targets/00/study_targets.npy"
            ids_path.parent.mkdir(parents=True)
            write_json(ids_path, expected_ids)
            np.save(values_path, targets)
            source = {"ids": "inputs/legacy_targets/00/study_ids.json",
                      "targets": "inputs/legacy_targets/00/study_targets.npy"}
            plan = {"legacy_target_sources": [source]}
            self.assertEqual(runner._validate_legacy_targets(output, plan, expected_ids, targets)["covered_studies"], 2)

            np.save(values_path, targets[::-1])
            with self.assertRaisesRegex(ValueError, "legacy targets do not align"):
                runner._validate_legacy_targets(output, plan, expected_ids, targets)
            np.save(values_path, targets[:1])
            write_json(ids_path, expected_ids[:1])
            with self.assertRaisesRegex(ValueError, "does not cover complete train"):
                runner._validate_legacy_targets(output, plan, expected_ids, targets)

    def _make_registered_fixture(self, base):
        base = Path(base)
        data = base / "raw"
        incumbent = base / "incumbent"
        eva = base / "eva"
        legacy = base / "legacy"
        partition_file = base / "partition.json"
        data.mkdir(parents=True)
        legacy.mkdir()

        train_rows, study_ids, targets, assignments = [], [], [], {}
        for fold in (0, 1, 2, "audit"):
            for side, label in (("a", "1"), ("b", ";".join(map(str, range(2, 11))))):
                subject, study = f"p{fold}{side}", f"s{fold}{side}"
                train_rows.append({"Dicom_id": f"d{fold}{side}", "Subject_id": subject,
                                   "Study_id": study, "Predict_class": label})
                row = np.zeros(10, dtype=np.float64)
                for class_id in map(int, label.split(";")):
                    row[class_id - 1] = 1
                study_ids.append([subject, study])
                targets.append(row)
                assignments[subject] = fold
        write_manifest(data / "train.csv", train_rows)
        write_manifest(data / "valid.csv", [{"Dicom_id": "dv", "Subject_id": "pv",
                                               "Study_id": "sv", "Predict_class": "1;2"}])
        write_manifest(data / "test.csv", [{"Dicom_id": "dt", "Subject_id": "pt", "Study_id": "st"}], labeled=False)
        train_targets = np.asarray(targets)
        probabilities = np.where(train_targets == 1, 0.9, 0.1)

        partition = {"subject_partition": assignments}
        write_json(partition_file, partition)
        legacy_targets = legacy / "source" / "study_targets.npy"
        legacy_ids = legacy / "source" / "study_ids.json"
        legacy_targets.parent.mkdir(parents=True)
        np.save(legacy_targets, train_targets)
        write_json(legacy_ids, study_ids)
        legacy_plan = {"data_root": str(data), "partition": str(partition_file),
                       "prior": [[1.0] + [0.0] * 9] + [[0.0] * 10 for _ in range(7)],
                       "input_locks": {
                           str(data / "train.csv"): runner._sha256(data / "train.csv"),
                           str(partition_file): runner._sha256(partition_file),
                           str(legacy_targets): runner._sha256(legacy_targets),
                           str(legacy_ids): runner._sha256(legacy_ids),
                       }}
        write_json(legacy / "plan.json", legacy_plan)
        write_json(legacy / "experiment.json", {"status": "complete", "stages": {}})
        write_json(legacy / "oof_study_ids.json", study_ids)
        components = np.zeros((len(study_ids), 8, 10), dtype=np.float64)
        components[:, 0, :] = probabilities
        np.save(legacy / "oof_component_probabilities.npy", components)
        legacy_valid = legacy / "valid.csv"
        legacy_valid.write_text("unused until OOF passes\n", encoding="utf-8")

        inc_files = ["recipe.json", "final_head.npz", "report.json", "validation_report.json",
                     "candidate_valid.csv"]
        for name in inc_files:
            path = incumbent / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("unused until required gate passes\n", encoding="utf-8")
        np.save(incumbent / "oof_mean_probabilities.npy", probabilities)
        np.save(incumbent / "oof_targets.npy", train_targets)
        write_json(incumbent / "oof_study_ids.json", study_ids)
        write_json(incumbent / "experiment.json", {"status": "complete", "candidate_promoted": True,
                                                     "test_csv_exported": True})
        test_export = incumbent / "submission_test_1414_cxrlt_study_20261108.csv"
        test_export.write_bytes(b"UNREADABLE TEST EXPORT SENTINEL")
        write_json(incumbent / "submission_verification.json", {
            "valid": True, "studies": 1414, "sha256": runner._sha256(test_export)})
        incumbent_plan = {"data_root": str(data), "old_oof": str(legacy), "old_valid": str(legacy_valid),
                          "partition": str(partition_file)}
        write_json(incumbent / "plan.json", incumbent_plan)
        for name in ("plan.json", "plan_lock.json", "experiment.json", "oof_report.json", "artifacts.json"):
            write_json(eva / name, {"fixture": True})
        write_json(eva / "experiment.json", {"status": "complete", "candidate_promoted": False,
                                              "test_csv_exported": False, "test_extracted": False})
        write_json(eva / "independent.json", {})
        eva_report = base / "eva-independent.json"
        write_json(eva_report, {"status": "verified_oof_only", "runner_status": "complete",
                                "selected_oof_candidate": None, "candidate_promoted": False,
                                "coverage": {"oof_heads_and_gates": "verified"}})
        return data, incumbent, eva, eva_report, train_targets, probabilities

    def test_prepare_locks_inputs_and_snapshot_run_stops_after_failed_oof(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            data, incumbent, eva, eva_report, targets, probabilities = self._make_registered_fixture(base)
            output = base / "run"
            deferred_test = incumbent / "submission_test_1414_cxrlt_study_20261108.csv"
            original_copy = runner._copy_lock

            def refuse_test_copy(source, destination):
                if Path(source) == deferred_test:
                    raise AssertionError("prepare attempted to read the deferred test export")
                return original_copy(source, destination)

            with patch.object(runner, "INCUMBENT", incumbent), patch.object(runner, "EVA", eva), \
                    patch.object(runner, "EVA_REPORT", eva_report), \
                    patch.object(runner, "_copy_lock", side_effect=refuse_test_copy):
                prepared = runner.prepare(output)
            plan = runner._json(prepared / "plan.json")
            self.assertEqual(plan["train_studies"], len(targets))
            self.assertTrue((prepared / "inputs/source/scripts/run_patient_smoothing.py").is_file())
            self.assertEqual(runner._sha256(prepared / "inputs/source/scripts/run_patient_smoothing.py"),
                             plan["source_locks"]["scripts/run_patient_smoothing.py"])
            self.assertFalse((prepared / plan["deferred_test_input"]["snapshot"]).exists())
            runner._verify(prepared, plan)

            command = [sys.executable, str(runner.ROOT / "scripts/run_patient_smoothing.py"),
                       "--output", str(prepared), "--resume"]
            with patch.object(runner, "INCUMBENT", incumbent), patch.object(runner, "EVA", eva), \
                    patch.object(runner, "EVA_REPORT", eva_report):
                completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            state = runner._json(prepared / "experiment.json")
            self.assertEqual(state["status"], "complete")
            self.assertFalse(state["valid_evaluated"])
            self.assertFalse(state["test_read"])
            self.assertFalse(state["candidate_promoted"])
            self.assertFalse((prepared / "valid_report.json").exists())
            self.assertFalse((prepared / "submission_test_1414_cxrlt_patient025_20261108.csv").exists())
            self.assertTrue((prepared / "oof_report.json").is_file())

            locked = prepared / "inputs/manifests/train.csv"
            original = locked.read_bytes()
            locked.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "locked input changed"):
                runner._verify(prepared, plan)

    def test_failed_valid_gate_never_reads_deferred_test_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            _, incumbent, eva, eva_report, _, _ = self._make_registered_fixture(base)
            with patch.object(runner, "INCUMBENT", incumbent), patch.object(runner, "EVA", eva), \
                    patch.object(runner, "EVA_REPORT", eva_report):
                output = runner.prepare(base / "run")
            probabilities = np.full((1, 10), 0.4)
            with patch.object(runner, "training_gate", return_value={"passed": True}), \
                    patch.object(runner, "load_submission", return_value=(probabilities, None, None)) as load, \
                    patch.object(runner, "compare_arrays", return_value={}), \
                    patch.object(runner, "validation_gate", return_value=False), \
                    patch.object(runner, "_copy_lock", side_effect=AssertionError("test copy before valid pass")):
                state = runner.run(output)
            self.assertEqual(state["status"], "complete")
            self.assertTrue(state["valid_evaluated"])
            self.assertFalse(state["test_read"])
            self.assertFalse(state["test_csv_exported"])
            self.assertEqual(load.call_count, 2)
            self.assertFalse((output / "deferred_inputs/incumbent_test.csv").exists())


if __name__ == "__main__":
    unittest.main()
