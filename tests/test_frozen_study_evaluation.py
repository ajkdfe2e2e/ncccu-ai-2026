"""Synthetic tests for the frozen study-probe selection and artifact boundary."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cxr.study_probe import fit_linear_probe, predict_linear_probe
from scripts.evaluate_frozen_study_probe import (
    _load_head, _save_head, _select_candidate, run_validation, train_candidates,
)


def validation_fixture(output):
    rng = np.random.default_rng(4)
    records, labels = [], []
    for index in range(20):
        label = float(index % 2)
        labels.append([label] * 10)
        records.append({
            "dicom_id": f"vd{index}", "subject_id": f"vp{index}",
            "study_id": f"vs{index}", "labels": [label] * 10,
        })
    features = rng.normal(size=(20, 4))
    payload = fit_linear_probe(features, np.asarray(labels), max_iter=10)
    head_path = Path(output) / "final_head.npz"
    _save_head(head_path, payload)
    recipe = {
        "status": "selected", "mode": "mean", "alpha": 0.2,
        "head_file": head_path.name,
        "head_sha256": hashlib.sha256(head_path.read_bytes()).hexdigest(),
    }
    recipe_path = Path(output) / "recipe.json"
    recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
    old = np.tile(np.linspace(0.1, 0.9, len(records))[:, None], (1, 10))
    return records, features, old, recipe, recipe_path


def comparison_result(defined_classes=10, lower=0.001, full_class_replicates=960):
    per_class = []
    for index in range(10):
        auc = {"baseline": 0.5, "candidate": 0.6}
        if defined_classes < 10 and index == 9:
            auc = {"baseline": None, "candidate": None}
        per_class.append({"metrics": {"auc": auc}})
    return {
        "macro_auc_defined_classes": defined_classes,
        "per_class": per_class,
        "macro": {"auc": {"delta": 0.002, "ci95_percentile": [lower, 0.004]}},
        "bootstrap": {"auc_defined_classes_histogram": {"10": full_class_replicates}},
    }


class FrozenStudyEvaluationTests(unittest.TestCase):
    def test_selection_requires_fixed_gain_and_three_positive_held_groups(self):
        eligible = {
            "mode": "views", "alpha": 1.0, "macro_auc": 0.81,
            "macro_auc_defined_classes": 10, "overall_gain": 0.001,
            "positive_held_groups": 3,
        }
        tied_preferred = {**eligible, "mode": "mean", "alpha": 0.2}
        self.assertEqual(_select_candidate([eligible, tied_preferred]), tied_preferred)
        self.assertIsNone(_select_candidate([
            {**eligible, "overall_gain": 0.00099},
            {**eligible, "positive_held_groups": 2},
            {**eligible, "macro_auc_defined_classes": 9},
        ]))

    def test_no_gain_candidate_is_rejected_and_each_head_excludes_held_patients(self):
        records, features, views, assignment = [], [], [], {}
        for fold_index, fold in enumerate((0, 1, 2, "audit")):
            for local_index in range(10):
                patient = f"p{fold_index}_{local_index}"
                label = float(local_index % 2)
                records.append({
                    "dicom_id": f"d{fold_index}_{local_index}",
                    "subject_id": patient,
                    "study_id": f"s{fold_index}_{local_index}",
                    "labels": [label] * 10,
                })
                features.append([float(len(features)), float(local_index)])
                views.append("AP")
                assignment[patient] = fold

        seen_fit_rows = []

        def fake_fit(x, y, **kwargs):
            seen_fit_rows.append(x[:, 0].copy())
            return {"marker": np.asarray([1.0])}

        def constant_predict(payload, x):
            return np.full((len(x), 10), 0.5)

        with tempfile.TemporaryDirectory() as temporary, \
                patch("scripts.evaluate_frozen_study_probe.fit_linear_probe", side_effect=fake_fit), \
                patch("scripts.evaluate_frozen_study_probe.predict_linear_probe", side_effect=constant_predict):
            result = train_candidates(
                temporary, records, np.asarray(features), views, assignment,
                np.full((len(records), 10), 0.5),
            )
            self.assertIsNone(result)
            report = json.loads((Path(temporary) / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "rejected")
            self.assertFalse((Path(temporary) / "recipe.json").exists())
            self.assertEqual(len(seen_fit_rows), 8)
            all_rows = set(range(len(records)))
            for call_index, trained_rows in enumerate(seen_fit_rows):
                held_fold = (0, 1, 2, "audit")[call_index % 4]
                held_rows = set(range((0, 1, 2, "audit").index(held_fold) * 10,
                                      ((0, 1, 2, "audit").index(held_fold) + 1) * 10))
                self.assertTrue(set(trained_rows.astype(int)) <= all_rows - held_rows)

    def test_overall_gain_uses_pooled_old_oof_not_last_held_group(self):
        records, features, views, assignment = [], [], [], {}
        old = []
        for fold_index, fold in enumerate((0, 1, 2, "audit")):
            for local_index in range(10):
                patient = f"gain_{fold_index}_{local_index}"
                label = float(local_index % 2)
                records.append({
                    "dicom_id": f"gd{fold_index}_{local_index}",
                    "subject_id": patient,
                    "study_id": f"gs{fold_index}_{local_index}",
                    "labels": [label] * 10,
                })
                features.append([float(len(features)), float(local_index)])
                views.append("AP")
                assignment[patient] = fold
                score = label if fold == "audit" else 1.0 - label
                old.append([0.9 if score else 0.1] * 10)

        def fake_fit(x, y, **kwargs):
            return {"marker": np.asarray([1.0])}

        def constant_predict(payload, x):
            return np.full((len(x), 10), 0.5)

        with tempfile.TemporaryDirectory() as temporary, \
                patch("scripts.evaluate_frozen_study_probe.fit_linear_probe", side_effect=fake_fit), \
                patch("scripts.evaluate_frozen_study_probe.predict_linear_probe", side_effect=constant_predict):
            recipe = train_candidates(
                temporary, records, np.asarray(features), views, assignment,
                np.asarray(old),
            )
            report = json.loads((Path(temporary) / "report.json").read_text(encoding="utf-8"))

        self.assertIsNotNone(recipe)
        selected = report["selected"]
        self.assertEqual((selected["mode"], selected["alpha"]), ("mean", 1.0))
        self.assertAlmostEqual(selected["overall_gain"], 0.25)
        self.assertEqual([round(item["gain"], 6) for item in selected["held_groups"]],
                         [0.5, 0.5, 0.5, -0.5])

    def test_head_npz_reload_preserves_predictions_exactly(self):
        rng = np.random.default_rng(73)
        features = rng.normal(size=(40, 5))
        targets = rng.integers(0, 2, size=(40, 10)).astype(float)
        payload = fit_linear_probe(features, targets, max_iter=15)
        expected = predict_linear_probe(payload, features)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "head.npz"
            _save_head(path, payload)
            actual = predict_linear_probe(_load_head(path), features)
        np.testing.assert_array_equal(expected, actual)

    def test_validation_promotion_requires_all_gate_conditions(self):
        scenarios = ((9, 0.001, 960, False), (10, -0.001, 960, False),
                     (10, 0.001, 960, True))
        for defined, lower, replicates, expected in scenarios:
            with self.subTest(defined=defined, lower=lower), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                records, features, old, recipe, _ = validation_fixture(output)
                mocked = comparison_result(defined, lower, replicates)
                with patch("scripts.evaluate_frozen_study_probe.compare_arrays", return_value=mocked) as compare:
                    report = run_validation(output, records, features, ["AP"] * 20, old, recipe)
                self.assertEqual(report["promoted"], expected)
                compare.assert_called_once()
                self.assertEqual(compare.call_args.kwargs, {"bootstrap": 1000, "seed": 20261108})
                self.assertTrue((output / "candidate_valid.csv").is_file())

    def test_validation_report_replay_checks_recipe_head_and_csv_hashes(self):
        for tamper in ("recipe", "head", "csv"):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                records, features, old, recipe, recipe_path = validation_fixture(output)
                with patch("scripts.evaluate_frozen_study_probe.compare_arrays",
                           return_value=comparison_result()):
                    report = run_validation(output, records, features, ["AP"] * 20, old, recipe)
                self.assertEqual(report["recipe_sha256"],
                                 hashlib.sha256(recipe_path.read_bytes()).hexdigest())
                self.assertEqual(report["head_sha256"], recipe["head_sha256"])

                # Completed reports replay without touching replacement inputs.
                altered_records = [dict(row, labels=None) for row in records]
                replay = run_validation(output, altered_records, np.empty((0, 0)), [],
                                        np.empty((0, 10)), recipe)
                self.assertEqual(replay, report)

                if tamper == "recipe":
                    changed = {**recipe, "alpha": 1.0}
                    recipe_path.write_text(json.dumps(changed), encoding="utf-8")
                    bad_recipe = changed
                elif tamper == "head":
                    head_path = output / recipe["head_file"]
                    head_path.write_bytes(head_path.read_bytes() + b"tampered")
                    bad_recipe = recipe
                else:
                    csv_path = output / "candidate_valid.csv"
                    csv_path.write_text(csv_path.read_text(encoding="utf-8") + "tampered\n",
                                        encoding="utf-8")
                    bad_recipe = recipe
                message = "registered final head" if tamper == "head" else "saved validation report"
                with self.assertRaisesRegex(ValueError, message):
                    run_validation(output, records, features, ["AP"] * 20, old, bad_recipe)


if __name__ == "__main__":
    unittest.main()
