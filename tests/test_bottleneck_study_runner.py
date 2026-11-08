"""Synthetic boundary tests for nested patient-group selection and gates."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import run_bottleneck_study_probe as nested


def fixture(rows_per_fold=40):
    folds = np.empty(rows_per_fold * len(nested.FOLDS), dtype=object)
    targets = np.empty((len(folds), 10), dtype=float)
    for fold_index, fold in enumerate(nested.FOLDS):
        start = fold_index * rows_per_fold
        end = start + rows_per_fold
        folds[start:end] = fold
        rng = np.random.default_rng(100 + fold_index)
        targets[start:end] = rng.integers(0, 2, size=(rows_per_fold, 10))
    return targets, folds


def validation_comparison(*, delta=0.001, ci=(0.0001, 0.02), histogram=950,
                           defined=10, missing_class=False):
    per_class = [{"metrics": {"auc": {"baseline": 0.5, "candidate": 0.51}}}
                 for _ in range(10)]
    if missing_class:
        per_class[-1]["metrics"]["auc"]["baseline"] = None
    return {
        "macro": {"auc": {"delta": delta, "ci95_percentile": list(ci)}},
        "macro_auc_defined_classes": defined,
        "per_class": per_class,
        "bootstrap": {"auc_defined_classes_histogram": {"10": histogram}},
    }


class BottleneckStudyProbeTests(unittest.TestCase):
    def test_inner_selection_and_scores_ignore_outer_held_labels(self):
        targets, folds = fixture()
        outer = nested.FOLDS[0]
        configs = ({"mode": "semantic", "l2": 0.01, "class_balance": False},
                   {"mode": "pca128", "l2": 0.01, "class_balance": False})

        def predict(training_folds, config, held):
            return 0.05 + 0.9 * targets[held]

        selected_before, rows_before = nested.inner_select(targets, folds, outer, predict, configs)
        targets[folds == outer] = 1 - targets[folds == outer]
        selected_after, rows_after = nested.inner_select(targets, folds, outer, predict, configs)
        self.assertEqual(selected_before, selected_after)
        self.assertEqual(rows_before, rows_after)

    def test_predict_callback_never_trains_on_outer_or_inner_held_fold(self):
        targets, folds = fixture()
        outer = nested.FOLDS[1]
        observations = []

        def predict(training_folds, config, held):
            held_folds = set(folds[held].tolist())
            observations.append((tuple(training_folds), held_folds))
            return 0.1 + 0.8 * targets[held]

        nested.inner_select(targets, folds, outer, predict,
                            configs=({"mode": "semantic", "l2": 0.01, "class_balance": False},))
        self.assertEqual(len(observations), len(nested.FOLDS) - 1)
        for training_folds, held_folds in observations:
            self.assertNotIn(outer, training_folds)
            self.assertEqual(len(held_folds), 1)
            held_fold = next(iter(held_folds))
            self.assertNotIn(held_fold, training_folds)

    def test_fixed_configs_and_ties_prefer_semantic(self):
        self.assertEqual(nested.CONFIGS, (
            {"mode": "semantic", "l2": 0.01, "class_balance": False},
            {"mode": "pca128", "l2": 0.01, "class_balance": False},
        ))
        rows = [{"config": dict(config), "macro_auc": 0.75,
                 "macro_auc_defined_classes": 10} for config in reversed(nested.CONFIGS)]
        self.assertEqual(nested.select_config(rows)["config"]["mode"], "semantic")
        with self.assertRaisesRegex(ValueError, "ten defined"):
            nested.select_config([dict(rows[0], macro_auc_defined_classes=9)])

    def test_nested_oof_covers_each_study_once_in_outer_prediction(self):
        targets, folds = fixture()
        final_counts = np.zeros(len(targets), dtype=int)

        def predict(training_folds, config, held):
            if len(training_folds) == len(nested.FOLDS) - 1:
                final_counts[held] += 1
            return 0.1 + 0.8 * targets[held]

        probabilities, reports = nested.nested_oof(
            targets, folds, predict,
            configs=({"mode": "semantic", "l2": 0.01, "class_balance": False},),
        )
        self.assertEqual(probabilities.shape, targets.shape)
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_array_equal(final_counts, np.ones(len(targets), dtype=int))
        self.assertEqual([row["outer_fold"] for row in reports], list(nested.FOLDS))

    def test_training_gate_requires_current_gain_three_groups_and_ten_classes(self):
        targets, folds = fixture()
        rng = np.random.default_rng(3)
        baseline = rng.random(targets.shape)
        legacy = np.clip(baseline - 0.05, 0, 1)
        candidate = baseline.copy()
        for fold in nested.FOLDS[:3]:
            held = folds == fold
            candidate[held] = 0.05 + 0.9 * targets[held]
        passed = nested.training_gate(targets, baseline, candidate, folds, legacy)
        self.assertTrue(passed["passed"])
        self.assertGreaterEqual(passed["gain"], 0.001)
        self.assertGreaterEqual(passed["legacy_gain"], 0.001)
        self.assertEqual(passed["positive_held_groups"], 3)
        self.assertTrue(all(group["defined_classes"] == 10 for group in passed["held_groups"]))

        two_group_candidate = baseline.copy()
        for fold in nested.FOLDS[:2]:
            held = folds == fold
            two_group_candidate[held] = 0.05 + 0.9 * targets[held]
        failed = nested.training_gate(targets, baseline, two_group_candidate, folds, legacy)
        self.assertEqual(failed["positive_held_groups"], 2)
        self.assertFalse(failed["passed"])

        missing_class_targets = targets.copy()
        missing_class_targets[folds == nested.FOLDS[0], 0] = 0
        missing = nested.training_gate(missing_class_targets, baseline, candidate, folds, legacy)
        self.assertFalse(missing["passed"])

    def test_training_gate_rejects_current_gain_below_point_zero_zero_one(self):
        def summary(auc):
            return {"macro_auc": auc, "macro_auc_defined_classes": 10}

        controlled = [summary(0.5), summary(0.5009), summary(0.49)]
        controlled.extend([summary(0.4), summary(0.5)] * len(nested.FOLDS))
        targets = np.zeros((len(nested.FOLDS), 10))
        folds = np.asarray(nested.FOLDS, dtype=object)
        arrays = np.zeros_like(targets)
        with patch.object(nested, "score", side_effect=controlled):
            result = nested.training_gate(targets, arrays, arrays, folds, arrays)
        self.assertAlmostEqual(result["gain"], 0.0009)
        self.assertEqual(result["positive_held_groups"], len(nested.FOLDS))
        self.assertFalse(result["passed"])

    def test_validation_gate_requires_both_baselines_ci_delta_and_bootstrap_support(self):
        comparisons = {"incumbent": validation_comparison(), "legacy": validation_comparison()}
        self.assertTrue(nested.validation_gate(comparisons))
        failures = (
            validation_comparison(delta=0.0009),
            validation_comparison(ci=(0.0, 0.02)),
            validation_comparison(histogram=949),
            validation_comparison(defined=9),
            validation_comparison(missing_class=True),
        )
        for failed in failures:
            with self.subTest(failed=failed):
                self.assertFalse(nested.validation_gate({"incumbent": failed,
                                                         "legacy": validation_comparison()}))

    def test_fit_cache_reuses_exact_fit_and_rejects_tampered_head(self):
        targets, folds = fixture(rows_per_fold=4)
        features = np.arange(len(targets) * 1054, dtype=float).reshape(len(targets), 1054)
        subjects = [f"p{i}" for i in range(len(targets))]
        config = {"mode": "semantic", "l2": 0.01, "class_balance": False}
        payload = {"mode_id": np.int64(0), "mean": np.zeros(30), "std": np.ones(30),
                   "weight": np.zeros((10, 30)), "bias": np.zeros(10),
                   "iterations": 2, "loss": 0.5}
        with tempfile.TemporaryDirectory() as directory, patch.object(
            nested, "fit_linear_probe", return_value=payload
        ) as fit:
            cache = nested.FitCache(directory, features, targets, folds, subjects, "plan-sha")
            held = np.flatnonzero(folds == nested.FOLDS[0])
            first = cache.predict(nested.FOLDS[1:], config, held)
            second = cache.predict(nested.FOLDS[1:], config, held)
            np.testing.assert_array_equal(first, second)
            self.assertEqual(fit.call_count, 1)
            self.assertEqual(cache.fit_count, 1)
            self.assertEqual(cache.cache_hits, 1)

            folder = Path(directory) / "1_2_audit" / nested.config_name(config)
            head = folder / "head.npz"
            head.write_bytes(head.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                cache.predict(nested.FOLDS[1:], config, held)

    def test_fit_cache_rejects_patient_shared_across_train_and_held(self):
        targets, folds = fixture(rows_per_fold=3)
        features = np.zeros((len(targets), 2))
        subjects = [f"p{i}" for i in range(len(targets))]
        held = int(np.flatnonzero(folds == nested.FOLDS[0])[0])
        train_row = int(np.flatnonzero(folds == nested.FOLDS[1])[0])
        subjects[train_row] = subjects[held]
        with tempfile.TemporaryDirectory() as directory:
            cache = nested.FitCache(directory, features, targets, folds, subjects, "plan-sha")
            with self.assertRaisesRegex(ValueError, "patient leakage"):
                cache.predict(nested.FOLDS[1:], {"mode": "semantic", "l2": 0.01, "class_balance": False}, [held])


if __name__ == "__main__":
    unittest.main()
