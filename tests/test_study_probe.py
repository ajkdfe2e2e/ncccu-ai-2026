"""Synthetic coverage for study feature pooling and the frozen encoder probe."""

import unittest

import numpy as np

from cxr.study_probe import fit_linear_probe, pool_study_features, predict_linear_probe


def record(dicom, study, subject="p1", labels=None):
    return {"dicom_id": dicom, "study_id": study, "subject_id": subject, "labels": labels}


def balanced_labels(rows=40):
    rng = np.random.default_rng(31)
    return rng.integers(0, 2, size=(rows, 10)).astype(float)


class StudyProbeTests(unittest.TestCase):
    def test_pool_mean_and_views_preserve_first_study_order(self):
        labels = [0.0] * 10
        records = [record("a", "s1", labels=labels), record("b", "s2", labels=labels),
                   record("c", "s1", labels=labels), record("d", "s1", labels=labels)]
        features = np.array([[1, 3], [10, 20], [3, 5], [9, 7]], dtype=float)
        views = ["AP", "LL", "PA LLD", "oblique"]
        studies, mean, targets = pool_study_features(records, features, views, "mean")
        self.assertEqual([item["study_id"] for item in studies], ["s1", "s2"])
        np.testing.assert_allclose(mean, [[13 / 3, 5], [10, 20]])
        self.assertEqual(targets.shape, (2, 10))

        _, view_features, _ = pool_study_features(records, features, views, "views")
        # s1 groups: AP/PA mean [2, 4], no lateral, other [9, 7].
        np.testing.assert_allclose(view_features[0], [2, 4, 0, 0, 9, 7, 1, 0, 1])
        np.testing.assert_allclose(view_features[1], [0, 0, 10, 20, 0, 0, 0, 1, 0])

    def test_pool_accepts_unlabeled_and_checks_consistency_and_rows(self):
        x = np.ones((2, 2))
        unlabeled = [record("a", "s1", labels=None), record("b", "s1", labels=None)]
        self.assertIsNone(pool_study_features(unlabeled, x, ["AP", "PA"], "mean")[2])
        with self.assertRaisesRegex(ValueError, "conflicting subjects"):
            pool_study_features([unlabeled[0], record("b", "s1", "p2")], x, ["AP", "PA"], "mean")
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            pool_study_features([record("a", "s1", labels=[0] * 10),
                                 record("b", "s1", labels=[1] + [0] * 9)], x, ["AP", "PA"], "mean")
        with self.assertRaisesRegex(ValueError, "duplicate dicom"):
            pool_study_features([unlabeled[0], record("a", "s2", labels=None)], x, ["AP", "PA"], "mean")
        with self.assertRaisesRegex(ValueError, "row count"):
            pool_study_features(unlabeled, x, ["AP"], "mean")

    def test_training_scaler_and_parameters_ignore_heldout(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(80, 4))
        y = balanced_labels(80)
        payload = fit_linear_probe(x, y, max_iter=20)
        altered_holdout = x.copy()
        altered_holdout[60:] += 10000
        same = fit_linear_probe(altered_holdout[:60], y[:60], max_iter=20)
        train_only = fit_linear_probe(x[:60], y[:60], max_iter=20)
        for key in ("mean", "std", "weight", "bias"):
            np.testing.assert_array_equal(same[key], train_only[key])
        np.testing.assert_allclose(payload["mean"], x.mean(axis=0))
        np.testing.assert_allclose(payload["std"], x.std(axis=0))

    def test_synthetic_signal_is_learned_and_outputs_are_finite(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(240, 10))
        y = (x > 0).astype(float)
        payload = fit_linear_probe(x[:180], y[:180], l2=0.01, max_iter=100)
        probabilities = predict_linear_probe(payload, x[180:])
        self.assertEqual(probabilities.shape, (60, 10))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertGreater(np.mean((probabilities >= 0.5) == y[180:]), 0.8)
        self.assertTrue(np.isfinite(payload["loss"]))

    def test_constant_features_and_class_support_validation(self):
        y = balanced_labels()
        payload = fit_linear_probe(np.ones((40, 3)), y, max_iter=20)
        self.assertTrue(np.isfinite(payload["weight"]).all())
        np.testing.assert_array_equal(payload["std"], np.ones(3))
        with self.assertRaisesRegex(ValueError, "both positive and negative"):
            fit_linear_probe(np.ones((40, 3)), np.zeros((40, 10)))

    def test_class_balance_constant_feature_matches_weighted_bce_optimum(self):
        targets = np.zeros((40, 10), dtype=float)
        for class_id in range(10):
            targets[(np.arange(4) + class_id * 3) % 40, class_id] = 1
        payload = fit_linear_probe(np.ones((40, 2)), targets, l2=0.01,
                                   max_iter=200, class_balance=True)
        expected_weight = np.full(10, 3.0)
        expected_normalizer = np.full(10, 1.2)
        np.testing.assert_allclose(payload["positive_weight"], expected_weight)
        np.testing.assert_allclose(payload["class_weight_normalizer"], expected_normalizer)
        self.assertIs(payload["class_balance"], True)
        probabilities = predict_linear_probe(payload, np.ones((1, 2)))[0]
        np.testing.assert_allclose(probabilities, np.full(10, 0.25), atol=1e-7, rtol=0)

    def test_class_balance_statistics_use_only_passed_training_rows(self):
        targets = np.zeros((60, 10), dtype=float)
        for class_id in range(10):
            targets[(np.arange(4) + class_id * 3) % 40, class_id] = 1
        train_x = np.ones((40, 2))
        first = fit_linear_probe(train_x, targets[:40], max_iter=100, class_balance=True)
        targets[40:] = 1
        second = fit_linear_probe(train_x.copy(), targets[:40], max_iter=100, class_balance=True)
        for key in ("mean", "std", "weight", "bias", "positive_weight", "class_weight_normalizer"):
            np.testing.assert_array_equal(first[key], second[key])
        np.testing.assert_array_equal(first["positive_weight"], np.full(10, 3.0))

    def test_explicit_false_preserves_default_payload_and_values(self):
        rng = np.random.default_rng(17)
        x = rng.normal(size=(50, 4))
        y = balanced_labels(50)
        implicit = fit_linear_probe(x, y, max_iter=25)
        explicit = fit_linear_probe(x, y, max_iter=25, class_balance=False)
        self.assertEqual(implicit.keys(), explicit.keys())
        for key, value in implicit.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, explicit[key])
            else:
                self.assertEqual(value, explicit[key])

    def test_class_balance_rejects_non_boolean_values(self):
        x = np.ones((40, 2))
        y = balanced_labels()
        for invalid in (0, 1, "true", None):
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, "boolean"):
                fit_linear_probe(x, y, class_balance=invalid)
        accepted = fit_linear_probe(x, y, max_iter=10, class_balance=np.bool_(True))
        self.assertIs(accepted["class_balance"], True)

    def test_prediction_rejects_bad_shape_and_nonfinite_inputs(self):
        payload = fit_linear_probe(np.arange(120).reshape(40, 3), balanced_labels(), max_iter=10)
        with self.assertRaisesRegex(ValueError, "shape"):
            predict_linear_probe(payload, np.ones((2, 4)))
        with self.assertRaisesRegex(ValueError, "finite"):
            predict_linear_probe(payload, [[np.nan, 0, 1]])
        with self.assertRaisesRegex(ValueError, "shape"):
            predict_linear_probe(payload, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
