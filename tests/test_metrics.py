"""Synthetic-only tests: no images, dataset files, or test-set statistics."""

import json
import unittest

import numpy as np

from cxr.metrics import aggregate_studies, evaluate, select_thresholds


def record(study="s1", subject="001", labels=None, dicom="d1"):
    return {"dicom_id": dicom, "subject_id": subject, "study_id": study,
            "path": "synthetic-not-opened.png", "labels": labels}


class AggregateStudiesTests(unittest.TestCase):
    def test_arithmetic_mean_first_seen_order_and_targets(self):
        labels = [1.0, 0.0] * 5
        records = [record("s2", labels=labels), record("s1", labels=[0.] * 10),
                   record("s2", labels=labels, dicom="d2")]
        probs = np.array([[0.2] * 10, [0.9] * 10, [0.8] * 10])
        studies, averaged, targets = aggregate_studies(records, probs)
        self.assertEqual([r["study_id"] for r in studies], ["s2", "s1"])
        self.assertIsNot(studies[0], records[0])
        self.assertEqual(studies[0]["dicom_id"], "d1")
        np.testing.assert_allclose(averaged, [[0.5] * 10, [0.9] * 10])
        np.testing.assert_array_equal(targets, [labels, [0.] * 10])
        np.testing.assert_array_equal(probs[0], [0.2] * 10)

    def test_unlabeled_and_empty(self):
        _, probs, targets = aggregate_studies([record()], np.zeros((1, 10)))
        self.assertEqual(probs.shape, (1, 10))
        self.assertIsNone(targets)
        records, probs, targets = aggregate_studies([], np.empty((0, 10)))
        self.assertEqual(records, [])
        self.assertEqual(probs.shape, (0, 10))
        self.assertIsNone(targets)

    def test_conflicting_subjects_or_labels_rejected(self):
        cases = [
            [record(), record(subject="002")],
            [record(labels=[0.] * 10), record(labels=[1.] * 10)],
            [record(labels=[0.] * 10), record()],
            [record("s1", labels=[0.] * 10), record("s2")],
        ]
        for records in cases:
            with self.subTest(records=records), self.assertRaises(ValueError):
                aggregate_studies(records, np.zeros((2, 10)))

    def test_bad_shapes_values_or_records_rejected(self):
        for probs in [np.zeros((1, 9)), np.zeros((2, 10)), np.full((1, 10), np.nan),
                      np.full((1, 10), np.inf), np.full((1, 10), -0.1), np.full((1, 10), 1.1)]:
            with self.subTest(probs=probs), self.assertRaises(ValueError):
                aggregate_studies([record()], probs)
        for bad_record in [{}, record(labels=[0.] * 9), record(labels=[float("nan")] * 10),
                           record(study=""), record(subject=None)]:
            with self.subTest(record=bad_record), self.assertRaises(ValueError):
                aggregate_studies([bad_record], np.zeros((1, 10)))


class EvaluateTests(unittest.TestCase):
    def test_perfect_inverse_and_tied_auc(self):
        targets = np.zeros((4, 10))
        targets[:, :3] = np.array([0, 0, 1, 1])[:, None]
        targets[:, 4] = 1
        probs = np.zeros((4, 10))
        probs[:, 0] = [0.1, 0.2, 0.8, 0.9]
        probs[:, 1] = [0.9, 0.8, 0.2, 0.1]
        probs[:, 2] = [0.1, 0.5, 0.5, 0.9]
        result = evaluate(targets, probs)
        self.assertEqual(result["macro_auc_defined_classes"], 3)
        self.assertAlmostEqual(result["macro_auc"], (1 + 0 + 0.875) / 3)
        self.assertEqual([r["id"] for r in result["per_class"]], list(range(1, 11)))
        self.assertEqual(result["per_class"][0]["f1"], 1.0)
        self.assertEqual(result["per_class"][0]["positives"], 2)
        self.assertIsNone(result["per_class"][3]["auc"])
        self.assertIsNone(result["per_class"][4]["auc"])
        json.dumps(result, allow_nan=False)

    def test_rank_auc_matches_pairwise_reference_with_ties(self):
        rng = np.random.default_rng(31)
        targets = rng.integers(0, 2, size=(23, 10))
        probs = rng.integers(0, 5, size=(23, 10)) / 4
        result = evaluate(targets, probs)
        for index in range(10):
            positive = probs[targets[:, index] == 1, index]
            negative = probs[targets[:, index] == 0, index]
            reference = np.mean((positive[:, None] > negative) + 0.5 * (positive[:, None] == negative))
            self.assertAlmostEqual(result["per_class"][index]["auc"], reference)

    def test_all_tied_auc_is_half(self):
        targets = np.tile([[0.] * 10, [1.] * 10], (2, 1))
        result = evaluate(targets, np.full((4, 10), 0.5))
        self.assertEqual(result["macro_auc"], 0.5)
        self.assertAlmostEqual(result["macro_f1"], 2 / 3)

    def test_no_defined_auc_and_empty_are_json_safe(self):
        for targets in [np.zeros((3, 10)), np.empty((0, 10))]:
            result = evaluate(targets, targets)
            self.assertIsNone(result["macro_auc"])
            self.assertEqual(result["macro_auc_defined_classes"], 0)
            self.assertEqual(result["macro_f1"], 0.0)
            json.dumps(result, allow_nan=False)

    def test_vector_threshold_and_inclusive_boundary(self):
        result = evaluate(np.ones((1, 10)), np.full((1, 10), 0.5), [0.5] * 5 + [0.6] * 5)
        self.assertEqual(result["macro_f1"], 0.5)

    def test_invalid_inputs(self):
        for target, probs, threshold in [
            (np.full((1, 10), 0.5), np.zeros((1, 10)), 0.5),
            (np.zeros((1, 10)), np.zeros((2, 10)), 0.5),
            (np.zeros((1, 10)), np.full((1, 10), np.nan), 0.5),
            (np.zeros((1, 10)), np.zeros((1, 10)), [0.5] * 9),
            (np.zeros((1, 10)), np.zeros((1, 10)), float("nan")),
            (np.zeros((1, 10)), np.zeros((1, 10)), 1.1),
        ]:
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                evaluate(target, probs, threshold)


class SelectThresholdsTests(unittest.TestCase):
    def test_calibration_improves_f1_and_is_deterministic(self):
        targets = np.array([[0.] * 10, [1.] * 10])
        probs = np.array([[0.1] * 10, [0.4] * 10])
        thresholds = select_thresholds(targets, probs, grid=[0.2, 0.4, 0.5])
        np.testing.assert_array_equal(thresholds, [0.4] * 10)
        self.assertEqual(evaluate(targets, probs, thresholds)["macro_f1"], 1.0)
        self.assertGreater(evaluate(targets, probs, thresholds)["macro_f1"],
                           evaluate(targets, probs)["macro_f1"])

    def test_ties_prefer_nearest_half_then_smaller(self):
        targets = np.zeros((2, 10))
        probs = np.zeros((2, 10))
        np.testing.assert_array_equal(select_thresholds(targets, probs), [0.5] * 10)
        np.testing.assert_array_equal(select_thresholds(targets, probs, grid=[0.75, 0.25]), [0.25] * 10)

    def test_invalid_grid_and_empty_validation(self):
        for grid in [[], [[0.5]], [np.nan], [-0.1], [1.1]]:
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                select_thresholds(np.zeros((1, 10)), np.zeros((1, 10)), grid)
        with self.assertRaises(ValueError):
            select_thresholds(np.empty((0, 10)), np.empty((0, 10)))


if __name__ == "__main__":
    unittest.main()
