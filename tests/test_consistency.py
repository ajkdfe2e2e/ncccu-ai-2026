"""Synthetic in-memory prediction comparison tests; no dataset or disk I/O."""

import json
import unittest

import numpy as np

from cxr.consistency import compare_predictions


class ComparePredictionsTests(unittest.TestCase):
    def test_perfect_and_empty(self):
        for expected in [np.full((4, 10), 0.5), np.empty((0, 10))]:
            result = compare_predictions(expected, expected)
            self.assertEqual(result, {"inconsistent_studies": 0,
                                      "max_probability_error": 0.0,
                                      "ranking_changed": False})
            json.dumps(result, allow_nan=False)

    def test_within_tolerance_without_other_changes(self):
        expected = np.tile(np.array([0.1, 0.2, 0.3])[:, None], (1, 10))
        actual = expected + 0.0005
        result = compare_predictions(actual, expected)
        self.assertEqual(result["inconsistent_studies"], 0)
        self.assertFalse(result["ranking_changed"])
        self.assertAlmostEqual(result["max_probability_error"], 0.0005)

    def test_exact_tolerance_allowed_and_greater_error_rejected(self):
        expected = np.full((2, 10), 0.125)
        actual = np.full((2, 10), 0.25)
        result = compare_predictions(actual, expected, tolerance=0.125)
        self.assertEqual(result["inconsistent_studies"], 0)
        result = compare_predictions(actual, expected, tolerance=0.124)
        self.assertEqual(result["inconsistent_studies"], 2)
        self.assertFalse(result["ranking_changed"])
        self.assertEqual(result["max_probability_error"], 0.125)

    def test_rank_inversion_within_tolerance_marks_both_endpoints(self):
        expected = np.full((3, 10), 0.2)
        expected[:, 0] = [0.2, 0.2004, 0.8]
        actual = expected.copy()
        actual[:2, 0] = expected[1::-1, 0]
        result = compare_predictions(actual, expected)
        self.assertEqual(result["inconsistent_studies"], 2)
        self.assertTrue(result["ranking_changed"])
        self.assertLess(result["max_probability_error"], 0.001)

    def test_tie_creation_and_breaking_count_all_affected_rows(self):
        for expected_column, actual_column in [
            ([0.2, 0.2, 0.2], [0.2, 0.2004, 0.2]),
            ([0.2, 0.2004, 0.2004], [0.2, 0.2, 0.2]),
        ]:
            expected = np.full((3, 10), 0.2)
            actual = expected.copy()
            expected[:, 3] = expected_column
            actual[:, 3] = actual_column
            result = compare_predictions(actual, expected)
            self.assertEqual(result["inconsistent_studies"], 3)
            self.assertTrue(result["ranking_changed"])
            self.assertLess(result["max_probability_error"], 0.001)

    def test_disease_class_order_within_row_is_irrelevant(self):
        expected = np.tile([0.2, 0.2004] + [0.3] * 8, (3, 1))
        actual = expected.copy()
        actual[:, :2] = expected[:, 1::-1]
        result = compare_predictions(actual, expected)
        self.assertEqual(result["inconsistent_studies"], 0)
        self.assertFalse(result["ranking_changed"])

    def test_probability_decision_flip_within_tolerance(self):
        expected = np.full((1, 10), 0.4999)
        actual = expected.copy()
        actual[0, 7] = 0.5
        result = compare_predictions(actual, expected)
        self.assertEqual(result["inconsistent_studies"], 1)
        self.assertFalse(result["ranking_changed"])

    def test_threshold_vectors_and_inclusive_decisions(self):
        expected = np.full((2, 10), 0.5)
        result = compare_predictions(expected, expected, actual_thresholds=[0.5] * 9 + [0.6],
                                     expected_thresholds=[0.5] * 10)
        self.assertEqual(result["inconsistent_studies"], 2)
        self.assertEqual(result["max_probability_error"], 0)
        self.assertFalse(result["ranking_changed"])
        result = compare_predictions(expected, expected, actual_thresholds=0.4,
                                     expected_thresholds=0.5)
        self.assertEqual(result["inconsistent_studies"], 0)

    def test_inconsistent_count_is_union_not_sum(self):
        expected = np.full((3, 10), 0.2)
        expected[:, 0] = [0.1, 0.2, 0.3]
        actual = expected.copy()
        actual[0, 0] = 0.6
        result = compare_predictions(actual, expected)
        self.assertEqual(result["inconsistent_studies"], 3)
        self.assertTrue(result["ranking_changed"])
        self.assertEqual(result["max_probability_error"], 0.5)

    def test_optimized_rank_check_matches_pairwise_reference(self):
        rng = np.random.default_rng(17)
        for count in [1, 2, 7, 30]:
            for _ in range(10):
                expected = rng.integers(0, 5, size=(count, 10)) / 4
                actual = rng.integers(0, 5, size=(count, 10)) / 4
                changed_rows = np.zeros(count, dtype=bool)
                for first in range(count):
                    for second in range(first + 1, count):
                        if np.any(np.sign(actual[first] - actual[second]) !=
                                  np.sign(expected[first] - expected[second])):
                            changed_rows[first] = changed_rows[second] = True
                result = compare_predictions(actual, expected, actual_thresholds=0,
                                             expected_thresholds=0, tolerance=1)
                self.assertEqual(result["inconsistent_studies"], int(changed_rows.sum()))
                self.assertEqual(result["ranking_changed"], bool(changed_rows.any()))

    def test_invalid_arrays(self):
        valid = np.zeros((2, 10))
        for invalid in [np.zeros((2, 9)), np.zeros(10), np.zeros((3, 10)),
                        np.full((2, 10), np.nan), np.full((2, 10), np.inf),
                        np.full((2, 10), -0.1), np.full((2, 10), 1.1)]:
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                compare_predictions(invalid, valid)
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                compare_predictions(valid, invalid)

    def test_invalid_thresholds_and_tolerance(self):
        valid = np.zeros((2, 10))
        for invalid in [[0.5] * 9, [[0.5] * 10], np.nan, np.inf, -0.1, 1.1]:
            for name in ["actual_thresholds", "expected_thresholds"]:
                with self.subTest(name=name, invalid=invalid), self.assertRaises(ValueError):
                    compare_predictions(valid, valid, **{name: invalid})
        for invalid in [-0.1, np.nan, np.inf, [0.1], "bad", None]:
            with self.subTest(tolerance=invalid), self.assertRaises(ValueError):
                compare_predictions(valid, valid, tolerance=invalid)


if __name__ == "__main__":
    unittest.main()
