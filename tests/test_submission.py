"""Synthetic CSV tests; never read or inspect any project data."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cxr.submission import COLUMNS, validate_submission, write_submission


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "submission.csv"
        self.records = [{"subject_id": "0001", "study_id": "0020"},
                        {"subject_id": "患者,二", "study_id": "0003"}]
        self.probs = np.array([[0.5] * 10, [0.1234567890123456] * 10])

    def read_rows(self):
        with self.path.open(encoding="utf-8", newline="") as stream:
            return list(csv.reader(stream))

    def save_rows(self, rows):
        with self.path.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)

    def test_round_trip_exact_columns_all_rows_and_string_ids(self):
        returned = write_submission(self.path, self.records, self.probs, [0.5] * 5 + [0.37] * 5)
        self.assertEqual(returned, self.path)
        rows = self.read_rows()
        self.assertEqual(rows[0], list(COLUMNS))
        self.assertEqual(len(rows), 21)
        self.assertEqual(rows[1][:3], ["0001", "0020", "1"])
        self.assertEqual(rows[10][:3], ["0001", "0020", "10"])
        self.assertEqual(rows[11][:3], ["患者,二", "0003", "1"])
        self.assertEqual([r[4] for r in rows[1:11]], ["0.5"] * 5 + ["0.37"] * 5)
        self.assertEqual([r[4] for r in rows[11:21]], ["0.5"] * 5 + ["0.37"] * 5)
        self.assertEqual(float(rows[11][3]), self.probs[1, 0])
        summary = validate_submission(self.path, self.records)
        self.assertEqual(summary, {"valid": True, "studies": 2, "rows": 20,
                                   "classes_per_study": 10, "default_threshold_rows": 10})
        json.dumps(summary, allow_nan=False)

    def test_scalar_default_blank_and_boundary_gate_thresholds(self):
        write_submission(self.path, self.records, self.probs)
        rows = self.read_rows()
        self.assertEqual([r[4] for r in rows[1:]], ["0.5"] * 20)
        self.assertEqual(validate_submission(self.path, self.records)["default_threshold_rows"], 20)
        rows[1][4] = ""
        rows[2][4] = "0"
        rows[3][4] = "1.0"
        rows[4][4] = "0.37"
        self.save_rows(rows)
        self.assertEqual(validate_submission(self.path, self.records)["default_threshold_rows"], 17)
        write_submission(self.path, self.records, self.probs, 0.37)
        self.assertEqual([r[4] for r in self.read_rows()[1:]], ["0.37"] * 20)
        self.assertEqual(validate_submission(self.path, self.records)["default_threshold_rows"], 0)

    def test_image_level_expected_records_and_reordered_rows(self):
        write_submission(self.path, self.records, self.probs, 0.5)
        rows = self.read_rows()
        self.save_rows(rows[:1] + rows[:0:-1])
        result = validate_submission(self.path, self.records + [self.records[0]])
        self.assertEqual(result["rows"], 20)

    def test_header_only_for_empty_expected_records(self):
        write_submission(self.path, [], np.empty((0, 10)), 0.5)
        self.assertEqual(validate_submission(self.path, [])["rows"], 0)
        with self.assertRaises(ValueError):
            validate_submission(self.path, self.records)

    def test_duplicate_missing_extra_and_wrong_id_rows(self):
        write_submission(self.path, self.records, self.probs, 0.5)
        baseline = self.read_rows()
        cases = [baseline + [baseline[1]], baseline[:-1], baseline[:1],
                 baseline[:2] + [baseline[1]] + baseline[3:]]
        for rows in cases:
            with self.subTest(rows=len(rows)):
                self.save_rows(rows)
                with self.assertRaises(ValueError):
                    validate_submission(self.path, self.records)
        for column, value in [(0, "1"), (1, "20"), (1, "9999"), (2, "0"),
                              (2, "11"), (2, "01"), (2, "1.0")]:
            rows = [r[:] for r in baseline]
            rows[1][column] = value
            self.save_rows(rows)
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                validate_submission(self.path, self.records)

    def test_invalid_probabilities_gates_and_columns(self):
        write_submission(self.path, self.records, self.probs, 0.5)
        baseline = self.read_rows()
        for column, values in [(3, ["nan", "inf", "-inf", "-0.01", "1.01", "", "bad"]),
                               (4, ["2", "-1", "-0.01", "1.01", "nan", "inf", "-inf", "True", "bad", " "])]:
            for value in values:
                rows = [r[:] for r in baseline]
                rows[1][column] = value
                self.save_rows(rows)
                with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                    validate_submission(self.path, self.records)
        for header in [list(reversed(COLUMNS)), list(COLUMNS) + ["Extra"], list(COLUMNS)[1:]]:
            self.save_rows([header] + baseline[1:])
            with self.assertRaises(ValueError):
                validate_submission(self.path, self.records)
        for row in [baseline[1] + ["extra"], baseline[1][:-1], []]:
            self.save_rows(baseline[:1] + [row] + baseline[2:])
            with self.assertRaises(ValueError):
                validate_submission(self.path, self.records)

    def test_invalid_utf8_and_malformed_csv(self):
        self.path.write_bytes(b"\xff\xfe")
        with self.assertRaises(ValueError):
            validate_submission(self.path, self.records)
        self.path.write_text(",".join(COLUMNS) + '\n"unclosed', encoding="utf-8")
        with self.assertRaises(ValueError):
            validate_submission(self.path, self.records)

    def test_writer_rejects_duplicate_conflicting_or_invalid_records(self):
        for records in [[self.records[0], self.records[0]],
                        [self.records[0], {"subject_id": "other", "study_id": "0020"}],
                        [{"subject_id": "", "study_id": "s"}], [{}]]:
            with self.subTest(records=records), self.assertRaises(ValueError):
                write_submission(self.path, records, np.zeros((len(records), 10)), 0.5)
        write_submission(self.path, self.records, self.probs, 0.5)
        with self.assertRaises(ValueError):
            validate_submission(self.path, [self.records[0], {"subject_id": "other", "study_id": "0020"}])

    def test_writer_validation_does_not_truncate_existing_output(self):
        self.path.write_text("keep", encoding="utf-8")
        for probs, thresholds in [(np.zeros((1, 10)), 0.5), (np.zeros((2, 9)), 0.5),
                                  (np.full((2, 10), np.nan), 0.5),
                                  (np.full((2, 10), 1.1), 0.5),
                                  (self.probs, [0.5] * 9), (self.probs, -0.1),
                                  (self.probs, float("inf"))]:
            with self.subTest(thresholds=thresholds), self.assertRaises(ValueError):
                write_submission(self.path, self.records, probs, thresholds)
            self.assertEqual(self.path.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
