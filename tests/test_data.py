"""Synthetic tests; never access the real dataset or decode an image."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cxr.data import read_records
from scripts.audit_data import audit_data, main


FIELDS = ["Dicom_id", "Subject_id", "Study_id", "ViewPosition", "PatientOrientation", "Rows", "Columns", "Predict_class"]


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / "train"
        self.images.mkdir()

    def row(self, dicom="pic_1", subject="subject_1", study="study_1", label="1;10"):
        return [dicom, subject, study, "AP", "Erect", "100", "200", label]

    def manifest(self, rows, name="train.csv", fields=None):
        path = self.root / name
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(FIELDS if fields is None else fields)
            writer.writerows(rows)
        return path

    def image(self, dicom="pic_1", subject="subject_1", study="study_1", suffix=".jpg", split="train"):
        path = self.root / split / subject / study / (dicom + suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Intentionally invalid bytes ensure the indexer never requires image decoding.
        path.write_bytes(b"synthetic placeholder, not an image")
        return path.resolve()

    def test_multihot_and_resolved_path(self):
        image = self.image(suffix=".JPEG")
        records = read_records(self.manifest([self.row()]), self.images)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(set(record), {"dicom_id", "subject_id", "study_id", "path", "labels"})
        self.assertEqual(record["path"], str(image))
        self.assertEqual(record["labels"], [1.0] + [0.0] * 8 + [1.0])
        self.assertTrue(all(type(label) is float for label in record["labels"]))

    def test_unlabeled_test(self):
        self.image()
        path = self.manifest([self.row()[:-1]], "test.csv", FIELDS[:-1])
        self.assertIsNone(read_records(path, self.images)[0]["labels"])

    def test_missing_labels_rejected(self):
        for label in ("", " ", "0", "11", "1;", ";1", "1;;2", "1,2", "1.0", "nan", "-1", "01", "1;1", "foo"):
            with self.subTest(label=label), self.assertRaises(ValueError):
                read_records(self.manifest([self.row(label=label)]), self.images)

    def test_train_and_valid_require_label_column(self):
        for name in ("train.csv", "valid.csv", "custom.csv"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Predict_class"):
                read_records(self.manifest([self.row()[:-1]], name, FIELDS[:-1]), self.images)

    def test_labeled_test_rejected(self):
        with self.assertRaisesRegex(ValueError, "test.csv"):
            read_records(self.manifest([self.row()], "test.csv"), self.images)

    def test_duplicate_dicom_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate Dicom_id"):
            read_records(self.manifest([self.row(), self.row(study="study_2")]), self.images)

    def test_conflicting_study_subject_rejected(self):
        with self.assertRaisesRegex(ValueError, "conflicting"):
            read_records(self.manifest([self.row(), self.row("pic_2", subject="subject_2")]), self.images)

    def test_conflicting_study_labels_rejected(self):
        with self.assertRaisesRegex(ValueError, "conflicting"):
            read_records(self.manifest([self.row(), self.row("pic_2", label="2")]), self.images)

    def test_label_order_is_irrelevant(self):
        self.image()
        self.image("pic_2")
        records = read_records(self.manifest([self.row(), self.row("pic_2", label="10; 1")]), self.images)
        self.assertEqual(records[0]["labels"], records[1]["labels"])

    def test_unsafe_identifiers_rejected_in_all_id_columns(self):
        for column in range(3):
            for bad in ("", "..", "../escape", "..\\escape", "/absolute", "C:\\escape", "C:escape", "x/y", "x\\y", "x.", "x ", " x", "x\x00", "CON"):
                row = self.row()
                row[column] = bad
                with self.subTest(column=column, bad=bad), self.assertRaises(ValueError):
                    read_records(self.manifest([row]), self.images)

    def test_missing_image_rejected(self):
        with self.assertRaisesRegex(FileNotFoundError, "missing image"):
            read_records(self.manifest([self.row()]), self.images)

    def test_missing_root_rejected(self):
        with self.assertRaises(FileNotFoundError):
            read_records(self.manifest([self.row()]), self.root / "absent")

    def test_ambiguous_image_rejected(self):
        self.image()
        self.image(suffix=".png")
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            read_records(self.manifest([self.row()]), self.images)

    def test_wrong_subject_not_used_as_fallback(self):
        self.image(subject="other_subject")
        with self.assertRaises(FileNotFoundError):
            read_records(self.manifest([self.row()]), self.images)

    def test_duplicate_headers_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate CSV headers"):
            read_records(self.manifest([], fields=FIELDS + ["Predict_class"]), self.images)

    def test_missing_id_header_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            read_records(self.manifest([], fields=FIELDS[1:]), self.images)

    def test_ragged_rows_rejected(self):
        for row in (self.row()[:-1], self.row() + ["extra"]):
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, "malformed"):
                read_records(self.manifest([row]), self.images)

    def test_empty_manifest(self):
        self.assertEqual(read_records(self.manifest([]), self.images), [])

    def test_symlink_outside_root_rejected(self):
        external = self.root / "outside.jpg"
        external.write_bytes(b"placeholder")
        link = self.images / "subject_1" / "study_1" / "pic_1.jpg"
        link.parent.mkdir(parents=True)
        try:
            link.symlink_to(external)
        except OSError:
            self.skipTest("Creating symlinks is not permitted on this platform")
        with self.assertRaisesRegex(ValueError, "escapes image_root"):
            read_records(self.manifest([self.row()]), self.images)

    def audit_fixture(self):
        self.manifest([self.row(), self.row("pic_2"), self.row("pic_3", "subject_3", "study_3", "2")])
        self.image()
        self.image("pic_2")
        self.manifest([self.row("pic_v", "subject_1", "study_v", "3")], "valid.csv")
        self.image("pic_v", "subject_1", "study_v", split="valid")
        test = self.row("pic_t", "subject_t", "study_t")[:-1]
        test[-2:] = ["not a dimension", "also ignored"]
        self.manifest([test], "test.csv", FIELDS[:-1])
        self.image("pic_t", "subject_t", "study_t", split="test")

    def test_audit_counts_study_prevalence_overlaps_missing(self):
        self.audit_fixture()
        report = audit_data(self.root)
        train = report["splits"]["train"]
        self.assertEqual((train["images"], train["studies"], train["patients"]), (3, 2, 2))
        self.assertEqual(train["missing_images_count"], 1)
        self.assertEqual(train["study_label_prevalence"]["1"], {"count": 1, "fraction": 0.5})
        self.assertEqual(report["overlaps"]["train_valid"]["patients"]["count"], 1)
        self.assertNotIn("study_label_prevalence", report["splits"]["test"])
        self.assertNotIn("declared_dimensions", report["splits"]["test"])

    def test_test_dimensions_never_requested(self):
        self.audit_fixture()
        from scripts.audit_data import _dimensions
        calls = []
        def dimensions(path):
            calls.append(path.name)
            self.assertNotEqual(path.name, "test.csv")
            return _dimensions(path)
        with patch("scripts.audit_data._dimensions", side_effect=dimensions):
            audit_data(self.root)
        self.assertEqual(calls, ["train.csv", "valid.csv"])

    def test_cli_writes_report_and_rejects_raw_output(self):
        self.audit_fixture()
        # Output must be outside raw data_root; use another synthetic temp directory.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as destination:
            output = Path(destination) / "nested" / "audit.json"
            self.assertEqual(main(["--data-root", str(self.root), "--output", str(output)]), 0)
            self.assertEqual(json.loads(output.read_text())["schema_version"], 1)
        with self.assertRaises(SystemExit) as result:
            main(["--data-root", str(self.root), "--output", str(self.root / "audit.json")])
        self.assertEqual(result.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
