"""Synthetic-only comparison tests: in-memory CSVs, no dataset or image access."""
import csv
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from cxr.metrics import evaluate
from cxr.submission import COLUMNS
from scripts import compare_runs as comparison


def csv_text(header, rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue()


def submission_text(studies, probabilities, gates=None, reverse=False):
    rows = []
    for i, study in enumerate(studies):
        for j in range(10):
            rows.append([study["subject_id"], study["study_id"], j + 1,
                         probabilities[i, j], "" if gates is None else gates[i, j]])
    if reverse:
        rows.reverse()
    return csv_text(COLUMNS, rows)


class MemoryCSVs:
    """Patch only file opens; any unlisted path is a test failure, never disk I/O."""
    def __init__(self, files):
        self.files = {str(Path(path).resolve()): value for path, value in files.items()}
        self.opened = []

    def __enter__(self):
        files, opened = self.files, self.opened

        def open_memory(path, mode="r", *args, **kwargs):
            key = str(path.resolve())
            if key not in files or mode not in ("r", "rb"):
                raise AssertionError(f"Unexpected filesystem access: {path}, {mode}")
            opened.append(key)
            text = files[key]
            return io.BytesIO(text.encode("utf-8")) if mode == "rb" else io.StringIO(text)

        self.patcher = patch.object(Path, "open", open_memory)
        self.patcher.start()
        return self

    def __exit__(self, *args):
        self.patcher.stop()


class ArrayComparisonTests(unittest.TestCase):
    def test_numpy_pairwise_auc_reference_with_ties_and_fixed_f1(self):
        rng = np.random.default_rng(30)
        y = rng.integers(0, 2, size=(24, 10))
        b = rng.integers(0, 5, size=y.shape) / 4
        c = rng.integers(0, 5, size=y.shape) / 4
        report = comparison.compare_arrays([str(i // 2) for i in range(24)], y, b, c,
                                            bootstrap=12)
        for j, item in enumerate(report["per_class"]):
            values = []
            for probs, run in ((b, "baseline"), (c, "candidate")):
                pos = probs[y[:, j] == 1, j]
                neg = probs[y[:, j] == 0, j]
                auc = np.mean((pos[:, None] > neg) + 0.5 * (pos[:, None] == neg))
                values.append(auc)
                self.assertAlmostEqual(item["metrics"]["auc"][run], auc)
                self.assertAlmostEqual(item["metrics"]["f1_fixed_0_5"][run],
                                       evaluate(y, probs)["per_class"][j]["f1"])
            self.assertAlmostEqual(item["metrics"]["auc"]["delta"], values[1] - values[0])
        self.assertEqual(report["patients"], 12)
        self.assertEqual(report["studies"], 24)
        json.dumps(report, allow_nan=False)

    def test_row_gates_and_threshold_only_improvement(self):
        y = np.tile(np.array([0, 1, 0, 1])[:, None], (1, 10))
        probs = np.tile(np.array([0.1, 0.4, 0.3, 0.6])[:, None], (1, 10))
        bg = np.full((4, 10), 0.5)
        cg = np.tile(np.array([0.2, 0.4, 0.4, 0.6])[:, None], (1, 10))
        report = comparison.compare_arrays(["a", "a", "b", "b"], y, probs, probs,
                                            bg, cg, bootstrap=8)
        self.assertEqual(report["macro"]["auc"]["delta"], 0)
        self.assertEqual(report["macro"]["f1_fixed_0_5"]["delta"], 0)
        declared = report["macro"]["f1_declared_gate"]
        self.assertAlmostEqual(declared["baseline"], 2 / 3)
        self.assertEqual(declared["candidate"], 1)
        self.assertAlmostEqual(declared["delta"], 1 / 3)
        self.assertIn("threshold-only", report["caveat"])
        self.assertIn("not unbiased", report["caveat"])

    def test_bootstrap_reproducible_paired_and_full_patient_clusters(self):
        groups = comparison._patient_groups(["001", "001", "002", "003", "003", "003"])

        class Draw:
            def integers(self, low, high, size):
                self.args = (low, high, size)
                return np.array([2, 0, 2])

        draw = Draw()
        indices = comparison._sample_patient_indices(groups, draw)
        np.testing.assert_array_equal(indices, [3, 4, 5, 0, 1, 3, 4, 5])
        self.assertEqual(draw.args, (0, 3, 3))
        y = np.tile(np.array([0, 1, 0, 1, 0, 1])[:, None], (1, 10))
        p = np.full((6, 10), 0.5)
        args = (["001", "001", "002", "003", "003", "003"], y, p, p)
        first = comparison.compare_arrays(*args, bootstrap=20)
        second = comparison.compare_arrays(*args, bootstrap=20)
        self.assertEqual(first, second)
        for metric in comparison.METRICS:
            self.assertEqual(first["macro"][metric]["ci95_percentile"], [0.0, 0.0])
        self.assertEqual(first["bootstrap"]["seed"], 2026)

    def test_bootstrap_matches_explicit_cluster_resampling_reference(self):
        y = np.tile(np.array([0, 1, 1, 0, 1])[:, None], (1, 10))
        b = np.tile(np.array([0.6, 0.3, 0.7, 0.4, 0.9])[:, None], (1, 10))
        c = np.tile(np.array([0.2, 0.8, 0.9, 0.3, 0.4])[:, None], (1, 10))
        groups = [np.array([0, 1]), np.array([2]), np.array([3, 4])]
        rng = np.random.default_rng(2026)
        expected = {"auc": [], "f1_fixed_0_5": []}
        for _ in range(31):
            indices = np.concatenate([groups[i] for i in rng.integers(0, 3, size=3)])
            base, candidate = evaluate(y[indices], b[indices]), evaluate(y[indices], c[indices])
            for name, source in (("auc", "macro_auc"), ("f1_fixed_0_5", "macro_f1")):
                if base[source] is not None:
                    expected[name].append(candidate[source] - base[source])
        report = comparison.compare_arrays(["a", "a", "b", "c", "c"], y, b, c, bootstrap=31)
        for name in expected:
            np.testing.assert_allclose(report["macro"][name]["ci95_percentile"],
                                       np.percentile(expected[name], [2.5, 97.5]))
            self.assertEqual(report["macro"][name]["defined_replicates"], len(expected[name]))

    def test_missing_auc_counts_and_zero_denominator_f1_are_json_safe(self):
        y = np.zeros((2, 10))
        y[1, 0] = 1
        report = comparison.compare_arrays(["a", "b"], y, y, y, bootstrap=40)
        self.assertEqual(report["macro_auc_defined_classes"], 1)
        self.assertEqual(report["macro_auc_missing_classes"], 9)
        macro = report["macro"]["auc"]
        self.assertGreater(macro["missing_replicates"], 0)
        self.assertGreater(macro["defined_replicates"], 0)
        self.assertEqual(macro["defined_replicates"] + macro["missing_replicates"], 40)
        missing = report["per_class"][1]["metrics"]["auc"]
        self.assertIsNone(missing["baseline"])
        self.assertIsNone(missing["delta"])
        self.assertIsNone(missing["ci95_percentile"])
        self.assertEqual(missing["missing_replicates"], 40)
        self.assertEqual(report["per_class"][1]["metrics"]["f1_fixed_0_5"]["baseline"], 0)
        all_missing = comparison.compare_arrays(["a", "b"], np.zeros((2, 10)), y, y, bootstrap=3)
        self.assertIsNone(all_missing["macro"]["auc"]["ci95_percentile"])
        self.assertEqual(all_missing["bootstrap"]["auc_defined_classes_histogram"], {"0": 3})
        json.dumps(all_missing, allow_nan=False)

    def test_invalid_array_arguments_rejected(self):
        good = np.zeros((2, 10))
        for kwargs in ({"bootstrap": 0}, {"bootstrap": 1.5}, {"bootstrap": True},
                       {"seed": -1}, {"seed": 1.5}, {"baseline_gates": np.ones((2, 9))},
                       {"candidate_gates": np.full((2, 10), np.nan)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                comparison.compare_arrays(["a", "b"], good, good, good, **kwargs)
        for ids, y, b in [(["a"], good, good), (["a", ""], good, good),
                           ([], good[:0], good[:0]), (["a", "b"], good + 0.5, good),
                           (["a", "b"], good, good - 0.1)]:
            with self.assertRaises(ValueError):
                comparison.compare_arrays(ids, y, b, b, bootstrap=1)


class CSVComparisonTests(unittest.TestCase):
    def setUp(self):
        self.root = comparison.ROOT / "__synthetic_compare_no_disk__"
        self.metadata = self.root / "raw" / "valid.csv"
        self.baseline = self.root / "baseline.csv"
        self.candidate = self.root / "candidate.csv"
        self.output = self.root / "report.json"
        self.meta_text = csv_text(["Dicom_id", "Subject_id", "Study_id", "Predict_class"],
                                  [["d1", "001", "s2", "1;3"],
                                   ["d2", "002", "s1", "2"],
                                   ["d3", "001", "s2", "1;3"]])
        self.studies = [{"subject_id": "001", "study_id": "s2"},
                        {"subject_id": "002", "study_id": "s1"}]
        self.probs = np.arange(20).reshape(2, 10) / 20

    def test_metadata_deduplicates_and_csv_alignment_reads_every_gate(self):
        gates = np.arange(20).reshape(2, 10) / 20
        text = submission_text(self.studies, self.probs, gates, reverse=True)
        with MemoryCSVs({self.metadata: self.meta_text, self.baseline: text}):
            studies, targets = comparison.load_validation_metadata(self.metadata)
            probs, actual_gates, summary = comparison.load_submission(self.baseline, studies)
        self.assertEqual([r["study_id"] for r in studies], ["s2", "s1"])
        self.assertEqual(studies[0]["subject_id"], "001")
        np.testing.assert_array_equal(targets[0], [1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(probs, self.probs)
        np.testing.assert_array_equal(actual_gates, gates)
        self.assertEqual(summary["rows"], 20)

    def test_strict_submission_coverage_rejects_missing_duplicate_and_unknown_rows(self):
        text = submission_text(self.studies, self.probs)
        lines = text.splitlines()
        for malformed in ("\n".join(lines[:-1]), text + lines[-1] + "\n",
                          text.replace("001,s2,", "003,s2,", 1),
                          text.replace(",0.0,", ",nan,", 1)):
            with self.subTest(malformed=malformed), MemoryCSVs({self.baseline: malformed}), self.assertRaises(ValueError):
                comparison.load_submission(self.baseline, self.studies)

    def test_metadata_nonvalidation_empty_and_conflicting_studies_rejected(self):
        # Refuse even opening a non-validation path, including test metadata.
        for name in ("test.csv", "train.csv", "renamed.csv"):
            with MemoryCSVs({}), self.assertRaises(ValueError):
                comparison.load_validation_metadata(self.root / name)
        header = self.meta_text.splitlines()[0] + "\n"
        for text in (header, self.meta_text.replace("d3,001,s2,1;3", "d3,001,s2,2")):
            with MemoryCSVs({self.metadata: text}), self.assertRaises(ValueError):
                comparison.load_validation_metadata(self.metadata)

    def test_full_report_hashes_default_gates_and_no_image_io(self):
        base = submission_text(self.studies, self.probs)
        candidate = submission_text(self.studies, self.probs, reverse=True)
        with MemoryCSVs({self.metadata: self.meta_text, self.baseline: base,
                         self.candidate: candidate}), patch.object(Path, "exists", return_value=False), \
                patch.object(Path, "mkdir") as mkdir, patch.object(Path, "write_text") as write:
            result = comparison.compare_runs(self.baseline, self.candidate, self.metadata,
                                             self.output, bootstrap=5)
        saved = json.loads(write.call_args.args[0])
        self.assertEqual(saved, result)
        self.assertEqual(result["inputs"]["baseline"]["sha256"], hashlib.sha256(base.encode()).hexdigest())
        self.assertEqual(result["inputs"]["candidate"]["sha256"], hashlib.sha256(candidate.encode()).hexdigest())
        self.assertEqual(result["inputs"]["baseline"]["validation"]["default_threshold_rows"], 20)
        self.assertEqual(result["macro"]["f1_fixed_0_5"], result["macro"]["f1_declared_gate"])
        mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_output_guards_before_any_read_or_write(self):
        forbidden = [self.baseline, self.candidate, self.metadata,
                     self.metadata.parent / "report.json", comparison.ROOT / "cxr_first" / "report.json"]
        for output in forbidden:
            with self.subTest(output=output), MemoryCSVs({}), patch.object(Path, "write_text") as write:
                with self.assertRaises(ValueError):
                    comparison.compare_runs(self.baseline, self.candidate, self.metadata, output, bootstrap=1)
                write.assert_not_called()
        with MemoryCSVs({}), patch.object(Path, "exists", return_value=True), \
                patch.object(Path, "samefile", return_value=True), self.assertRaises(ValueError):
            comparison.compare_runs(self.baseline, self.candidate, self.metadata, self.output, bootstrap=1)

    def test_invalid_submission_does_not_open_output(self):
        base = submission_text(self.studies, self.probs)
        with MemoryCSVs({self.metadata: self.meta_text, self.baseline: base, self.candidate: base + "bad\n"}), \
                patch.object(Path, "exists", return_value=False), patch.object(Path, "write_text") as write:
            with self.assertRaises(ValueError):
                comparison.compare_runs(self.baseline, self.candidate, self.metadata, self.output, bootstrap=1)
            write.assert_not_called()

    def test_cli_defaults(self):
        report = {"studies": 2, "patients": 2, "macro": {}, "caveat": comparison.CAVEAT}
        with patch.object(comparison, "compare_runs", return_value=report) as compare, \
                patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(comparison.main(["--baseline", str(self.baseline), "--candidate", str(self.candidate),
                                              "--metadata", str(self.metadata), "--output", str(self.output)]), 0)
        self.assertEqual(compare.call_args.kwargs, {"bootstrap": 500, "seed": 2026})


if __name__ == "__main__":
    unittest.main()
