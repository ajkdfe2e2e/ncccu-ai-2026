import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.improve_best_ensemble import improve


def _write_bundle(folder, keys, targets, probabilities, report):
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "study_targets.npy", targets)
    np.save(folder / "study_probabilities.npy", probabilities)
    (folder / "study_ids.json").write_text(json.dumps(keys), encoding="utf-8")
    (folder / "report.json").write_text(json.dumps(report), encoding="utf-8")


def _data(rows=12):
    targets = (np.arange(rows)[:, None] % 2) * np.ones((1, 10), dtype=np.float64)
    keys = [[f"subject-{row // 2}", f"study-{row}"] for row in range(rows)]
    baseline = np.where(targets == 1, 0.2, 0.8)
    candidate = np.where(targets == 1, 0.9, 0.1)
    return keys, targets, baseline, candidate


class ImproveBestEnsembleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.keys, self.targets, self.baseline, self.candidate = _data()
        self.baseline_dir = self.root / "baseline"
        _write_bundle(self.baseline_dir, self.keys, self.targets, self.baseline, {
            "local_labels": [f"label-{index}" for index in range(10)],
            "selected": {"macro_auc": 0.0},
            "per_class_selection": [{"class": index + 1, "source": "original-source"}
                                    for index in range(10)],
            "models": {"original-source": {"macro_auc": 0.75}},
        })

    def tearDown(self):
        self.temp.cleanup()

    def _candidate(self, *, keys=None, targets=None, probabilities=None, report=None, name="candidate"):
        folder = self.root / name
        _write_bundle(
            folder,
            self.keys if keys is None else keys,
            self.targets if targets is None else targets,
            self.candidate if probabilities is None else probabilities,
            {"model": "candidate-model"} if report is None else report,
        )
        return folder

    def test_aligns_rows_and_applies_selected_coefficients(self):
        permutation = np.arange(len(self.keys))[::-1]
        folder = self._candidate(
            keys=[self.keys[index] for index in permutation],
            targets=self.targets[permutation],
            probabilities=self.candidate[permutation],
        )
        output = self.root / "result"

        report = improve(self.baseline_dir, [folder], output)

        self.assertEqual(report["selected"]["macro_auc"], 1.0)
        alphas = [item["alpha_candidate"] for item in report["recipe"]]
        self.assertTrue(all(alpha in (0.5, 0.75, 1.0) for alpha in alphas))
        expected = np.column_stack([
            alpha * self.candidate[:, index] + (1.0 - alpha) * self.baseline[:, index]
            for index, alpha in enumerate(alphas)
        ])
        np.testing.assert_array_equal(np.load(output / "study_probabilities.npy"), expected)
        self.assertEqual(report["fixed_half_blends"][0]["macro_auc"], 1.0)
        self.assertEqual(report["computed_baseline_metrics"]["macro_auc"], 0.0)
        recipe = json.loads((output / "recipe.json").read_text(encoding="utf-8"))
        self.assertEqual(len(recipe["baseline_selection"]["per_class_selection"]), 10)
        self.assertIn("original-source", recipe["baseline_selection"]["sources"])
        self.assertIn("combined_selection", report["cross_validation"])

    def test_rejects_targets_that_mismatch_after_id_alignment(self):
        permutation = np.arange(len(self.keys))[::-1]
        mismatched = self.targets.copy()
        mismatched[0, 0] = 1 - mismatched[0, 0]
        folder = self._candidate(
            keys=[self.keys[index] for index in permutation],
            targets=mismatched[permutation],
            probabilities=self.candidate[permutation],
        )
        with self.assertRaisesRegex(ValueError, "targets differ.*after ID alignment"):
            improve(self.baseline_dir, [folder], self.root / "result")

    def test_partial_candidate_changes_only_mapped_classes(self):
        indices = [8, 2]
        probabilities = self.candidate[:, indices]
        folder = self._candidate(probabilities=probabilities,
                                  report={"model": "partial-model", "local_label_indices": indices})

        report = improve(self.baseline_dir, [folder], self.root / "result")

        self.assertEqual(report["candidates"][0]["available_classes"], [9, 3])
        self.assertEqual(set(report["candidates"][0]["selected_classes"]), {3, 9})
        self.assertEqual(report["recipe"][2]["source"], "partial-model")
        self.assertEqual(report["recipe"][8]["source"], "partial-model")
        for index in set(range(10)) - set(indices):
            np.testing.assert_array_equal(np.load(self.root / "result/study_probabilities.npy")[:, index],
                                          self.baseline[:, index])

    def test_full_candidate_with_permuted_class_mapping_is_scored_canonically(self):
        indices = list(reversed(range(10)))
        folder = self._candidate(probabilities=self.candidate[:, indices],
                                  report={"model": "permuted-full", "local_label_indices": indices})

        report = improve(self.baseline_dir, [folder], self.root / "result")

        self.assertEqual(report["candidates"][0]["metrics"]["macro_auc"], 1.0)

    def test_rejects_duplicate_candidate_ids(self):
        duplicate_keys = [list(key) for key in self.keys]
        duplicate_keys[1] = duplicate_keys[0]
        folder = self._candidate(keys=duplicate_keys)
        with self.assertRaisesRegex(ValueError, "duplicate subject/study IDs"):
            improve(self.baseline_dir, [folder], self.root / "result")

    def test_rejects_nonfinite_candidate_probabilities(self):
        invalid = self.candidate.copy()
        invalid[0, 0] = np.nan
        folder = self._candidate(probabilities=invalid)
        with self.assertRaisesRegex(ValueError, "finite values"):
            improve(self.baseline_dir, [folder], self.root / "result")

    def test_keeps_immutable_baseline_when_candidate_does_not_improve(self):
        good_baseline = np.where(self.targets == 1, 0.9, 0.1)
        folder = self._candidate(probabilities=1.0 - good_baseline)
        _write_bundle(self.baseline_dir, self.keys, self.targets, good_baseline, {})

        report = improve(self.baseline_dir, [folder], self.root / "result")

        self.assertEqual(report["selected"]["delta_vs_immutable_baseline"], 0.0)
        self.assertTrue(all(item["alpha_candidate"] == 0.0 for item in report["recipe"]))
        np.testing.assert_array_equal(np.load(self.root / "result/study_probabilities.npy"), good_baseline)

    def test_rejects_output_overlapping_an_input_or_raw_root(self):
        folder = self._candidate()
        with self.assertRaisesRegex(ValueError, "overlap an input folder"):
            improve(self.baseline_dir, [folder], self.baseline_dir / "nested")
        raw_root = self.root / "raw"
        raw_root.mkdir()
        with self.assertRaisesRegex(ValueError, "inside raw data"):
            improve(self.baseline_dir, [folder], raw_root / "generated", data_root=raw_root)

    def test_rejects_nonempty_output_directory(self):
        folder = self._candidate()
        output = self.root / "occupied"
        output.mkdir()
        (output / "keep.txt").write_text("existing", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not empty"):
            improve(self.baseline_dir, [folder], output)


if __name__ == "__main__":
    unittest.main()
