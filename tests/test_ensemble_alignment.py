import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.eval_ensemble import _align


class EnsembleAlignmentTests(unittest.TestCase):
    def test_reorders_targets_and_probabilities_together(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            keys = [("patient-1", "study-1"), ("patient-2", "study-2")]
            targets = np.array([[0, 1] * 5, [1, 0] * 5])
            probabilities = np.array([[0.1, 0.8] * 5, [0.7, 0.2] * 5])
            (folder / "study_ids.json").write_text(json.dumps(keys[::-1]))
            np.save(folder / "study_targets.npy", targets[::-1])
            np.save(folder / "study_probabilities.npy", probabilities[::-1])
            np.testing.assert_array_equal(_align(folder, keys, targets), probabilities)

    def test_rejects_target_mismatch_after_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            keys = [("patient-1", "study-1"), ("patient-2", "study-2")]
            targets = np.array([[0, 1] * 5, [1, 0] * 5])
            (folder / "study_ids.json").write_text(json.dumps(keys[::-1]))
            np.save(folder / "study_targets.npy", targets)
            np.save(folder / "study_probabilities.npy", np.full((2, 10), 0.5))
            with self.assertRaisesRegex(ValueError, "after ID alignment"):
                _align(folder, keys, targets)
