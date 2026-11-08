"""Synthetic boundary checks for the bottleneck run verifier."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cxr.bottleneck_probe import INPUT_WIDTH, fit_feature_transform
from scripts.verify_bottleneck_study_probe import (
    EXPECTED_CONFIGS,
    _config_name,
    _require_known_config,
    _same_config,
    _select,
    _validate_report_path,
    _verify_head_transform,
)


class BottleneckVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_torch_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_torch_threads)

    def test_configuration_identity_names_tie_and_plan_boundary(self):
        semantic, pca = EXPECTED_CONFIGS
        self.assertEqual(_config_name(semantic), "semantic_l2_0.01")
        self.assertEqual(_config_name(pca), "pca128_l2_0.01")
        self.assertFalse(_same_config(semantic, pca))
        _require_known_config(semantic, {"configs": list(EXPECTED_CONFIGS)})
        with self.assertRaisesRegex(ValueError, "fixed bottleneck plan"):
            _require_known_config({**semantic, "mode": "other"},
                                  {"configs": list(EXPECTED_CONFIGS)})
        rows = [
            {"config": pca, "macro_auc": 0.8, "macro_auc_defined_classes": 10},
            {"config": semantic, "macro_auc": 0.8, "macro_auc_defined_classes": 10},
        ]
        self.assertEqual(_select(rows)["config"]["mode"], "semantic")

    def test_saved_pca_transform_is_rederived_from_training_rows(self):
        rng = np.random.default_rng(241)
        features = rng.normal(size=(138, INPUT_WIDTH))
        train, held = features[:132], features[132:]
        transform = fit_feature_transform(train, "pca128")
        errors = {}
        _verify_head_transform(transform, train, EXPECTED_CONFIGS[1], errors, "synthetic_head")
        self.assertIn("synthetic_head.deterministic_projection", errors)

        leaked_transform = fit_feature_transform(np.vstack((train, held)), "pca128")
        with self.assertRaisesRegex(ValueError, "gap_mean_from_training_rows"):
            _verify_head_transform(leaked_transform, train, EXPECTED_CONFIGS[1], {},
                                   "heldout_leak")

    def test_semantic_transform_rejects_pca_metadata(self):
        features = np.ones((3, INPUT_WIDTH))
        with self.assertRaisesRegex(ValueError, "PCA fields"):
            _verify_head_transform({"mode_id": np.int64(0), "components": np.ones((1, 1))},
                                   features, EXPECTED_CONFIGS[0], {}, "semantic_head")

    def test_report_path_cannot_overwrite_locked_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / "experiment"
            experiment.mkdir()
            locked = root / "input.bin"
            locked.write_bytes(b"locked")
            plan = {"input_locks": {str(locked): "unused"}, "source_locks": {}}
            with self.assertRaisesRegex(ValueError, "locked"):
                _validate_report_path(locked, experiment, plan, {})
            valid = _validate_report_path(root / "report.json", experiment, plan, {})
            self.assertEqual(valid, (root / "report.json").resolve())


if __name__ == "__main__":
    unittest.main()
