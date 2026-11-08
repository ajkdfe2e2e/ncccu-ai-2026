import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.run_grouped_generalization import (
    aligned_predictions, choose_configuration, qualifying, stage_command,
)
from scripts.train_rad_dino_grouped import select_records, validate_config, verify_image_inventory


class GroupedGeneralizationTests(unittest.TestCase):
    def test_selection_requires_effect_size_and_multiple_improving_folds(self):
        self.assertFalse(qualifying([0.01, -0.001, -0.001]))
        self.assertFalse(qualifying([0.0009, 0.0009, 0.0009]))
        self.assertTrue(qualifying([0.004, 0.003, -0.001]))
        with self.assertRaises(ValueError):
            qualifying([0.01, np.nan, 0.01])

    def test_raw_AB_selection_precedes_selected_configuration_EMA(self):
        selected = choose_configuration({
            "A/raw": [.8, .8, .8], "A/ema": [.85, .85, .85],
            "B/raw": [.803, .804, .802], "B/ema": [.802, .803, .801],
        })
        self.assertEqual((selected["config"], selected["variant"]), ("B", "raw"))
        self.assertTrue(selected["qualifies_for_audit"])

    def test_ties_keep_baseline_and_audit_closed(self):
        selected = choose_configuration({key: [.8, .8, .8] for key in (
            "A/raw", "A/ema", "B/raw", "B/ema")})
        self.assertEqual((selected["config"], selected["variant"]), ("A", "raw"))
        self.assertFalse(selected["qualifies_for_audit"])

    def test_baseline_ema_can_qualify_when_lower_lr_does_not(self):
        selected = choose_configuration({
            "A/raw": [.8, .8, .8], "A/ema": [.803, .804, .805],
            "B/raw": [.79, .79, .79], "B/ema": [.9, .9, .9],
        })
        self.assertEqual((selected["config"], selected["variant"]), ("A", "ema"))
        self.assertTrue(selected["qualifies_for_audit"])

    def test_stage_command_uses_snapshot_public_init_and_fixed_budget(self):
        plan = {"data_root": "/raw", "model_dir": "/public", "epochs": 3,
                "batch_size": 2, "accumulation_steps": 8, "workers": 0,
                "training_seed": 2026, "ema_decay": .999, "device": "cpu"}
        name, command = stage_command(plan, Path("/experiment"), "B", "cv", 2)
        self.assertEqual(name, "cv_B_fold2")
        self.assertIn(str(Path("/experiment") / "source/scripts/train_rad_dino_grouped.py"), command)
        self.assertEqual(command[command.index("--head-lr") + 1], "0.0001")
        self.assertEqual(command[command.index("--epochs") + 1], "3")
        self.assertEqual(command[command.index("--fold") + 1], "2")
        self.assertNotIn("--init-checkpoint", command)

    def test_alignment_rejects_missing_studies_and_mismatched_targets(self):
        records = [{"subject_id": "p1", "study_id": "s1", "labels": [0] * 10},
                   {"subject_id": "p2", "study_id": "s2", "labels": [1] * 10}]
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "study_ids.json").write_text(json.dumps([["p2", "s2"], ["p1", "s1"]]))
            np.save(folder / "study_probabilities.npy", [[.9] * 10, [.1] * 10])
            np.save(folder / "study_targets.npy", [[1] * 10, [0] * 10])
            keys, _, probabilities, metrics = aligned_predictions(folder, records)
            self.assertEqual(keys, [("p1", "s1"), ("p2", "s2")])
            self.assertEqual(metrics["macro_auc"], 1)
            self.assertAlmostEqual(probabilities[0, 0], .1)
            np.save(folder / "study_targets.npy", [[0] * 10, [1] * 10])
            with self.assertRaisesRegex(ValueError, "labels differ"):
                aligned_predictions(folder, records)
            (folder / "study_ids.json").write_text(json.dumps([["p2", "s2"], ["p3", "s3"]]))
            with self.assertRaisesRegex(ValueError, "coverage"):
                aligned_predictions(folder, records)

    def test_cv_requires_fold_and_valid_hyperparameters(self):
        cfg = argparse.Namespace(epochs=3, batch_size=2, accumulation_steps=8,
                                 workers=0, seed=2026, head_lr=1e-4,
                                 ema_decay=.999, phase="cv", fold=None)
        with self.assertRaisesRegex(ValueError, "fold"):
            validate_config(cfg)
        cfg.fold = 0
        validate_config(cfg)
        cfg.head_lr = float("nan")
        with self.assertRaisesRegex(ValueError, "head-lr"):
            validate_config(cfg)

    def test_phase_rejects_unrecognized_modes(self):
        with self.assertRaisesRegex(ValueError, "phase"):
            select_records([], {}, "test", None)

    def test_image_inventory_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.jpg"
            path.write_bytes(b"original")
            stat = path.stat()
            records = [{"dicom_id": "image", "path": str(path)}]
            inventory = {"image": {"path": str(path), "bytes": stat.st_size,
                                    "mtime_ns": stat.st_mtime_ns}}
            verify_image_inventory(records, inventory)
            path.write_bytes(b"changed and larger")
            with self.assertRaisesRegex(ValueError, "locked inventory"):
                verify_image_inventory(records, inventory)


if __name__ == "__main__":
    unittest.main()
