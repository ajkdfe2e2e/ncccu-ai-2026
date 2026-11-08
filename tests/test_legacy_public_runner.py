import json
from pathlib import Path
import unittest

import numpy as np

from cxr.anchored_fusion import flatten_legacy_recipe
from scripts.predict_best_ensemble import replay_recipe
from scripts.train_legacy_public_fusion import (
    MODEL_ORDER,
    apply_delta,
    gate_decision,
    patient_folds,
)


class LegacyPublicRunnerTests(unittest.TestCase):
    def test_patient_folds_keep_each_subject_together(self):
        studies = [
            {"subject_id": subject, "study_id": f"study-{index}"}
            for index, subject in enumerate(("p1", "p2", "p1", "p3", "p4", "p2", "p5"))
        ]
        assignment, folds = patient_folds(studies)
        self.assertEqual(set(assignment), {row["subject_id"] for row in studies})
        self.assertEqual(len(set(folds.tolist())), 3)
        for subject, fold in assignment.items():
            subject_folds = {folds[i] for i, row in enumerate(studies) if row["subject_id"] == subject}
            self.assertEqual(subject_folds, {fold})

    def test_apply_delta_preserves_zero_mass_and_identical_weights_exactly(self):
        rng = np.random.default_rng(7)
        anchor = 0.2 + 0.6 * rng.random((6, 10))
        probabilities = rng.random((6, 5, 10))
        before = np.full((5, 10), 0.2)
        after = before.copy()
        mass = np.full(10, 0.5)
        mass[[0, 7, 9]] = 0.0

        unchanged = apply_delta(anchor, probabilities, before, before, mass)
        np.testing.assert_array_equal(unchanged, anchor)

        after[:, 1] = [0.4, 0.3, 0.1, 0.1, 0.1]
        candidate = apply_delta(anchor, probabilities, before, after, mass)
        for class_index in (0, 7, 9):
            np.testing.assert_array_equal(candidate[:, class_index], anchor[:, class_index])

    def test_guardrail_rejects_undefined_low_gain_and_interval_crossing_zero(self):
        def comparison(defined, delta, lower):
            return {
                "macro_auc_defined_classes": defined,
                "macro": {"auc": {"delta": delta, "ci95_percentile": [lower, 0.02]}},
            }

        self.assertFalse(gate_decision(comparison(9, 0.02, 0.01)))
        self.assertFalse(gate_decision(comparison(10, 0.0009, 0.0001)))
        self.assertFalse(gate_decision(comparison(10, 0.01, 0.0)))
        self.assertFalse(gate_decision(comparison(10, 0.01, -0.001)))
        self.assertTrue(gate_decision(comparison(10, 0.001, 0.00001)))
        undefined_interval = comparison(10, 0.01, 0.01)
        undefined_interval["macro"]["auc"]["ci95_percentile"] = None
        self.assertFalse(gate_decision(undefined_interval))

    def test_real_legacy_recipe_replay_keeps_raw_and_tta_sources_distinct(self):
        root = Path(__file__).resolve().parents[1]
        recipe = json.loads((root / "docs/best_ensemble_recipe_20261105.json").read_text(encoding="utf-8"))
        rng = np.random.default_rng(19)
        sources = {name: rng.random((12, 10)) for name in MODEL_ORDER}
        _, replayed = replay_recipe(sources, recipe)
        weights = flatten_legacy_recipe(recipe, MODEL_ORDER)
        stacked = np.stack([sources[name] for name in MODEL_ORDER], axis=1)
        expanded = np.einsum("nmj,mj->nj", stacked, weights)

        np.testing.assert_allclose(expanded, replayed, rtol=0, atol=1e-14)
        raw = MODEL_ORDER.index("rad_dino_finetuned")
        tta = MODEL_ORDER.index("rad_dino_tta")
        self.assertTrue(np.any(weights[raw] > 0))
        self.assertTrue(np.any(weights[tta] > 0))
        self.assertFalse(np.array_equal(weights[raw], weights[tta]))


if __name__ == "__main__":
    unittest.main()
