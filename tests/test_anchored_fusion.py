import unittest

import numpy as np

from cxr.anchored_fusion import (
    apply_public_weights,
    fit_anchored_public,
    flatten_legacy_recipe,
)
from scripts.predict_best_ensemble import replay_recipe


class AnchoredFusionTests(unittest.TestCase):
    def _recipe(self):
        baseline = []
        candidates = []
        sources = [
            ("yaLLMan/rad-dino-chexpert fine-tuned on the local train split", 0.5),
            ("immutable_saved_ensemble", 0.0),
            ("mamounyosef/chest-xray-bench/medmae_vitb_nih_B_768_s2_seed1337", 0.25),
            ("yaLLMan/rad-dino-chexpert fine-tuned on the local train split", 0.25),
            ("mamounyosef/chest-xray-bench/medmae_vitb_nih_B_768_s2_seed1337", 0.5),
            ("yaLLMan/rad-dino-chexpert fine-tuned on the local train split", 0.25),
            ("yaLLMan/rad-dino-chexpert fine-tuned on the local train split", 0.25),
            ("yaLLMan/rad-dino-chexpert fine-tuned on the local train split", 0.5),
            ("RussellALA/convnext-chexpert-seed1", 0.5),
            ("yaLLMan/rad-dino-chexpert fine-tuned on the local train split", 0.25),
        ]
        for class_id in range(1, 11):
            if class_id % 3 == 0:
                baseline.append({"class": class_id, "kind": "external_blend", "external_model": "rad_dino_vitB_768",
                                 "base_model": "rad_dino_finetuned", "alpha_external": 0.4})
            elif class_id % 3 == 1:
                baseline.append({"class": class_id, "kind": "full_model_blend", "left": "resnet18_baseline",
                                 "right": "rad_dino_finetuned", "alpha_left": 0.3})
            else:
                baseline.append({"class": class_id, "kind": "full_model_blend", "left": "rad_dino_finetuned",
                                 "right": "densenet121_chexpert", "alpha_left": 0.2})
            source, alpha = sources[class_id - 1]
            candidates.append({"class": class_id, "source": source, "alpha_candidate": alpha})
        return {"baseline_per_class_selection": baseline, "classes": candidates}

    def test_flatten_matches_legacy_replay_and_keeps_raw_and_tta_distinct(self):
        recipe = self._recipe()
        order = ["resnet18_baseline", "rad_dino_finetuned", "rad_dino_tta", "densenet121_chexpert",
                 "rad_dino_vitB_768", "medmae_vitb_nih_B_768_s2_seed1337", "russellala_convnext"]
        rng = np.random.default_rng(19)
        sources = {name: rng.random((17, 10)) for name in order}
        _, expected = replay_recipe(sources, recipe)
        weights = flatten_legacy_recipe(recipe, order)
        actual = np.einsum("nmj,mj->nj", np.stack([sources[name] for name in order], axis=1), weights)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
        self.assertGreater(weights[order.index("rad_dino_tta"), 0], 0)
        self.assertGreater(weights[order.index("rad_dino_finetuned"), 0], 0)
        self.assertFalse(np.array_equal(weights[order.index("rad_dino_tta")],
                                        weights[order.index("rad_dino_finetuned")]))
        np.testing.assert_allclose(weights.sum(axis=0), 1.0, rtol=0, atol=1e-12)

    def test_flatten_rejects_unknown_sources_duplicates_and_bad_coefficients(self):
        recipe = self._recipe()
        recipe["classes"][0]["source"] = "unknown"
        with self.assertRaises(ValueError):
            flatten_legacy_recipe(recipe, ["resnet18_baseline", "rad_dino_finetuned", "rad_dino_tta",
                                           "densenet121_chexpert", "rad_dino_vitB_768",
                                           "medmae_vitb_nih_B_768_s2_seed1337", "russellala_convnext"])
        recipe = self._recipe()
        recipe["classes"][1]["class"] = 1
        with self.assertRaises(ValueError):
            flatten_legacy_recipe(recipe, ["resnet18_baseline", "rad_dino_finetuned", "rad_dino_tta",
                                           "densenet121_chexpert", "rad_dino_vitB_768",
                                           "medmae_vitb_nih_B_768_s2_seed1337", "russellala_convnext"])
        recipe = self._recipe()
        recipe["baseline_per_class_selection"][0]["alpha_left"] = 1.2
        with self.assertRaises(ValueError):
            flatten_legacy_recipe(recipe, ["resnet18_baseline", "rad_dino_finetuned", "rad_dino_tta",
                                           "densenet121_chexpert", "rad_dino_vitB_768",
                                           "medmae_vitb_nih_B_768_s2_seed1337", "russellala_convnext"])

    def _fit_data(self):
        rng = np.random.default_rng(5)
        n = 80
        labels = np.tile(np.array([0, 1], dtype=np.float64)[:, None], (n // 2, 10))
        probs = np.full((n, 3, 10), np.nan, dtype=np.float64)
        for label in range(10):
            signal = labels[:, label]
            probs[:, 0, label] = np.clip(0.25 + 0.5 * signal + rng.normal(0, 0.2, n), 0, 1)
            probs[:, 1, label] = np.clip(0.75 - 0.5 * signal + rng.normal(0, 0.2, n), 0, 1)
        prior = np.zeros((3, 10), dtype=np.float64)
        prior[0, :] = 0.7
        prior[1, :] = 0.3
        return probs, labels, prior

    def test_rho_zero_is_exact_prior_and_apply_masks_inactive_classes(self):
        probs, labels, prior = self._fit_data()
        prior[:, 9] = 0
        probs[:, :, 9] = np.nan
        fitted = fit_anchored_public(probs, labels, prior, rho=0.0, pair_count=128)
        weights = np.asarray(fitted["weights"])
        np.testing.assert_array_equal(weights, prior)
        output = apply_public_weights(probs, weights)
        self.assertTrue(np.isfinite(output).all())
        np.testing.assert_array_equal(output[:, 9], np.zeros(len(labels)))

    def test_fit_is_deterministic_decreases_objective_and_obeys_trust_floor(self):
        probs, labels, prior = self._fit_data()
        kwargs = {"rho": 0.2, "l2": 1.0, "temperature": 0.1, "pair_count": 512, "seed": 11}
        first = fit_anchored_public(probs, labels, prior, **kwargs)
        second = fit_anchored_public(probs, labels, prior, **kwargs)
        weights = np.asarray(first["weights"])
        np.testing.assert_array_equal(weights, np.asarray(second["weights"]))
        self.assertTrue(np.all(weights >= 0.8 * prior - 1e-12))
        np.testing.assert_allclose(weights.sum(axis=0), 1.0, rtol=0, atol=1e-10)
        self.assertTrue(all(item["objective_after"] <= item["objective_before"] + 1e-12
                            for item in first["classes"] if item["active"]))
        fused = apply_public_weights(probs, weights)
        self.assertTrue(np.isfinite(fused).all())

    def test_available_zero_prior_public_heads_can_receive_anchored_mass(self):
        probs, labels, _ = self._fit_data()
        rng = np.random.default_rng(23)
        five = np.empty((len(labels), 5, 10), dtype=np.float64)
        for class_id in range(10):
            signal = labels[:, class_id]
            five[:, 0, class_id] = np.clip(0.75 - 0.5 * signal + rng.normal(0, 0.15, len(labels)), 0, 1)
            for model in range(1, 5):
                five[:, model, class_id] = np.clip(0.25 + 0.5 * signal + rng.normal(0, 0.15, len(labels)), 0, 1)
        prior = np.zeros((5, 10), dtype=np.float64)
        prior[0, :] = 1.0
        fitted = fit_anchored_public(five, labels, prior, rho=0.2, pair_count=256, seed=7)
        weights = np.asarray(fitted["weights"])
        self.assertTrue(np.all(weights >= 0.8 * prior - 1e-12))
        np.testing.assert_allclose(weights.sum(axis=0), 1.0, rtol=0, atol=1e-10)
        self.assertTrue(np.all(weights[1:, :] >= 0))
        self.assertTrue(np.any(weights[1:, :] > 0))
        self.assertTrue(np.isfinite(apply_public_weights(five, weights)).all())

    def test_rejects_bad_targets_prior_support_and_invalid_nan_mapping(self):
        probs, labels, prior = self._fit_data()
        bad_labels = labels.copy()
        bad_labels[0, 0] = 0.5
        with self.assertRaises(ValueError):
            fit_anchored_public(probs, bad_labels, prior)
        bad_prior = prior.copy()
        bad_prior[0, 0] = 0.6
        with self.assertRaises(ValueError):
            fit_anchored_public(probs, labels, bad_prior)
        unavailable_prior = prior.copy()
        unavailable_prior[2, :] = 0.1
        unavailable_prior[0, :] -= 0.1
        with self.assertRaises(ValueError):
            fit_anchored_public(probs, labels, unavailable_prior)
        partial = probs.copy()
        partial[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            fit_anchored_public(partial, labels, prior)
        invalid = probs.copy()
        invalid[0, 0, 0] = 2.0
        with self.assertRaises(ValueError):
            fit_anchored_public(invalid, labels, prior)


if __name__ == "__main__":
    unittest.main()
