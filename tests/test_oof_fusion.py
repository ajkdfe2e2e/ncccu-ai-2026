import copy
import unittest

import numpy as np

from cxr.oof_fusion import (
    apply_simplex_fusion,
    fit_simplex_fusion,
    uniform_simplex_recipe,
)


def _fixture():
    targets = np.tile(np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)[:, None], (1, 10))
    good = np.tile(np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])[:, None], (1, 10))
    reverse = 1.0 - good
    return np.stack([good, reverse], axis=1), targets


class OOFSimplexFusionTests(unittest.TestCase):
    def test_fit_improves_pairwise_loss_and_suppresses_adversarial_branch(self):
        probabilities, targets = _fixture()
        recipe = fit_simplex_fusion(probabilities, targets, pair_count=1000)
        weights = np.asarray(recipe["weights"])
        self.assertGreater(weights[0].mean(), weights[1].mean())
        self.assertLess(weights[1].mean(), 0.5)
        for entry in recipe["classes"]:
            self.assertLessEqual(entry["objective_after"], entry["objective_before"] + 1e-12)
        fused = apply_simplex_fusion(probabilities, recipe)
        uniform = apply_simplex_fusion(probabilities, uniform_simplex_recipe(probabilities))
        self.assertTrue(np.all(np.diff(fused[:, 0]) >= 0))
        self.assertLess(float(np.mean(np.logaddexp(0, -(fused[4:, 0].mean() - fused[:4, 0].mean())))),
                        float(np.mean(np.logaddexp(0, -(uniform[4:, 0].mean() - uniform[:4, 0].mean())))))

    def test_complementary_predictions_can_improve_loss_over_either_branch(self):
        targets = np.tile(np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)[:, None], (1, 10))
        first = np.tile(np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])[:, None], (1, 10))
        second = np.tile(np.array([0.2, 0.1, 0.4, 0.3, 0.9, 0.8, 0.7, 0.6])[:, None], (1, 10))
        probs = np.stack([first, second], axis=1)
        recipe = fit_simplex_fusion(probs, targets, l2=0.0, pair_count=1000)
        fused = apply_simplex_fusion(probs, recipe)
        pairwise = lambda scores: np.logaddexp(0.0, -(scores[4:, None, 0] - scores[None, :4, 0]) / 0.1).mean()
        self.assertLess(pairwise(fused), min(pairwise(first), pairwise(second)))

    def test_simplex_constraints_and_exact_partial_head_masking(self):
        probabilities, targets = _fixture()
        probabilities[:, 1, 3] = np.nan
        recipe = fit_simplex_fusion(probabilities, targets)
        weights = np.asarray(recipe["weights"])
        coverage = np.asarray(recipe["coverage"])
        self.assertTrue(np.all(weights >= 0))
        np.testing.assert_allclose(weights.sum(axis=0), np.ones(10))
        self.assertTrue(np.array_equal(weights[~coverage], np.zeros(np.count_nonzero(~coverage))))
        self.assertEqual(weights[1, 3], 0.0)
        self.assertTrue(np.isfinite(apply_simplex_fusion(probabilities, recipe)).all())

    def test_deterministic_seed_and_fit(self):
        probabilities, targets = _fixture()
        first = fit_simplex_fusion(probabilities, targets, seed=4, pair_count=5)
        second = fit_simplex_fusion(probabilities, targets, seed=4, pair_count=5)
        self.assertEqual(first, second)

    def test_single_available_branch_gets_unit_weight(self):
        probabilities, targets = _fixture()
        probabilities[:, 1, :] = np.nan
        recipe = fit_simplex_fusion(probabilities, targets)
        np.testing.assert_array_equal(np.asarray(recipe["weights"])[0], np.ones(10))
        np.testing.assert_array_equal(apply_simplex_fusion(probabilities, recipe), probabilities[:, 0, :])

    def test_rejects_invalid_input_values_shapes_and_labels(self):
        probabilities, targets = _fixture()
        cases = []
        partial = probabilities.copy()
        partial[0, 0, 0] = np.nan
        cases.append((partial, targets))
        infinite = probabilities.copy()
        infinite[0, 0, 0] = np.inf
        cases.append((infinite, targets))
        out_of_range = probabilities.copy()
        out_of_range[0, 0, 0] = 2.0
        cases.append((out_of_range, targets))
        bad_targets = targets.copy()
        bad_targets[0, 0] = 0.5
        cases.append((probabilities, bad_targets))
        cases.append((probabilities[:, :, :9], targets))
        for probs, labels in cases:
            with self.subTest(shape=probs.shape):
                with self.assertRaises(ValueError):
                    fit_simplex_fusion(probs, labels)
        one_class = targets.copy()
        one_class[:, 0] = 1
        with self.assertRaisesRegex(ValueError, "positives and negatives"):
            fit_simplex_fusion(probabilities, one_class)

    def test_rejects_invalid_recipes_and_coverage_mismatch(self):
        probabilities, targets = _fixture()
        recipe = fit_simplex_fusion(probabilities, targets)
        bad = copy.deepcopy(recipe)
        bad["weights"][0][0] = -0.1
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            apply_simplex_fusion(probabilities, bad)
        bad = copy.deepcopy(recipe)
        bad["weights"][0][0] += 0.2
        with self.assertRaisesRegex(ValueError, "sum to one"):
            apply_simplex_fusion(probabilities, bad)
        bad = copy.deepcopy(recipe)
        bad["coverage"][0][0] = False
        with self.assertRaisesRegex(ValueError, "exactly zero"):
            apply_simplex_fusion(probabilities, bad)
        probabilities[:, 1, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "availability does not match"):
            apply_simplex_fusion(probabilities, recipe)

    def test_rejects_invalid_optimizer_parameters(self):
        probabilities, targets = _fixture()
        for value in (0, -1, 1.5, True):
            with self.subTest(max_iter=value):
                with self.assertRaisesRegex(ValueError, "max_iter"):
                    fit_simplex_fusion(probabilities, targets, max_iter=value)


if __name__ == "__main__":
    unittest.main()
