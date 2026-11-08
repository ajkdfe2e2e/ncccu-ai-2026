"""Synthetic tests for train-only bottleneck feature transforms."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cxr.bottleneck_probe import (
    GAP_WIDTH,
    INPUT_WIDTH,
    LOGIT_WIDTH,
    PCA_WIDTH,
    apply_feature_transform,
    fit_bottleneck_probe,
    fit_feature_transform,
    predict_bottleneck_probe,
)
from cxr.study_probe import fit_linear_probe, predict_linear_probe


def synthetic(rows=144, seed=19):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(rows, INPUT_WIDTH))
    targets = rng.integers(0, 2, size=(rows, 10)).astype(np.float64)
    return features, targets


class BottleneckProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_torch_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_torch_threads)

    def test_semantic_matches_existing_probe_and_ignores_gap(self):
        features, targets = synthetic(64)
        payload = fit_bottleneck_probe(features, targets, mode="semantic", max_iter=15)
        direct = fit_linear_probe(features[:, GAP_WIDTH:], targets, max_iter=15,
                                  class_balance=False)
        self.assertEqual(int(payload["mode_id"]), 0)
        self.assertFalse(any(key in payload for key in ("gap_mean", "gap_std", "components")))
        for key, expected in direct.items():
            if isinstance(expected, np.ndarray):
                np.testing.assert_array_equal(payload[key], expected)
            else:
                self.assertEqual(payload[key], expected)
        np.testing.assert_array_equal(
            predict_bottleneck_probe(payload, features),
            predict_linear_probe(direct, features[:, GAP_WIDTH:]),
        )

        changed_gap = features.copy()
        changed_gap[:, :GAP_WIDTH] = np.nan_to_num(changed_gap[:, :GAP_WIDTH] * 1000 + 77)
        np.testing.assert_array_equal(predict_bottleneck_probe(payload, changed_gap),
                                      predict_bottleneck_probe(payload, features))

    def test_pca_is_train_only_oriented_and_has_fixed_projection_shape(self):
        features, _ = synthetic(144)
        training = features[:132]
        transform = fit_feature_transform(training, "pca128")
        changed_heldout = features.copy()
        changed_heldout[132:, :GAP_WIDTH] += 1e6
        repeated = fit_feature_transform(changed_heldout[:132], "pca128")
        for key in ("mode_id", "gap_mean", "gap_std", "components"):
            np.testing.assert_array_equal(transform[key], repeated[key])

        components = transform["components"]
        self.assertEqual(components.shape, (GAP_WIDTH, PCA_WIDTH))
        self.assertEqual(np.linalg.matrix_rank(components), PCA_WIDTH)
        np.testing.assert_allclose(components.T @ components, np.eye(PCA_WIDTH),
                                   atol=1e-12, rtol=1e-12)
        pivots = np.argmax(np.abs(components), axis=0)
        self.assertTrue(np.all(components[pivots, np.arange(PCA_WIDTH)] > 0))
        projected = apply_feature_transform(transform, features)
        self.assertEqual(projected.shape, (len(features), PCA_WIDTH + LOGIT_WIDTH))
        np.testing.assert_array_equal(projected[:, PCA_WIDTH:], features[:, GAP_WIDTH:])

    def test_pca_payload_roundtrips_through_numpy_archive_exactly(self):
        features, targets = synthetic(140, seed=21)
        payload = fit_bottleneck_probe(features, targets, mode="pca128", max_iter=12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head.npz"
            np.savez(path, **payload)
            with np.load(path, allow_pickle=False) as archive:
                restored = {key: archive[key].copy() for key in archive.files}
        self.assertEqual(set(restored), set(payload))
        for key, value in payload.items():
            self.assertNotIn(np.asarray(value).dtype.kind, "OUSV")
            np.testing.assert_array_equal(restored[key], value)
        np.testing.assert_array_equal(predict_bottleneck_probe(payload, features),
                                      predict_bottleneck_probe(restored, features))

    def test_validation_rejects_bad_inputs_modes_and_class_balance(self):
        features, targets = synthetic(48)
        for bad in (np.ones((48, 1053)), np.ones((48, 1055)),
                    np.full((48, INPUT_WIDTH), np.inf)):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                fit_bottleneck_probe(bad, targets, mode="semantic", max_iter=5)
        for mode in ("unknown", None, 1):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "mode"):
                fit_bottleneck_probe(features, targets, mode=mode, max_iter=5)
        for value in (True, 0, 1, "false", None):
            with self.subTest(class_balance=value), self.assertRaisesRegex(ValueError, "class_balance"):
                fit_bottleneck_probe(features, targets, mode="semantic", class_balance=value)
        with self.assertRaisesRegex(ValueError, "shape"):
            predict_bottleneck_probe({"mode_id": np.int64(0)}, np.ones((2, 3)))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            apply_feature_transform({"mode_id": np.int64(2)}, features)


if __name__ == "__main__":
    unittest.main()
