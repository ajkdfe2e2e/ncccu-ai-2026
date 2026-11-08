"""Linear probes over semantic logits or train-only GAP principal components."""

from collections.abc import Mapping

import numpy as np
import torch

from cxr.study_probe import NUM_CLASSES, fit_linear_probe, predict_linear_probe


INPUT_WIDTH = 1054
GAP_WIDTH = 1024
LOGIT_WIDTH = 30
PCA_WIDTH = 128
MODE_SEMANTIC = 0
MODE_PCA128 = 1


def _features(values, name="features"):
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric matrix") from exc
    if array.ndim != 2 or array.shape[1] != INPUT_WIDTH:
        raise ValueError(f"{name} must have shape (N, {INPUT_WIDTH})")
    if not len(array):
        raise ValueError(f"{name} must contain at least one row")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _mode_id(mode):
    if mode == "semantic":
        return MODE_SEMANTIC
    if mode == "pca128":
        return MODE_PCA128
    raise ValueError("mode must be 'semantic' or 'pca128'")


def fit_feature_transform(training_features, mode):
    """Fit transform metadata using features from training rows only."""
    mode_id = _mode_id(mode)
    x = _features(training_features, "training_features")
    transform = {"mode_id": np.int64(mode_id)}
    if mode_id == MODE_SEMANTIC:
        return transform

    gap = x[:, :GAP_WIDTH]
    mean = gap.mean(axis=0)
    std = gap.std(axis=0)
    std[std == 0] = 1.0
    standardized = (gap - mean) / std
    tensor = torch.as_tensor(standardized, dtype=torch.float64, device="cpu")
    covariance = tensor.T @ tensor / len(tensor)
    _, eigenvectors = torch.linalg.eigh(covariance)
    components = eigenvectors[:, -PCA_WIDTH:].flip(1).cpu().numpy().copy()

    pivots = np.argmax(np.abs(components), axis=0)
    signs = np.sign(components[pivots, np.arange(PCA_WIDTH)])
    signs[signs == 0] = 1.0
    components *= signs[None, :]
    transform.update(gap_mean=mean, gap_std=std, components=components)
    return transform


def apply_feature_transform(transform, features):
    """Apply saved semantic or PCA transform metadata to 1054D features."""
    if not isinstance(transform, Mapping):
        raise ValueError("transform must be a mapping")
    try:
        raw_mode_id = np.asarray(transform["mode_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("transform is missing a valid mode_id") from exc
    if raw_mode_id.shape != () or raw_mode_id.dtype.kind not in "iu":
        raise ValueError("transform mode_id must be an integer scalar")
    mode_id = int(raw_mode_id)
    x = _features(features)
    logits = x[:, GAP_WIDTH:]
    if mode_id == MODE_SEMANTIC:
        if any(key in transform for key in ("gap_mean", "gap_std", "components")):
            raise ValueError("semantic transform must not contain PCA fields")
        return logits.copy()
    if mode_id != MODE_PCA128:
        raise ValueError("transform mode_id is unsupported")
    try:
        mean = np.asarray(transform["gap_mean"], dtype=np.float64)
        std = np.asarray(transform["gap_std"], dtype=np.float64)
        components = np.asarray(transform["components"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("PCA transform is missing valid arrays") from exc
    if (mean.shape != (GAP_WIDTH,) or std.shape != (GAP_WIDTH,)
            or components.shape != (GAP_WIDTH, PCA_WIDTH)):
        raise ValueError("PCA transform arrays have incompatible shapes")
    if (not all(np.isfinite(value).all() for value in (mean, std, components))
            or np.any(std <= 0)):
        raise ValueError("PCA transform arrays must be finite with positive standard deviations")
    projected = ((x[:, :GAP_WIDTH] - mean) / std) @ components
    return np.concatenate((projected, logits), axis=1)


def fit_bottleneck_probe(features, targets, *, mode, l2=0.01, max_iter=300,
                         seed=20261108, class_balance=False):
    """Fit an ordinary BCE linear probe over one of the fixed bottlenecks."""
    if not isinstance(class_balance, (bool, np.bool_)) or bool(class_balance):
        raise ValueError("bottleneck probe supports only class_balance=False")
    x = _features(features, "training_features")
    transform = fit_feature_transform(x, mode)
    transformed = apply_feature_transform(transform, x)
    payload = fit_linear_probe(transformed, targets, l2=l2, max_iter=max_iter,
                               seed=seed, class_balance=False)
    payload.update(transform)
    return payload


def predict_bottleneck_probe(payload, features):
    """Predict probabilities with a saved bottleneck and linear-head payload."""
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a mapping")
    transformed = apply_feature_transform(payload, features)
    return predict_linear_probe(payload, transformed)
