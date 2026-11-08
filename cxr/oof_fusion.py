"""Patient-OOF-trained per-class simplex fusion for multilabel probabilities."""

import math
from numbers import Integral, Real

import numpy as np


NUM_CLASSES = 10


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite_real(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _inputs(probabilities, targets):
    try:
        probs = np.asarray(probabilities, dtype=np.float64)
        labels = np.asarray(targets, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("probabilities and targets must be numeric arrays") from exc
    if probs.ndim != 3 or probs.shape[0] == 0 or probs.shape[1] == 0 or probs.shape[2] != NUM_CLASSES:
        raise ValueError("probabilities must have shape (N, M, 10) with N and M positive")
    if labels.shape != (probs.shape[0], NUM_CLASSES):
        raise ValueError("targets must have shape (N, 10) matching probabilities")
    if not np.isfinite(labels).all() or ((labels != 0) & (labels != 1)).any():
        raise ValueError("targets must contain finite binary labels (0 or 1)")
    if not np.isfinite(probs[~np.isnan(probs)]).all():
        raise ValueError("probabilities may contain NaN for unavailable heads, but not infinity")
    available = np.empty((probs.shape[1], NUM_CLASSES), dtype=bool)
    for model in range(probs.shape[1]):
        for label in range(NUM_CLASSES):
            column = probs[:, model, label]
            missing = np.isnan(column)
            if missing.any() and not missing.all():
                raise ValueError("a model/class head must be fully available or all NaN")
            available[model, label] = not missing.all()
            if not missing.all() and ((column < 0) | (column > 1)).any():
                raise ValueError("available probabilities must be in [0, 1]")
    if not available.any(axis=0).all():
        raise ValueError("every class must have at least one available model")
    for label in range(NUM_CLASSES):
        if not np.any(labels[:, label] == 1) or not np.any(labels[:, label] == 0):
            raise ValueError(f"class {label + 1} must contain positives and negatives")
    return probs, labels, available


def _project_simplex(vector):
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    indexes = np.arange(1, len(vector) + 1)
    valid = ordered - cumulative / indexes > 0
    rho = np.flatnonzero(valid)[-1]
    threshold = cumulative[rho] / (rho + 1)
    return np.maximum(vector - threshold, 0.0)


def _loss_gradient(weights, differences, uniform, temperature, l2):
    margins = differences @ weights / temperature
    loss = np.logaddexp(0.0, -margins).mean()
    residual = -1.0 / (1.0 + np.exp(np.clip(margins, -700.0, 700.0)))
    gradient = differences.T @ residual / (len(differences) * temperature)
    delta = weights - uniform
    return float(loss + 0.5 * l2 * (delta @ delta)), gradient + l2 * delta


def _fit_class(differences, temperature, l2, max_iter, tolerance):
    count = differences.shape[1]
    uniform = np.full(count, 1.0 / count, dtype=np.float64)
    weights = uniform.copy()
    objective_before, gradient = _loss_gradient(weights, differences, uniform, temperature, l2)
    objective = objective_before
    converged = False
    iterations = 0
    step = 1.0
    for iteration in range(1, max_iter + 1):
        accepted = False
        trial_step = step
        for _ in range(60):
            candidate = _project_simplex(weights - trial_step * gradient)
            candidate_objective, candidate_gradient = _loss_gradient(
                candidate, differences, uniform, temperature, l2
            )
            displacement = candidate - weights
            if candidate_objective <= objective + gradient @ displacement + (displacement @ displacement) / (2 * trial_step) + 1e-14:
                accepted = True
                break
            trial_step *= 0.5
        if not accepted:
            break
        iterations = iteration
        improvement = objective - candidate_objective
        weights, objective, gradient = candidate, candidate_objective, candidate_gradient
        step = min(trial_step * 1.5, 1e6)
        if np.linalg.norm(displacement) <= tolerance or improvement <= tolerance * max(1.0, abs(objective)):
            converged = True
            break
    return weights, {
        "objective_before": float(objective_before),
        "objective_after": float(objective),
        "iterations": int(iterations),
        "converged": bool(converged),
    }


def fit_simplex_fusion(
    probabilities,
    targets,
    *,
    seed=20261106,
    pair_count=32768,
    temperature=0.1,
    l2=0.1,
    max_iter=300,
    tolerance=1e-8,
):
    """Fit per-class simplex weights using a fixed sampled pairwise logistic loss."""
    probs, labels, available = _inputs(probabilities, targets)
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    pair_count = _positive_int(pair_count, "pair_count")
    temperature = _finite_real(temperature, "temperature", positive=True)
    l2 = _finite_real(l2, "l2", nonnegative=True)
    max_iter = _positive_int(max_iter, "max_iter")
    tolerance = _finite_real(tolerance, "tolerance", positive=True)
    weights = np.zeros((probs.shape[1], NUM_CLASSES), dtype=np.float64)
    rng = np.random.default_rng(seed)
    diagnostics = []
    for label in range(NUM_CLASSES):
        positive = np.flatnonzero(labels[:, label] == 1)
        negative = np.flatnonzero(labels[:, label] == 0)
        models = np.flatnonzero(available[:, label])
        total_pairs = len(positive) * len(negative)
        if total_pairs <= pair_count:
            pos_index = np.repeat(positive, len(negative))
            neg_index = np.tile(negative, len(positive))
        else:
            pos_index = rng.choice(positive, size=pair_count, replace=True)
            neg_index = rng.choice(negative, size=pair_count, replace=True)
        differences = probs[pos_index[:, None], models[None, :], label] - probs[neg_index[:, None], models[None, :], label]
        class_weights, info = _fit_class(differences, temperature, l2, max_iter, tolerance)
        weights[models, label] = class_weights
        diagnostics.append({
            "class_id": label + 1,
            "available_models": [int(index) for index in models],
            "pair_count": int(len(pos_index)),
            **info,
        })
    return {
        "version": 1,
        "weights": weights.tolist(),
        "coverage": available.tolist(),
        "parameters": {
            "seed": int(seed),
            "pair_count": int(pair_count),
            "temperature": float(temperature),
            "l2": float(l2),
            "max_iter": int(max_iter),
            "tolerance": float(tolerance),
        },
        "classes": diagnostics,
        "objective": "mean_pairwise_logistic_plus_uniform_l2",
        "objective_note": "AUC surrogate; optimization does not guarantee improved AUC.",
    }


def _validate_recipe(recipe, models):
    if not isinstance(recipe, dict) or recipe.get("version") != 1:
        raise ValueError("recipe must be a version 1 fusion recipe")
    try:
        weights = np.asarray(recipe["weights"], dtype=np.float64)
        raw_coverage = np.asarray(recipe["coverage"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("recipe must contain numeric weights and coverage") from exc
    if raw_coverage.dtype.kind not in "biu" or not np.isin(raw_coverage, (0, 1)).all():
        raise ValueError("recipe coverage must contain only boolean values")
    coverage = raw_coverage.astype(bool)
    if weights.shape != (models, NUM_CLASSES) or coverage.shape != (models, NUM_CLASSES):
        raise ValueError("recipe weights and coverage must have shape (M, 10)")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("recipe weights must be finite and nonnegative")
    if (weights[~coverage] != 0).any():
        raise ValueError("unsupported model/class weights must be exactly zero")
    if not coverage.any(axis=0).all():
        raise ValueError("recipe must cover all ten classes")
    if not np.allclose(weights.sum(axis=0), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("recipe weights must sum to one for every class")
    return weights, coverage


def apply_simplex_fusion(probabilities, recipe):
    """Apply a validated recipe while safely masking unavailable model heads."""
    try:
        probs = np.asarray(probabilities, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("probabilities must be a numeric N x M x 10 array") from exc
    if probs.ndim != 3 or probs.shape[0] == 0 or probs.shape[2] != NUM_CLASSES:
        raise ValueError("probabilities must have shape (N, M, 10) with N positive")
    weights, coverage = _validate_recipe(recipe, probs.shape[1])
    actual = np.empty_like(coverage)
    for model in range(probs.shape[1]):
        for label in range(NUM_CLASSES):
            column = probs[:, model, label]
            missing = np.isnan(column)
            if missing.any() and not missing.all():
                raise ValueError("a model/class head must be fully available or all NaN")
            actual[model, label] = not missing.all()
            if not missing.all() and (not np.isfinite(column).all() or ((column < 0) | (column > 1)).any()):
                raise ValueError("available probabilities must be finite and in [0, 1]")
    if not np.array_equal(actual, coverage):
        raise ValueError("probability availability does not match recipe coverage")
    result = np.zeros((probs.shape[0], NUM_CLASSES), dtype=np.float64)
    for model in range(probs.shape[1]):
        for label in range(NUM_CLASSES):
            if coverage[model, label]:
                result[:, label] += probs[:, model, label] * weights[model, label]
    return result


def uniform_simplex_recipe(probabilities):
    """Build a validated uniform recipe for each class's available model heads."""
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 3 or probs.shape[0] == 0 or probs.shape[1] == 0 or probs.shape[2] != NUM_CLASSES:
        raise ValueError("probabilities must have shape (N, M, 10) with N and M positive")
    # Reuse inference validation without requiring targets.
    available = np.empty((probs.shape[1], NUM_CLASSES), dtype=bool)
    for model in range(probs.shape[1]):
        for label in range(NUM_CLASSES):
            column = probs[:, model, label]
            missing = np.isnan(column)
            if missing.any() and not missing.all():
                raise ValueError("a model/class head must be fully available or all NaN")
            available[model, label] = not missing.all()
            if not missing.all() and (not np.isfinite(column).all() or ((column < 0) | (column > 1)).any()):
                raise ValueError("available probabilities must be finite and in [0, 1]")
    if not available.any(axis=0).all():
        raise ValueError("every class must have at least one available model")
    weights = np.zeros_like(available, dtype=np.float64)
    for label in range(NUM_CLASSES):
        weights[available[:, label], label] = 1.0 / available[:, label].sum()
    return {"version": 1, "weights": weights.tolist(), "coverage": available.tolist()}
