"""Flatten the frozen legacy ensemble and fit trust-anchored public weights."""

import math
from numbers import Integral, Real

import numpy as np

from cxr.oof_fusion import NUM_CLASSES, _project_simplex


_CANDIDATES = {
    "yaLLMan/rad-dino-chexpert fine-tuned on the local train split": "rad_dino_tta",
    "RussellALA/convnext-chexpert-seed1": "russellala_convnext",
    "mamounyosef/chest-xray-bench/medmae_vitb_nih_B_768_s2_seed1337":
        "medmae_vitb_nih_B_768_s2_seed1337",
}
_KNOWN_SOURCES = {
    "resnet18_baseline", "rad_dino_finetuned", "rad_dino_tta",
    "densenet121_chexpert", "rad_dino_vitB_768",
    "medmae_vitb_nih_B_768_s2", "medmae_vitb_nih_B_768_s2_seed1337",
    "russellala_convnext",
}


def _coefficient(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _entries_by_class(entries, name):
    if not isinstance(entries, list) or len(entries) != NUM_CLASSES:
        raise ValueError(f"{name} must contain exactly one entry per class")
    by_class = {}
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError(f"{name} entries must be objects")
        class_id = item.get("class")
        if isinstance(class_id, bool) or not isinstance(class_id, Integral) or class_id not in range(1, NUM_CLASSES + 1):
            raise ValueError(f"{name} contains an invalid class id")
        if class_id in by_class:
            raise ValueError(f"{name} contains duplicate class {class_id}")
        by_class[int(class_id)] = item
    if set(by_class) != set(range(1, NUM_CLASSES + 1)):
        raise ValueError(f"{name} must cover classes 1..10")
    return by_class


def flatten_legacy_recipe(recipe, model_order):
    """Expand the saved nested recipe into model-by-class weights.

    ``rad_dino_finetuned`` (original-view probabilities) and ``rad_dino_tta``
    (horizontal-flip TTA probabilities) are independent sources.
    """
    if not isinstance(recipe, dict):
        raise ValueError("recipe must be an object")
    if not isinstance(model_order, (list, tuple)) or not model_order:
        raise ValueError("model_order must be a non-empty list of source names")
    if any(not isinstance(name, str) or name not in _KNOWN_SOURCES for name in model_order):
        raise ValueError("model_order contains an unknown source")
    if len(set(model_order)) != len(model_order):
        raise ValueError("model_order must not contain duplicate sources")
    order = {name: index for index, name in enumerate(model_order)}
    baseline_entries = _entries_by_class(recipe.get("baseline_per_class_selection"), "baseline_per_class_selection")
    candidate_entries = _entries_by_class(recipe.get("classes"), "classes")
    weights = np.zeros((len(model_order), NUM_CLASSES), dtype=np.float64)

    def add(class_index, source, amount):
        if not isinstance(source, str) or source not in _KNOWN_SOURCES:
            raise ValueError(f"unknown recipe source: {source}")
        if source not in order:
            raise ValueError(f"model_order is missing required source: {source}")
        weights[order[source], class_index] += amount

    for class_id, item in baseline_entries.items():
        j = class_id - 1
        kind = item.get("kind")
        if kind == "full_model_blend":
            left, right = item.get("left"), item.get("right")
            alpha = _coefficient(item.get("alpha_left"), "alpha_left")
        elif kind == "external_blend":
            left, right = item.get("external_model"), item.get("base_model")
            alpha = _coefficient(item.get("alpha_external"), "alpha_external")
        elif kind == "single_model":
            left = right = item.get("model")
            alpha = 1.0
        else:
            raise ValueError(f"unsupported baseline formula: {kind}")
        add(j, left, alpha)
        add(j, right, 1.0 - alpha)

    for class_id, item in candidate_entries.items():
        j = class_id - 1
        alpha = _coefficient(item.get("alpha_candidate"), "alpha_candidate")
        source_name = item.get("source")
        source = _CANDIDATES.get(source_name)
        if source is None and source_name != "immutable_saved_ensemble":
            raise ValueError(f"unknown candidate source: {source_name}")
        if source_name == "immutable_saved_ensemble":
            if alpha != 0.0:
                raise ValueError("immutable_saved_ensemble can only be used with alpha_candidate=0")
            continue
        weights[:, j] *= 1.0 - alpha
        if alpha:
            add(j, source, alpha)

    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("flattened weights must be finite and nonnegative")
    if not np.allclose(weights.sum(axis=0), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("flattened recipe must sum to one for every class")
    return weights


def _validate_inputs(probabilities, targets, prior):
    try:
        probs = np.asarray(probabilities, dtype=np.float64)
        labels = np.asarray(targets, dtype=np.float64)
        q0 = np.asarray(prior, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("probabilities, targets, and prior must be numeric arrays") from exc
    if probs.ndim != 3 or probs.shape[0] == 0 or probs.shape[1] == 0 or probs.shape[2] != NUM_CLASSES:
        raise ValueError("probabilities must have shape (N, M, 10) with N and M positive")
    if labels.shape != (probs.shape[0], NUM_CLASSES):
        raise ValueError("targets must have shape (N, 10) matching probabilities")
    if q0.shape != (probs.shape[1], NUM_CLASSES):
        raise ValueError("prior must have shape (M, 10) matching probabilities")
    if not np.isfinite(labels).all() or ((labels != 0) & (labels != 1)).any():
        raise ValueError("targets must contain finite binary labels")
    if not np.isfinite(q0).all() or (q0 < 0).any():
        raise ValueError("prior must be finite and nonnegative")
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
    active = q0.sum(axis=0) > 0
    for label in range(NUM_CLASSES):
        total = float(q0[:, label].sum())
        if active[label] and not np.isclose(total, 1.0, rtol=0.0, atol=1e-10):
            raise ValueError(f"active prior for class {label + 1} must sum to one")
        if not active[label] and total != 0.0:
            raise ValueError(f"inactive prior for class {label + 1} must be all zero")
        if np.any((q0[:, label] > 0) & ~available[:, label]):
            raise ValueError(f"prior assigns weight to an unavailable head for class {label + 1}")
        if active[label] and (not np.any(labels[:, label] == 1) or not np.any(labels[:, label] == 0)):
            raise ValueError(f"active class {label + 1} must contain positives and negatives")
    return probs, labels, q0, available, active


def _pairwise_loss_gradient(weights, differences, temperature, l2, prior):
    margins = differences @ weights / temperature
    loss = np.logaddexp(0.0, -margins).mean()
    residual = -1.0 / (1.0 + np.exp(np.clip(margins, -700.0, 700.0)))
    gradient = differences.T @ residual / (len(differences) * temperature)
    delta = weights - prior
    return float(loss + 0.5 * l2 * (delta @ delta)), gradient + l2 * delta


def fit_anchored_public(
    probabilities,
    targets,
    prior,
    *,
    rho=0.2,
    l2=1.0,
    temperature=0.1,
    pair_count=32768,
    seed=20261107,
    max_iter=300,
    tolerance=1e-8,
):
    """Fit per-class public weights while retaining at least ``1-rho`` trust in prior."""
    probs, labels, q0, available, active = _validate_inputs(probabilities, targets, prior)
    rho = _coefficient(rho, "rho")
    for value, name in ((l2, "l2"), (temperature, "temperature"), (tolerance, "tolerance")):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
    l2, temperature, tolerance = float(l2), float(temperature), float(tolerance)
    if l2 < 0:
        raise ValueError("l2 must be nonnegative")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if isinstance(pair_count, bool) or not isinstance(pair_count, Integral) or pair_count <= 0:
        raise ValueError("pair_count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(max_iter, bool) or not isinstance(max_iter, Integral) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")

    weights = q0.copy()
    diagnostics = []
    rng = np.random.default_rng(int(seed))
    for label in range(NUM_CLASSES):
        if not active[label]:
            diagnostics.append({"class_id": label + 1, "active": False, "before": [], "after": [], "changes": [],
                                "objective_before": None, "objective_after": None, "iterations": 0, "converged": True})
            continue
        models = np.flatnonzero(available[:, label])
        positive = np.flatnonzero(labels[:, label] == 1)
        negative = np.flatnonzero(labels[:, label] == 0)
        total_pairs = len(positive) * len(negative)
        if total_pairs <= pair_count:
            pos_index = np.repeat(positive, len(negative))
            neg_index = np.tile(negative, len(positive))
        else:
            pos_index = rng.choice(positive, size=int(pair_count), replace=True)
            neg_index = rng.choice(negative, size=int(pair_count), replace=True)
        differences = probs[pos_index[:, None], models[None, :], label] - probs[neg_index[:, None], models[None, :], label]
        anchor = q0[models, label]
        variable = anchor.copy()

        def evaluate(value):
            q = (1.0 - rho) * anchor + rho * value
            objective, grad_q = _pairwise_loss_gradient(q, differences, temperature, l2, anchor)
            return objective, grad_q * rho

        objective_before, gradient = evaluate(variable)
        objective = objective_before
        converged = rho == 0.0
        iterations = 0
        step = 1.0
        while rho > 0.0 and iterations < max_iter:
            accepted = False
            trial_step = step
            for _ in range(60):
                candidate = _project_simplex(variable - trial_step * gradient)
                candidate_objective, candidate_gradient = evaluate(candidate)
                displacement = candidate - variable
                bound = objective + gradient @ displacement + (displacement @ displacement) / (2.0 * trial_step) + 1e-14
                if candidate_objective <= bound:
                    accepted = True
                    break
                trial_step *= 0.5
            if not accepted:
                break
            iterations += 1
            improvement = objective - candidate_objective
            variable, objective, gradient = candidate, candidate_objective, candidate_gradient
            step = min(trial_step * 1.5, 1e6)
            if np.linalg.norm(displacement) <= tolerance or improvement <= tolerance * max(1.0, abs(objective)):
                converged = True
                break
        after = (1.0 - rho) * anchor + rho * variable
        weights[models, label] = after
        diagnostics.append({
            "class_id": label + 1,
            "active": True,
            "available_models": [int(i) for i in models],
            "pair_count": int(len(pos_index)),
            "before": anchor.tolist(),
            "after": after.tolist(),
            "changes": (after - anchor).tolist(),
            "objective_before": float(objective_before),
            "objective_after": float(objective),
            "iterations": int(iterations),
            "converged": bool(converged),
        })
    return {
        "version": 1,
        "weights": weights.tolist(),
        "parameters": {"rho": rho, "l2": l2, "temperature": temperature, "pair_count": int(pair_count),
                       "seed": int(seed), "max_iter": int(max_iter), "tolerance": tolerance},
        "classes": diagnostics,
        "objective": "mean_pairwise_logistic_plus_anchored_l2",
        "objective_note": "The public simplex remains within rho of its normalized legacy prior.",
    }


def apply_public_weights(probabilities, weights):
    """Apply an (M, 10) weight matrix, ignoring unavailable zero-weight heads safely."""
    try:
        probs = np.asarray(probabilities, dtype=np.float64)
        matrix = np.asarray(weights, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("probabilities and weights must be numeric arrays") from exc
    if probs.ndim != 3 or probs.shape[0] == 0 or probs.shape[1] == 0 or probs.shape[2] != NUM_CLASSES:
        raise ValueError("probabilities must have shape (N, M, 10) with N and M positive")
    if matrix.shape != (probs.shape[1], NUM_CLASSES) or not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("weights must be finite, nonnegative, and shaped (M, 10)")
    available = np.empty_like(matrix, dtype=bool)
    for model in range(probs.shape[1]):
        for label in range(NUM_CLASSES):
            column = probs[:, model, label]
            missing = np.isnan(column)
            if missing.any() and not missing.all():
                raise ValueError("a model/class head must be fully available or all NaN")
            available[model, label] = not missing.all()
            if not missing.all() and (not np.isfinite(column).all() or ((column < 0) | (column > 1)).any()):
                raise ValueError("available probabilities must be finite and in [0, 1]")
    if ((matrix > 0) & ~available).any():
        raise ValueError("positive weights cannot reference unavailable probability heads")
    totals = matrix.sum(axis=0)
    active = totals > 0
    if not np.allclose(totals[active], 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("active class weights must sum to one")
    result = np.zeros((probs.shape[0], NUM_CLASSES), dtype=np.float64)
    for model in range(probs.shape[1]):
        for label in range(NUM_CLASSES):
            if matrix[model, label] > 0:
                result[:, label] += probs[:, model, label] * matrix[model, label]
    return result
