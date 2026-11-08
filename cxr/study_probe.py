"""Study-level feature pooling and a small frozen-feature linear probe."""

from collections.abc import Mapping

import numpy as np
import torch
from torch import nn


NUM_CLASSES = 10


def _matrix(values, name, *, cols=None, rows=None):
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric matrix") from exc
    if array.ndim != 2 or (cols is not None and array.shape[1] != cols):
        shape = f"(N, {cols})" if cols is not None else "two-dimensional"
        raise ValueError(f"{name} must have shape {shape}")
    if rows is not None and array.shape[0] != rows:
        raise ValueError(f"{name} row count must be {rows}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _id(value, name):
    if value is None or not str(value).strip():
        raise ValueError(f"{name} must not be missing or empty")
    return str(value)


def _view_group(value):
    view = "" if value is None else str(value).strip().upper()
    if view in {"AP", "PA", "PA LLD"}:
        return 0
    if view in {"LATERAL", "LL"}:
        return 1
    return 2


def pool_study_features(records, image_features, views, mode):
    """Pool row-aligned image features by study, preserving first-seen order.

    ``mean`` averages all image vectors. ``views`` concatenates means for AP/PA,
    lateral, and other/empty views, then appends three group-availability flags.
    """
    records = list(records)
    features = _matrix(image_features, "image_features", rows=len(records))
    if features.shape[1] == 0:
        raise ValueError("image_features must contain at least one feature")
    views = list(views)
    if len(views) != len(records):
        raise ValueError("views row count must match records")
    if mode not in {"mean", "views"}:
        raise ValueError("mode must be 'mean' or 'views'")

    studies, positions, sums, counts, labels_by_study = [], {}, [], [], []
    seen_dicoms = set()
    label_status = set()
    group_sums, group_counts = [], []
    for row, (record, vector, view) in enumerate(zip(records, features, views)):
        if not isinstance(record, Mapping):
            raise ValueError("records must contain mappings")
        if not {"subject_id", "study_id", "dicom_id", "labels"} <= record.keys():
            raise ValueError("record is missing one or more required fields")
        dicom = _id(record["dicom_id"], "dicom_id")
        if dicom in seen_dicoms:
            raise ValueError(f"duplicate dicom_id {dicom!r}")
        seen_dicoms.add(dicom)
        subject, study = _id(record["subject_id"], "subject_id"), _id(record["study_id"], "study_id")
        raw_labels = record["labels"]
        labels = None if raw_labels is None else _matrix([raw_labels], "labels", cols=NUM_CLASSES)[0].copy()
        if labels is not None and ((labels < 0) | (labels > 1)).any():
            raise ValueError("labels must contain values in [0, 1]")
        label_status.add(labels is not None)

        if study in positions:
            index = positions[study]
            if studies[index]["subject_id"] != subject:
                raise ValueError(f"study {study!r} has conflicting subjects")
            prior = labels_by_study[index]
            if (prior is None) != (labels is None) or (prior is not None and not np.array_equal(prior, labels)):
                raise ValueError(f"study {study!r} has conflicting labels")
        else:
            index = len(studies)
            positions[study] = index
            studies.append(dict(record))
            sums.append(np.zeros(features.shape[1], dtype=np.float64))
            counts.append(0)
            labels_by_study.append(labels)
            group_sums.append(np.zeros((3, features.shape[1]), dtype=np.float64))
            group_counts.append(np.zeros(3, dtype=np.int64))
        sums[index] += vector
        counts[index] += 1
        group = _view_group(view)
        group_sums[index][group] += vector
        group_counts[index][group] += 1

    if len(label_status) > 1:
        raise ValueError("cannot mix labeled and unlabeled records")
    if not studies:
        width = features.shape[1] if mode == "mean" else features.shape[1] * 3 + 3
        return [], np.empty((0, width), dtype=np.float64), None
    if mode == "mean":
        pooled = np.asarray([total / count for total, count in zip(sums, counts)])
    else:
        rows = []
        for total, count in zip(group_sums, group_counts):
            means = np.zeros_like(total)
            available = count > 0
            means[available] = total[available] / count[available, None]
            rows.append(np.concatenate((means.ravel(), available.astype(np.float64))))
        pooled = np.asarray(rows)
    targets = np.asarray(labels_by_study, dtype=np.float64) if True in label_status else None
    return studies, pooled, targets


def fit_linear_probe(features, targets, *, l2=0.01, max_iter=200, seed=20261108,
                     class_balance=False):
    """Fit ten independent BCE logits with train-fold-only standardization."""
    x = _matrix(features, "features")
    y = _matrix(targets, "targets", cols=NUM_CLASSES, rows=len(x))
    if not len(x) or x.shape[1] == 0:
        raise ValueError("training features must be nonempty")
    if ((y != 0) & (y != 1)).any():
        raise ValueError("targets must contain binary labels")
    if not np.isfinite(l2) or l2 < 0:
        raise ValueError("l2 must be a finite nonnegative value")
    if not isinstance(max_iter, (int, np.integer)) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    if not isinstance(class_balance, (bool, np.bool_)):
        raise ValueError("class_balance must be a boolean")
    class_balance = bool(class_balance)
    if np.any(y.sum(axis=0) == 0) or np.any(y.sum(axis=0) == len(y)):
        raise ValueError("each class must contain both positive and negative training samples")

    if class_balance:
        positives = y.sum(axis=0)
        negatives = len(y) - positives
        positive_weight = np.clip(np.sqrt(negatives / positives), 1.0, 5.0)
        class_weight_normalizer = (negatives + positive_weight * positives) / len(y)

    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std == 0] = 1.0
    standardized = (x - mean) / std
    torch.manual_seed(int(seed))
    tx = torch.as_tensor(standardized, dtype=torch.float64, device="cpu")
    ty = torch.as_tensor(y, dtype=torch.float64, device="cpu")
    model = nn.Linear(x.shape[1], NUM_CLASSES, dtype=torch.float64, device="cpu")
    with torch.no_grad():
        model.weight.zero_()
        freq = ty.mean(dim=0).clamp(1e-12, 1 - 1e-12)
        model.bias.copy_(torch.log(freq / (1 - freq)))
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=int(max_iter),
                                  line_search_fn="strong_wolfe")
    calls = 0

    def closure():
        nonlocal calls
        calls += 1
        optimizer.zero_grad()
        logits = model(tx)
        if class_balance:
            per_class = nn.functional.binary_cross_entropy_with_logits(
                logits, ty, reduction="none",
                pos_weight=torch.as_tensor(positive_weight, dtype=torch.float64),
            ).mean(dim=0)
            loss = (per_class / torch.as_tensor(class_weight_normalizer, dtype=torch.float64)).mean()
        else:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, ty, reduction="mean")
        loss = loss + 0.5 * float(l2) * model.weight.square().sum() / NUM_CLASSES
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        final_logits = model(tx)
        if class_balance:
            final_per_class = nn.functional.binary_cross_entropy_with_logits(
                final_logits, ty, reduction="none",
                pos_weight=torch.as_tensor(positive_weight, dtype=torch.float64),
            ).mean(dim=0)
            final_loss = (final_per_class / torch.as_tensor(class_weight_normalizer, dtype=torch.float64)).mean()
        else:
            final_loss = nn.functional.binary_cross_entropy_with_logits(final_logits, ty, reduction="mean")
        final_loss += 0.5 * float(l2) * model.weight.square().sum() / NUM_CLASSES
        weight = model.weight.cpu().numpy().copy()
        bias = model.bias.cpu().numpy().copy()
    payload = {
        "mean": mean.copy(), "std": std.copy(), "weight": weight, "bias": bias,
        "l2": float(l2), "iterations": int(optimizer.state[model.weight].get("n_iter", 0)),
        "closure_calls": int(calls), "loss": float(final_loss.item()), "seed": int(seed),
    }
    if class_balance:
        payload.update(
            class_balance=True,
            positive_weight=positive_weight.copy(),
            class_weight_normalizer=class_weight_normalizer.copy(),
        )
    return payload


def predict_linear_probe(payload, features):
    """Return sigmoid probabilities from a fitted probe payload."""
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a mapping")
    try:
        mean = np.asarray(payload["mean"], dtype=np.float64)
        std = np.asarray(payload["std"], dtype=np.float64)
        weight = np.asarray(payload["weight"], dtype=np.float64)
        bias = np.asarray(payload["bias"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("payload is missing valid model arrays") from exc
    if mean.ndim != 1 or std.shape != mean.shape or weight.shape != (NUM_CLASSES, len(mean)) or bias.shape != (NUM_CLASSES,):
        raise ValueError("payload arrays have incompatible shapes")
    if not all(np.isfinite(array).all() for array in (mean, std, weight, bias)) or np.any(std <= 0):
        raise ValueError("payload arrays must be finite and standard deviations positive")
    x = _matrix(features, "features", cols=len(mean))
    logits = ((x - mean) / std) @ weight.T + bias
    positive = logits >= 0
    probabilities = np.empty_like(logits)
    probabilities[positive] = 1 / (1 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    probabilities[~positive] = exp_logits / (1 + exp_logits)
    return probabilities
