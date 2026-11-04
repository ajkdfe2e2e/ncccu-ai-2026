"""NumPy-only multilabel study aggregation and validation-set metrics.

Class IDs are 1 through 10 (array columns 0 through 9). Predictions use ``probability >=
threshold``. AUC excludes classes without both outcomes from its macro mean;
F1 includes all ten classes and is zero when its denominator is zero.
"""

from collections.abc import Mapping

import numpy as np


NUM_CLASSES = 10


def _matrix(values, name, *, rows=None, binary=False):
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric N x {NUM_CLASSES} matrix") from exc
    if result.ndim != 2 or result.shape[1] != NUM_CLASSES:
        raise ValueError(f"{name} must have shape (N, {NUM_CLASSES})")
    if rows is not None and result.shape[0] != rows:
        raise ValueError(f"{name} row count must be {rows}")
    if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        raise ValueError(f"{name} must contain finite values in [0, 1]")
    if binary and ((result != 0) & (result != 1)).any():
        raise ValueError(f"{name} must contain binary labels (0 or 1)")
    return result


def _thresholds(values):
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("thresholds must be a scalar or ten numeric values") from exc
    if result.ndim == 0:
        result = np.full(NUM_CLASSES, float(result), dtype=np.float64)
    if result.shape != (NUM_CLASSES,):
        raise ValueError("thresholds must be a scalar or have shape (10,)")
    if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        raise ValueError("thresholds must contain finite values in [0, 1]")
    return result


def _identifier(value, name):
    if value is None:
        raise ValueError(f"{name} must not be missing")
    result = str(value)
    if not result.strip():
        raise ValueError(f"{name} must not be empty")
    return result


def aggregate_studies(records, probabilities):
    """Average image probabilities per study in first-seen input order.

    ``records`` contains mappings with dicom_id, subject_id, study_id, path,
    and labels (ten finite values in [0, 1], or None). The first record of
    each study is copied into the output without mutating the inputs. Its
    image-specific fields therefore describe the first image, not all images.
    A study ID cannot belong to multiple subjects or have inconsistent labels.
    Mixing labeled and unlabeled records is rejected, even across studies.
    Empty input returns [], an empty (0, 10) matrix, and None.
    """
    records = list(records)
    probabilities = _matrix(probabilities, "probabilities", rows=len(records))
    studies = []
    sums = []
    counts = []
    labels_by_study = []
    positions = {}
    label_status = set()
    for record, probability in zip(records, probabilities):
        if not isinstance(record, Mapping):
            raise ValueError("records must contain mappings")
        if not {"dicom_id", "subject_id", "study_id", "path", "labels"} <= record.keys():
            raise ValueError("record is missing one or more required fields")
        study_id = _identifier(record["study_id"], "study_id")
        subject_id = _identifier(record["subject_id"], "subject_id")
        raw_labels = record["labels"]
        labels = None if raw_labels is None else _matrix([raw_labels], "labels")[0].copy()
        label_status.add(labels is not None)
        if study_id in positions:
            index = positions[study_id]
            if _identifier(studies[index]["subject_id"], "subject_id") != subject_id:
                raise ValueError(f"study {study_id!r} has conflicting subjects")
            prior = labels_by_study[index]
            if (prior is None) != (labels is None) or (
                prior is not None and not np.array_equal(prior, labels)
            ):
                raise ValueError(f"study {study_id!r} has conflicting labels")
            sums[index] += probability
            counts[index] += 1
        else:
            positions[study_id] = len(studies)
            studies.append(dict(record))
            sums.append(probability.copy())
            counts.append(1)
            labels_by_study.append(labels)
    if len(label_status) > 1:
        raise ValueError("cannot mix labeled and unlabeled studies")
    if not studies:
        return [], np.empty((0, NUM_CLASSES), dtype=np.float64), None
    averaged = np.asarray(sums) / np.asarray(counts)[:, None]
    targets = np.asarray(labels_by_study, dtype=np.float64) if True in label_status else None
    return studies, averaged, targets


def _rank_auc(target, probability):
    """Mann-Whitney AUC using average ranks for exactly tied scores."""
    positives = int(np.sum(target))
    negatives = len(target) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(probability, kind="mergesort")
    sorted_scores = probability[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    ends = np.r_[starts[1:], len(target)]
    average_ranks = (starts + ends + 1) / 2.0
    ranks = np.empty(len(target), dtype=np.float64)
    ranks[order] = np.repeat(average_ranks, ends - starts)
    positive_rank_sum = ranks[target == 1].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def _f1(target, predicted):
    true_positives = int(np.count_nonzero((target == 1) & predicted))
    denominator = int(np.count_nonzero(target == 1) + np.count_nonzero(predicted))
    return 2.0 * true_positives / denominator if denominator else 0.0


def evaluate(targets, probabilities, thresholds=0.5):
    """Return JSON-safe macro AUC/F1 and per-class metrics for binary targets.

    ``macro_auc_defined_classes`` reports how many of the ten classes have a
    defined AUC. ``macro_auc`` is None if that count is zero. All returned
    numbers are native Python types; no NaN values are emitted.
    """
    targets = _matrix(targets, "targets", binary=True)
    probabilities = _matrix(probabilities, "probabilities", rows=len(targets))
    thresholds = _thresholds(thresholds)
    predicted = probabilities >= thresholds
    per_class = []
    for class_id in range(NUM_CLASSES):
        target = targets[:, class_id]
        per_class.append({
            "id": class_id + 1,
            "auc": _rank_auc(target, probabilities[:, class_id]),
            "f1": _f1(target, predicted[:, class_id]),
            "positives": int(target.sum()),
        })
    defined_auc = [item["auc"] for item in per_class if item["auc"] is not None]
    return {
        "macro_auc": float(np.mean(defined_auc)) if defined_auc else None,
        "macro_auc_defined_classes": len(defined_auc),
        "macro_f1": float(np.mean([item["f1"] for item in per_class])),
        "per_class": per_class,
    }


def select_thresholds(targets, probs, grid=None):
    """Fit ten thresholds on VALIDATION data only by maximizing macro F1.

    Each class is optimized independently, which maximizes their macro mean.
    The default grid is 0.00, 0.01, ..., 1.00. Ties favor the candidate nearest
    0.5, then the smaller candidate. This routine cannot establish data split
    provenance: callers must never supply held-out test labels or tune on test
    predictions. Returns a float64 array of shape (10,).
    """
    targets = _matrix(targets, "targets", binary=True)
    probabilities = _matrix(probs, "probabilities", rows=len(targets))
    if not len(targets):
        raise ValueError("threshold calibration requires nonempty validation data")
    try:
        candidates = np.asarray(np.linspace(0, 1, 101) if grid is None else grid, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("grid must be a nonempty vector of numeric thresholds") from exc
    if candidates.ndim != 1 or not candidates.size:
        raise ValueError("grid must be a nonempty vector")
    if not np.isfinite(candidates).all() or ((candidates < 0) | (candidates > 1)).any():
        raise ValueError("grid must contain finite values in [0, 1]")
    candidates = sorted(set(candidates.tolist()), key=lambda value: (abs(value - 0.5), value))
    selected = np.empty(NUM_CLASSES, dtype=np.float64)
    for class_id in range(NUM_CLASSES):
        scores = [_f1(targets[:, class_id], probabilities[:, class_id] >= value) for value in candidates]
        selected[class_id] = candidates[int(np.argmax(scores))]
    return selected
