"""Prediction equivalence with per-class ordering across studies, not classes."""

import numpy as np

from .metrics import NUM_CLASSES, _matrix, _thresholds


def _ranking_changes(actual, expected):
    """Mark both endpoints of every changed pairwise sign/tie relation.

    For each class, sort studies by expected score and group exact ties. A row
    changes a relation if its actual score is <= an earlier group's maximum,
    >= a later group's minimum, or differs from a member of its own tie group.
    These conditions exactly compare all cross-study pairwise signs, using
    O(N log N) time and O(N) auxiliary memory instead of an N x N matrix.
    """
    changed = np.zeros(len(actual), dtype=bool)
    if len(actual) < 2:
        return changed
    for column in range(NUM_CLASSES):
        order = np.argsort(expected[:, column], kind="mergesort")
        reference = expected[order, column]
        scores = actual[order, column]
        starts = np.r_[0, np.flatnonzero(reference[1:] != reference[:-1]) + 1]
        ends = np.r_[starts[1:], len(scores)]
        lengths = ends - starts
        minima = np.minimum.reduceat(scores, starts)
        maxima = np.maximum.reduceat(scores, starts)
        prior_max = np.r_[-np.inf, np.maximum.accumulate(maxima)[:-1]]
        later_min = np.r_[np.minimum.accumulate(minima[::-1])[::-1][1:], np.inf]
        broken = (scores <= np.repeat(prior_max, lengths))
        broken |= scores >= np.repeat(later_min, lengths)
        broken |= np.repeat(minima != maxima, lengths)
        changed[order] |= broken
    return changed


def compare_predictions(actual, expected, actual_thresholds=0.5,
                        expected_thresholds=0.5, tolerance=0.001):
    """Compare aligned N x 10 prediction matrices and return JSON-safe counts.

    Inputs must be finite probabilities in [0, 1]; each threshold argument may
    be a scalar or length-ten vector in [0, 1]. Decisions use score >= threshold.
    Absolute error exactly equal to tolerance is allowed. Ranking comparison
    is exact (no tolerance): for every class, every pair of studies must retain
    its ordering AND tie status. Disease classes within a row are never ranked.

    ``inconsistent_studies`` counts the union of rows with excessive error,
    different decisions, or any changed ranking relation (both pair endpoints).
    ``max_probability_error`` is the largest absolute error, or 0 for empty
    input. ``ranking_changed`` indicates any changed relation. Row alignment
    and validation/test provenance are the caller's responsibility.
    """
    actual = _matrix(actual, "actual")
    expected = _matrix(expected, "expected", rows=len(actual))
    actual_thresholds = _thresholds(actual_thresholds)
    expected_thresholds = _thresholds(expected_thresholds)
    try:
        tolerance_array = np.asarray(tolerance, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("tolerance must be a finite nonnegative scalar") from exc
    if tolerance_array.ndim != 0 or not np.isfinite(tolerance_array) or tolerance_array < 0:
        raise ValueError("tolerance must be a finite nonnegative scalar")
    tolerance = float(tolerance_array)
    errors = np.abs(actual - expected)
    inconsistent = np.any(errors > tolerance, axis=1)
    inconsistent |= np.any((actual >= actual_thresholds) != (expected >= expected_thresholds), axis=1)
    ranking_rows = _ranking_changes(actual, expected)
    inconsistent |= ranking_rows
    return {
        "inconsistent_studies": int(np.count_nonzero(inconsistent)),
        "max_probability_error": float(errors.max()) if errors.size else 0.0,
        "ranking_changed": bool(np.any(ranking_rows)),
    }
