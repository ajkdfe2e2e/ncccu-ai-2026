"""Fixed, within-split patient probability smoothing."""
from __future__ import annotations

import numpy as np


ALPHA = 0.25
NUM_CLASSES = 10


def _probabilities(values):
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("probabilities must be numeric") from exc
    if result.ndim != 2 or result.shape[1] != NUM_CLASSES:
        raise ValueError("probabilities must have shape (N, 10)")
    if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        raise ValueError("probabilities must be finite and in [0, 1]")
    return result


def smooth_patient_probabilities(probabilities, subjects, studies, *, alpha=ALPHA):
    """Apply the registered patient mean independently within one split."""
    if alpha != ALPHA:
        raise ValueError(f"alpha is fixed at {ALPHA}")
    p = _probabilities(probabilities)
    subjects, studies = list(subjects), list(studies)
    if len(subjects) != len(p) or len(studies) != len(p):
        raise ValueError("subject and study IDs must match probability rows")
    seen_pairs, study_owners = set(), {}
    groups = {}
    for i, (subject, study) in enumerate(zip(subjects, studies)):
        if not isinstance(subject, str) or not subject or subject != subject.strip():
            raise ValueError("subject IDs must be nonempty strings")
        if not isinstance(study, str) or not study or study != study.strip():
            raise ValueError("study IDs must be nonempty strings")
        pair = subject, study
        if pair in seen_pairs:
            raise ValueError(f"duplicate subject/study pair: {pair!r}")
        if study in study_owners and study_owners[study] != subject:
            raise ValueError(f"study ID has conflicting subjects: {study!r}")
        seen_pairs.add(pair)
        study_owners[study] = subject
        groups.setdefault(subject, []).append(i)
    output = p.copy()
    for indices in groups.values():
        if len(indices) > 1:
            mean = p[indices].mean(axis=0)
            output[indices] = (1.0 - ALPHA) * p[indices] + ALPHA * mean
    return output
