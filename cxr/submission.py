"""UTF-8 submission I/O with exact study/class coverage validation.

Class IDs are 1..10, in probability-column order. Subject and study IDs are
handled as strings without stripping or numeric conversion (e.g. ``001`` is
not ``1``). Callers should likewise load source identifiers as strings.
"""

import csv
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .metrics import NUM_CLASSES, _identifier, _matrix, _thresholds


COLUMNS = ("Subject_id", "Study_id", "Predict_class", "Probability", "Gate")
CLASS_IDS = tuple(str(value) for value in range(1, NUM_CLASSES + 1))


def _study_pairs(records, *, unique):
    pairs = []
    seen = set()
    study_subjects = {}
    for record in records:
        if not isinstance(record, Mapping) or not {"subject_id", "study_id"} <= record.keys():
            raise ValueError("records must include subject_id and study_id")
        subject = _identifier(record["subject_id"], "subject_id")
        study = _identifier(record["study_id"], "study_id")
        if study in study_subjects and study_subjects[study] != subject:
            raise ValueError(f"study {study!r} has conflicting subjects")
        study_subjects[study] = subject
        pair = (subject, study)
        if pair in seen:
            if unique:
                raise ValueError(f"duplicate study {study!r}; aggregate images before writing")
        else:
            seen.add(pair)
            pairs.append(pair)
    return pairs


def write_submission(path, study_records, probabilities, thresholds=0.5):
    """Write all ten classes per unique study; return the output Path.

    Preserves study input order and writes classes 1..10 in ascending order.
    ``thresholds`` is a scalar or ten-vector; Gate contains the class threshold
    itself, NOT a binary prediction. Probabilities must be a finite N x 10
    matrix in [0, 1]. Input
    validation finishes before opening the destination, so invalid inputs do
    not truncate an existing file. The destination's parent must exist.
    """
    records = list(study_records)
    pairs = _study_pairs(records, unique=True)
    probabilities = _matrix(probabilities, "probabilities", rows=len(pairs))
    thresholds = _thresholds(thresholds)
    destination = Path(path)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        for (subject, study), row in zip(pairs, probabilities):
            for index, probability in enumerate(row):
                writer.writerow((subject, study, index + 1, repr(float(probability)),
                                 repr(float(thresholds[index]))))
    return destination


def validate_submission(path, expected_records):
    """Validate exact columns, IDs, class coverage, uniqueness and values.

    ``expected_records`` may be image-level (repeated study IDs are deduplicated)
    or already aggregated. Every expected study must appear exactly ten times,
    once per class 1..10, and no unexpected subject/study pair may appear.
    Gate is a class threshold: blank means the default 0.5; otherwise it must
    parse to a finite float in [0, 1], as must Probability. Gate is not a binary
    prediction. File row order is not constrained. The JSON-safe summary counts
    rows with effective default threshold 0.5 (explicit or blank).
    """
    pairs = _study_pairs(expected_records, unique=False)
    expected_pairs = set(pairs)
    seen = set()
    default_threshold_rows = 0
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, None)
            if header != list(COLUMNS):
                raise ValueError(f"submission columns must be exactly {list(COLUMNS)!r}")
            for line, row in enumerate(reader, start=2):
                if len(row) != len(COLUMNS):
                    raise ValueError(f"line {line}: expected exactly five fields")
                subject, study, class_id, raw_probability, gate = row
                if (subject, study) not in expected_pairs:
                    raise ValueError(f"line {line}: unexpected subject/study IDs {(subject, study)!r}")
                if class_id not in CLASS_IDS:
                    raise ValueError(f"line {line}: Predict_class must be an integer ID from 1 through 10")
                key = (subject, study, class_id)
                if key in seen:
                    raise ValueError(f"line {line}: duplicate study/class row")
                try:
                    probability = float(raw_probability)
                except ValueError as exc:
                    raise ValueError(f"line {line}: invalid Probability") from exc
                if not np.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError(f"line {line}: Probability must be finite and in [0, 1]")
                try:
                    threshold = 0.5 if gate == "" else float(gate)
                except ValueError as exc:
                    raise ValueError(f"line {line}: invalid Gate threshold") from exc
                if not np.isfinite(threshold) or not 0 <= threshold <= 1:
                    raise ValueError(f"line {line}: Gate must be blank or a finite threshold in [0, 1]")
                seen.add(key)
                default_threshold_rows += int(threshold == 0.5)
    except (UnicodeError, csv.Error) as exc:
        raise ValueError(f"invalid UTF-8 submission CSV: {exc}") from exc
    expected_count = len(expected_pairs) * NUM_CLASSES
    if len(seen) != expected_count:
        raise ValueError(f"incomplete study/class coverage: expected {expected_count} rows, found {len(seen)}")
    return {
        "valid": True,
        "studies": len(pairs),
        "rows": len(seen),
        "classes_per_study": NUM_CLASSES,
        "default_threshold_rows": default_threshold_rows,
    }
