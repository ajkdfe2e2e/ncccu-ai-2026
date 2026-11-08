"""Deterministic shuffled, patient-disjoint partitions for labeled CXR records.

Patients are ranked once with a seeded stable hash and placed into exact-size
audit and fold buckets. Labels are used only for support validation, never to
select or optimize the split.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence


LABEL_COUNT = 10
DEFAULT_SEED = 20261105


def _collect(records: Sequence[Mapping]) -> dict:
    if not records:
        raise ValueError("records must contain at least one labeled image")

    seen_images: set[str] = set()
    studies: dict[str, dict] = {}
    patients: dict[str, dict] = {}
    for row_number, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"record {row_number} must be a mapping")
        required = ("subject_id", "study_id", "dicom_id", "labels")
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(f"record {row_number} is missing fields: {missing}")
        subject = record["subject_id"]
        study = record["study_id"]
        dicom = record["dicom_id"]
        if not isinstance(subject, str) or not subject:
            raise ValueError(f"record {row_number} has invalid subject_id")
        if not isinstance(study, str) or not study:
            raise ValueError(f"record {row_number} has invalid study_id")
        if not isinstance(dicom, str) or not dicom:
            raise ValueError(f"record {row_number} has invalid dicom_id")
        if dicom in seen_images:
            raise ValueError(f"duplicate image ID {dicom!r}")
        seen_images.add(dicom)

        labels = record["labels"]
        if labels is None or isinstance(labels, (str, bytes)):
            raise ValueError(f"record {row_number} must have ten binary labels")
        try:
            values = tuple(labels)
        except TypeError as exc:
            raise ValueError(f"record {row_number} must have ten binary labels") from exc
        if len(values) != LABEL_COUNT or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value not in (0, 1)
            for value in values
        ):
            raise ValueError(f"record {row_number} must have ten binary labels")
        labels_tuple = tuple(int(value) for value in values)

        if study in studies:
            current = studies[study]
            if current["subject_id"] != subject:
                raise ValueError(f"study {study!r} has ambiguous patient ownership")
            if current["labels"] != labels_tuple:
                raise ValueError(f"study {study!r} has inconsistent labels")
            current["images"] += 1
        else:
            studies[study] = {
                "subject_id": subject,
                "labels": labels_tuple,
                "images": 1,
            }

        patient = patients.setdefault(subject, {"studies": set()})
        patient["studies"].add(study)

    # Recompute patient totals by unique study, avoiding image-count weighting.
    for patient in patients.values():
        patient["positive_studies"] = [0] * LABEL_COUNT
        patient["study_count"] = len(patient["studies"])
        patient["image_count"] = sum(studies[study]["images"] for study in patient["studies"])
        for study in patient["studies"]:
            for index, value in enumerate(studies[study]["labels"]):
                patient["positive_studies"][index] += value

    return {"studies": studies, "patients": patients}


def _stable_rank(subject: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{subject}".encode("utf-8")).digest()


def _partition_names(folds: int) -> list[str | int]:
    return ["audit", *range(folds)]


def _capacity_sizes(patient_count: int, audit_fraction: float, folds: int) -> list[int]:
    audit_count = int(math.floor(patient_count * audit_fraction + 0.5))
    development_count = patient_count - audit_count
    if audit_count < 1 or development_count < folds:
        raise ValueError("patient count is too small for a non-empty audit set and folds")
    capacities = [audit_count]
    base, remainder = divmod(development_count, folds)
    capacities.extend(base + (index < remainder) for index in range(folds))
    return capacities


def _shuffle_assignment(meta: dict, seed: int, audit_fraction: float, folds: int) -> dict[str, str | int]:
    patients = meta["patients"]
    studies = meta["studies"]
    names = _partition_names(folds)
    capacities = _capacity_sizes(len(patients), audit_fraction, folds)
    label_patients = [0] * LABEL_COUNT
    for patient in patients.values():
        for index, count in enumerate(patient["positive_studies"]):
            label_patients[index] += count > 0

    partition_count = folds + 1
    for index in range(LABEL_COUNT):
        negative_patients = sum(
            any(studies[study]["labels"][index] == 0 for study in patient["studies"])
            for patient in patients.values()
        )
        if label_patients[index] < partition_count or negative_patients < partition_count:
            raise ValueError(
                f"insufficient positive or negative patients for label {index + 1}: "
                f"positive={label_patients[index]}, negative={negative_patients}, "
                f"partitions={partition_count}"
            )

    order = sorted(patients, key=lambda subject: (_stable_rank(subject, seed), subject))
    result: dict[str, str | int] = {}
    offset = 0
    for name, capacity in zip(names, capacities):
        for subject in order[offset:offset + capacity]:
            result[subject] = name
        offset += capacity

    return result


def _summary(records: Sequence[Mapping], meta: dict, assignment: Mapping[str, str | int], folds: int) -> dict:
    names = _partition_names(folds)
    summary = {}
    for name in names:
        subjects = {subject for subject, value in assignment.items() if value == name}
        study_ids = {record["study_id"] for record in records if record["subject_id"] in subjects}
        positive_counts = [0] * LABEL_COUNT
        for study_id in study_ids:
            for index, value in enumerate(meta["studies"][study_id]["labels"]):
                positive_counts[index] += value
        positive_patients = [0] * LABEL_COUNT
        for subject in subjects:
            for index, count in enumerate(meta["patients"][subject]["positive_studies"]):
                positive_patients[index] += count > 0
        summary[str(name)] = {
            "images": sum(1 for record in records if record["subject_id"] in subjects),
            "studies": len(study_ids),
            "patients": len(subjects),
            "study_positive_counts": {str(index + 1): count for index, count in enumerate(positive_counts)},
            "positive_patient_counts": {str(index + 1): count for index, count in enumerate(positive_patients)},
        }
    return summary


def _validate_partition(records: Sequence[Mapping], partition: Mapping, folds: int) -> dict:
    meta = _collect(records)
    if not isinstance(partition, Mapping) or "subject_partition" not in partition:
        raise ValueError("partition must contain a subject_partition mapping")
    assignment = partition["subject_partition"]
    if not isinstance(assignment, Mapping):
        raise ValueError("subject_partition must be a mapping")
    expected = set(meta["patients"])
    actual = set(assignment)
    if actual != expected:
        missing, extra = sorted(expected - actual), sorted(actual - expected)
        raise ValueError(f"subject_partition must cover records exactly (missing={missing}, extra={extra})")
    allowed = {"audit", *range(folds)}
    invalid = [
        value for value in assignment.values()
        if isinstance(value, bool) or not isinstance(value, (str, int)) or value not in allowed
    ]
    if invalid:
        raise ValueError(f"invalid partition values: {sorted(invalid, key=str)}")
    return meta


def _check_class_support(summary: Mapping, folds: int) -> None:
    for name in _partition_names(folds):
        counts = summary[str(name)]["study_positive_counts"]
        study_count = summary[str(name)]["studies"]
        unsupported = [
            str(label)
            for label, positive in counts.items()
            if positive == 0 or positive == study_count
        ]
        if unsupported:
            raise ValueError(
                f"partition {name!r} lacks positive or negative study support for labels {unsupported}"
            )


def make_patient_partition(
    records: Sequence[Mapping],
    seed: int = DEFAULT_SEED,
    audit_fraction: float = 0.2,
    folds: int = 3,
) -> dict:
    """Assign patients to one audit set and ``folds`` development folds.

    The only inputs to stratification are training-record labels and patient,
    study, and image counts. Input rows are never modified.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError("folds must be an integer of at least 2")
    if isinstance(audit_fraction, bool) or not isinstance(audit_fraction, (int, float)):
        raise ValueError("audit_fraction must be between 0 and 1")
    if not math.isfinite(audit_fraction) or not 0 < audit_fraction < 1:
        raise ValueError("audit_fraction must be between 0 and 1")
    meta = _collect(records)
    assignment = _shuffle_assignment(meta, seed, float(audit_fraction), folds)
    summary = _summary(records, meta, assignment, folds)
    _check_class_support(summary, folds)
    return {
        "subject_partition": assignment,
        "summary": summary,
        "seed": seed,
        "folds": folds,
        "audit_fraction": float(audit_fraction),
    }


def _check_requested_fold(fold: int, folds: int) -> None:
    if isinstance(fold, bool) or not isinstance(fold, int) or not 0 <= fold < folds:
        raise ValueError(f"fold must be an integer from 0 to {folds - 1}")


def partition_records(records: Sequence[Mapping], partition: Mapping, fold: int) -> tuple[list[Mapping], list[Mapping]]:
    """Return development and evaluation rows for one CV fold; audit excluded."""
    folds = partition.get("folds") if isinstance(partition, Mapping) else None
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError("partition must specify a valid folds count")
    _check_requested_fold(fold, folds)
    assignment = partition["subject_partition"]
    _validate_partition(records, partition, folds)
    train_records = [record for record in records if assignment[record["subject_id"]] not in {"audit", fold}]
    eval_records = [record for record in records if assignment[record["subject_id"]] == fold]
    if not train_records or not eval_records:
        raise ValueError(f"fold {fold} has an empty development or evaluation partition")
    return train_records, eval_records


def audit_records(records: Sequence[Mapping], partition: Mapping) -> tuple[list[Mapping], list[Mapping]]:
    """Return development rows and the held-out audit rows."""
    folds = partition.get("folds") if isinstance(partition, Mapping) else None
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError("partition must specify a valid folds count")
    _validate_partition(records, partition, folds)
    assignment = partition["subject_partition"]
    development = [record for record in records if assignment[record["subject_id"]] != "audit"]
    audit = [record for record in records if assignment[record["subject_id"]] == "audit"]
    if not development or not audit:
        raise ValueError("audit and development partitions must both be non-empty")
    return development, audit
