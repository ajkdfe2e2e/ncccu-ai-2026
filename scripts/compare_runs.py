"""Descriptive paired comparison of validation submissions (NumPy + stdlib only).

Only validation CSV metadata is read: no images, checkpoints or test data.
Example: python scripts/compare_runs.py --baseline runs/base/valid.csv
    --candidate runs/new/valid.csv --metadata cxr_first/valid.csv
    --output runs/comparison.json --bootstrap 500
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cxr.data import _read_metadata
from cxr.metrics import NUM_CLASSES, _identifier, _matrix, _rank_auc
from cxr.safety import safe_output
from cxr.submission import validate_submission

METRICS = ("auc", "f1_fixed_0_5", "f1_declared_gate")
CAVEAT = (
    "Descriptive validation-selected comparison, not unbiased significance or "
    "generalization evidence. Models and Gate thresholds may have been selected on "
    "these validation labels. Paired patient-cluster percentile intervals do not "
    "correct selection bias. Declared-Gate F1 changes can be threshold-only; compare "
    "fixed-0.5 F1 and threshold-independent AUC before attributing gains to probabilities."
)


def load_validation_metadata(path):
    """Deduplicate image metadata into first-seen labeled studies without images."""
    if Path(path).name.casefold() != "valid.csv":
        raise ValueError("--metadata must be the official validation manifest named valid.csv")
    records = _read_metadata(path)
    studies = {}
    for record in records:
        studies.setdefault((record["subject_id"], record["study_id"]), record)
    if not studies:
        raise ValueError("validation metadata must contain at least one study")
    studies = list(studies.values())
    targets = _matrix([r["labels"] for r in studies], "targets", binary=True)
    return studies, targets


def load_submission(path, studies):
    """Strict validation precedes reconstruction; Gate is read on EVERY row."""
    validation = validate_submission(path, studies)
    positions = {(r["subject_id"], r["study_id"]): i for i, r in enumerate(studies)}
    probabilities = np.empty((len(studies), NUM_CLASSES), dtype=np.float64)
    gates = np.empty_like(probabilities)
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            i = positions[(row["Subject_id"], row["Study_id"])]
            j = int(row["Predict_class"]) - 1
            probabilities[i, j] = float(row["Probability"])
            gates[i, j] = 0.5 if row["Gate"] == "" else float(row["Gate"])
    return probabilities, gates, validation


def _scores(targets, probabilities, gates):
    """Rows are metrics, columns are classes; undefined AUC is internal NaN."""
    auc = [_rank_auc(targets[:, j], probabilities[:, j]) for j in range(NUM_CLASSES)]
    values = [np.array([np.nan if x is None else x for x in auc])]
    for threshold in (0.5, gates):
        predicted = probabilities >= threshold
        numerator = 2 * np.count_nonzero((targets == 1) & predicted, axis=0)
        denominator = np.sum(targets, axis=0) + np.count_nonzero(predicted, axis=0)
        values.append(np.divide(numerator, denominator, out=np.zeros(NUM_CLASSES),
                                where=denominator != 0))
    return np.asarray(values)


def _macro(values):
    defined = np.isfinite(values)
    count = defined.sum(axis=-1)
    return np.divide(np.where(defined, values, 0).sum(axis=-1), count,
                     out=np.full(count.shape, np.nan), where=count != 0)


def _number(value):
    return float(value) if np.isfinite(value) else None


def _comparison(baseline, candidate):
    return {"baseline": _number(baseline), "candidate": _number(candidate),
            "delta": _number(candidate - baseline)}


def _interval(values):
    defined = values[np.isfinite(values)]
    return {
        "ci95_percentile": np.percentile(defined, [2.5, 97.5]).tolist() if len(defined) else None,
        "defined_replicates": int(len(defined)),
        "missing_replicates": int(len(values) - len(defined)),
    }


def _patient_groups(subject_ids):
    groups = {}
    for i, subject in enumerate(subject_ids):
        groups.setdefault(_identifier(subject, "subject_id"), []).append(i)
    return [np.asarray(indices, dtype=np.int64) for indices in groups.values()]


def _sample_patient_indices(groups, rng):
    """Sample P patients; duplicate a selected patient's entire study cluster."""
    return np.concatenate([groups[i] for i in rng.integers(0, len(groups), size=len(groups))])


def compare_arrays(subject_ids, targets, baseline, candidate, baseline_gates=None,
                   candidate_gates=None, *, bootstrap=500, seed=2026):
    """Compare aligned N x 10 arrays and bootstrap paired patient clusters.

    AUC macro excludes one-outcome classes in each replicate for BOTH runs;
    this shared defined subset can differ between replicates. F1 follows
    cxr.metrics: zero denominator means zero, and all ten classes are included.
    """
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, (int, np.integer)) or bootstrap < 1:
        raise ValueError("bootstrap must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    targets = _matrix(targets, "targets", binary=True)
    n = len(targets)
    if not n:
        raise ValueError("comparison requires nonempty validation studies")
    subject_ids = list(subject_ids)
    if len(subject_ids) != n:
        raise ValueError("subject_ids must match the study row count")
    baseline = _matrix(baseline, "baseline", rows=n)
    candidate = _matrix(candidate, "candidate", rows=n)
    baseline_gates = _matrix(np.full_like(targets, 0.5) if baseline_gates is None else baseline_gates,
                             "baseline_gates", rows=n)
    candidate_gates = _matrix(np.full_like(targets, 0.5) if candidate_gates is None else candidate_gates,
                              "candidate_gates", rows=n)
    groups = _patient_groups(subject_ids)
    base_scores = _scores(targets, baseline, baseline_gates)
    candidate_scores = _scores(targets, candidate, candidate_gates)
    base_macro, candidate_macro = _macro(base_scores), _macro(candidate_scores)
    class_deltas = np.empty((bootstrap, len(METRICS), NUM_CLASSES))
    macro_deltas = np.empty((bootstrap, len(METRICS)))
    auc_counts = np.empty(bootstrap, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for repeat in range(bootstrap):
        indices = _sample_patient_indices(groups, rng)
        y = targets[indices]
        b = _scores(y, baseline[indices], baseline_gates[indices])
        c = _scores(y, candidate[indices], candidate_gates[indices])
        # Both runs have identical targets, hence identical defined AUC masks.
        class_deltas[repeat] = c - b
        macro_deltas[repeat] = _macro(c) - _macro(b)
        auc_counts[repeat] = np.count_nonzero(np.isfinite(b[0]))
    defined_auc = int(np.count_nonzero(np.isfinite(base_scores[0])))
    per_class = []
    for j in range(NUM_CLASSES):
        per_class.append({
            "id": j + 1,
            "positives": int(targets[:, j].sum()),
            "negatives": int(n - targets[:, j].sum()),
            "metrics": {name: {**_comparison(base_scores[k, j], candidate_scores[k, j]),
                               **_interval(class_deltas[:, k, j])}
                        for k, name in enumerate(METRICS)},
        })
    return {
        "schema_version": 1,
        "comparison": "candidate minus baseline",
        "caveat": CAVEAT,
        "studies": n,
        "patients": len(groups),
        "studies_per_patient": {"minimum": min(map(len, groups)), "maximum": max(map(len, groups))},
        "definitions": {
            "auc": "Mann-Whitney rank AUC with average ranks for exact ties; requires both outcomes.",
            "macro_auc": "Mean over classes with both outcomes, same subset for both runs in each replicate; subset may vary across replicates.",
            "f1": "2TP/(2TP+FP+FN); zero denominator is 0; macro averages all 10 classes.",
            "prediction": "Probability >= threshold; fixed threshold 0.5 and each submission row's Gate (blank = 0.5).",
            "bootstrap": "Sample P patients uniformly with replacement; include ALL studies of each selected patient, repeated for repeated selections; use identical draws for both runs. Gates remain fixed, not refitted.",
            "interval": "2.5th and 97.5th percentiles of defined paired replicate deltas, NumPy linear interpolation; null if none defined. No p-values or significance claims.",
        },
        "macro_auc_defined_classes": defined_auc,
        "macro_auc_missing_classes": NUM_CLASSES - defined_auc,
        "macro": {name: {**_comparison(base_macro[k], candidate_macro[k]),
                          **_interval(macro_deltas[:, k])} for k, name in enumerate(METRICS)},
        "per_class": per_class,
        "bootstrap": {
            "repeats": int(bootstrap), "seed": int(seed), "unit": "patient",
            "paired": True, "confidence_level": 0.95,
            "auc_defined_classes_histogram": {
                str(count): int(np.count_nonzero(auc_counts == count))
                for count in range(NUM_CLASSES + 1) if np.any(auc_counts == count)
            },
        },
    }


def _fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(Path(path).resolve()), "sha256": digest.hexdigest()}


def compare_runs(baseline, candidate, metadata, output, *, bootstrap=500, seed=2026):
    """Validate, compare and write JSON; refuse raw-root or input overwrites."""
    inputs = [Path(baseline), Path(candidate), Path(metadata)]
    destination = safe_output(output, data_root=Path(metadata).resolve().parent, inputs=inputs)
    # safe_output resolves symlinks; samefile additionally catches existing hard links.
    if destination.exists() and any(destination.samefile(source) for source in inputs):
        raise ValueError("Output would overwrite input through a filesystem alias")
    studies, targets = load_validation_metadata(metadata)
    b, bg, bv = load_submission(baseline, studies)
    c, cg, cv = load_submission(candidate, studies)
    report = compare_arrays([r["subject_id"] for r in studies], targets, b, c, bg, cg,
                            bootstrap=bootstrap, seed=seed)
    report["inputs"] = {
        "baseline": {**_fingerprint(baseline), "validation": bv},
        "candidate": {**_fingerprint(candidate), "validation": cv},
        "metadata": _fingerprint(metadata),
    }
    # Finish all validation, computation and JSON serialization before opening output.
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(serialized, encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True,
                        help="Official labeled valid.csv; its parent is protected as a raw-data root")
    parser.add_argument("--output", type=Path, required=True, help="JSON report outside raw-data root")
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    try:
        report = compare_runs(args.baseline, args.candidate, args.metadata, args.output,
                              bootstrap=args.bootstrap, seed=args.seed)
    except (ValueError, OSError, csv.Error) as exc:
        parser.error(str(exc))
    print(json.dumps({"studies": report["studies"], "patients": report["patients"],
                      "macro": report["macro"], "caveat": report["caveat"]}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
