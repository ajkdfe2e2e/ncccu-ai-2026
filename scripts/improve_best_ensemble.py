"""Evaluate new validation candidates against an immutable saved ensemble.

All model selection uses validation labels. The optional patient-group folds
reduce direct fitting/evaluation overlap, but the anchor ensemble was itself
selected on this validation set and therefore these diagnostics are not an
unbiased estimate of generalization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cxr.metrics import NUM_CLASSES, _rank_auc, evaluate
from cxr.safety import fresh_run_directory, safe_output
from cxr.submission import write_submission

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
FOLD_COUNT = 3
CLASS_LABELS = (
    "Enlarged Cardiomediastinum", "Pneumothorax", "Consolidation", "Pneumonia",
    "Edema", "Cardiomegaly", "Atelectasis", "Lung Opacity", "Pleural Effusion", "No Finding",
)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _keys(path: Path):
    raw = _read_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"study_ids.json must contain a list: {path}")
    keys = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"each study ID must be a [subject_id, study_id] pair: {path}")
        key = tuple(str(value) for value in item)
        if not all(value.strip() for value in key):
            raise ValueError(f"study IDs must not be empty: {path}")
        keys.append(key)
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate subject/study IDs: {path}")
    if len({study for _, study in keys}) != len(keys):
        raise ValueError(f"study IDs must belong to only one subject: {path}")
    return keys


def _matrix(path: Path, name: str, *, rows: int, columns: int | None = None,
            binary: bool = False):
    try:
        values = np.asarray(np.load(path), dtype=np.float64)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"could not load {name}: {path}") from exc
    if values.ndim != 2 or values.shape[0] != rows:
        raise ValueError(f"{name} must have {rows} rows: {path}")
    if columns is not None and values.shape[1] != columns:
        raise ValueError(f"{name} must have {columns} columns: {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values: {path}")
    if binary:
        if ((values != 0) & (values != 1)).any():
            raise ValueError(f"{name} must contain binary labels: {path}")
    elif ((values < 0) | (values > 1)).any():
        raise ValueError(f"{name} must contain values in [0, 1]: {path}")
    return values


def _load_aligned(folder: Path, expected_keys, expected_targets):
    ids_path = folder / "study_ids.json"
    probability_path = folder / "study_probabilities.npy"
    target_path = folder / "study_targets.npy"
    keys = _keys(ids_path)
    probabilities = _matrix(probability_path, "probabilities", rows=len(keys))
    targets = _matrix(target_path, "targets", rows=len(keys), columns=NUM_CLASSES, binary=True)
    if probabilities.shape[1] not in (NUM_CLASSES,):
        report_path = folder / "report.json"
        report = _read_json(report_path) if report_path.is_file() else {}
        indices = report.get("local_label_indices")
        if indices is None:
            raise ValueError(f"partial predictions require report.local_label_indices: {folder}")
    else:
        report_path = folder / "report.json"
        report = _read_json(report_path) if report_path.is_file() else {}
        indices = report.get("local_label_indices", list(range(NUM_CLASSES)))
    try:
        indices = [int(value) for value in indices]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"local_label_indices must be integer class indices: {folder}") from exc
    if (len(indices) != probabilities.shape[1] or len(set(indices)) != len(indices)
            or any(index < 0 or index >= NUM_CLASSES for index in indices)):
        raise ValueError(f"local_label_indices do not match prediction columns: {folder}")

    if len(keys) != len(expected_keys) or set(keys) != set(expected_keys):
        raise ValueError(f"study IDs differ from the immutable baseline: {folder}")
    positions = {key: i for i, key in enumerate(keys)}
    order = np.asarray([positions[key] for key in expected_keys], dtype=np.int64)
    # Align both data matrices before comparing labels. A candidate can be in
    # any row order, but its targets must match the corresponding baseline row.
    aligned_targets = targets[order]
    if not np.array_equal(aligned_targets, expected_targets):
        raise ValueError(f"targets differ from the immutable baseline after ID alignment: {folder}")
    return probabilities[order], indices, report


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path):
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _candidate(folder: Path, keys, targets):
    probabilities, indices, report = _load_aligned(folder, keys, targets)
    expanded = np.full((len(keys), NUM_CLASSES), np.nan, dtype=np.float64)
    expanded[:, indices] = probabilities
    return {
        "name": report.get("model") or folder.name,
        "folder": _portable_path(folder),
        "indices": indices,
        "probabilities": expanded,
        "provenance": {
            "model": report.get("model", folder.name),
            "folder": _portable_path(folder),
            "model_revision": report.get("model_revision") or report.get("source_revision"),
            "weights_sha256": report.get("weights_sha256") or report.get("checkpoint_sha256"),
            "checkpoint_sha256": report.get("checkpoint_sha256"),
            "horizontal_flip_tta": report.get("horizontal_flip_tta"),
            "probabilities_sha256": _sha256(folder / "study_probabilities.npy"),
            "report_sha256": _sha256(folder / "report.json") if (folder / "report.json").is_file() else None,
        },
    }


def _best_alpha(target, baseline, candidate):
    valid = np.isfinite(candidate)
    if not valid.any():
        return 0.0, _rank_auc(target, baseline)
    trials = []
    for alpha in ALPHAS:
        scores = baseline.copy()
        scores[valid] = alpha * candidate[valid] + (1.0 - alpha) * baseline[valid]
        auc = _rank_auc(target, scores)
        trials.append((float("-inf") if auc is None else auc, alpha))
    # Stable tie handling keeps the simpler baseline (alpha=0) preferred.
    score, alpha = max(trials, key=lambda value: (value[0], -value[1]))
    return alpha, None if not np.isfinite(score) else float(score)


def _fold_ids(keys):
    subjects = sorted({subject for subject, _ in keys})
    # Deterministic subject-level assignment prevents a patient's studies from
    # crossing folds while avoiding dependence on filesystem row ordering.
    subject_fold = {subject: index % FOLD_COUNT for index, subject in enumerate(subjects)}
    return np.asarray([subject_fold[subject] for subject, _ in keys], dtype=np.int64)


def _cross_validated(candidate, baseline, targets, keys):
    folds = _fold_ids(keys)
    oof = baseline.copy()
    fold_reports = []
    for fold in range(FOLD_COUNT):
        fit = folds != fold
        held = folds == fold
        if not fit.any() or not held.any():
            raise ValueError("three nonempty patient-group folds are required")
        alphas = []
        for class_index in range(NUM_CLASSES):
            alpha, _ = _best_alpha(targets[fit, class_index], baseline[fit, class_index],
                                   candidate[fit, class_index])
            alphas.append(alpha)
            valid = np.isfinite(candidate[:, class_index])
            oof[held, class_index] = baseline[held, class_index]
            use_candidate = held & valid
            oof[use_candidate, class_index] = (
                alpha * candidate[use_candidate, class_index]
                + (1.0 - alpha) * baseline[use_candidate, class_index]
            )
        fold_reports.append({
            "fold": fold,
            "fit_studies": int(fit.sum()),
            "held_studies": int(held.sum()),
            "held_subjects": int(len({keys[index][0] for index in np.flatnonzero(held)})),
            "alpha_by_class": alphas,
            "held_metrics": evaluate(targets[held], oof[held]),
        })
    return {"folds": fold_reports, "out_of_fold_metrics": evaluate(targets, oof)}


def _cross_validated_selection(candidates, baseline, targets, keys):
    folds = _fold_ids(keys)
    oof = baseline.copy()
    fold_reports = []
    for fold in range(FOLD_COUNT):
        fit, held = folds != fold, folds == fold
        if not fit.any() or not held.any():
            raise ValueError("three nonempty patient-group folds are required")
        selections = []
        for class_index in range(NUM_CLASSES):
            baseline_auc = _rank_auc(targets[fit, class_index], baseline[fit, class_index])
            best_auc = float("-inf") if baseline_auc is None else baseline_auc
            winner, alpha = None, 0.0
            for candidate in candidates:
                values = candidate["probabilities"][:, class_index]
                if not np.isfinite(values[fit]).all():
                    continue
                candidate_alpha, auc = _best_alpha(
                    targets[fit, class_index], baseline[fit, class_index], values[fit]
                )
                auc = float("-inf") if auc is None else auc
                if auc > best_auc:
                    best_auc, winner, alpha = auc, candidate, candidate_alpha
            if winner is not None:
                values = winner["probabilities"][:, class_index]
                oof[held, class_index] = alpha * values[held] + (1.0 - alpha) * baseline[held, class_index]
            selections.append({
                "class": class_index + 1,
                "source": "immutable_saved_ensemble" if winner is None else winner["name"],
                "alpha_candidate": float(alpha),
            })
        fold_reports.append({
            "fold": fold,
            "fit_studies": int(fit.sum()),
            "held_studies": int(held.sum()),
            "held_subjects": int(len({keys[index][0] for index in np.flatnonzero(held)})),
            "selection_by_class": selections,
            "held_metrics": evaluate(targets[held], oof[held]),
        })
    return {"folds": fold_reports, "out_of_fold_metrics": evaluate(targets, oof)}


def _safe_output_path(output, data_root, protected_dirs):
    target = safe_output(output, data_root=data_root, inputs=protected_dirs)
    for source in protected_dirs:
        source = Path(source).resolve()
        if target == source or source in target.parents or target in source.parents:
            raise ValueError(f"output must not overlap an input folder: {target} and {source}")
    if target.exists() and any(target.iterdir()):
        raise ValueError(f"output directory is not empty; choose a fresh path: {target}")
    return target


def improve(baseline_folder: Path, candidate_folders, output: Path, *, data_root: Path | None = None):
    baseline_folder = Path(baseline_folder).resolve()
    candidate_folders = [Path(folder).resolve() for folder in candidate_folders]
    protected_dirs = [baseline_folder, *candidate_folders]
    data_root = Path(data_root) if data_root is not None else ROOT / "cxr_first"
    output = _safe_output_path(output, data_root, protected_dirs)
    baseline_keys = _keys(baseline_folder / "study_ids.json")
    targets = _matrix(baseline_folder / "study_targets.npy", "baseline targets",
                      rows=len(baseline_keys), columns=NUM_CLASSES, binary=True)
    baseline = _matrix(baseline_folder / "study_probabilities.npy", "baseline probabilities",
                       rows=len(baseline_keys), columns=NUM_CLASSES)
    baseline_report_path = baseline_folder / "report.json"
    baseline_report = _read_json(baseline_report_path) if baseline_report_path.is_file() else {}
    baseline_metrics = evaluate(targets, baseline)
    baseline_source = {
        "name": "immutable_saved_ensemble",
        "folder": _portable_path(baseline_folder),
        "probabilities_sha256": _sha256(baseline_folder / "study_probabilities.npy"),
        "report_sha256": _sha256(baseline_report_path) if baseline_report_path.is_file() else None,
        "reported_macro_auc": baseline_report.get("selected", {}).get("macro_auc"),
    }

    # The baseline is always a per-class option, so adding a candidate cannot
    # silently replace a class when no candidate blend improves its AUC.
    chosen = baseline.copy()
    chosen_auc = [_rank_auc(targets[:, index], baseline[:, index]) for index in range(NUM_CLASSES)]
    recipe = [{"class": index + 1, "label": None, "source": "immutable_saved_ensemble",
               "alpha_candidate": 0.0} for index in range(NUM_CLASSES)]
    report = {
        "selection": "validation-only; per-class alpha selected on supplied validation labels",
        "baseline": baseline_source,
        "computed_baseline_metrics": baseline_metrics,
        "candidates": [],
        "fixed_half_blends": [],
        "selected": {},
        "recipe": [],
        "cross_validation": {},
        "cross_validation_limitation": (
            "The immutable baseline ensemble was selected using this validation set. Fold coefficients are fit only "
            "on the other two patient groups, but baseline preselection on all validation labels biases both the "
            "per-candidate and combined-selection diagnostics; neither is an unbiased generalization estimate."
        ),
    }

    candidates = [_candidate(folder, baseline_keys, targets) for folder in candidate_folders]
    for candidate in candidates:
        full = candidate["probabilities"]
        candidate_report = {
            "name": candidate["name"], "provenance": candidate["provenance"],
            "available_classes": [index + 1 for index in candidate["indices"]],
            "metrics": evaluate(targets, full) if len(candidate["indices"]) == NUM_CLASSES else {
                "per_class": [{"class": index + 1,
                               "auc": _rank_auc(targets[:, index], full[:, index])}
                              for index in candidate["indices"]]
            },
        }
        fixed = baseline.copy()
        for class_index in candidate["indices"]:
            fixed[:, class_index] = 0.5 * full[:, class_index] + 0.5 * baseline[:, class_index]
        report["fixed_half_blends"].append({
            "candidate": candidate["name"], "macro_auc": evaluate(targets, fixed)["macro_auc"]
        })
        changed = []
        for class_index in candidate["indices"]:
            alpha, auc = _best_alpha(targets[:, class_index], baseline[:, class_index],
                                    full[:, class_index])
            if auc is not None:
                if auc > chosen_auc[class_index]:
                    chosen[:, class_index] = alpha * full[:, class_index] + (1.0 - alpha) * baseline[:, class_index]
                    chosen_auc[class_index] = auc
                    recipe[class_index] = {
                        "class": class_index + 1, "label": None, "source": candidate["name"],
                        "alpha_candidate": float(alpha),
                        "provenance": candidate["provenance"],
                    }
                    changed.append(class_index + 1)
        candidate_report["selected_classes"] = changed
        report["candidates"].append(candidate_report)
        report["cross_validation"][candidate["name"]] = _cross_validated(full, baseline, targets, baseline_keys)
    report["cross_validation"]["combined_selection"] = _cross_validated_selection(
        candidates, baseline, targets, baseline_keys
    )

    for index, item in enumerate(recipe):
        item["label"] = baseline_report.get("local_labels", CLASS_LABELS)[index]
        item.setdefault("provenance", baseline_source)
    metrics = evaluate(targets, chosen)
    report["selected"] = {"metrics": metrics, "macro_auc": metrics["macro_auc"],
                          "delta_vs_immutable_baseline": metrics["macro_auc"] - baseline_metrics["macro_auc"]}
    report["recipe"] = recipe
    output = fresh_run_directory(output, data_root)
    np.save(output / "study_probabilities.npy", chosen)
    np.save(output / "study_targets.npy", targets)
    (output / "study_ids.json").write_text(json.dumps(baseline_keys, indent=2) + "\n", encoding="utf-8")
    studies = [{"subject_id": subject, "study_id": study} for subject, study in baseline_keys]
    write_submission(output / "valid.csv", studies, chosen)
    (output / "recipe.json").write_text(json.dumps({
        "formula": "selected[class] = alpha_candidate * candidate[class] + (1 - alpha_candidate) * immutable_baseline[class]",
        "baseline": baseline_source,
        "baseline_computed_metrics": baseline_metrics,
        "baseline_selection": {
            "per_class_selection": baseline_report.get("per_class_selection", []),
            "sources": baseline_report.get("models", {}),
            "global_pair_blends": baseline_report.get("global_pair_blends", {}),
            "benchmark_global_blends": baseline_report.get("benchmark_global_blends", {}),
        },
        "classes": recipe,
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "outputs/ensemble-valid")
    parser.add_argument("--candidate", type=Path, action="append", default=[],
                        help="candidate output folder; may be specified more than once")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/improved-ensemble-valid")
    parser.add_argument("--data-root", type=Path, default=Path(r"G:\cxr_first(1)\cxr_first"),
                        help="raw-data root protected from generated output")
    args = parser.parse_args()
    report = improve(args.baseline, args.candidate, args.output, data_root=args.data_root)
    print(json.dumps(report["selected"], indent=2), flush=True)


if __name__ == "__main__":
    main()
