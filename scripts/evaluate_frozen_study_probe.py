"""Patient-held-out evaluation for a frozen study-level feature probe."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np

from cxr.metrics import evaluate
from cxr.study_probe import fit_linear_probe, pool_study_features, predict_linear_probe
from cxr.submission import write_submission
from scripts.compare_runs import compare_arrays


NUM_CLASSES = 10
FOLDS = (0, 1, 2, "audit")
MODES = ("mean", "views")
ALPHAS = (0.2, 1.0)
L2 = 0.01
MAX_ITER = 200
SEED = 20261108
RECIPE_NAME = "recipe.json"
REPORT_NAME = "report.json"


def _matrix(values, name, *, rows=None):
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric matrix") from exc
    if array.ndim != 2 or array.shape[1] != NUM_CLASSES:
        raise ValueError(f"{name} must have shape (N, 10)")
    if rows is not None and len(array) != rows:
        raise ValueError(f"{name} row count must be {rows}")
    if not np.isfinite(array).all() or ((array < 0) | (array > 1)).any():
        raise ValueError(f"{name} must contain finite probabilities in [0, 1]")
    return array


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n",
                                     dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _load_head(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key].copy() for key in archive.files}
    for key in ("l2", "iterations", "closure_calls", "loss", "seed"):
        if key in payload and payload[key].shape == ():
            payload[key] = payload[key].item()
    return payload


def _save_head(path: Path, payload: Mapping) -> None:
    arrays = {}
    for key, value in payload.items():
        if isinstance(value, (str, bytes)) or value is None:
            raise ValueError(f"head payload field {key!r} is not a NumPy-safe scalar or array")
        arrays[key] = np.asarray(value)
        if arrays[key].dtype.hasobject:
            raise ValueError(f"head payload field {key!r} requires pickle")
    np.savez(path, **arrays)


def _study_subject_ids(studies):
    return [str(row["subject_id"]) for row in studies]


def _study_keys(studies):
    return [(str(row["subject_id"]), str(row["study_id"])) for row in studies]


def _aligned_train_inputs(records, image_features, views, old_probabilities):
    if any(row.get("labels") is None for row in records):
        raise ValueError("training records must include labels")
    results = {}
    for mode in MODES:
        studies, features, targets = pool_study_features(records, image_features, views, mode)
        if targets is None:
            raise ValueError("study pooling returned no training targets")
        results[mode] = (studies, np.asarray(features, dtype=np.float64),
                         np.asarray(targets, dtype=np.float64))
    keys = _study_keys(results["mean"][0])
    if any(_study_keys(value[0]) != keys for value in results.values()):
        raise ValueError("study pooling modes returned different study orders")
    old = _matrix(old_probabilities, "old_probabilities", rows=len(keys))
    if any(((targets != 0) & (targets != 1)).any() for _, _, targets in results.values()):
        raise ValueError("training targets must be binary")
    return results, old


def _fold_memberships(studies, assignment):
    if not isinstance(assignment, Mapping):
        raise ValueError("assignment must map patient IDs to fold names")
    patients = set(_study_subject_ids(studies))
    missing = patients - set(map(str, assignment.keys()))
    if missing:
        raise ValueError(f"assignment is missing {len(missing)} training patients")
    normalized = {str(patient): fold for patient, fold in assignment.items()}
    invalid = {patient: fold for patient, fold in normalized.items() if fold not in FOLDS}
    if invalid:
        raise ValueError(f"assignment has unsupported fold values: {invalid}")
    folds = np.asarray([normalized[subject] for subject in _study_subject_ids(studies)], dtype=object)
    if set(folds.tolist()) != set(FOLDS):
        raise ValueError("assignment must populate folds 0, 1, 2, and 'audit'")
    return folds


def _metric_summary(targets, probabilities):
    result = evaluate(targets, probabilities)
    class_aucs = [entry["auc"] for entry in result["per_class"]]
    return {
        "macro_auc": result["macro_auc"],
        "macro_auc_defined_classes": result["macro_auc_defined_classes"],
        "per_class_auc": class_aucs,
    }


def _candidate_eligibility(candidate):
    return (
        candidate["macro_auc"] is not None
        and candidate["macro_auc_defined_classes"] == NUM_CLASSES
        and candidate["overall_gain"] is not None
        and candidate["overall_gain"] >= 0.001
        and candidate["positive_held_groups"] >= 3
    )


def _select_candidate(candidates):
    eligible = [item for item in candidates if _candidate_eligibility(item)]
    if not eligible:
        return None
    return sorted(eligible, key=lambda item: (
        -item["macro_auc"], item["alpha"], MODES.index(item["mode"]),
    ))[0]


def _cached_training_result(output: Path):
    recipe_path = output / RECIPE_NAME
    head_path = output / "final_head.npz"
    report_path = output / REPORT_NAME
    if recipe_path.exists() and head_path.exists():
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
        if recipe.get("head_sha256") != _sha256(head_path):
            raise ValueError("saved final head hash does not match recipe")
        if recipe.get("status") != "selected":
            raise ValueError("saved recipe has an unsupported status")
        return recipe
    if recipe_path.exists() != head_path.exists():
        raise ValueError("incomplete saved recipe/final head pair")
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") == "rejected":
            return None
        raise ValueError("existing training report is incomplete; refusing to refit")
    return False


def train_candidates(output, records, image_features, views, assignment,
                     old_probabilities) -> dict | None:
    """Fit patient-held-out heads and register the selected recipe before validation."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cached = _cached_training_result(output)
    if cached is not False:
        return cached

    pooled, old = _aligned_train_inputs(records, image_features, views, old_probabilities)
    studies = pooled["mean"][0]
    folds = _fold_memberships(studies, assignment)
    fold_reports = {}
    mode_oof = {}

    for mode in MODES:
        _, features, targets = pooled[mode]
        oof = np.full((len(studies), NUM_CLASSES), np.nan, dtype=np.float64)
        fold_reports[mode] = {}
        for fold in FOLDS:
            held = folds == fold
            train = ~held
            if not held.any() or not train.any():
                raise ValueError(f"fold {fold!r} must have nonempty train and held studies")
            if set(_study_subject_ids([studies[i] for i in np.flatnonzero(train)])) & set(
                    _study_subject_ids([studies[i] for i in np.flatnonzero(held)])):
                raise ValueError(f"patient leakage in fold {fold!r}")
            payload = fit_linear_probe(features[train], targets[train], l2=L2,
                                       max_iter=MAX_ITER, seed=SEED)
            fold_dir = output / mode / str(fold)
            fold_dir.mkdir(parents=True, exist_ok=True)
            head_path = fold_dir / "head.npz"
            _save_head(head_path, payload)
            reloaded = _load_head(head_path)
            expected = predict_linear_probe(payload, features[held])
            actual = predict_linear_probe(reloaded, features[held])
            if not np.array_equal(expected, actual):
                raise ValueError(f"fold {fold!r} head changed after exact reload")
            oof[held] = actual
            fold_reports[mode][str(fold)] = {
                "training_studies": int(train.sum()),
                "training_patients": len(set(_study_subject_ids([studies[i] for i in np.flatnonzero(train)]))),
                "held_studies": int(held.sum()),
                "held_patients": len(set(_study_subject_ids([studies[i] for i in np.flatnonzero(held)]))),
                "head_sha256": _sha256(head_path),
            }
        if not np.isfinite(oof).all():
            raise ValueError(f"{mode} OOF predictions do not cover every study")
        mode_oof[mode] = oof
        np.save(output / f"oof_{mode}_probabilities.npy", oof, allow_pickle=False)

    targets = pooled["mean"][2]
    np.save(output / "oof_targets.npy", targets, allow_pickle=False)
    (output / "oof_study_ids.json").write_text(
        json.dumps(_study_keys(studies), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    overall_old_metrics = _metric_summary(targets, old)
    candidates = []
    for mode in MODES:
        for alpha in ALPHAS:
            candidate_probs = (1.0 - alpha) * old + alpha * mode_oof[mode]
            np.save(output / f"oof_{mode}_alpha_{alpha:g}.npy", candidate_probs,
                    allow_pickle=False)
            overall = _metric_summary(targets, candidate_probs)
            held_groups = []
            for fold in FOLDS:
                held = folds == fold
                held_old_metrics = _metric_summary(targets[held], old[held])
                new_metrics = _metric_summary(targets[held], candidate_probs[held])
                gain = None
                if (held_old_metrics["macro_auc"] is not None
                        and new_metrics["macro_auc"] is not None):
                    gain = float(new_metrics["macro_auc"] - held_old_metrics["macro_auc"])
                held_groups.append({
                    "fold": fold, "baseline_macro_auc": held_old_metrics["macro_auc"],
                    "candidate_macro_auc": new_metrics["macro_auc"], "gain": gain,
                    "defined_classes": new_metrics["macro_auc_defined_classes"],
                })
            candidates.append({
                "mode": mode, "alpha": alpha,
                "macro_auc": overall["macro_auc"],
                "macro_auc_defined_classes": overall["macro_auc_defined_classes"],
                "overall_gain": (
                    float(overall["macro_auc"] - overall_old_metrics["macro_auc"])
                    if overall["macro_auc"] is not None
                    and overall_old_metrics["macro_auc"] is not None
                    else None
                ),
                "per_class_auc": overall["per_class_auc"],
                "held_groups": held_groups,
                "positive_held_groups": sum(item["gain"] is not None and item["gain"] > 0
                                             for item in held_groups),
            })

    selected = _select_candidate(candidates)
    report = {
        "status": "selected" if selected else "rejected",
        "selection_method": "four fixed mode/alpha candidates on patient-held-out OOF predictions",
        "selection_caveat": "Finite-candidate OOF selection estimate; not unbiased generalization evidence.",
        "folds": fold_reports,
        "candidates": candidates,
        "selected": selected,
        "promotion_rule": {"overall_oof_gain_minimum": 0.001,
                           "minimum_positive_held_groups": 3,
                           "required_defined_classes": 10},
    }
    if selected is None:
        _json_write(output / REPORT_NAME, report)
        return None

    mode = selected["mode"]
    _, features, all_targets = pooled[mode]
    final_payload = fit_linear_probe(features, all_targets, l2=L2,
                                     max_iter=MAX_ITER, seed=SEED)
    final_head = output / "final_head.npz"
    _save_head(final_head, final_payload)
    reloaded = _load_head(final_head)
    if not np.array_equal(predict_linear_probe(final_payload, features),
                          predict_linear_probe(reloaded, features)):
        raise ValueError("final head changed after exact reload")

    recipe = {
        "schema_version": 1, "status": "selected", "mode": mode,
        "alpha": selected["alpha"], "l2": L2, "max_iter": MAX_ITER,
        "seed": SEED, "head_file": final_head.name,
        "head_sha256": _sha256(final_head),
        "selection_evidence": selected,
        "selection_caveat": report["selection_caveat"],
    }
    _json_write(output / REPORT_NAME, report)
    _json_write(output / RECIPE_NAME, recipe)
    return recipe


def _replay_validation(output: Path, recipe: dict):
    report_path = output / "validation_report.json"
    csv_path = output / "candidate_valid.csv"
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("recipe_sha256") != _sha256(output / RECIPE_NAME):
        raise ValueError("saved validation report belongs to a different recipe")
    if report.get("head_sha256") != recipe["head_sha256"]:
        raise ValueError("saved validation report belongs to a different final head")
    if not csv_path.is_file() or report.get("candidate_valid_csv_sha256") != _sha256(csv_path):
        raise ValueError("saved validation report does not match candidate_valid.csv")
    return report


def run_validation(output, records, image_features, views, old_probabilities,
                   recipe) -> dict:
    """Evaluate only the registered candidate and write its validation CSV."""
    output = Path(output)
    recipe_path = output / RECIPE_NAME
    if not recipe_path.is_file():
        raise ValueError("registered recipe must be saved before validation")
    saved_recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if recipe != saved_recipe:
        raise ValueError("validation recipe differs from the registered recipe")
    if recipe.get("status") != "selected" or recipe.get("mode") not in MODES:
        raise ValueError("recipe is not an eligible frozen probe candidate")
    head_path = output / str(recipe.get("head_file", ""))
    if not head_path.is_file() or _sha256(head_path) != recipe.get("head_sha256"):
        raise ValueError("registered final head is missing or has a hash mismatch")
    cached = _replay_validation(output, recipe)
    if cached is not None:
        return cached

    if any(row.get("labels") is None for row in records):
        raise ValueError("validation records must include labels")
    studies, features, targets = pool_study_features(records, image_features, views,
                                                      recipe["mode"])
    if targets is None or ((targets != 0) & (targets != 1)).any():
        raise ValueError("validation targets must be binary and present")
    old = _matrix(old_probabilities, "old_probabilities", rows=len(studies))
    head = _load_head(head_path)
    new_probs = predict_linear_probe(head, features)
    candidate = (1.0 - float(recipe["alpha"])) * old + float(recipe["alpha"]) * new_probs
    if not np.isfinite(candidate).all() or ((candidate < 0) | (candidate > 1)).any():
        raise ValueError("candidate probabilities are invalid")

    csv_path = output / "candidate_valid.csv"
    write_submission(csv_path, studies, candidate, 0.5)
    np.save(output / "candidate_valid_probabilities.npy", candidate, allow_pickle=False)
    comparison = compare_arrays(
        _study_subject_ids(studies), targets, old, candidate,
        bootstrap=1000, seed=SEED,
    )
    auc = comparison["macro"]["auc"]
    ci = auc.get("ci95_percentile")
    full_auc_replicates = int(comparison["bootstrap"]["auc_defined_classes_histogram"].get("10", 0))
    all_classes_defined = comparison["macro_auc_defined_classes"] == NUM_CLASSES and all(
        item["metrics"]["auc"]["baseline"] is not None
        and item["metrics"]["auc"]["candidate"] is not None
        for item in comparison["per_class"]
    )
    promoted = bool(
        all_classes_defined and auc["delta"] is not None and auc["delta"] >= 0.001
        and ci is not None and ci[0] > 0 and full_auc_replicates >= 950
    )
    report = {
        "status": "complete", "promoted": promoted,
        "recipe_sha256": _sha256(recipe_path),
        "head_sha256": recipe["head_sha256"], "mode": recipe["mode"],
        "alpha": recipe["alpha"], "comparison": comparison,
        "all_ten_class_aucs_defined": all_classes_defined,
        "full_ten_class_bootstrap_replicates": full_auc_replicates,
        "promotion_rule": {"macro_auc_gain_minimum": 0.001,
                           "ci95_lower_strictly_positive": True,
                           "all_ten_class_aucs_defined": True,
                           "minimum_full_class_bootstrap_replicates": 950},
        "candidate_valid_csv_sha256": _sha256(csv_path),
        "old_online_score_user_reported": 78.53,
        "new_online_score": None,
        "caveat": "Validation is a single fixed-candidate comparison; it was not used to choose mode or alpha.",
    }
    _json_write(output / "validation_report.json", report)
    return report
