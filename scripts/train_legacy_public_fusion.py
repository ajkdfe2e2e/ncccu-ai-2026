"""Train a bounded public subensemble around the preserved legacy fusion.

CPU only. Frozen local-model contributions are never used during fitting.
Official validation is opened only after one final training recipe is saved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cxr.anchored_fusion import (
    apply_public_weights, fit_anchored_public, flatten_legacy_recipe,
)
from cxr.data import _read_metadata
from cxr.metrics import evaluate
from cxr.safety import fresh_run_directory
from cxr.submission import validate_submission, write_submission
from scripts.compare_runs import compare_arrays, load_submission
from scripts.predict_best_ensemble import load_scores, replay_recipe, unique_studies

MODEL_ORDER = (
    "resnet18_baseline", "rad_dino_finetuned", "rad_dino_tta",
    "densenet121_chexpert", "russellala_convnext", "rad_dino_vitB_768",
    "medmae_vitb_nih_B_768_s2", "medmae_vitb_nih_B_768_s2_seed1337",
)
PUBLIC_ORDER = MODEL_ORDER[3:]
BRANCHES = ("densenet121", *PUBLIC_ORDER[1:])
WEIGHT_PATHS = (
    "weights/medical/itsomk-chexpert-densenet121/pytorch_model.safetensors",
    "weights/medical/russellala-convnext-chexpert-seed1/model.safetensors",
    *[f"weights/medical/chest-xray-bench/{name}/model.safetensors" for name in PUBLIC_ORDER[2:]],
)
VALID_FOLDERS = (
    "outputs/itsomk-densenet121-valid", "outputs/russellala-convnext-valid",
    "outputs/chest-xray-bench-valid/rad_dino_vitB_768",
    "outputs/chest-xray-bench-valid/medmae_vitb_nih_B_768_s2",
    "outputs/chest-xray-bench-new-valid-20261105/medmae_vitb_nih_B_768_s2_seed1337",
)
FIT_PARAMETERS = dict(rho=0.2, l2=1.0, temperature=0.1, pair_count=32768,
                      seed=20261107, max_iter=300, tolerance=1e-8)
OLD_CSV = ROOT / "outputs/submission_test_1414_best_20261105.csv"
OLD_HASH = "23f27c9f319ad7153cd1bd9b12064b2ed54111fdb6b388b794246e72e3a6d285"
CAVEAT = (
    "Public-only ranking objective, not full-ensemble ranking optimization. "
    "Local models and their coefficients are frozen; no local in-sample "
    "predictions enter fitting. Public pretraining overlap is unknown. "
    "The old recipe was selected on official validation; this comparison "
    "and its interval are descriptive, not independent generalization proof."
)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patient_folds(studies):
    subjects = sorted({row["subject_id"] for row in studies},
                      key=lambda subject: hashlib.sha256(f"20261107:{subject}".encode()).digest())
    if len(subjects) < 3:
        raise ValueError("at least three patients are required")
    assignment = {subject: index % 3 for index, subject in enumerate(subjects)}
    return assignment, np.asarray([assignment[row["subject_id"]] for row in studies])


def public_prior(legacy_weights):
    weights = np.asarray(legacy_weights, dtype=np.float64)
    if weights.shape != (len(MODEL_ORDER), 10):
        raise ValueError("legacy weights must match the locked eight-source order")
    public = weights[3:].copy()
    mass = public.sum(axis=0)
    prior = np.divide(public, mass, out=np.zeros_like(public), where=mass > 0)
    return mass, prior


def apply_delta(anchor, probabilities, before, after, mass):
    """Keep fixed classes bitwise intact and apply only the public delta."""
    anchor = np.asarray(anchor, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    before, after, mass = map(lambda x: np.asarray(x, dtype=np.float64), (before, after, mass))
    if (anchor.ndim != 2 or anchor.shape[1] != 10
            or probabilities.shape != (len(anchor), len(PUBLIC_ORDER), 10)
            or before.shape != (len(PUBLIC_ORDER), 10) or after.shape != before.shape
            or mass.shape != (10,) or not np.isfinite(mass).all()
            or ((mass < 0) | (mass > 1)).any()):
        raise ValueError("inconsistent public delta inputs")
    # Both applications validate coverage, finite weights and simplex masses.
    apply_public_weights(probabilities, before)
    apply_public_weights(probabilities, after)
    if not np.isfinite(anchor).all() or ((anchor < 0) | (anchor > 1)).any():
        raise ValueError("anchor probabilities must be finite and in [0,1]")
    delta = (after - before) * mass
    result = anchor.copy()
    for label in range(10):
        if not np.any(delta[:, label]):
            continue
        used = np.flatnonzero(delta[:, label])
        result[:, label] += probabilities[:, used, label] @ delta[used, label]
    if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        raise ValueError("candidate is not a finite probability matrix")
    return result


def active_metrics(targets, probabilities, active):
    metrics = evaluate(targets, probabilities)
    per_class = [row for index, row in enumerate(metrics["per_class"]) if active[index]]
    aucs = [row["auc"] for row in per_class if row["auc"] is not None]
    return {"scope": "normalized public subensemble only; not complete ensemble",
            "macro_auc_active_classes": float(np.mean(aucs)) if aucs else None,
            "defined_classes": len(aucs), "per_class": per_class}


def load_public_training(data_root, source, old_manifest):
    records = _read_metadata(data_root / "train.csv")
    studies = unique_studies(records)
    positions = {(r["subject_id"], r["study_id"]): i for i, r in enumerate(studies)}
    targets = np.asarray([r["labels"] for r in studies], dtype=np.float64)
    matrix = np.full((len(studies), len(PUBLIC_ORDER), 10), np.nan)
    experiment = read_json(source / "experiment.json")
    source_plan_path = source / "plan.json"
    if sha256(source_plan_path) != experiment["plan_sha256"]:
        raise ValueError("public source plan differs from its registered fingerprint")
    source_plan = read_json(source_plan_path)
    train_sha = sha256(data_root / "train.csv")
    if source_plan["train_metadata_sha256"] != train_sha:
        raise ValueError("public source plan uses a different training manifest")
    locks = {str(source / "experiment.json"): sha256(source / "experiment.json"),
             str(source_plan_path): sha256(source_plan_path)}
    fingerprints = old_manifest["model_fingerprints"]
    for model, (branch, weight_path) in enumerate(zip(BRANCHES, WEIGHT_PATHS)):
        expected_weight = fingerprints[str(Path(weight_path))]
        seen = set()
        for phase in ("development", "audit"):
            folder = source / "components" / phase / branch
            stage = experiment["stages"][f"{phase}_{branch}"]
            if stage["status"] != "complete":
                raise ValueError("source cache stage did not complete")
            for relative in ("report.json", "study_ids.json", "study_targets.npy", "study_probabilities.npy"):
                path = folder / relative
                if sha256(path) != stage["artifacts"][relative]:
                    raise ValueError(f"source cache fingerprint changed: {path}")
                locks[str(path)] = sha256(path)
            report = read_json(folder / "report.json")
            if (report.get("local_fit") is not False or report.get("train_csv_sha256") != train_sha
                    or report["model_sha256"].get(Path(weight_path).name) != expected_weight):
                raise ValueError("public cache training metadata or model identity differs")
            keys = [tuple(row) for row in read_json(folder / "study_ids.json")]
            values = np.load(folder / "study_probabilities.npy", allow_pickle=False)
            labels = np.load(folder / "study_targets.npy", allow_pickle=False)
            indices = report["local_label_indices"]
            if (indices != source_plan["class_indices"][branch]
                    or len(set(indices)) != len(indices) or any(j not in range(10) for j in indices)
                    or len(keys) != len(set(keys)) or seen.intersection(keys)
                    or values.shape != (len(keys), len(indices)) or labels.shape != (len(keys), 10)
                    or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any()):
                raise ValueError("public training cache shape, coverage or probabilities invalid")
            seen.update(keys)
            order = np.asarray([positions[key] for key in keys])
            if not np.array_equal(labels, targets[order]):
                raise ValueError("public cache labels do not match the training manifest")
            matrix[order[:, None], model, np.asarray(indices)[None, :]] = values
        if seen != set(positions):
            raise ValueError("public caches do not cover the entire training set")
    return studies, targets, matrix, locks


def gate_decision(comparison):
    auc = comparison["macro"]["auc"]
    interval = auc["ci95_percentile"]
    return (comparison["macro_auc_defined_classes"] == 10
            and auc["delta"] >= 0.001 and interval is not None and interval[0] > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root.resolve(strict=True)
    output = fresh_run_directory(args.output, data_root)
    state = {"status": "preflight", "pid": os.getpid(), "created_utc": now()}
    write_json(output / "run.json", state)
    try:
        recipe_path = ROOT / "docs/best_ensemble_recipe_20261105.json"
        test_root = ROOT / "outputs/best-ensemble-test-20261105"
        old_manifest = read_json(test_root / "run_manifest.json")
        if sha256(OLD_CSV) != OLD_HASH or sha256(recipe_path) != old_manifest["recipe_sha256"]:
            raise ValueError("legacy submission or recipe differs from the submitted anchor")
        legacy = read_json(recipe_path)
        old_weights = flatten_legacy_recipe(legacy, MODEL_ORDER)
        mass, prior = public_prior(old_weights)
        studies, targets, probabilities, locks = load_public_training(
            data_root, ROOT / "outputs/oof-fusion-20261106", old_manifest)
        active = mass > 0
        assignment, folds = patient_folds(studies)
        for path in (recipe_path, OLD_CSV, data_root / "train.csv", data_root / "valid.csv",
                     test_root / "run_manifest.json", ROOT / "docs/LEGACY_ANCHORED_TRAINING_20261107.md",
                     ROOT / "cxr/anchored_fusion.py", Path(__file__), ROOT / "cxr/oof_fusion.py",
                     ROOT / "scripts/predict_best_ensemble.py", ROOT / "scripts/compare_runs.py",
                     ROOT / "outputs/best-ensemble-improvement-20261105/study_probabilities.npy",
                     ROOT / "outputs/best-ensemble-improvement-20261105/study_ids.json"):
            locks[str(path)] = sha256(path)
        # Freeze validation caches without opening their labels before fitting.
        for folder in VALID_FOLDERS:
            for filename in ("report.json", "study_ids.json", "study_probabilities.npy"):
                path = ROOT / folder / filename
                locks[str(path)] = sha256(path)
        plan = {"created_utc": now(), "model_order": MODEL_ORDER, "public_order": PUBLIC_ORDER,
                "old_weights": old_weights.tolist(), "public_mass": mass.tolist(),
                "public_prior": prior.tolist(), "fit_parameters": FIT_PARAMETERS,
                "training_images": sum(1 for _ in _read_metadata(data_root / "train.csv")),
                "training_studies": len(studies), "training_patients": len(assignment),
                "fold_assignment": assignment, "input_locks": locks,
                "validation_gate": {"minimum_auc_gain": 0.001, "descriptive_ci95_lower_above": 0,
                                    "bootstrap": 1000, "seed": 20261107},
                "local_train_predictions_used": False, "caveat": CAVEAT}
        write_json(output / "plan.json", plan)
        np.save(output / "public_training_probabilities.npy", probabilities)
        write_json(output / "training_study_ids.json", [[r["subject_id"], r["study_id"]] for r in studies])
        diagnostics = []
        cross_fitted = np.zeros_like(targets)
        for fold in range(3):
            state.update(status="patient_cross_fit", fold=fold, updated_utc=now())
            write_json(output / "run.json", state)
            held = folds == fold
            fit = ~held
            fitted = fit_anchored_public(probabilities[fit], targets[fit], prior, **FIT_PARAMETERS)
            weights = np.asarray(fitted["weights"])
            cross_fitted[held] = apply_public_weights(probabilities[held], weights)
            old_public = apply_public_weights(probabilities[held], prior)
            diagnostics.append({"fold": fold, "training_patients": len({studies[i]["subject_id"] for i in np.flatnonzero(fit)}),
                                "held_patients": len({studies[i]["subject_id"] for i in np.flatnonzero(held)}),
                                "prior": active_metrics(targets[held], old_public, active),
                                "candidate": active_metrics(targets[held], cross_fitted[held], active),
                                "fit": fitted})
            print(f"patient fold {fold + 1}/3 completed", flush=True)
        write_json(output / "cross_fit.json", {"folds": diagnostics,
                   "pooled": active_metrics(targets, cross_fitted, active), "caveat": CAVEAT})
        state.update(status="full_training_fit", updated_utc=now())
        write_json(output / "run.json", state)
        trained = fit_anchored_public(probabilities, targets, prior, **FIT_PARAMETERS)
        new_public = np.asarray(trained["weights"])
        full_weights = old_weights.copy()
        full_weights[3:, active] = new_public[:, active] * mass[active]
        if (not np.array_equal(full_weights[:3], old_weights[:3])
                or not np.allclose(full_weights.sum(axis=0), 1, rtol=0, atol=1e-12)
                or np.any(full_weights < 0)):
            raise ValueError("fitted weights changed fixed local contributions or convexity")
        trained.update(model_order=MODEL_ORDER, public_order=PUBLIC_ORDER,
                       full_weights=full_weights.tolist(), public_mass=mass.tolist(),
                       public_prior=prior.tolist(), plan_sha256=sha256(output / "plan.json"),
                       validation_labels_read=False, caveat=CAVEAT)
        write_json(output / "trained_recipe.json", trained)
        recipe_sha = sha256(output / "trained_recipe.json")
        # The full fit and recipe are locked before opening official-valid labels.
        state.update(status="validation_guardrail", recipe_sha256=recipe_sha, updated_utc=now())
        write_json(output / "run.json", state)
        valid_records = _read_metadata(data_root / "valid.csv")
        valid_studies = unique_studies(valid_records)
        y_valid = np.asarray([r["labels"] for r in valid_studies], dtype=np.float64)
        anchor = load_scores(ROOT / "outputs/best-ensemble-improvement-20261105", valid_studies)
        old_metrics = evaluate(y_valid, anchor)
        if abs(old_metrics["macro_auc"] - 0.8203251712650804) > 1e-12:
            raise ValueError("legacy validation anchor no longer reproduces its recorded score")
        public_valid = np.stack([load_scores(ROOT / folder, valid_studies) for folder in VALID_FOLDERS], axis=1)
        for folder, weight_path in zip(VALID_FOLDERS, WEIGHT_PATHS):
            if read_json(ROOT / folder / "report.json")["weights_sha256"] != old_manifest["model_fingerprints"][str(Path(weight_path))]:
                raise ValueError("validation public model differs from the training cache")
        candidate = apply_delta(anchor, public_valid, prior, new_public, mass)
        write_submission(output / "candidate_valid.csv", valid_studies, candidate)
        comparison = compare_arrays([r["subject_id"] for r in valid_studies], y_valid, anchor, candidate,
                                    bootstrap=1000, seed=20261107)
        write_json(output / "validation_comparison.json", comparison)
        passed = gate_decision(comparison)
        for path, expected in locks.items():
            if sha256(path) != expected:
                raise ValueError(f"locked source changed during fitting: {path}")
        if sha256(output / "trained_recipe.json") != recipe_sha:
            raise ValueError("trained recipe changed after validation opened")
        report = {"status": "complete", "created_utc": now(), "guardrail_passed": passed,
                  "candidate_promoted": False, "old_online_score_user_reported": 78.53,
                  "candidate_online_score": None, "comparison": comparison["macro"],
                  "trained_recipe_sha256": recipe_sha, "caveat": CAVEAT,
                  "local_weights_unchanged": True, "old_submission_preserved": sha256(OLD_CSV) == OLD_HASH}
        report["maximum_absolute_probability_change"] = float(np.abs(candidate - anchor).max())
        report["training_summary"] = {"studies": len(studies), "patients": len(assignment),
                                      "public_components": len(PUBLIC_ORDER),
                                      "maximum_normalized_weight_change": float(np.abs(new_public - prior).max())}
        if passed:
            test_records = _read_metadata(data_root / "test.csv")
            test_studies = unique_studies(test_records)
            if any(r["labels"] is not None for r in test_records):
                raise ValueError("test manifest unexpectedly includes labels")
            components = test_root / "components"
            sources = {
                "resnet18_baseline": load_submission(components / "resnet18/test.csv", test_studies)[0],
                "rad_dino_finetuned": load_scores(components / "rad_dino", test_studies, "original_study_probabilities.npy"),
                "rad_dino_tta": load_scores(components / "rad_dino", test_studies),
                "densenet121_chexpert": load_scores(components / "densenet121", test_studies),
                "russellala_convnext": load_scores(components / "russellala_convnext", test_studies),
            }
            sources.update({name: load_scores(components / name / name, test_studies) for name in PUBLIC_ORDER[2:]})
            _, old_test = replay_recipe(sources, legacy)
            delivered, _, _ = load_submission(OLD_CSV, test_studies)
            if not np.array_equal(delivered, old_test):
                raise ValueError("old test recipe does not exactly reproduce the delivered submission")
            public_test = np.stack([sources[name] for name in PUBLIC_ORDER], axis=1)
            test_scores = apply_delta(delivered, public_test, prior, new_public, mass)
            destination = output / "submission_test_1414_legacy_public_trained_20261107.csv"
            temporary = destination.with_suffix(".csv.tmp")
            write_submission(temporary, test_studies, test_scores)
            validation = validate_submission(temporary, test_records)
            reloaded, _, _ = load_submission(temporary, test_studies)
            if not np.array_equal(test_scores, reloaded):
                raise ValueError("new CSV did not roundtrip exactly")
            if destination.exists():
                raise FileExistsError(destination)
            temporary.rename(destination)
            report.update(candidate_promoted=True, csv=str(destination), csv_sha256=sha256(destination),
                          validation=validation, csv_roundtrip_exact=True, old_test_replay_exact=True)
        else:
            report["decision"] = "retain old ensemble; candidate failed the prespecified validation guardrail"
        write_json(output / "report.json", report)
        state.update(status="complete", guardrail_passed=passed, updated_utc=now())
        write_json(output / "run.json", state)
        print(json.dumps(report, indent=2), flush=True)
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", updated_utc=now())
        write_json(output / "run.json", state)
        raise


if __name__ == "__main__":
    main()
