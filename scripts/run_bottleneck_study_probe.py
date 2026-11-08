"""Nested patient-group selection of compact CXR-LT feature bottlenecks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cxr.data import read_records
from cxr.metrics import evaluate
from cxr.safety import fresh_run_directory, safe_output
from cxr.study_probe import pool_study_features
from cxr.bottleneck_probe import (
    fit_bottleneck_probe as fit_linear_probe,
    predict_bottleneck_probe as predict_linear_probe,
)
from cxr.submission import validate_submission, write_submission
from scripts.compare_runs import compare_arrays, load_submission
from scripts.evaluate_frozen_study_probe import (
    FOLDS, _fold_memberships, _json_write, _load_head, _save_head, _sha256,
)

SEED = 20261108
MAX_ITER = 300
CONFIGS = tuple({"mode": mode, "l2": 0.01, "class_balance": False}
                for mode in ("semantic", "pca128"))
FILES = (
    "scripts/run_bottleneck_study_probe.py", "scripts/evaluate_frozen_study_probe.py",
    "scripts/compare_runs.py", "cxr/__init__.py", "cxr/data.py", "cxr/metrics.py",
    "cxr/safety.py", "cxr/submission.py", "cxr/study_probe.py", "cxr/bottleneck_probe.py",
    "docs/BOTTLENECK_STUDY_HEAD_PLAN_20261108.md",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def config_name(config):
    return f"{config['mode']}_l2_{config['l2']:g}"


def score(targets, probabilities):
    result = evaluate(targets, probabilities)
    return {key: result[key] for key in ("macro_auc", "macro_auc_defined_classes", "macro_f1")} | {
        "per_class_auc": [item["auc"] for item in result["per_class"]]
    }


def select_config(rows):
    eligible = [row for row in rows if row["macro_auc_defined_classes"] == 10
                and row["macro_auc"] is not None and np.isfinite(row["macro_auc"])]
    if len(eligible) != len(rows) or not eligible:
        raise ValueError("all fixed configurations must have ten defined inner AUCs")
    return sorted(eligible, key=lambda row: (
        -row["macro_auc"], 0 if row["config"]["mode"] == "semantic" else 1,
    ))[0]


def inner_select(targets, folds, outer_fold, predict, configs=CONFIGS):
    """Select without accessing outer-fold targets or fitting on outer patients."""
    remaining = [fold for fold in FOLDS if fold != outer_fold]
    inner_mask = folds != outer_fold
    rows = []
    for config in configs:
        probabilities = np.full_like(targets, np.nan, dtype=np.float64)
        for inner_fold in remaining:
            training_folds = tuple(fold for fold in remaining if fold != inner_fold)
            held = np.flatnonzero(folds == inner_fold)
            probabilities[held] = predict(training_folds, config, held)
        rows.append({"config": dict(config), **score(targets[inner_mask], probabilities[inner_mask])})
    return select_config(rows), rows


def nested_oof(targets, folds, predict, callback=None, configs=CONFIGS):
    predictions = np.full_like(targets, np.nan, dtype=np.float64)
    reports = []
    for outer_fold in FOLDS:
        selected, rows = inner_select(targets, folds, outer_fold, predict, configs)
        training_folds = tuple(fold for fold in FOLDS if fold != outer_fold)
        held = np.flatnonzero(folds == outer_fold)
        predictions[held] = predict(training_folds, selected["config"], held)
        report = {"outer_fold": outer_fold, "selected_config": selected["config"],
                  "inner_selection": rows, "held_studies": int(len(held))}
        reports.append(report)
        if callback:
            callback(reports)
    if not np.isfinite(predictions).all():
        raise ValueError("nested predictions do not cover every training study")
    return predictions, reports


def training_gate(targets, baseline, candidate, folds, legacy):
    old, new, older = score(targets, baseline), score(targets, candidate), score(targets, legacy)
    groups = []
    for fold in FOLDS:
        held = folds == fold
        before, after = score(targets[held], baseline[held]), score(targets[held], candidate[held])
        groups.append({"fold": fold, "baseline_macro_auc": before["macro_auc"],
                       "candidate_macro_auc": after["macro_auc"],
                       "defined_classes": after["macro_auc_defined_classes"],
                       "gain": None if before["macro_auc"] is None or after["macro_auc"] is None
                       else after["macro_auc"] - before["macro_auc"]})
    gain = None if old["macro_auc"] is None or new["macro_auc"] is None else new["macro_auc"] - old["macro_auc"]
    legacy_gain = None if older["macro_auc"] is None or new["macro_auc"] is None else new["macro_auc"] - older["macro_auc"]
    positive = sum(row["gain"] is not None and row["gain"] > 0 for row in groups)
    passed = bool(old["macro_auc_defined_classes"] == new["macro_auc_defined_classes"] == 10
                  and all(row["defined_classes"] == 10 for row in groups)
                  and gain is not None and gain >= 0.001 and positive >= 3
                  and legacy_gain is not None and legacy_gain >= 0.001)
    return {"passed": passed, "baseline": old, "candidate": new, "legacy": older,
            "gain": gain, "legacy_gain": legacy_gain, "positive_held_groups": positive,
            "held_groups": groups}


def validation_gate(comparisons):
    for name in ("incumbent", "legacy"):
        comparison = comparisons[name]
        auc = comparison["macro"]["auc"]
        interval = auc.get("ci95_percentile")
        if not (comparison["macro_auc_defined_classes"] == 10
                and all(row["metrics"]["auc"]["baseline"] is not None
                        and row["metrics"]["auc"]["candidate"] is not None
                        for row in comparison["per_class"])
                and auc["delta"] is not None and auc["delta"] >= 0.001
                and interval is not None and interval[0] > 0
                and comparison["bootstrap"]["auc_defined_classes_histogram"].get("10", 0) >= 950):
            return False
    return True


class FitCache:
    """Memoize each exact training-fold set, never a held label or statistic."""

    def __init__(self, folder, features, targets, folds, subjects, plan_sha256):
        self.folder, self.x, self.y = Path(folder), features, targets
        self.folds, self.subjects, self.plan_sha256 = folds, np.asarray(subjects), plan_sha256
        self.fit_count = 0
        self.cache_hits = 0

    def predict(self, training_folds, config, held_indices):
        training_folds = tuple(fold for fold in FOLDS if fold in training_folds)
        if not training_folds or len(training_folds) >= len(FOLDS):
            raise ValueError("CV fit needs a proper nonempty training-fold subset")
        held_indices = np.asarray(held_indices, dtype=np.int64)
        if held_indices.ndim != 1 or len(held_indices) == 0 or np.any(held_indices < 0) or np.any(held_indices >= len(self.y)):
            raise ValueError("held indices must select nonempty valid rows")
        train = np.isin(self.folds, np.asarray(training_folds, dtype=object))
        if np.any(train[held_indices]) or set(self.subjects[train]) & set(self.subjects[held_indices]):
            raise ValueError("patient leakage in requested fit")
        identity = {"plan_sha256": self.plan_sha256, "training_folds": list(training_folds),
                    "config": dict(config), "max_iter": MAX_ITER, "seed": SEED,
                    "training_studies": int(train.sum()),
                    "training_patients": len(set(self.subjects[train]))}
        group_name = "_".join(str(fold) for fold in training_folds)
        folder = self.folder / group_name / config_name(config)
        head_path, metadata_path = folder / "head.npz", folder / "fit.json"
        if metadata_path.exists():
            metadata = read_json(metadata_path)
            if metadata["identity"] != identity or _sha256(head_path) != metadata["head_sha256"]:
                raise ValueError("cached fit fingerprint mismatch")
            payload = _load_head(head_path)
            self.cache_hits += 1
        else:
            if head_path.exists():
                raise ValueError("uncommitted cached head; inspect before resuming")
            started = time.monotonic()
            payload = fit_linear_probe(self.x[train], self.y[train], l2=config["l2"],
                                       class_balance=config["class_balance"], mode=config["mode"],
                                       max_iter=MAX_ITER, seed=SEED)
            folder.mkdir(parents=True, exist_ok=True)
            _save_head(head_path, payload)
            reloaded = _load_head(head_path)
            expected = predict_linear_probe(payload, self.x[held_indices])
            if not np.array_equal(expected, predict_linear_probe(reloaded, self.x[held_indices])):
                raise ValueError("saved head does not replay exactly")
            metadata = {"identity": identity, "head_sha256": _sha256(head_path),
                        "elapsed_seconds": time.monotonic() - started,
                        "iterations": int(payload["iterations"]), "loss": float(payload["loss"])}
            _json_write(metadata_path, metadata)
            self.fit_count += 1
            _json_write(self.folder.parent / "fit_progress.json", {
                "updated_utc": now(), "training_folds": list(training_folds), "config": config,
                "fit_count_this_process": self.fit_count,
                "unique_registered_heads": len(list(self.folder.rglob("fit.json"))),
                "last_fit_seconds": metadata["elapsed_seconds"],
            })
            print(json.dumps({"event": "fit_complete", "training_folds": list(training_folds),
                              "config": config, "fit_count_this_process": self.fit_count,
                              "elapsed_seconds": metadata["elapsed_seconds"]}), flush=True)
        return predict_linear_probe(payload, self.x[held_indices])


def load_split(plan, split):
    source = Path(plan["source_experiment"])
    data_root = Path(plan["data_root"])
    if split == "train":
        records = read_json(source / "train_records.json")["records"]
    else:
        records = read_records(data_root / f"{split}.csv", data_root / split)
    cache = source / f"{split}_features"
    identities = read_json(cache / "input.json")["images"]
    if [row["dicom_id"] for row in records] != [row[0] for row in identities]:
        raise ValueError(f"{split} feature rows do not match manifest order")
    for record, (dicom, path, size, mtime) in zip(records, identities):
        stat = Path(path).stat()
        if str(Path(record["path"]).resolve()) != str(Path(path).resolve()) or stat.st_size != size or stat.st_mtime_ns != mtime:
            raise ValueError(f"cached image identity changed: {dicom}")
    progress = read_json(cache / "progress.json")
    if progress.get("status") != "complete" or progress["completed"] != len(records):
        raise ValueError("source features are incomplete")
    features = np.load(cache / "image_features.npy", mmap_mode="r", allow_pickle=False)
    if features.shape != (len(records), 1054) or features.dtype != np.float32:
        raise ValueError("unexpected source feature dimensions or dtype")
    studies, pooled, targets = pool_study_features(records, features, [""] * len(records), "mean")
    return studies, pooled, targets


def environment():
    return {"python": sys.version, **{key: importlib.metadata.version(key) for key in ("numpy", "torch")}}


def verify_plan(output, plan):
    if environment() != plan["environment"]:
        raise ValueError("software environment changed")
    for path, digest in plan["input_locks"].items():
        if _sha256(Path(path)) != digest:
            raise ValueError(f"input fingerprint changed: {path}")
    for path, digest in plan["source_locks"].items():
        if _sha256(output / "source" / path) != digest:
            raise ValueError(f"source snapshot changed: {path}")


def prepare(output, source):
    source = Path(source).resolve()
    source_plan = read_json(source / "plan.json")
    state = read_json(source / "experiment.json")
    if state.get("status") != "complete" or not state.get("candidate_promoted") or not state.get("test_csv_exported"):
        raise ValueError("source incumbent must have completed and exported test")
    output = fresh_run_directory(output, source_plan["data_root"])
    paths = list(source_plan["input_locks"])
    paths += [source / "source" / name for name in source_plan["source_locks"]]
    paths += [source / name for name in (
        "plan.json", "experiment.json", "recipe.json", "final_head.npz", "report.json",
        "validation_report.json", "train_records.json", "oof_mean_probabilities.npy",
        "oof_targets.npy", "oof_study_ids.json", "candidate_valid.csv",
        "submission_test_1414_cxrlt_study_20261108.csv",
    )]
    for split in ("train", "valid", "test"):
        paths += [source / f"{split}_features" / name for name in ("input.json", "progress.json", "image_features.npy")]
    input_locks = {str(Path(path).resolve()): _sha256(Path(path)) for path in paths}
    for path, digest in source_plan["input_locks"].items():
        if input_locks[str(Path(path).resolve())] != digest:
            raise ValueError(f"source experiment input has changed: {path}")
    for name, digest in source_plan["source_locks"].items():
        if input_locks[str((source / "source" / name).resolve())] != digest:
            raise ValueError(f"source experiment snapshot has changed: {name}")
    for split in ("train", "valid", "test"):
        cache = source / f"{split}_features"
        if input_locks[str((cache / "image_features.npy").resolve())] != read_json(cache / "progress.json")["sha256"]:
            raise ValueError("source feature cache fingerprint mismatch")
    source_locks = {}
    for name in FILES:
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        source_locks[name] = _sha256(target)
    plan = {"created_utc": now(), "source_experiment": str(source),
            "data_root": source_plan["data_root"], "partition": source_plan["partition"],
            "legacy_valid": source_plan["old_valid"], "legacy_oof": source_plan["old_oof"],
            "configs": list(CONFIGS), "max_iter": MAX_ITER, "seed": SEED,
            "environment": environment(), "input_locks": input_locks, "source_locks": source_locks,
            "selection": "Inner patient CV chooses one of two fixed feature bottlenecks; outer CV evaluates the selector.",
            "caveat": "Historical/adaptive reuse of training groups and validation; public pretraining overlap unknown."}
    _json_write(output / "plan.json", plan)
    _json_write(output / "experiment.json", {"status": "prepared", "created_utc": now()})
    return output


def legacy_oof(plan, studies):
    folder = Path(plan["legacy_oof"])
    ids = read_json(folder / "oof_study_ids.json")
    if ids != [[row["subject_id"], row["study_id"]] for row in studies]:
        raise ValueError("legacy OOF order differs from current training studies")
    components = np.load(folder / "oof_component_probabilities.npy", allow_pickle=False)
    prior = np.asarray(read_json(folder / "plan.json")["prior"])
    if components.shape != (len(studies), 8, 10) or prior.shape != (8, 10):
        raise ValueError("legacy matrix shape mismatch")
    if np.any(~np.isfinite(components) & (prior[None] != 0)):
        raise ValueError("legacy baseline needs an unavailable output")
    return (np.where(np.isfinite(components), components, 0) * prior[None]).sum(axis=1)


def run(output):
    plan = read_json(output / "plan.json")
    state = read_json(output / "experiment.json")
    verify_plan(output, plan)
    if state["status"] == "complete":
        for name, digest in read_json(output / "artifacts.json").items():
            if _sha256(output / name) != digest:
                raise ValueError(f"completed artifact changed: {name}")
        print(json.dumps(state), flush=True)
        return

    def stage(phase, **extra):
        state.update(status="running", phase=phase, pid=os.getpid(), updated_utc=now(), error=None, **extra)
        _json_write(output / "experiment.json", state)

    def finish(decision, promoted=False):
        verify_plan(output, plan)
        artifacts = {str(path.relative_to(output)): _sha256(path) for path in output.rglob("*")
                     if path.is_file() and "source" not in path.relative_to(output).parts
                     and path.name not in {"artifacts.json", "experiment.json", "launcher.json"}
                     and not path.name.endswith(".log")}
        _json_write(output / "artifacts.json", artifacts)
        state.update(status="complete", phase=None, updated_utc=now(), error=None,
                     decision=decision, candidate_promoted=promoted, test_csv_exported=promoted)
        _json_write(output / "experiment.json", state)

    torch.set_num_threads(4)
    try:
        stage("load_train_cache")
        studies, features, targets = load_split(plan, "train")
        source = Path(plan["source_experiment"])
        keys = [[row["subject_id"], row["study_id"]] for row in studies]
        if keys != read_json(source / "oof_study_ids.json") or not np.array_equal(targets, np.load(source / "oof_targets.npy", allow_pickle=False)):
            raise ValueError("incumbent OOF studies or targets are misaligned")
        baseline = np.load(source / "oof_mean_probabilities.npy", allow_pickle=False)
        legacy = legacy_oof(plan, studies)
        folds = _fold_memberships(studies, read_json(plan["partition"])["subject_partition"])
        subjects = [row["subject_id"] for row in studies]
        fits = FitCache(output / "fits", features, targets, folds, subjects, _sha256(output / "plan.json"))
        stage("nested_patient_cv", completed_outer_folds=0, total_outer_folds=4)

        def progress(rows):
            _json_write(output / "nested_progress.json", rows)
            stage("nested_patient_cv", completed_outer_folds=len(rows), total_outer_folds=4)

        nested, outer_reports = nested_oof(targets, folds, fits.predict, progress)
        np.save(output / "nested_probabilities.npy", nested, allow_pickle=False)
        _json_write(output / "oof_study_ids.json", keys)
        gate = training_gate(targets, baseline, nested, folds, legacy)
        _json_write(output / "nested_report.json", {"outer_selection": outer_reports, "gate": gate,
                    "caveat": plan["caveat"], "fit_count_this_process": fits.fit_count,
                    "cache_hits_this_process": fits.cache_hits,
                    "unique_registered_heads": len(list((output / "fits").rglob("fit.json")))})
        if not gate["passed"]:
            finish("retain incumbent; nested selector did not pass the training gate")
            return

        stage("select_final_config")
        candidate_rows = []
        for config in CONFIGS:
            predictions = np.full_like(targets, np.nan)
            for fold in FOLDS:
                train_folds = tuple(other for other in FOLDS if other != fold)
                held = np.flatnonzero(folds == fold)
                predictions[held] = fits.predict(train_folds, config, held)
            np.save(output / f"final_cv_{config_name(config)}.npy", predictions, allow_pickle=False)
            candidate_rows.append({"config": dict(config), **score(targets, predictions)})
        chosen = select_config(candidate_rows)
        _json_write(output / "final_selection.json", {"candidates": candidate_rows, "selected": chosen,
                    "caveat": "Full-train CV used for final parameter choice, not unbiased evaluation."})
        if chosen["macro_auc"] - gate["baseline"]["macro_auc"] < 0.001:
            finish("retain incumbent; final configuration CV gain below fixed threshold")
            return
        stage("fit_final_head")
        head_path = output / "final_head.npz"
        recipe_path = output / "recipe.json"
        if recipe_path.exists():
            recipe = read_json(recipe_path)
            if recipe["config"] != chosen["config"] or recipe["head_sha256"] != _sha256(head_path) or recipe["source_plan_sha256"] != _sha256(output / "plan.json"):
                raise ValueError("registered final recipe or head changed")
            head = _load_head(head_path)
        else:
            if head_path.exists():
                raise ValueError("uncommitted final head; inspect before resuming")
            payload = fit_linear_probe(features, targets, **chosen["config"], max_iter=MAX_ITER, seed=SEED)
            _save_head(head_path, payload)
            head = _load_head(head_path)
            if not np.array_equal(predict_linear_probe(payload, features), predict_linear_probe(head, features)):
                raise ValueError("final saved head changed predictions")
            recipe = {"mode": "mean", "alpha": 1.0, "config": chosen["config"],
                      "head_file": head_path.name, "head_sha256": _sha256(head_path),
                      "max_iter": MAX_ITER, "seed": SEED, "source_plan_sha256": _sha256(output / "plan.json")}
            _json_write(recipe_path, recipe)
        recipe_sha = _sha256(output / "recipe.json")

        stage("validate_fixed_candidate")
        valid_studies, valid_features, valid_targets = load_split(plan, "valid")
        current_valid, _, _ = load_submission(source / "candidate_valid.csv", valid_studies)
        older_valid, _, _ = load_submission(plan["legacy_valid"], valid_studies)
        candidate = predict_linear_probe(head, valid_features)
        valid_csv = output / "candidate_valid.csv"
        write_submission(valid_csv, valid_studies, candidate)
        replay, _, _ = load_submission(valid_csv, valid_studies)
        if not np.array_equal(candidate, replay):
            raise ValueError("candidate validation CSV does not replay exactly")
        comparisons = {name: compare_arrays([row["subject_id"] for row in valid_studies],
                       valid_targets, old, candidate, bootstrap=1000, seed=SEED)
                       for name, old in (("incumbent", current_valid), ("legacy", older_valid))}
        promoted = validation_gate(comparisons)
        _json_write(output / "validation_report.json", {"comparisons": comparisons, "promoted": promoted,
                    "recipe_sha256": recipe_sha, "head_sha256": recipe["head_sha256"],
                    "candidate_valid_csv_sha256": _sha256(valid_csv), "new_online_score": None,
                    "caveat": "Single fixed candidate; historical validation reuse; descriptive intervals only."})
        if not promoted:
            finish("retain incumbent; fixed validation gate failed")
            return

        stage("predict_test_from_cache")
        test_studies, test_features, _ = load_split(plan, "test")
        test_probabilities = predict_linear_probe(head, test_features)
        csv_path = output / "submission_test_1414_cxrlt_bottleneck_20261108.csv"
        write_submission(csv_path, test_studies, test_probabilities)
        verification = validate_submission(csv_path, test_studies)
        reloaded, _, _ = load_submission(csv_path, test_studies)
        if not np.array_equal(test_probabilities, reloaded):
            raise ValueError("test CSV does not replay exactly")
        _json_write(output / "submission_verification.json", {**verification, "sha256": _sha256(csv_path),
                    "online_score": None, "replay_max_error": 0.0})
        finish("promote locally; fixed validation gate passed", promoted=True)
    except Exception as error:
        state.update(status="failed", updated_utc=now(), error=repr(error), traceback=traceback.format_exc())
        _json_write(output / "experiment.json", state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-experiment", type=Path, default=ROOT / "outputs/cxrlt-supervised-study-probe-20261108")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = safe_output(args.output)
    if not args.resume:
        output = prepare(output, args.source_experiment)
    if args.prepare_only:
        print(json.dumps({"status": "prepared", "output": str(output)}), flush=True)
        return
    snapshot = output / "source"
    if ROOT != snapshot:
        import subprocess
        raise SystemExit(subprocess.call([sys.executable, "-u", str(snapshot / "scripts/run_bottleneck_study_probe.py"),
                                         "--output", str(output), "--resume"]))
    run(output)


if __name__ == "__main__":
    main()
