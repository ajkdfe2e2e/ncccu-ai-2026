"""Complete patient-OOF inputs, then fit one trust-anchored legacy ensemble.

All local predictions used for fitting exclude the predicted patient's labels
from base-model training. Previously opened cohorts are training data here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cxr.anchored_fusion import apply_public_weights, fit_anchored_public, flatten_legacy_recipe
from cxr.data import _read_metadata
from cxr.metrics import evaluate
from cxr.patient_splits import audit_records, partition_records
from cxr.safety import fresh_run_directory, safe_output
from cxr.submission import validate_submission, write_submission
from scripts.compare_runs import compare_arrays, load_submission
from scripts.predict_best_ensemble import load_scores, replay_recipe, unique_studies
from scripts.run_grouped_generalization import environment_versions, process_alive
from scripts.train_legacy_public_fusion import (
    FIT_PARAMETERS, MODEL_ORDER, OLD_CSV, OLD_HASH, PUBLIC_ORDER, VALID_FOLDERS,
    load_public_training, read_json, sha256, write_json,
)

SPECS = (("cv", 0, "cv_A_fold0"), ("cv", 1, "cv_A_fold1"),
         ("cv", 2, "cv_A_fold2"), ("audit", None, "audit_A_raw"))
FILES = ("report.json", "study_ids.json", "study_targets.npy", "study_probabilities.npy")
CAVEAT = (
    "Base predictions are patient-disjoint, but meta-model fit metrics are training diagnostics. "
    "Previously opened audit is training data. Existing full-train deployment models are "
    "reconstructions' counterparts, not identical fold models. Official validation and legacy "
    "prior were used in past selection; comparison intervals are descriptive, not unbiased "
    "generalization evidence. Public-pretraining patient overlap remains unknown."
)


def now():
    return datetime.now(timezone.utc).isoformat()


def split_records(records, partition, phase, fold):
    if phase == "cv" and fold in range(3):
        train, held = partition_records(records, partition, fold)
    elif phase == "audit" and fold is None:
        train, held = audit_records(records, partition)
    else:
        raise ValueError("invalid patient stage")
    if not train or not held or ({r["subject_id"] for r in train} & {r["subject_id"] for r in held}):
        raise ValueError("base fitting and evaluation patients overlap or are empty")
    return train, held


def aligned_component(folder, studies, *, filename="study_probabilities.npy"):
    keys = [tuple(row) for row in read_json(folder / "study_ids.json")]
    expected = [(r["subject_id"], r["study_id"]) for r in studies]
    values = np.load(folder / filename, allow_pickle=False)
    labels = np.load(folder / "study_targets.npy", allow_pickle=False)
    if (len(keys) != len(set(keys)) or set(keys) != set(expected)
            or values.shape != (len(keys), 10) or labels.shape != values.shape
            or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any()):
        raise ValueError("local component coverage, shape or scores differ")
    positions = {key: i for i, key in enumerate(keys)}
    order = [positions[key] for key in expected]
    if not np.array_equal(labels[order], np.asarray([r["labels"] for r in studies])):
        raise ValueError("local component labels differ from locked train")
    return values[order]


def verify_local_report(report, records, partition, phase, fold, train_sha, partition_sha):
    train, held = split_records(records, partition, phase, fold)
    if (report.get("status") != "complete" or report.get("smoke") is not False
            or report.get("phase") != phase or report.get("fold") != fold
            or report.get("train_metadata_sha256") != train_sha
            or report.get("partition_sha256") != partition_sha
            or report.get("train_subjects") != sorted({r["subject_id"] for r in train})
            or report.get("held_subjects") != sorted({r["subject_id"] for r in held})):
        raise ValueError("local component provenance differs from requested patient split")
    return unique_studies(held)


def weight_delta(anchor, components, prior, trained):
    before = apply_public_weights(components, prior)
    after = apply_public_weights(components, trained)
    if np.asarray(anchor).shape != before.shape:
        raise ValueError("anchor shape mismatch")
    result = np.asarray(anchor, dtype=np.float64).copy()
    for j in range(10):
        delta = np.asarray(trained)[:, j] - np.asarray(prior)[:, j]
        used = np.flatnonzero(delta)
        if len(used):
            result[:, j] += components[:, used, j] @ delta[used]
    if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        raise ValueError("invalid full-fusion candidate scores")
    if not np.allclose(result, after, atol=1e-12, rtol=0):
        raise ValueError("saved anchor does not match the locked complete-source prior")
    return result


def gate_passed(comparison):
    auc = comparison["macro"]["auc"]
    interval = auc["ci95_percentile"]
    return (comparison["macro_auc_defined_classes"] == 10 and auc["delta"] >= 0.001
            and interval is not None and interval[0] > 0
            and auc["defined_replicates"] >= 950)


def source_locks(source, source_state):
    locks = {}
    for _, _, stage_name in SPECS:
        stage = source_state["stages"][stage_name]
        if stage["status"] != "complete":
            raise ValueError("required local source stage is incomplete")
        folder = source / "runs" / stage_name
        for name in ("report.json", *[f"raw/{f}" for f in (*FILES, "last.pt")]):
            path = folder / name
            if sha256(path) != stage["artifacts"][str(Path(name))]:
                raise ValueError(f"source artifact changed: {path}")
            locks[str(path)] = sha256(path)
    return locks


def prepare(args):
    data_root = args.data_root.resolve(strict=True)
    source = ROOT / "outputs/generalization-rad-dino-20261105-v2"
    state = read_json(source / "experiment.json")
    source_plan = read_json(source / "plan.json")
    if state.get("status") != "complete":
        raise ValueError("source grouped experiment is not complete")
    if sha256(source / "plan.json") != state["plan_sha256"]:
        raise ValueError("source plan fingerprint differs")
    if sha256(source / "partition.json") != source_plan["partition_sha256"]:
        raise ValueError("patient partition differs")
    if sha256(data_root / "train.csv") != source_plan["train_metadata_sha256"]:
        raise ValueError("training manifest differs")
    test_manifest = ROOT / "outputs/best-ensemble-test-20261105/run_manifest.json"
    old_manifest = read_json(test_manifest)
    recipe_path = ROOT / "docs/best_ensemble_recipe_20261105.json"
    if sha256(OLD_CSV) != OLD_HASH or sha256(recipe_path) != old_manifest["recipe_sha256"]:
        raise ValueError("old deployed recipe or CSV differs")
    studies, _, public, locks = load_public_training(
        data_root, ROOT / "outputs/oof-fusion-20261106", old_manifest)
    locks.update(source_locks(source, state))
    for relative, expected in old_manifest["model_fingerprints"].items():
        path = ROOT / relative
        if sha256(path) != expected:
            raise ValueError(f"deployed source fingerprint differs: {path}")
        locks[str(path)] = expected
    code = (
        "scripts/run_legacy_full_oof.py", "scripts/train_resnet_grouped.py",
        "scripts/predict_rad_dino_oof_tta.py", "scripts/train_legacy_public_fusion.py",
        "scripts/train_rad_dino_grouped.py", "scripts/run_rad_dino_finetune.py",
        "scripts/run_grouped_generalization.py", "scripts/run_aggressive_experiment.py",
        "scripts/predict_best_ensemble.py", "scripts/compare_runs.py", "train.py",
        *[str(path.relative_to(ROOT)) for path in (ROOT / "cxr").glob("*.py")],
    )
    for path in (
        recipe_path, OLD_CSV, test_manifest, source / "plan.json", source / "partition.json",
        source / "experiment.json", source / "image_inventory.json",
        ROOT / "docs/LEGACY_FULL_OOF_PLAN_20261107.md", ROOT / "weights/resnet18-f37072fd.pth",
        data_root / "train.csv", data_root / "valid.csv", data_root / "test.csv",
        ROOT / "outputs/dataset-verification-20261104-180020/valid.csv",
        *[ROOT / relative for relative in code],
    ):
        locks[str(path)] = sha256(path)
    for folder in (*VALID_FOLDERS, "outputs/rad-dino-finetuned-valid-independent",
                   "outputs/rad-dino-best-tta-valid-20261105", "outputs/best-ensemble-improvement-20261105"):
        for filename in ("report.json", "study_ids.json", "study_probabilities.npy"):
            path = ROOT / folder / filename
            locks[str(path)] = sha256(path)
    # Test caches are frozen as files only; no test images or labels enter fitting.
    for path in (ROOT / "outputs/best-ensemble-test-20261105/components").rglob("*"):
        if path.is_file() and path.suffix in (".json", ".npy", ".csv"):
            locks[str(path)] = sha256(path)
    output = fresh_run_directory(args.output, data_root)
    (output / "logs").mkdir()
    (output / "runs").mkdir()
    for relative in code:
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    prior = flatten_legacy_recipe(read_json(recipe_path), MODEL_ORDER)
    np.save(output / "public_training_probabilities.npy", public)
    plan = {
        "created_utc": now(), "data_root": str(data_root), "source": str(source),
        "partition": str(source / "partition.json"), "model_dir": source_plan["model_dir"],
        "model_order": MODEL_ORDER, "prior": prior.tolist(), "fit_parameters": FIT_PARAMETERS,
        "input_locks": locks, "train_studies": len(studies), "train_patients": 11055,
        "previous_audit_is_training": True, "fit_metrics_are_training_diagnostics": True,
        "resnet_epochs": 5, "resnet_batch_size": 8,
        "rad_tta_batch_size": source_plan["batch_size"],
        "gate": {"minimum_gain": 0.001, "ci95_lower_above": 0,
                 "patient_bootstrap": 1000, "minimum_defined_replicates": 950, "seed": 20261107},
        "official_valid_labels_read": False, "caveat": CAVEAT,
        "environment_versions": environment_versions(),
        "public_matrix_sha256": sha256(output / "public_training_probabilities.npy"),
    }
    write_json(output / "plan.json", plan)
    write_json(output / "experiment.json", {"status": "prepared", "created_utc": now(),
               "plan_sha256": sha256(output / "plan.json"), "stages": {}})
    print(f"Prepared complete-fusion experiment: {output}", flush=True)
    return output


def verify_locks(plan):
    if environment_versions() != plan["environment_versions"]:
        raise ValueError("locked software environment changed")
    for path, expected in plan["input_locks"].items():
        if sha256(path) != expected:
            raise ValueError(f"locked experiment input changed: {path}")


def stage_artifacts(folder, kind):
    paths = [folder / name for name in FILES]
    paths.append(folder / ("final.pt" if kind == "resnet" else "original_study_probabilities.npy"))
    if read_json(folder / "report.json").get("status") != "complete":
        raise ValueError("local stage did not complete")
    return {str(path.relative_to(folder)): sha256(path) for path in paths}


def stage_folder(output, state, name):
    folder = Path(state["stages"].get(name, {}).get("output", output / "runs" / name))
    if folder.resolve().parent != (output / "runs").resolve():
        raise ValueError("stage output is outside the experiment runs directory")
    return folder


def run_stage(output, plan, state, kind, phase, fold, source_name):
    name = f"{kind}_{source_name}"
    previous = state["stages"].get(name, {})
    folder = stage_folder(output, state, name)
    if previous.get("status") == "complete":
        if previous["artifacts"] != stage_artifacts(folder, kind):
            raise ValueError("completed stage artifact changed")
        return folder
    if previous.get("pid") and process_alive(previous["pid"]):
        raise RuntimeError(f"stage {name} still has a live process; do not duplicate it")
    attempts = list(previous.get("previous_attempts", []))
    if folder.exists() and any(folder.iterdir()) and (kind == "tta" or not (folder / "resume.pt").is_file()):
        attempts.append({"output": str(folder), "status": previous.get("status", "interrupted")})
        attempt = 2
        while True:
            candidate = output / "runs" / f"{name}_attempt{attempt}"
            if not candidate.exists():
                folder = candidate
                break
            attempt += 1
    command = [sys.executable, "-u", str(ROOT / "scripts" / (
        "train_resnet_grouped.py" if kind == "resnet" else "predict_rad_dino_oof_tta.py")),
        "--data-root", plan["data_root"], "--partition", plan["partition"],
        "--output", str(folder), "--phase", phase, "--device", "cuda"]
    if fold is not None:
        command += ["--fold", str(fold)]
    if kind == "resnet":
        command += ["--pretrained", str(ROOT / "weights/resnet18-f37072fd.pth"),
                    "--cache", str(ROOT / ".cache/legacy-resnet224")]
        if folder.exists() and any(folder.iterdir()):
            command += ["--resume"]
    else:
        command += ["--source-stage", str(Path(plan["source"]) / "runs" / source_name),
                    "--model-dir", plan["model_dir"], "--batch-size", str(plan["rad_tta_batch_size"])]
    stage = {"status": "running", "started_utc": now(), "command": command,
             "output": str(folder), "previous_attempts": attempts}
    state["stages"][name] = stage
    state.update(status="running", current_stage=name, updated_utc=now())
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
                       TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4")
    with (output / "logs" / f"{name}.log").open("a", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        stage["pid"] = child.pid
        write_json(output / "experiment.json", state)
        print(f"Started {name}, pid={child.pid}", flush=True)
        result = child.wait()
    stage.update(returncode=result, ended_utc=now())
    if result:
        stage["status"] = "failed"
        write_json(output / "experiment.json", state)
        raise RuntimeError(f"stage failed: {name}; see its preserved log")
    stage.update(status="complete", artifacts=stage_artifacts(folder, kind))
    write_json(output / "experiment.json", state)
    return folder


def locked_recipe(output, plan_sha, state):
    final = output / "trained_recipe.json"
    pending = output / "trained_recipe.pending.json"
    path = final if final.exists() else pending if pending.exists() else None
    if path is None:
        if state.get("recipe_sha256"):
            raise ValueError("registered recipe is missing")
        return None
    fitted = read_json(path)
    actual = sha256(path)
    expected = state.get("recipe_sha256")
    if (fitted.get("plan_sha256") != plan_sha
            or fitted.get("official_valid_labels_read") is not False
            or (expected is not None and actual != expected)
            or (path == final and expected is None)):
        raise ValueError("previously fitted recipe differs from locked state")
    # Journal the training-only recipe before publishing it. Interrupted writes
    # recover from the pending file without refitting after validation is opened.
    if expected is None:
        state.update(recipe_sha256=actual, status="validation_guardrail", updated_utc=now())
        write_json(output / "experiment.json", state)
    if path == pending:
        pending.replace(final)
    return fitted


def fit_complete(output, plan, state):
    records = _read_metadata(Path(plan["data_root"]) / "train.csv")
    studies = unique_studies(records)
    positions = {(r["subject_id"], r["study_id"]): i for i, r in enumerate(studies)}
    targets = np.asarray([r["labels"] for r in studies])
    matrix = np.full((len(studies), len(MODEL_ORDER), 10), np.nan)
    if sha256(output / "public_training_probabilities.npy") != plan["public_matrix_sha256"]:
        raise ValueError("saved public training matrix changed")
    matrix[:, 3:] = np.load(output / "public_training_probabilities.npy", allow_pickle=False)
    partition = read_json(plan["partition"])
    train_sha = sha256(Path(plan["data_root"]) / "train.csv")
    partition_sha = sha256(plan["partition"])
    seen = set()
    for phase, fold, source_name in SPECS:
        _, held = split_records(records, partition, phase, fold)
        held_studies = unique_studies(held)
        keys = {(r["subject_id"], r["study_id"]) for r in held_studies}
        if seen & keys:
            raise ValueError("held groups contain duplicate studies")
        seen.update(keys)
        order = [positions[(r["subject_id"], r["study_id"])] for r in held_studies]
        for kind in ("resnet", "tta"):
            folder = stage_folder(output, state, f"{kind}_{source_name}")
            verify_local_report(read_json(folder / "report.json"), records, partition,
                                phase, fold, train_sha, partition_sha)
            if stage_artifacts(folder, kind) != state["stages"][f"{kind}_{source_name}"]["artifacts"]:
                raise ValueError("completed base predictions changed")
            if kind == "resnet":
                matrix[order, 0] = aligned_component(folder, held_studies)
            else:
                matrix[order, 1] = aligned_component(folder, held_studies, filename="original_study_probabilities.npy")
                matrix[order, 2] = aligned_component(folder, held_studies)
    if seen != set(positions) or not np.isfinite(matrix[:, :3]).all():
        raise ValueError("local patient OOF predictions do not cover complete train")
    verify_locks(plan)
    np.save(output / "oof_component_probabilities.npy", matrix)
    write_json(output / "oof_study_ids.json", [[r["subject_id"], r["study_id"]] for r in studies])
    fitted = locked_recipe(output, sha256(output / "plan.json"), state)
    if fitted is None:
        fitted = fit_anchored_public(matrix, targets, plan["prior"], **plan["fit_parameters"])
        fitted.update(model_order=MODEL_ORDER, plan_sha256=sha256(output / "plan.json"),
                      official_valid_labels_read=False, caveat=CAVEAT)
        write_json(output / "trained_recipe.pending.json", fitted)
        fitted = locked_recipe(output, sha256(output / "plan.json"), state)
    write_json(output / "meta_training_diagnostics.json", {
        "scope": "meta-model training metrics on patient-disjoint base predictions; not independent meta-OOF",
        "old_prior": evaluate(targets, apply_public_weights(matrix, plan["prior"])),
        "trained": evaluate(targets, apply_public_weights(matrix, fitted["weights"])), "caveat": CAVEAT})
    return fitted


def deployed_sources(studies, split):
    if split == "valid":
        sources = {
            "resnet18_baseline": load_submission(ROOT / "outputs/dataset-verification-20261104-180020/valid.csv", studies)[0],
            "rad_dino_finetuned": load_scores(ROOT / "outputs/rad-dino-finetuned-valid-independent", studies),
            "rad_dino_tta": load_scores(ROOT / "outputs/rad-dino-best-tta-valid-20261105", studies),
        }
        sources.update({name: load_scores(ROOT / folder, studies) for name, folder in zip(PUBLIC_ORDER, VALID_FOLDERS)})
    else:
        components = ROOT / "outputs/best-ensemble-test-20261105/components"
        sources = {
            "resnet18_baseline": load_submission(components / "resnet18/test.csv", studies)[0],
            "rad_dino_finetuned": load_scores(components / "rad_dino", studies, "original_study_probabilities.npy"),
            "rad_dino_tta": load_scores(components / "rad_dino", studies),
            "densenet121_chexpert": load_scores(components / "densenet121", studies),
            "russellala_convnext": load_scores(components / "russellala_convnext", studies),
        }
        sources.update({name: load_scores(components / name / name, studies) for name in PUBLIC_ORDER[2:]})
    return sources


def compare_and_export(output, plan, state, fitted):
    data_root = Path(plan["data_root"])
    records = _read_metadata(data_root / "valid.csv")
    studies = unique_studies(records)
    y = np.asarray([r["labels"] for r in studies])
    legacy = read_json(ROOT / "docs/best_ensemble_recipe_20261105.json")
    sources = deployed_sources(studies, "valid")
    _, replayed = replay_recipe(sources, legacy)
    anchor = load_scores(ROOT / "outputs/best-ensemble-improvement-20261105", studies)
    if not np.array_equal(replayed, anchor):
        raise ValueError("complete validation sources do not replay the exact legacy anchor")
    components = np.stack([sources[name] for name in MODEL_ORDER], axis=1)
    candidate = weight_delta(anchor, components, plan["prior"], fitted["weights"])
    comparison = compare_arrays([r["subject_id"] for r in studies], y, anchor, candidate,
                                bootstrap=1000, seed=20261107)
    write_submission(output / "candidate_valid.csv", studies, candidate)
    reloaded, gates, _ = load_submission(output / "candidate_valid.csv", studies)
    if not np.array_equal(candidate, reloaded) or not np.all(gates == 0.5):
        raise ValueError("candidate validation CSV failed exact roundtrip")
    write_json(output / "validation_comparison.json", comparison)
    passed = gate_passed(comparison)
    report = {"status": "complete", "created_utc": now(), "guardrail_passed": passed,
              "candidate_promoted": False, "comparison": comparison["macro"],
              "old_online_score_user_reported": 78.53, "candidate_online_score": None,
              "recipe_sha256": state["recipe_sha256"], "caveat": CAVEAT}
    if passed:
        test_records = _read_metadata(data_root / "test.csv")
        test_studies = unique_studies(test_records)
        test_sources = deployed_sources(test_studies, "test")
        _, old_test = replay_recipe(test_sources, legacy)
        delivered, _, _ = load_submission(OLD_CSV, test_studies)
        if not np.array_equal(old_test, delivered):
            raise ValueError("legacy test submission no longer replays exactly")
        values = weight_delta(delivered, np.stack([test_sources[name] for name in MODEL_ORDER], axis=1),
                              plan["prior"], fitted["weights"])
        destination = output / "submission_test_1414_legacy_full_oof_20261107.csv"
        if destination.exists():
            raise FileExistsError(destination)
        temporary = destination.with_suffix(".csv.tmp")
        write_submission(temporary, test_studies, values)
        validation = validate_submission(temporary, test_records)
        roundtrip, gates, _ = load_submission(temporary, test_studies)
        if not np.array_equal(roundtrip, values) or not np.all(gates == 0.5):
            raise ValueError("new test CSV failed exact numeric roundtrip")
        temporary.rename(destination)
        report.update(candidate_promoted=True, csv=str(destination), csv_sha256=sha256(destination),
                      csv_validation=validation, old_test_replay_exact=True)
    else:
        report["decision"] = "retain old ensemble; complete OOF-trained candidate failed fixed guardrail"
    verify_locks(plan)
    if sha256(output / "trained_recipe.json") != state["recipe_sha256"]:
        raise ValueError("recipe changed after validation opened")
    write_json(output / "report.json", report)
    state.update(status="complete", current_stage=None, guardrail_passed=passed, updated_utc=now())
    write_json(output / "experiment.json", state)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"G:\cxr_first(1)\cxr_first"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = (safe_output(args.output, args.data_root).resolve(strict=True) if args.resume else prepare(args))
    plan, state = read_json(output / "plan.json"), read_json(output / "experiment.json")
    if sha256(output / "plan.json") != state["plan_sha256"]:
        raise ValueError("experiment plan changed")
    verify_locks(plan)
    if args.prepare_only or state["status"] == "complete":
        return
    old_pid = state.get("runner_pid")
    if old_pid and old_pid != os.getpid() and process_alive(old_pid):
        raise RuntimeError("this experiment already has a live runner")
    state.update(runner_pid=os.getpid(), updated_utc=now())
    write_json(output / "experiment.json", state)
    try:
        for kind in ("resnet", "tta"):
            for phase, fold, source_name in SPECS:
                run_stage(output, plan, state, kind, phase, fold, source_name)
        fitted = fit_complete(output, plan, state)
        compare_and_export(output, plan, state, fitted)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", updated_utc=now())
        write_json(output / "experiment.json", state)
        raise


if __name__ == "__main__":
    main()
