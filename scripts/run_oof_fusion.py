"""Fit a preregistered train-only OOF ensemble, then audit the locked recipe.

The active grouped Rad-DINO experiment remains unchanged. This runner waits for
it, reuses A/raw OOF predictions, and serializes subsequent GPU jobs. The fusion
protocol must be registered before the shared audit has been opened. Neither
official valid/test nor legacy full-train local checkpoints are inputs.
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
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cxr.data import _read_metadata
from cxr.metrics import _matrix, _rank_auc, evaluate
from cxr.oof_fusion import apply_simplex_fusion, fit_simplex_fusion, uniform_simplex_recipe
from cxr.patient_splits import audit_records, partition_records
from cxr.safety import fresh_run_directory, safe_output
from scripts.run_aggressive_experiment import atomic_json, file_sha256
from scripts.run_grouped_generalization import (
    SNAPSHOT_FILES, environment_versions, expected_studies, process_alive,
)

PUBLIC_BRANCHES = (
    "densenet121", "rad_dino_vitB_768", "medmae_vitb_nih_B_768_s2",
    "medmae_vitb_nih_B_768_s2_seed1337", "russellala_convnext",
)
EXTRA_SOURCES = (
    "scripts/run_oof_fusion.py", "scripts/predict_grouped_public.py",
    "scripts/eval_chest_xray_bench.py", "cxr/oof_fusion.py",
)
FIT_PARAMETERS = {
    "seed": 20261106, "pair_count": 32768, "temperature": 0.1,
    "l2": 0.1, "max_iter": 300, "tolerance": 1e-8,
}
SAVED_FILES = ("report.json", "study_ids.json", "study_targets.npy", "study_probabilities.npy")


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_unopened_audit(source):
    state = read_json(Path(source) / "experiment.json")
    if state.get("audit_opened") or (Path(source) / "audit_comparison.json").exists():
        raise ValueError("fusion must be preregistered before any source audit is opened")
    return state


def register_plan(output, source, plan):
    """Persist the hypothesis before checking the final audit boundary."""
    atomic_json(output / "plan.json", plan)
    state = {"status": "registering", "created_utc": now(),
             "plan_sha256": file_sha256(output / "plan.json"),
             "source_audit_unopened_at_registration": False,
             "audit_opened": False, "stages": {}}
    atomic_json(output / "experiment.json", state)
    try:
        require_unopened_audit(source)
    except BaseException as error:
        state.update(status="registration_rejected", error=f"{type(error).__name__}: {error}")
        atomic_json(output / "experiment.json", state)
        raise
    state.update(status="prepared", source_audit_unopened_at_registration=True,
                 registered_utc=now())
    atomic_json(output / "experiment.json", state)


def public_inputs(medical_root):
    """Pin existing public files against the previously recorded manifest."""
    repositories = read_json(ROOT / "docs/public_model_manifest.json")["repositories"]
    required = {
        "itsomk/chexpert-densenet121": ["config.json", "pytorch_model.safetensors"],
        "RussellALA/convnext-chexpert-seed1": ["config.json", "model.safetensors"],
        "mamounyosef/chest-xray-bench": ["modeling.py", *[
            f"{branch}/{name}" for branch in PUBLIC_BRANCHES[1:4]
            for name in ("config.json", "model.safetensors")]],
    }
    fingerprints = {}
    mappings = {}
    labels = [
        "Enlarged Cardiomediastinum", "Pneumothorax", "Consolidation", "Pneumonia",
        "Edema", "Cardiomegaly", "Atelectasis", "Lung Opacity", "Pleural Effusion", "No Finding",
    ]
    for repo, names in required.items():
        info = repositories[repo]
        directory = Path(medical_root) / Path(info["directory"]).name
        for name in names:
            path = directory / name
            if file_sha256(path) != info["files"][name]:
                raise ValueError(f"public checkpoint differs from pinned source: {repo}/{name}")
            fingerprints[str(path.resolve())] = info["files"][name]
        if "densenet" in repo:
            cfg = read_json(directory / "config.json")
            if not all(label in cfg["labels"] for label in labels):
                raise ValueError("DenseNet lacks required published labels")
            mappings["densenet121"] = list(range(10))
        elif "convnext" in repo:
            cfg = read_json(directory / "config.json")
            ordered = [cfg["id2label"][str(i)] for i in range(len(cfg["id2label"]))]
            mappings["russellala_convnext"] = [labels.index(label) for label in ordered]
        else:
            for branch in PUBLIC_BRANCHES[1:4]:
                mappings[branch] = [labels.index(label) for label in read_json(directory / branch / "config.json")["tasks"]]
    return fingerprints, mappings


def prepare(cfg):
    source = cfg.source.resolve(strict=True)
    source_state = require_unopened_audit(source)
    source_plan = read_json(source / "plan.json")
    if file_sha256(source / "plan.json") != source_state["plan_sha256"]:
        raise ValueError("source training plan changed")
    data_root = Path(source_plan["data_root"]).resolve(strict=True)
    medical_root = cfg.medical_root.resolve(strict=True)
    public_files, mappings = public_inputs(medical_root)
    output = fresh_run_directory(cfg.output, data_root)
    for name in ("logs", "runs", "components"):
        (output / name).mkdir()
    source_hashes = {}
    for relative in dict.fromkeys((*SNAPSHOT_FILES, *EXTRA_SOURCES)):
        original = ROOT / relative
        expected = source_plan["source_snapshot"].get(relative)
        if expected is not None and file_sha256(original) != expected:
            raise ValueError("existing locked training source was modified")
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
        source_hashes[relative] = file_sha256(destination)
    # Recheck at the registration boundary, after any slower file hashing.
    require_unopened_audit(source)
    submission = ROOT / "outputs/submission_test_1414_best_20261105.csv"
    plan = {
        "schema_version": 1, "created_utc": now(), "project_root": str(ROOT),
        "source_experiment": str(source), "source_plan_sha256": file_sha256(source / "plan.json"),
        "partition_sha256": file_sha256(source / "partition.json"),
        "data_root": str(data_root), "medical_root": str(medical_root),
        "train_metadata_sha256": source_plan["train_metadata_sha256"],
        "source_snapshot": source_hashes, "public_files": public_files,
        "environment_versions": environment_versions(), "device": cfg.device,
        "model_order": ["rad_dino_A_raw", *PUBLIC_BRANCHES], "class_indices": mappings,
        "public_batch_size": 4, "public_workers": 0,
        "main_model": "fixed A/raw; no A/B or EMA selection for fusion",
        "fit_parameters": FIT_PARAMETERS,
        "fit_objective": "mean sampled positive-negative logistic ranking loss + 0.5*l2*distance_squared_to_uniform_available_weights",
        "weight_constraints": "nonnegative per-class simplex; unsupported model/class has exact zero weight",
        "transforms": "untransformed probabilities; image mean within study; no TTA; no calibration or threshold fitting",
        "optimizer": "deterministic projected gradient with backtracking; uniform initialization; fixed stopping limits",
        "selection_policy": "one fixed fit; no subset, hyperparameter, or meta-CV search; OOF fit metrics are training diagnostics",
        "comparison_family": ["source development-selected training change versus A/raw", "fixed OOF learned fusion versus A/raw"],
        "audit_rule": {"confidence_level": 0.975, "family_size": 2, "family_alpha": 0.05,
                       "correction": "Bonferroni for two prespecified contrasts; recompute source interval at 97.5%",
                       "bootstrap_repeats": 4000, "seed": 20261106,
                       "minimum_defined_replicates": 3800, "minimum_gain": 0.001},
        "audit_policy": "register before source audit; fit uses development only; recipe hash before own audit access; both contrasts reported regardless of signs; no retuning",
        "uniform_fusion_policy": "descriptive comparator only; never selected as a fallback after audit",
        "legacy_local_checkpoints_allowed": False, "official_valid_read": False, "test_read": False,
        "public_pretraining_overlap": "unknown; patient grouping does not resolve public pretraining overlap",
        "existing_submission": {"path": str(submission), "sha256": file_sha256(submission)} if submission.exists() else None,
        "scope": "fresh development-trained models; audit is not the old 0.820325 validation cohort or the hidden test set",
    }
    register_plan(output, source, plan)
    return output


def load_locked(output):
    state = read_json(output / "experiment.json")
    if state.get("source_audit_unopened_at_registration") is not True:
        raise ValueError("fusion protocol registration did not complete before source audit access")
    if file_sha256(output / "plan.json") != state["plan_sha256"]:
        raise ValueError("fusion plan changed")
    plan = read_json(output / "plan.json")
    if environment_versions() != plan["environment_versions"]:
        raise ValueError("locked execution environment changed")
    source = Path(plan["source_experiment"])
    for name, expected in (("plan.json", plan["source_plan_sha256"]),
                           ("partition.json", plan["partition_sha256"])):
        if file_sha256(source / name) != expected:
            raise ValueError("source experiment plan or partition changed")
    source_plan = read_json(source / "plan.json")
    for name, expected in source_plan["public_files"].items():
        if file_sha256(Path(source_plan["model_dir"]) / name) != expected:
            raise ValueError("main model public initialization changed")
    if file_sha256(source / "image_inventory.json") != source_plan["image_inventory_sha256"]:
        raise ValueError("source image inventory changed")
    if file_sha256(Path(plan["data_root"]) / "train.csv") != plan["train_metadata_sha256"]:
        raise ValueError("training metadata changed")
    for relative, expected in plan["source_snapshot"].items():
        if file_sha256(output / "source" / relative) != expected:
            raise ValueError(f"fusion source snapshot changed: {relative}")
        if file_sha256(ROOT / relative) != expected:
            raise ValueError("executing fusion source differs from its lock; resume using the immutable snapshot")
    for path, expected in plan["public_files"].items():
        if file_sha256(path) != expected:
            raise ValueError(f"public initialization changed: {path}")
    return plan, state


def verify_source_stage(source, source_state, name):
    entry = source_state["stages"].get(name)
    if not entry or entry["status"] != "complete":
        raise ValueError(f"source stage has not completed: {name}")
    folder = Path(source) / "runs" / name
    for relative, expected in entry["artifacts"].items():
        if file_sha256(folder / relative) != expected:
            raise ValueError(f"source stage artifact changed: {name}/{relative}")
    return folder


def aligned_component(folder, records, class_indices):
    expected, targets = expected_studies(records)
    keys = [tuple(map(str, item)) for item in read_json(folder / "study_ids.json")]
    if len(keys) != len(set(keys)) or set(keys) != set(expected):
        raise ValueError("component study coverage differs from requested patients")
    indices = list(class_indices)
    if not indices or len(set(indices)) != len(indices) or any(type(j) is not int or j not in range(10) for j in indices):
        raise ValueError("invalid class mapping")
    probabilities = np.load(folder / "study_probabilities.npy", allow_pickle=False)
    saved_targets = np.load(folder / "study_targets.npy", allow_pickle=False)
    if probabilities.shape != (len(keys), len(indices)) or saved_targets.shape != (len(keys), 10):
        raise ValueError("component shape mismatch")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("component probabilities must be finite in [0, 1]")
    positions = {key: i for i, key in enumerate(keys)}
    order = [positions[key] for key in expected]
    if not np.array_equal(saved_targets[order], targets):
        raise ValueError("component targets disagree with locked training metadata")
    report = read_json(folder / "report.json")
    if report.get("local_label_indices", list(range(10))) != indices:
        raise ValueError("component report class mapping differs from locked plan")
    full = np.full((len(expected), 10), np.nan, dtype=np.float64)
    full[:, indices] = probabilities[order]
    return expected, targets, full


def collect_oof(plan, source_state, records, partition):
    keys, values, targets = [], [], []
    lineage = []
    for fold in range(3):
        fit, held = partition_records(records, partition, fold)
        folder = verify_source_stage(plan["source_experiment"], source_state, f"cv_A_fold{fold}")
        report = read_json(folder / "report.json")
        if (report.get("phase"), report.get("fold"), report.get("head_lr")) != ("cv", fold, 1e-3):
            raise ValueError("OOF source is not the fixed A fold configuration")
        if report.get("partition_sha256") != plan["partition_sha256"]:
            raise ValueError("OOF source used a different patient partition")
        current_keys, y, probabilities = aligned_component(folder / "raw", held, list(range(10)))
        fit_subjects = {row["subject_id"] for row in fit}
        if fit_subjects & {key[0] for key in current_keys}:
            raise ValueError("OOF fit and prediction patients overlap")
        keys.extend(current_keys)
        targets.extend(y)
        values.extend(probabilities)
        lineage.append({"fold": fold, "fit_patients": len(fit_subjects), "held_studies": len(current_keys),
                        "raw_probabilities_sha256": file_sha256(folder / "raw/study_probabilities.npy")})
    expected, expected_targets = expected_studies(audit_records(records, partition)[0])
    if len(keys) != len(set(keys)) or set(keys) != set(expected):
        raise ValueError("OOF sources do not cover development exactly")
    positions = {key: i for i, key in enumerate(keys)}
    order = [positions[key] for key in expected]
    if not np.array_equal(np.asarray(targets)[order], expected_targets):
        raise ValueError("OOF labels changed")
    return expected, expected_targets, np.asarray(values)[order], lineage


def bootstrap_family(subject_ids, targets, baseline, candidates, *, repeats=4000, seed=20261106,
                     confidence=0.975, minimum_defined=3800):
    """Paired patient clusters; keep only replicates with all ten AUCs defined."""
    y = _matrix(targets, "targets", binary=True)
    b = _matrix(baseline, "baseline", rows=len(y))
    if not candidates or len(subject_ids) != len(y) or not len(y):
        raise ValueError("aligned nonempty comparison arrays required")
    if repeats < 1 or not 1 <= minimum_defined <= repeats or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap limits")
    arrays = {name: _matrix(value, name, rows=len(y)) for name, value in candidates.items()}
    groups = {}
    for index, subject in enumerate(subject_ids):
        groups.setdefault(str(subject), []).append(index)
    clusters = [np.asarray(value, dtype=np.int64) for value in groups.values()]

    def aucs(labels, predictions):
        values = [_rank_auc(labels[:, j], predictions[:, j]) for j in range(10)]
        return None if any(value is None for value in values) else float(np.mean(values))

    base_point = aucs(y, b)
    if base_point is None:
        raise ValueError("all ten classes require both outcomes")
    deltas = {name: [] for name in arrays}
    rng = np.random.default_rng(seed)
    for repeat in range(repeats):
        sampled = np.concatenate([clusters[index] for index in rng.integers(len(clusters), size=len(clusters))])
        labels = y[sampled]
        base = aucs(labels, b[sampled])
        if base is None:
            continue
        for name, probabilities in arrays.items():
            deltas[name].append(aucs(labels, probabilities[sampled]) - base)
        if (repeat + 1) % 500 == 0:
            print(f"patient bootstrap {repeat + 1}/{repeats}", flush=True)
    tail = (1 - confidence) * 50
    comparisons = {}
    for name, probabilities in arrays.items():
        point = aucs(y, probabilities)
        values = deltas[name]
        interval = np.percentile(values, [tail, 100 - tail]).tolist() if len(values) >= minimum_defined else None
        comparisons[name] = {"baseline": base_point, "candidate": point, "delta": point - base_point,
                             "paired_patient_interval": interval, "defined_replicates": len(values)}
    return {"comparisons": comparisons, "bootstrap": {"repeats": repeats, "seed": seed,
            "confidence_level": confidence, "family_size": 2, "correction": "Bonferroni",
            "unit": "patient", "paired": True, "patients": len(clusters),
            "undefined_policy": "discard a draw if any of ten classes lacks either outcome; no replacement draws",
            "minimum_defined_replicates": minimum_defined},
            "caveat": "Approximate percentile intervals conditional on fitted models; no upstream-overlap guarantee or hidden-test score."}


def run_stage(output, state, name, command, folder, artifacts):
    load_locked(output)
    previous = state["stages"].get(name)
    if previous and previous.get("pid") and process_alive(previous["pid"]):
        raise RuntimeError(f"stage process is still active; refusing a duplicate: {name}")
    if previous and previous["status"] == "complete":
        if previous["command"] != command or any(file_sha256(folder / key) != value for key, value in previous["artifacts"].items()):
            raise ValueError(f"cached stage differs from locked computation: {name}")
        return folder
    if folder.exists() and any(folder.iterdir()):
        raise ValueError(f"incomplete output must be inspected before resuming: {folder}")
    folder.parent.mkdir(parents=True, exist_ok=True)
    stage = {"status": "running", "started_utc": now(), "command": command}
    state["stages"][name] = stage
    state.update(status="running", current_stage=name)
    atomic_json(output / "experiment.json", state)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    print(f"Starting {name}", flush=True)
    with (output / "logs" / f"{name}.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=output / "source", env=env, stdout=log, stderr=subprocess.STDOUT)
        stage["pid"] = child.pid
        atomic_json(output / "experiment.json", state)
        code = child.wait()
    stage.update(ended_utc=now(), returncode=code)
    if code:
        stage["status"] = "failed"
        raise RuntimeError(f"stage {name} failed; see logs/{name}.log")
    stage["artifacts"] = {relative: file_sha256(folder / relative) for relative in artifacts}
    stage["status"] = "complete"
    atomic_json(output / "experiment.json", state)
    print(f"Completed {name}", flush=True)
    return folder


def public_stage(output, state, plan, phase, branch):
    folder = output / "components" / phase / branch
    command = [sys.executable, "-u", str(output / "source/scripts/predict_grouped_public.py"),
               "--data-root", plan["data_root"], "--medical-root", plan["medical_root"],
               "--partition", str(Path(plan["source_experiment"]) / "partition.json"),
               "--phase", phase, "--branch", branch, "--output", str(folder),
               "--device", plan["device"], "--batch-size", str(plan["public_batch_size"]),
               "--workers", str(plan["public_workers"])]
    return run_stage(output, state, f"{phase}_{branch}", command, folder, SAVED_FILES)


def lock_recipe(output, state, plan, records, partition, source_state):
    destination = output / "fit"
    if state.get("recipe_sha256"):
        for name, expected in state["fit_artifacts"].items():
            if file_sha256(destination / name) != expected:
                raise ValueError("locked fusion fit artifact changed")
        return read_json(destination / "recipe.json")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("incomplete fusion fit must be inspected")
    development, _ = audit_records(records, partition)
    keys, targets, base, lineage = collect_oof(plan, source_state, records, partition)
    tensors = [base]
    for branch in PUBLIC_BRANCHES:
        folder = public_stage(output, state, plan, "development", branch)
        tensors.append(aligned_component(folder, development, plan["class_indices"][branch])[2])
    tensor = np.stack(tensors, axis=1)
    recipe = fit_simplex_fusion(tensor, targets, **plan["fit_parameters"])
    recipe.update(model_order=plan["model_order"], plan_sha256=state["plan_sha256"],
                  fitted_utc=now(), training_studies=len(keys),
                  training_patients=len({key[0] for key in keys}))
    fused = apply_simplex_fusion(tensor, recipe)
    uniform = apply_simplex_fusion(tensor, uniform_simplex_recipe(tensor))
    destination.mkdir()
    atomic_json(destination / "recipe.json", recipe)
    atomic_json(destination / "study_ids.json", keys)
    np.save(destination / "study_targets.npy", targets)
    np.save(destination / "component_probabilities.npy", tensor)
    np.save(destination / "study_probabilities.npy", fused)
    atomic_json(destination / "report.json", {
        "status": "complete", "interpretation": "fusion fitting data; these are not independent validation metrics",
        "raw_A_oof_metrics": evaluate(targets, base), "uniform_oof_metrics": evaluate(targets, uniform),
        "fitted_oof_metrics": evaluate(targets, fused), "oof_lineage": lineage,
        "audit_used_for_fit": False, "official_valid_read": False, "test_read": False})
    state["fit_artifacts"] = {str(path.name): file_sha256(path) for path in destination.iterdir()}
    state["recipe_sha256"] = file_sha256(destination / "recipe.json")
    state["status"] = "recipe_locked"
    atomic_json(output / "experiment.json", state)
    print("Fusion recipe locked from development OOF predictions", flush=True)
    return recipe


def audit_baseline(output, state, plan, source_state):
    source = Path(plan["source_experiment"])
    if source_state["stages"].get("audit_A_raw", {}).get("status") == "complete":
        folder = verify_source_stage(source, source_state, "audit_A_raw")
        return folder / "raw"
    source_plan = read_json(source / "plan.json")
    folder = output / "runs/audit_A_raw"
    command = [sys.executable, "-u", str(output / "source/scripts/train_rad_dino_grouped.py"),
               "--data-root", plan["data_root"], "--model-dir", source_plan["model_dir"],
               "--partition", str(source / "partition.json"), "--output", str(folder),
               "--phase", "audit", "--head-lr", "0.001", "--epochs", "3", "--batch-size", "2",
               "--accumulation-steps", "8", "--workers", "4", "--seed", "2026",
               "--ema-decay", "0.999", "--evaluation-variants", "raw", "--device", plan["device"]]
    artifacts = ["report.json", *[f"raw/{name}" for name in (*SAVED_FILES, "last.pt")]]
    run_stage(output, state, "audit_refit_A_raw", command, folder, artifacts)
    return folder / "raw"


def run(output):
    plan, state = load_locked(output)
    if state["status"] == "complete":
        for relative, expected in state["final_artifacts"].items():
            if file_sha256(output / relative) != expected:
                raise ValueError("completed fusion result changed")
        for relative, expected in state["fit_artifacts"].items():
            if file_sha256(output / "fit" / relative) != expected:
                raise ValueError("completed fusion recipe or fit data changed")
        print("Fusion experiment already complete", flush=True)
        return
    if state.get("runner_pid") and state["runner_pid"] != os.getpid() and process_alive(state["runner_pid"]):
        raise RuntimeError("another fusion runner is already active")
    state.update(runner_pid=os.getpid(), status="waiting_for_source")
    atomic_json(output / "experiment.json", state)
    source = Path(plan["source_experiment"])
    previous = None
    while True:
        source_state = read_json(source / "experiment.json")
        status = source_state["status"]
        if status == "complete":
            break
        if status == "failed":
            raise RuntimeError("source training failed; preserve its logs and fix the dependency first")
        pid = source_state.get("runner_pid")
        if not pid or not process_alive(pid):
            raise RuntimeError("source training is incomplete and its runner is not active")
        current = (status, source_state.get("current_stage"))
        if current != previous:
            print(f"Waiting for existing GPU experiment: {current}", flush=True)
            previous = current
        atomic_json(output / "dependency_progress.json", {"checked_utc": now(), "status": status,
                    "current_stage": source_state.get("current_stage"), "GPU_jobs_started_here": False})
        time.sleep(30)
    plan, _ = load_locked(output)
    partition = read_json(source / "partition.json")
    records = _read_metadata(Path(plan["data_root"]) / "train.csv")
    recipe = lock_recipe(output, state, plan, records, partition, source_state)
    if file_sha256(output / "fit/recipe.json") != state["recipe_sha256"]:
        raise ValueError("fusion recipe changed before audit")
    state.update(audit_opened=True, status="auditing", audit_started_utc=now())
    atomic_json(output / "experiment.json", state)
    _, audit = audit_records(records, partition)
    base_folder = audit_baseline(output, state, plan, source_state)
    keys, targets, base = aligned_component(base_folder, audit, list(range(10)))
    tensors = [base]
    for branch in PUBLIC_BRANCHES:
        folder = public_stage(output, state, plan, "audit", branch)
        tensors.append(aligned_component(folder, audit, plan["class_indices"][branch])[2])
    tensor = np.stack(tensors, axis=1)
    fused = apply_simplex_fusion(tensor, recipe)
    uniform = apply_simplex_fusion(tensor, uniform_simplex_recipe(tensor))
    candidates = {"oof_learned_fusion": fused}
    source_selection = read_json(source / "selection.json")
    if file_sha256(source / "selection.json") != source_state["selection_sha256"]:
        raise ValueError("source development selection changed")
    selected = source_selection["selected"]
    if selected["qualifies_for_audit"]:
        folder = verify_source_stage(source, source_state, f"audit_{selected['config']}_{selected['variant']}")
        candidates["source_training_change"] = aligned_component(folder / selected["variant"], audit, list(range(10)))[2]
    rule = plan["audit_rule"]
    comparison = bootstrap_family([key[0] for key in keys], targets, base, candidates,
                                  repeats=rule["bootstrap_repeats"], seed=rule["seed"],
                                  confidence=rule["confidence_level"], minimum_defined=rule["minimum_defined_replicates"])
    for item in comparison["comparisons"].values():
        interval = item["paired_patient_interval"]
        item["supports_prespecified_gain"] = item["delta"] >= rule["minimum_gain"] and interval is not None and interval[0] > 0
    audit_output = output / "audit"
    audit_output.mkdir(exist_ok=True)
    atomic_json(audit_output / "study_ids.json", keys)
    np.save(audit_output / "study_targets.npy", targets)
    np.save(audit_output / "baseline_probabilities.npy", base)
    np.save(audit_output / "study_probabilities.npy", fused)
    np.save(audit_output / "uniform_probabilities.npy", uniform)
    atomic_json(audit_output / "comparison.json", comparison)
    preserved = plan["existing_submission"]
    if preserved and file_sha256(preserved["path"]) != preserved["sha256"]:
        raise ValueError("previously delivered test CSV changed during the experiment")
    report = {
        "status": "complete", "completed_utc": now(), "recipe_sha256": state["recipe_sha256"],
        "baseline_metrics": evaluate(targets, base), "fusion_metrics": evaluate(targets, fused),
        "uniform_metrics_descriptive_only": evaluate(targets, uniform), "paired_comparisons": comparison,
        "source_training_contrast": "reported above" if selected["qualifies_for_audit"] else "not eligible under its locked development rule; unused family slot retained",
        "scope": plan["scope"], "public_pretraining_overlap": "unknown",
        "official_valid_read": False, "test_read": False, "existing_submission_changed": False,
        "post_audit_tuning_allowed": False,
    }
    atomic_json(output / "report.json", report)
    state.update(status="complete", runner_pid=None, current_stage=None, ended_utc=now(),
                 final_artifacts={str(path.relative_to(output)): file_sha256(path)
                                  for path in [output / "report.json", *audit_output.iterdir()]})
    atomic_json(output / "experiment.json", state)
    print(json.dumps(report), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=ROOT / "outputs/generalization-rad-dino-20261105-v2")
    p.add_argument("--medical-root", type=Path, default=ROOT / "weights/medical")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    cfg = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    output = safe_output(cfg.output) if cfg.resume else prepare(cfg)
    if not cfg.prepare_only:
        try:
            run(output)
        except BaseException as error:
            state = read_json(output / "experiment.json")
            if state.get("runner_pid") == os.getpid():
                state.update(status="failed", runner_pid=None, failed_utc=now(), error=f"{type(error).__name__}: {error}")
                atomic_json(output / "experiment.json", state)
            raise


if __name__ == "__main__":
    main()
