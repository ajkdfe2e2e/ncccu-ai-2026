"""Run a preregistered train-only, patient-grouped Rad-DINO A/B experiment.

Development folds choose the configuration. Only a qualifying candidate and a
fresh baseline are then trained on development patients and compared once on
the locked audit patients. Official valid/test are never read by this runner.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import version
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

from cxr.data import _read_metadata, read_records
from cxr.metrics import evaluate
from cxr.patient_splits import make_patient_partition, partition_records, audit_records
from cxr.safety import fresh_run_directory, safe_output
from scripts.compare_runs import compare_arrays
from scripts.run_aggressive_experiment import atomic_json, file_sha256

PUBLIC_SHA = "a81fea07272e6cff836b838ba4c12dd26d128b5fb16b3eb2e8c089a4a5ca14f8"
SNAPSHOT_FILES = (
    "scripts/run_grouped_generalization.py", "scripts/train_rad_dino_grouped.py",
    "scripts/run_rad_dino_finetune.py", "scripts/run_aggressive_experiment.py",
    "scripts/compare_runs.py", "cxr/__init__.py", "cxr/data.py",
    "cxr/metrics.py", "cxr/optimization.py", "cxr/patient_splits.py", "cxr/safety.py",
    "cxr/submission.py",
)
CONFIGS = {"A": 1e-3, "B": 1e-4}
EXPECTED_COUNTS = {"images": 30344, "studies": 18513, "patients": 11055}


def now():
    return datetime.now(timezone.utc).isoformat()


def environment_versions():
    return {"python": sys.version, **{name: version(name) for name in (
        "numpy", "torch", "transformers", "Pillow")}}


def process_alive(pid):
    """Check liveness without Windows os.kill(pid, 0), which can terminate it."""
    if sys.platform == "win32":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, check=True,
        )
        return any(len(row) > 1 and row[1] == str(pid)
                   for row in csv.reader(result.stdout.splitlines()))
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def expected_studies(records):
    result = {}
    for row in records:
        result.setdefault((str(row["subject_id"]), str(row["study_id"])), row["labels"])
    keys = sorted(result)
    return keys, np.asarray([result[key] for key in keys], dtype=np.float64)


def aligned_predictions(folder, records):
    folder = Path(folder)
    expected, targets = expected_studies(records)
    ids = [tuple(map(str, row)) for row in json.loads(
        (folder / "study_ids.json").read_text(encoding="utf-8"))]
    probabilities = np.load(folder / "study_probabilities.npy", allow_pickle=False)
    saved_targets = np.load(folder / "study_targets.npy", allow_pickle=False)
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("prediction coverage differs from locked evaluation studies")
    if probabilities.shape != (len(ids), 10) or saved_targets.shape != probabilities.shape:
        raise ValueError("prediction or target shape mismatch")
    order_by_id = {key: i for i, key in enumerate(ids)}
    order = [order_by_id[key] for key in expected]
    probabilities, saved_targets = probabilities[order], saved_targets[order]
    if not np.array_equal(targets, saved_targets):
        raise ValueError("prediction labels differ from locked metadata")
    metrics = evaluate(targets, probabilities)
    if metrics["macro_auc_defined_classes"] != 10:
        raise ValueError("all ten classes must have a defined AUC")
    return expected, targets, probabilities, metrics


def qualifying(fold_deltas, minimum_gain=0.001):
    values = np.asarray(fold_deltas, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("three finite paired fold deltas required")
    return bool(values.mean() >= minimum_gain and np.count_nonzero(values > 0) >= 2)


def choose_configuration(scores, minimum_gain=0.001):
    """Choose A/B raw first, then its EMA, using only development folds."""
    required = {"A/raw", "A/ema", "B/raw", "B/ema"}
    if set(scores) != required:
        raise ValueError("all preregistered CV configurations are required")
    vectors = {key: np.asarray(value, dtype=np.float64) for key, value in scores.items()}
    if any(value.shape != (3,) or not np.isfinite(value).all()
           or ((value < 0) | (value > 1)).any() for value in vectors.values()):
        raise ValueError("each configuration needs three finite AUCs in [0, 1]")
    ab_delta = vectors["B/raw"] - vectors["A/raw"]
    config = "B" if qualifying(ab_delta, minimum_gain) else "A"
    ema_delta = vectors[f"{config}/ema"] - vectors[f"{config}/raw"]
    variant = "ema" if qualifying(ema_delta, minimum_gain) else "raw"
    total_delta = vectors[f"{config}/{variant}"] - vectors["A/raw"]
    return {
        "config": config, "variant": variant,
        "head_lr": CONFIGS[config], "A_to_B_fold_deltas": ab_delta.tolist(),
        "raw_to_ema_fold_deltas": ema_delta.tolist(),
        "candidate_vs_baseline_fold_deltas": total_delta.tolist(),
        "qualifies_for_audit": (config, variant) != ("A", "raw")
            and qualifying(total_delta, minimum_gain),
        "rule": {"minimum_mean_auc_gain": minimum_gain,
                 "minimum_positive_folds": 2, "folds": 3,
                 "tie_preference": "A/raw", "epoch_policy": "fixed_final_epoch"},
    }


def stage_command(plan, output, config, phase, fold=None, variant=None):
    name = f"cv_{config}_fold{fold}" if phase == "cv" else f"audit_{config}_{variant}"
    command = [
        sys.executable, "-u", str(output / "source/scripts/train_rad_dino_grouped.py"),
        "--data-root", plan["data_root"], "--model-dir", plan["model_dir"],
        "--partition", str(output / "partition.json"),
        "--output", str(output / "runs" / name), "--phase", phase,
        "--head-lr", str(CONFIGS[config]), "--epochs", str(plan["epochs"]),
        "--batch-size", str(plan["batch_size"]),
        "--accumulation-steps", str(plan["accumulation_steps"]),
        "--workers", str(plan["workers"]), "--seed", str(plan["training_seed"]),
        "--ema-decay", str(plan["ema_decay"]), "--device", plan["device"],
    ]
    if fold is not None:
        command += ["--fold", str(fold)]
    if variant is not None:
        command += ["--evaluation-variants", variant]
    return name, command


def stage_artifacts(folder):
    paths = [folder / "report.json"]
    report = json.loads(paths[0].read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise ValueError("stage did not complete")
    for variant in report["variants"]:
        paths += [folder / variant / name for name in (
            "report.json", "study_ids.json", "study_probabilities.npy",
            "study_targets.npy", "last.pt",
        )]
    return {str(path.relative_to(folder)): file_sha256(path) for path in paths}


def run_stage(plan, output, state, config, phase, fold=None, variant=None):
    load_locked(output)
    name, command = stage_command(plan, output, config, phase, fold, variant)
    folder = output / "runs" / name
    previous = state["stages"].get(name)
    if previous and previous.get("status") == "complete":
        if stage_artifacts(folder) != previous["artifacts"]:
            raise ValueError(f"completed stage artifacts changed: {name}")
        print(f"Reusing verified stage {name}", flush=True)
        return folder
    if folder.exists() and any(folder.iterdir()):
        raise ValueError(f"incomplete stage exists; preserve it and investigate: {folder}")
    stage = {"status": "running", "started_utc": now(), "command": command}
    state["stages"][name] = stage
    state.update(status="running", current_stage=name)
    atomic_json(output / "experiment.json", state)
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
                       TOKENIZERS_PARALLELISM="false")
    print(f"Starting {name}", flush=True)
    with (output / "logs" / f"{name}.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   cwd=output / "source", env=environment)
        stage["pid"] = process.pid
        atomic_json(output / "experiment.json", state)
        result = process.wait()
    stage.update(returncode=result, ended_utc=now())
    if result:
        stage["status"] = "failed"
        state["status"] = "failed"
        atomic_json(output / "experiment.json", state)
        raise RuntimeError(f"stage failed: {name}; see {output / 'logs' / (name + '.log')}")
    stage["artifacts"] = stage_artifacts(folder)
    stage["status"] = "complete"
    atomic_json(output / "experiment.json", state)
    print(f"Completed {name}", flush=True)
    return folder


def prepare(cfg):
    data_root = cfg.data_root.resolve(strict=True)
    model_dir = cfg.model_dir.resolve(strict=True)
    checkpoint = model_dir / "pytorch_model.bin"
    if file_sha256(checkpoint) != PUBLIC_SHA:
        raise ValueError("public checkpoint does not match the pinned initialization")
    records = read_records(data_root / "train.csv", data_root / "train")
    counts = {"images": len(records), "studies": len({row["study_id"] for row in records}),
              "patients": len({row["subject_id"] for row in records})}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"training dataset counts differ from preregistration: {counts}")
    partition = make_patient_partition(records, seed=20261105, audit_fraction=0.2, folds=3)
    partition.update(train_metadata_sha256=file_sha256(data_root / "train.csv"),
                     public_checkpoint_sha256=PUBLIC_SHA)
    output = fresh_run_directory(cfg.output, data_root)
    (output / "logs").mkdir()
    (output / "runs").mkdir()
    snapshots = {}
    for relative in SNAPSHOT_FILES:
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
        snapshots[relative] = file_sha256(target)
    inventory = {}
    for row in records:
        stat = Path(row["path"]).stat()
        inventory[row["dicom_id"]] = {
            "path": row["path"], "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        }
    atomic_json(output / "image_inventory.json", inventory)
    partition.update(image_inventory_path=str(output / "image_inventory.json"),
                     image_inventory_sha256=file_sha256(output / "image_inventory.json"))
    atomic_json(output / "partition.json", partition)
    plan = {
        "schema_version": 1, "created_utc": now(), "data_root": str(data_root),
        "model_dir": str(model_dir), "train_metadata_sha256": partition["train_metadata_sha256"],
        "public_files": {name: file_sha256(model_dir / name) for name in (
            "pytorch_model.bin", "config.json", "preprocessor_config.json", "modeling_rad_dino.py")},
        "environment_versions": environment_versions(),
        "partition_sha256": file_sha256(output / "partition.json"),
        "image_inventory_sha256": file_sha256(output / "image_inventory.json"),
        "image_identity_policy": "resolved path, size and nanosecond mtime checked per stage; not content hashes",
        "source_snapshot": snapshots, "epochs": 3, "batch_size": 2,
        "accumulation_steps": 8, "workers": cfg.workers, "device": cfg.device,
        "training_seed": 2026, "ema_decay": 0.999, "configs": CONFIGS,
        "selection_rule": {"minimum_mean_auc_gain": 0.001, "minimum_positive_folds": 2},
        "audit_rule": {"minimum_auc_gain": 0.001, "paired_patient_ci95_lower_above": 0,
                       "bootstrap_repeats": 1000, "seed": 20261105},
        "evaluation_policy": "fixed final epoch; no interim validation; raw A/B then winner EMA",
        "legacy_local_checkpoints_allowed": False, "official_valid_read": False, "test_read": False,
        "public_pretraining_overlap": "unknown; patient grouping cannot resolve upstream overlap",
        "audit_policy": "one locked comparison; not opened if CV candidate does not qualify",
    }
    atomic_json(output / "plan.json", plan)
    state = {"status": "prepared", "started_utc": now(), "stages": {},
             "plan_sha256": file_sha256(output / "plan.json"), "audit_opened": False}
    atomic_json(output / "experiment.json", state)
    print(json.dumps({"output": str(output), "partition_summary": partition["summary"]}), flush=True)
    return output


def load_locked(output):
    state = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
    if file_sha256(output / "plan.json") != state["plan_sha256"]:
        raise ValueError("locked plan changed")
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    if environment_versions() != plan["environment_versions"]:
        raise ValueError("Python or primary package versions differ from the locked environment")
    if file_sha256(output / "partition.json") != plan["partition_sha256"]:
        raise ValueError("locked partition changed")
    if file_sha256(output / "image_inventory.json") != plan["image_inventory_sha256"]:
        raise ValueError("locked image inventory changed")
    if file_sha256(Path(plan["data_root"]) / "train.csv") != plan["train_metadata_sha256"]:
        raise ValueError("training metadata changed")
    for relative, expected in plan["source_snapshot"].items():
        if file_sha256(output / "source" / relative) != expected:
            raise ValueError("source snapshot changed")
        if file_sha256(ROOT / relative) != expected:
            raise ValueError("live orchestration differs from the locked source; use the source snapshot to resume")
    for relative, expected in plan["public_files"].items():
        if file_sha256(Path(plan["model_dir"]) / relative) != expected:
            raise ValueError("public initialization changed")
    return plan, state, json.loads((output / "partition.json").read_text(encoding="utf-8"))


def run(output):
    plan, state, partition = load_locked(output)
    if state["status"] == "complete":
        print("Experiment already complete", flush=True)
        return
    if state.get("runner_pid"):
        old_pid = state["runner_pid"]
        if old_pid != os.getpid() and process_alive(old_pid):
            raise RuntimeError("another runner is already active")
    state["runner_pid"] = os.getpid()
    records = _read_metadata(Path(plan["data_root"]) / "train.csv")
    scores = {f"{config}/{variant}": [] for config in CONFIGS for variant in ("raw", "ema")}
    all_predictions = {key: [] for key in scores}
    all_keys, all_targets = [], []
    for fold in range(3):
        _, held = partition_records(records, partition, fold)
        expected, targets = expected_studies(held)
        all_keys.extend(expected)
        all_targets.append(targets)
        for config in CONFIGS:
            folder = run_stage(plan, output, state, config, "cv", fold)
            for variant in ("raw", "ema"):
                _, _, probabilities, metrics = aligned_predictions(folder / variant, held)
                key = f"{config}/{variant}"
                scores[key].append(metrics["macro_auc"])
                all_predictions[key].append(probabilities)
        atomic_json(output / "development_progress.json", {"completed_folds": fold + 1, "scores": scores})
    development = output / "development"
    development.mkdir(exist_ok=True)
    targets = np.concatenate(all_targets)
    atomic_json(development / "study_ids.json", all_keys)
    np.save(development / "study_targets.npy", targets)
    oof_metrics = {}
    for key, values in all_predictions.items():
        probabilities = np.concatenate(values)
        np.save(development / (key.replace("/", "_") + "_probabilities.npy"), probabilities)
        oof_metrics[key] = evaluate(targets, probabilities)
    selected = choose_configuration(scores)
    selection = {"selected": selected, "fold_scores": scores,
                 "fold_mean": {key: float(np.mean(value)) for key, value in scores.items()},
                 "pooled_oof_metrics": oof_metrics,
                 "caveat": "development scores choose parameters; final audit is separate"}
    atomic_json(output / "selection.json", selection)
    state["selection_sha256"] = file_sha256(output / "selection.json")
    if not selected["qualifies_for_audit"]:
        report = {"status": "complete", "outcome": "no_candidate_passed_development_rule",
                  "audit_opened": False, "selection": selection,
                  "official_valid_read": False, "test_read": False,
                  "existing_submission_changed": False}
    else:
        state["status"] = "audit_candidate_locked"
        state["audit_opened"] = True
        atomic_json(output / "experiment.json", state)
        _, audit = audit_records(records, partition)
        baseline_folder = run_stage(plan, output, state, "A", "audit", variant="raw")
        chosen_folder = run_stage(plan, output, state, selected["config"], "audit", variant=selected["variant"])
        keys, targets, baseline, baseline_metrics = aligned_predictions(baseline_folder / "raw", audit)
        _, _, candidate, candidate_metrics = aligned_predictions(chosen_folder / selected["variant"], audit)
        comparison = compare_arrays([key[0] for key in keys], targets, baseline, candidate,
                                    bootstrap=1000, seed=20261105)
        comparison["caveat"] = (
            "Single preregistered patient-audit comparison after development-only selection. "
            "Intervals describe paired patient resampling conditional on fitted models; "
            "upstream pretraining overlap remains unknown. No post-audit retuning."
        )
        atomic_json(output / "audit_comparison.json", comparison)
        delta = comparison["macro"]["auc"]["delta"]
        interval = comparison["macro"]["auc"]["ci95_percentile"]
        accepted = delta >= 0.001 and interval is not None and interval[0] > 0
        report = {"status": "complete", "outcome": "audit_gain_confirmed" if accepted else "audit_gain_not_confirmed",
                  "audit_opened": True, "selection": selection,
                  "audit_baseline": baseline_metrics, "audit_candidate": candidate_metrics,
                  "audit_delta": delta, "audit_ci95": interval,
                  "official_valid_read": False, "test_read": False,
                  "existing_submission_changed": False,
                  "scope": "fresh A/B training protocol comparison; not comparison against legacy seven-model ensemble"}
    report["completed_utc"] = now()
    atomic_json(output / "report.json", report)
    state.update(status="complete", current_stage=None, ended_utc=now(), runner_pid=None)
    atomic_json(output / "experiment.json", state)
    print(json.dumps(report), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path(r"G:\cxr_first(1)\cxr_first"))
    p.add_argument("--model-dir", type=Path, default=ROOT / "weights/medical/rad-dino-chexpert-upstream")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    cfg = p.parse_args(argv)
    if cfg.workers < 0:
        p.error("workers must be nonnegative")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if cfg.resume:
        output = safe_output(cfg.output, cfg.data_root)
    else:
        output = prepare(cfg)
    if not cfg.prepare_only:
        run(output)


if __name__ == "__main__":
    main()
