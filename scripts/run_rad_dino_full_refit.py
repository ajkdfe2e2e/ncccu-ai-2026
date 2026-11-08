"""Run the locked Rad-DINO A/EMA full-train refit and test submission."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PLAN_ROOT = ROOT / "outputs" / "generalization-rad-dino-20261105-v2"
DEFAULT_OUTPUT = ROOT / "outputs" / "rad-dino-A-ema-fulltrain-20261107"
DEFAULT_CSV = ROOT / "outputs" / "submission_test_1414_rad_dino_A_ema_fulltrain_20261107.csv"
SNAPSHOT_FILES = (
    "scripts/run_rad_dino_full_refit.py",
    "scripts/train_rad_dino_grouped.py",
    "scripts/run_rad_dino_finetune.py",
    "scripts/run_aggressive_experiment.py",
    "scripts/eval_rad_dino_checkpoint.py",
    "scripts/supervise_grouped_experiments.py",
    "cxr/__init__.py",
    "cxr/data.py",
    "cxr/metrics.py",
    "cxr/optimization.py",
    "cxr/patient_splits.py",
    "cxr/safety.py",
    "cxr/submission.py",
    "cxr/training_state.py",
)
EXPECTED_CSVS = (
    "submission_test_1414_best_20261105.csv",
    "submission_test_1414_rad_dino_A_ema_20261107.csv",
)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def locked_inputs(root=ROOT):
    """Verify the completed selection and hashes without importing training code."""
    base = root / "outputs" / "generalization-rad-dino-20261105-v2"
    plan_path, selection_path = base / "plan.json", base / "selection.json"
    report_path, partition_path = base / "report.json", base / "partition.json"
    inventory_path = base / "image_inventory.json"
    plan, selection = read_json(plan_path), read_json(selection_path)
    report, partition = read_json(report_path), read_json(partition_path)
    selected = selection.get("selected", {})
    if report.get("status") != "complete" or report.get("outcome") != "audit_gain_confirmed":
        raise ValueError("locked generalization experiment is not a completed A/EMA selection")
    if (selected.get("config"), selected.get("variant"), selected.get("head_lr")) != ("A", "ema", 0.001):
        raise ValueError("locked selection is not configuration A / EMA / head_lr 0.001")
    lock_hashes = {
        "plan.json": sha256(plan_path), "selection.json": sha256(selection_path),
        "report.json": sha256(report_path), "partition.json": sha256(partition_path),
        "image_inventory.json": sha256(inventory_path),
    }
    if lock_hashes["partition.json"] != plan.get("partition_sha256"):
        raise ValueError("partition hash differs from locked plan")
    if lock_hashes["image_inventory.json"] != plan.get("image_inventory_sha256"):
        raise ValueError("image inventory hash differs from locked plan")
    if partition.get("train_metadata_sha256") != plan.get("train_metadata_sha256"):
        raise ValueError("train metadata hash differs between plan and partition")
    inventory_decl = partition.get("image_inventory_sha256")
    if inventory_decl != lock_hashes["image_inventory.json"]:
        raise ValueError("partition image inventory hash differs")
    data_root = Path(plan["data_root"])
    model_dir = Path(plan["model_dir"])
    train_metadata = data_root / "train.csv"
    test_metadata = data_root / "test.csv"
    model_checkpoint = model_dir / "pytorch_model.bin"
    if sha256(train_metadata) != plan["train_metadata_sha256"]:
        raise ValueError("train.csv changed since the locked selection")
    if sha256(model_checkpoint) != partition.get("public_checkpoint_sha256"):
        raise ValueError("public model checkpoint changed since the locked selection")
    for name, expected in plan.get("public_files", {}).items():
        if sha256(model_dir / name) != expected:
            raise ValueError(f"locked public model file changed: {name}")
    expected = {"epochs": 3, "batch_size": 2, "accumulation_steps": 8,
                "training_seed": 2026, "ema_decay": 0.999, "device": "cuda"}
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError("locked A/EMA training parameters differ from the requested fixed settings")
    return {
        "plan": plan, "partition": partition, "data_root": str(data_root.resolve()),
        "model_dir": str(model_dir.resolve()), "train_metadata_sha256": plan["train_metadata_sha256"],
        "test_metadata_sha256": sha256(test_metadata),
        "public_checkpoint_sha256": partition["public_checkpoint_sha256"],
        "image_inventory_sha256": lock_hashes["image_inventory.json"],
        "lock_hashes": lock_hashes,
    }


def training_command(python, snapshot, binding, output, resume=False):
    plan = binding["plan"]
    command = [
        str(python), "-u", str(snapshot / "scripts/train_rad_dino_grouped.py"),
        "--data-root", binding["data_root"], "--model-dir", binding["model_dir"],
        "--partition", str(snapshot / "locked/partition.json"),
        "--output", str(output / "train"), "--phase", "full", "--head-lr", "0.001",
        "--epochs", "3", "--batch-size", "2", "--accumulation-steps", "8",
        "--workers", "0", "--seed", "2026", "--ema-decay", "0.999",
        "--evaluation-variants", "raw", "ema", "--device", "cuda",
    ]
    if resume:
        command.append("--resume")
    return command


def evaluation_command(python, snapshot, binding, output):
    return [
        str(python), "-u", str(snapshot / "scripts/eval_rad_dino_checkpoint.py"),
        "--data-root", binding["data_root"], "--model-dir", binding["model_dir"],
        "--split", "test", "--checkpoint", str(output / "train/ema/last.pt"),
        "--output", str(output / "test"), "--batch-size", "2", "--device", "cuda",
    ]


def copy_snapshot(root, output, binding):
    source = output / "source"
    hashes = {}
    for name in SNAPSHOT_FILES:
        src, dst = root / name, source / name
        if not src.is_file():
            raise FileNotFoundError(f"required launch snapshot source is missing: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        hashes[name] = sha256(dst)
    locked = source / "locked"
    locked.mkdir(parents=True, exist_ok=True)
    base = root / "outputs" / "generalization-rad-dino-20261105-v2"
    for name in ("plan.json", "selection.json", "report.json", "partition.json", "image_inventory.json"):
        shutil.copy2(base / name, locked / name)
    manifest = {"created_utc": now(), "files": hashes,
                "locked_evidence": binding["lock_hashes"]}
    atomic_json(source / "snapshot.json", manifest)
    return {**manifest, "manifest_sha256": sha256(source / "snapshot.json")}


def _state_path(output):
    return output / "run.json"


def _record_stage(output, state, name, command):
    stage = {"status": "running", "started_utc": now(), "command": [str(x) for x in command]}
    state["stages"][name] = stage
    state["status"] = name
    state["heartbeat_utc"] = now()
    atomic_json(_state_path(output), state)
    log_path = output / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        kwargs = {"stdin": subprocess.DEVNULL, "stdout": log, "stderr": subprocess.STDOUT,
                  "cwd": str(output / "source"), "close_fds": True}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        process = subprocess.Popen(command, **kwargs)
        stage["pid"] = process.pid
        stage["log"] = str(log_path)
        atomic_json(_state_path(output), state)
        result = process.wait()
    stage["returncode"] = int(result)
    stage["ended_utc"] = now()
    stage["status"] = "complete" if result == 0 else "failed"
    if result:
        state["status"] = "failed"
        state["error"] = f"{name} exited with code {result}; see {log_path}"
    state["heartbeat_utc"] = now()
    atomic_json(_state_path(output), state)
    if result:
        raise RuntimeError(state["error"])
    return stage


def _verify_training(output):
    report_path = output / "train/report.json"
    report = read_json(report_path)
    expected = {"status": "complete", "phase": "full", "train_images": 30344,
                "train_patients": 11055, "independent_validation_metrics": None,
                "eligible_for_oof": False, "includes_former_audit_patients": True,
                "epochs": 3, "head_lr": 0.001, "batch_size": 2,
                "accumulation_steps": 8, "seed": 2026, "ema_decay": 0.999,
                "evaluation_images": 0}
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("full-refit report does not match locked training contract")
    expected_files = (output / "train/ema/last.pt", output / "train/raw/last.pt")
    hashes = {path.relative_to(output).as_posix(): sha256(path) for path in expected_files}
    variants = report.get("variants", {})
    for variant in ("ema", "raw"):
        if variants.get(variant, {}).get("metrics") is not None:
            raise ValueError(f"full-refit {variant} checkpoint unexpectedly has metrics")
        if variants.get(variant, {}).get("checkpoint_sha256") != hashes[f"train/{variant}/last.pt"]:
            raise ValueError(f"full-refit {variant} checkpoint hash mismatch")
    history = report.get("history")
    if (not isinstance(history, list) or len(history) != 3
            or [item.get("epoch") for item in history] != [1, 2, 3]):
        raise ValueError("full-refit report is missing its three-epoch history")
    return {"report_sha256": sha256(report_path), "checkpoint_hashes": hashes}


def _quarantine_test_output(test_output):
    if not test_output.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    target = test_output.with_name(f"{test_output.name}.failed-{stamp}")
    test_output.rename(target)
    return str(target)


def export_submission(output, csv_path, data_root):
    """Build and validate test CSV in this light process; never overwrite."""
    import numpy as np
    from cxr.data import _read_metadata
    from cxr.safety import safe_output
    from cxr.submission import validate_submission, write_submission

    test_dir = output / "test"
    report = read_json(test_dir / "report.json")
    ids = read_json(test_dir / "study_ids.json")
    probabilities = np.load(test_dir / "study_probabilities.npy", allow_pickle=False)
    records = _read_metadata(Path(data_root) / "test.csv")
    ordered_studies = []
    seen = set()
    for record in records:
        key = (record["subject_id"], record["study_id"])
        if key not in seen:
            ordered_studies.append(key)
            seen.add(key)
    if (len(ordered_studies), int(report.get("test_images", -1))) != (1414, 2218) or len(records) != 2218:
        raise ValueError("test metadata/evaluation counts differ from 1414 studies, 14140 rows, 2218 images")
    normalized_ids = [tuple(pair) for pair in ids]
    if normalized_ids != ordered_studies:
        raise ValueError("evaluation study IDs do not cover test studies in source order")
    if probabilities.shape != (1414, 10) or not np.isfinite(probabilities).all():
        raise ValueError("test probability matrix must be finite with shape (1414, 10)")
    if report.get("split") != "test" or report.get("test_studies") != 1414:
        raise ValueError("evaluation report does not describe the expected test split")
    if report.get("test_images") != 2218:
        raise ValueError("evaluation must use the 2218-image test inventory")
    destination = safe_output(csv_path, data_root=data_root,
                              inputs=(test_dir / "study_probabilities.npy", output / "train/ema/last.pt"))
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite submission CSV: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp-{os.getpid()}-{time.time_ns()}")
    study_records = [{"subject_id": subject, "study_id": study} for subject, study in ordered_studies]
    try:
        write_submission(temporary, study_records, probabilities, thresholds=0.5)
        summary = validate_submission(temporary, records)
        if summary != {"valid": True, "studies": 1414, "rows": 14140,
                       "classes_per_study": 10, "default_threshold_rows": 14140}:
            raise ValueError(f"unexpected submission validation result: {summary}")
        with temporary.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            parsed = []
            for index, row in enumerate(reader):
                expected_subject, expected_study = ordered_studies[index // 10]
                if (row["Subject_id"], row["Study_id"], row["Predict_class"], row["Gate"]) != (
                        expected_subject, expected_study, str(index % 10 + 1), "0.5"):
                    raise ValueError(f"submission order or Gate mismatch at row {index + 1}")
                parsed.append(float(row["Probability"]))
        if not np.array_equal(np.asarray(parsed, dtype=np.float64).reshape(1414, 10), probabilities.astype(np.float64)):
            raise ValueError("submission probability float roundtrip changed values")
        os.rename(temporary, destination)
    except BaseException:
        if temporary.exists():
            failed = temporary.with_name(temporary.name + ".failed")
            os.rename(temporary, failed)
        raise
    return {"path": str(destination), "sha256": sha256(destination), "validation": summary}


def _capture_existing_csvs(root):
    found = {}
    for name in EXPECTED_CSVS:
        path = root / "outputs" / name
        if path.is_file():
            found[name] = sha256(path)
    if len(found) != len(EXPECTED_CSVS):
        raise FileNotFoundError("expected baseline and previous A/EMA test CSVs must both be present")
    return found


def run(output=DEFAULT_OUTPUT, csv_path=DEFAULT_CSV, *, resume=False, root=ROOT, python=None):
    from cxr.safety import fresh_run_directory, safe_output
    from scripts.supervise_grouped_experiments import _execution_state, process_alive

    root, output, csv_path = Path(root).resolve(), Path(output).resolve(), Path(csv_path).resolve()
    python = Path(python or (root / ".venv/Scripts/python.exe")).resolve()
    binding = locked_inputs(root)
    safe_output(output, data_root=binding["data_root"])
    safe_output(csv_path, data_root=binding["data_root"], inputs=(output / "train/ema/last.pt",))
    if not python.is_file():
        raise FileNotFoundError(f"project Python environment is missing: {python}")
    if resume:
        if not _state_path(output).is_file():
            raise FileNotFoundError("--resume requires this runner's run.json")
        state = read_json(_state_path(output))
        if state.get("binding") != _binding_record(binding, csv_path):
            raise ValueError("resume binding differs from this locked run")
        pid = state.get("runner_pid")
        if pid and process_alive(pid):
            raise RuntimeError(f"runner PID {pid} is already active")
        for name, stage in state.get("stages", {}).items():
            child_pid = stage.get("pid")
            if stage.get("status") == "running" and child_pid and process_alive(child_pid):
                raise RuntimeError(f"{name} process PID {child_pid} is still active")
        if state.get("status") == "complete":
            _verify_completed_csv(state, csv_path, binding["data_root"])
            if _capture_existing_csvs(root) != state.get("existing_submission_csvs"):
                raise ValueError("a pre-existing submission CSV changed since this run completed")
            return state
        snapshot = state["snapshot"]
        _verify_snapshot(output / "source", snapshot)
        if csv_path.exists():
            _verify_completed_csv(state, csv_path, binding["data_root"])
            return state
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError(f"run directory is not empty; choose --resume or another output: {output}")
        if csv_path.exists():
            raise FileExistsError(f"refusing to overwrite submission CSV: {csv_path}")
        originals = _capture_existing_csvs(root)
        fresh_run_directory(output, binding["data_root"])
        snapshot = copy_snapshot(root, output, binding)
        state = {
            "status": "starting", "runner_pid": os.getpid(), "created_utc": now(),
            "heartbeat_utc": now(), "execution_state_held": False,
            "binding": _binding_record(binding, csv_path), "snapshot": snapshot,
            "existing_submission_csvs": originals, "stages": {},
        }
        atomic_json(_state_path(output), state)
    state["runner_pid"] = os.getpid()
    _execution_state(True)
    state["execution_state_held"] = os.name == "nt"
    atomic_json(_state_path(output), state)
    try:
        train_report = output / "train/report.json"
        if train_report.is_file() and read_json(train_report).get("status") == "complete":
            train_artifacts = _verify_training(output)
        else:
            resume_checkpoint = output / "train/resume.pt"
            if resume and not resume_checkpoint.is_file():
                raise FileNotFoundError("failed full refit has no epoch-boundary resume.pt; refusing a fresh overwrite")
            command = training_command(python, output / "source", binding, output,
                                      resume=bool(resume and resume_checkpoint.is_file()))
            _record_stage(output, state, "training", command)
            train_artifacts = _verify_training(output)
        state["training_artifacts"] = train_artifacts
        state["status"] = "training_complete"
        state["heartbeat_utc"] = now()
        atomic_json(_state_path(output), state)

        test_output = output / "test"
        if _verified_test_output(test_output, state.get("test_artifacts")):
            state["test_output_reused"] = True
        else:
            _verify_test_manifest(binding)
            if test_output.exists():
                quarantined = _quarantine_test_output(test_output)
                state.setdefault("quarantined_test_outputs", []).append(quarantined)
                atomic_json(_state_path(output), state)
            command = evaluation_command(python, output / "source", binding, output)
            for attempt in (1, 2):
                try:
                    _record_stage(output, state, f"test_inference_attempt_{attempt}", command)
                    break
                except RuntimeError:
                    if test_output.exists():
                        quarantined = _quarantine_test_output(test_output)
                        state.setdefault("quarantined_test_outputs", []).append(quarantined)
                        atomic_json(_state_path(output), state)
                    if attempt == 2:
                        raise
        _verify_test_manifest(binding)
        evaluation_report = read_json(test_output / "report.json")
        checkpoint_hash = sha256(output / "train/ema/last.pt")
        if (evaluation_report.get("checkpoint_sha256") != checkpoint_hash
                or evaluation_report.get("split") != "test"
                or evaluation_report.get("horizontal_flip_tta") is not False
                or evaluation_report.get("stored_validation_metrics") is not None):
            raise ValueError("test evaluation provenance differs from the fixed EMA checkpoint contract")
        state["test_artifacts"] = {
            "report_sha256": sha256(test_output / "report.json"),
            "study_ids_sha256": sha256(test_output / "study_ids.json"),
            "study_probabilities_sha256": sha256(test_output / "study_probabilities.npy"),
            "report": {"split": evaluation_report.get("split"),
                       "test_images": evaluation_report.get("test_images"),
                       "test_studies": evaluation_report.get("test_studies")},
        }
        if (evaluation_report.get("test_images"), evaluation_report.get("test_studies")) != (2218, 1414):
            raise ValueError("test inference output count differs from locked requirement")
        state["status"] = "test_inference_complete"
        atomic_json(_state_path(output), state)

        current_originals = _capture_existing_csvs(root)
        if current_originals != state["existing_submission_csvs"]:
            raise ValueError("one of the pre-existing submission CSVs changed during the run")
        _verify_test_manifest(binding)
        csv_result = export_submission(output, csv_path, binding["data_root"])
        state["submission"] = csv_result
        state["status"] = "complete"
        state["ended_utc"] = now()
        state["heartbeat_utc"] = state["ended_utc"]
        atomic_json(_state_path(output), state)
        return state
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = f"{type(error).__name__}: {error}"
        state["ended_utc"] = now()
        state["heartbeat_utc"] = state["ended_utc"]
        atomic_json(_state_path(output), state)
        raise
    finally:
        state["execution_state_held"] = False
        _execution_state(False)
        atomic_json(_state_path(output), state)


def _binding_record(binding, csv_path):
    return {
        "config": "A", "variant": "ema", "head_lr": 0.001,
        "epochs": 3, "batch_size": 2, "accumulation_steps": 8,
        "workers": 0, "seed": 2026, "ema_decay": 0.999, "device": "cuda",
        "train_metadata_sha256": binding["train_metadata_sha256"],
        "public_checkpoint_sha256": binding["public_checkpoint_sha256"],
        "image_inventory_sha256": binding["image_inventory_sha256"],
        "test_metadata_sha256": binding["test_metadata_sha256"],
        "lock_hashes": binding["lock_hashes"], "csv_path": str(csv_path),
    }


def _verify_snapshot(source, snapshot):
    manifest_path = source / "snapshot.json"
    if not manifest_path.is_file() or sha256(manifest_path) != snapshot["manifest_sha256"]:
        raise ValueError("immutable source snapshot manifest changed")
    manifest = read_json(manifest_path)
    for name, expected in manifest["files"].items():
        if sha256(source / name) != expected:
            raise ValueError(f"immutable source snapshot changed: {name}")
    for name, expected in manifest["locked_evidence"].items():
        if sha256(source / "locked" / name) != expected:
            raise ValueError(f"immutable locked evidence changed: {name}")


def _verified_test_output(test_output, recorded):
    if not isinstance(recorded, dict) or not test_output.is_dir():
        return False
    paths = {
        "report_sha256": test_output / "report.json",
        "study_ids_sha256": test_output / "study_ids.json",
        "study_probabilities_sha256": test_output / "study_probabilities.npy",
    }
    try:
        return all(path.is_file() and sha256(path) == recorded.get(key)
                   for key, path in paths.items())
    except OSError:
        return False


def _verify_completed_csv(state, csv_path, data_root):
    submission = state.get("submission", {})
    if Path(submission.get("path", "")).resolve() != Path(csv_path).resolve():
        raise ValueError("completed runner records a different CSV path")
    if not Path(csv_path).is_file() or sha256(csv_path) != submission.get("sha256"):
        raise ValueError("completed submission CSV is missing or its hash changed")
    from cxr.data import _read_metadata
    from cxr.submission import validate_submission
    validation = validate_submission(csv_path, _read_metadata(Path(data_root) / "test.csv"))
    if validation.get("studies") != 1414 or validation.get("rows") != 14140:
        raise ValueError("completed CSV no longer passes the test submission contract")


def _verify_test_manifest(binding):
    if sha256(Path(binding["data_root"]) / "test.csv") != binding["test_metadata_sha256"]:
        raise ValueError("test.csv changed since run preflight")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    state = run(args.output, args.csv, resume=args.resume)
    print(json.dumps({"status": state["status"], "output": str(args.output.resolve()),
                      "csv": state.get("submission", {}).get("path")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
