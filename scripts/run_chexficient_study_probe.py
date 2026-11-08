"""Prepare or run the fixed CheXficient frozen-feature study probe."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cxr.metrics import evaluate
from cxr.safety import safe_output
from cxr.study_probe import fit_linear_probe, pool_study_features, predict_linear_probe
from scripts.run_bottleneck_study_probe import legacy_oof, training_gate, validation_gate
from scripts.evaluate_frozen_study_probe import (
    FOLDS, _fold_memberships, _json_write, _load_head, _save_head, _sha256,
)


SEED = 20261108
MAX_ITER = 200
L2 = 0.01
CLASS_BALANCE = False
CHEXFICIENT_BATCH_SIZE = 4
CHEXFICIENT_PREPROCESS_WORKERS = 2
FEATURE_WIDTHS = {"chexficient": 512, "joint": 1566}
CANDIDATES = ("chexficient", "joint")
PLAN_PATH = ROOT / "docs/CHEXFICIENT_STUDY_PROBE_PLAN_20261108.md"
STAGE_STATE_PATHS = (
    ROOT / "outputs/eva-study-probe-20261108-v2/experiment.json",
    ROOT / "outputs/cxrlt-patient-smoothing-20261108-v2/experiment.json",
)
DEFAULT_SOURCE = ROOT / "outputs/cxrlt-supervised-study-probe-20261108"
DEFAULT_OUTPUT = ROOT / "outputs/chexficient-study-probe-20261108"

SOURCE_FILES = (
    "scripts/run_chexficient_study_probe.py",
    "scripts/run_frozen_study_probe.py",
    "scripts/evaluate_frozen_study_probe.py",
    "scripts/run_bottleneck_study_probe.py",
    "scripts/compare_runs.py",
    "scripts/run_eva_study_probe.py", "scripts/run_patient_smoothing.py",
    "cxr/__init__.py", "cxr/data.py", "cxr/metrics.py", "cxr/safety.py",
    "cxr/submission.py", "cxr/study_probe.py", "cxr/bottleneck_probe.py",
    "cxr/patient_smoothing.py", "cxr/eva_x_transfer.py", "cxr/cxrlt_transfer.py",
    "cxr/chexficient_transfer.py",
    "docs/CHEXFICIENT_STUDY_PROBE_PLAN_20261108.md",
    "docs/chexficient_source_manifest_20261108.json",
)
MANIFEST_PATH = ROOT / "docs/chexficient_source_manifest_20261108.json"
MODEL_DIR = ROOT / "weights/medical/chexficient"
CHEXFICIENT_REVISION = "34bb5ea905c4cb504d2b73c9a6687a5d5082f52a"
CHEXFICIENT_WEIGHT_SHA256 = "f691262c3f77c3c850bebff420180602cf4ca5d5214449e377366f3205548336"
CHEXFICIENT_WEIGHT_BYTES = 780793036
CHEXFICIENT_ADAPTER_SHA256 = "f45aca9b42d21953aa459e43e5b665df1f22f0228e6457f5583bfd1d1ee3e6c3"
ENGINEERING_EVIDENCE = (
    ROOT / "outputs/chexficient-preparation-20261108/cpu_engineering_check-v2.json",
    ROOT / "outputs/chexficient-preparation-20261108/gpu_engineering_check.json",
)
INDEPENDENT_EVIDENCE = (
    ROOT / "outputs/eva-study-probe-20261108-v2-independent-oof-verification-v2.json",
    ROOT / "outputs/cxrlt-patient-smoothing-20261108-v2-independent-verification.json",
)
CHEXFICIENT_MODEL_FILES = (
    "config.json", "configuration_chexficient.py", "dinov2/__init__.py",
    "dinov2/layers/__init__.py", "dinov2/layers/_utils.py",
    "dinov2/layers/attention.py", "dinov2/layers/block.py",
    "dinov2/layers/dino_head.py", "dinov2/layers/drop_path.py",
    "dinov2/layers/layer_scale.py", "dinov2/layers/mlp.py",
    "dinov2/layers/patch_embed.py", "dinov2/layers/swiglu_ffn.py",
    "dinov2/models/__init__.py", "dinov2/models/vision_transformer.py",
    "DINOv2-LICENSE.txt", "model.safetensors", "modeling_chexficient.py",
    "preprocessor_config.json", "projection.py", "README.md",
    "upstream-LICENSE", "upstream-models_clip.py", "upstream-README.md",
)
INCUMBENT_ARTIFACTS = (
    "plan.json", "experiment.json", "recipe.json", "final_head.npz",
    "report.json", "validation_report.json", "submission_verification.json", "train_records.json",
    "oof_mean_probabilities.npy", "oof_targets.npy", "oof_study_ids.json",
    "candidate_valid.csv",
)
SPLIT_CACHE_FILES = ("input.json", "progress.json", "image_features.npy")


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return _sha256(Path(path))


def atomic_json(path, value):
    _json_write(Path(path), value)


def _copy_snapshot_file(source, destination, input_locks, snapshot_locks, snapshot_root):
    source = Path(source).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = sha256(source)
    input_locks[str(source)] = digest
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256(destination) != digest:
        raise ValueError(f"snapshot copy hash mismatch: {source}")
    destination.chmod(destination.stat().st_mode & ~stat.S_IWRITE)
    snapshot_locks[destination.relative_to(snapshot_root).as_posix()] = digest
    return destination


def environment():
    packages = ("numpy", "torch", "torchvision", "Pillow", "timm", "transformers")
    return {"python": sys.version,
            **{name: importlib.metadata.version(name) for name in packages}}


def execution_settings(batch_size=CHEXFICIENT_BATCH_SIZE,
                       preprocess_workers=CHEXFICIENT_PREPROCESS_WORKERS):
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("CheXficient batch size must be a positive integer")
    if not isinstance(preprocess_workers, int) or preprocess_workers < 0:
        raise ValueError("CheXficient preprocessing workers must be a nonnegative integer")
    return {"batch_size": batch_size, "preprocess_workers": preprocess_workers}


def _require_preparation_authority(plan_path=PLAN_PATH, stage_state_paths=STAGE_STATE_PATHS):
    if not Path(plan_path).is_file():
        raise FileNotFoundError(f"fixed CheXficient protocol is required before preparation: {plan_path}")
    registrations = []
    for stage_state_path in stage_state_paths:
        stage_state_path = Path(stage_state_path).resolve()
        stage_root = stage_state_path.parent
        required = ("experiment.json", "plan.json", "artifacts.json")
        missing = [name for name in required if not (stage_root / name).is_file()]
        if missing:
            raise FileNotFoundError(f"terminal stage registration is incomplete: {stage_root}: {', '.join(missing)}")
        stage = read_json(stage_root / "experiment.json")
        if stage.get("status") != "complete":
            raise ValueError("CheXficient preparation requires completed EVA-X and patient-smoothing stages")
        if stage.get("candidate_promoted") is not False or stage.get("test_csv_exported") is True:
            raise ValueError("a prerequisite stage promoted or exported test; revise the fixed comparison protocol")
        if stage.get("test_read") is True or stage.get("test_extracted") is True:
            raise ValueError("a prerequisite stage read test inputs; revise the fixed comparison protocol")
        stage_plan = stage_root / "plan.json"
        plan = read_json(stage_plan)
        lock_path = stage_root / "plan_lock.json"
        if lock_path.exists():
            if sha256(stage_plan) != read_json(lock_path).get("sha256"):
                raise ValueError("EVA-X prerequisite plan hash does not match plan_lock.json")
        elif plan.get("plan_sha256") != _json_hash({key: value for key, value in plan.items()
                                                       if key != "plan_sha256"}):
            raise ValueError("patient-smoothing prerequisite plan self-hash mismatch")
        artifacts = read_json(stage_root / "artifacts.json")
        if not isinstance(artifacts, dict) or not artifacts:
            raise ValueError("prerequisite stage artifacts registration is empty or invalid")
        for relative, digest in artifacts.items():
            path = stage_root / relative
            if not path.is_file() or sha256(path) != digest:
                raise ValueError(f"prerequisite stage artifact changed or is missing: {relative}")
        alignment_path = stage_root / "alignment_report.json"
        if alignment_path.exists():
            alignment = read_json(alignment_path)
            if alignment.get("sources") != 22 or alignment.get("covered_studies") != 18513:
                raise ValueError("patient-smoothing legacy target audit is incomplete")
        registrations.append({"root": str(stage_root), "state": stage,
                              "plan_sha256": sha256(stage_plan),
                              "artifacts_sha256": sha256(stage_root / "artifacts.json")})
    return registrations


def _relative_copy_name(index, path):
    path = Path(path)
    safe_name = "".join(char if char.isalnum() or char in "._-" else "_" for char in path.name)
    return Path("locked_inputs") / f"{index:03d}-{safe_name}"


def prepare(output, source=DEFAULT_SOURCE, *, plan_path=PLAN_PATH,
            stage_state_paths=STAGE_STATE_PATHS, batch_size=CHEXFICIENT_BATCH_SIZE,
            preprocess_workers=CHEXFICIENT_PREPROCESS_WORKERS):
    """Register the fixed inputs and immutable source snapshot without running it."""
    stage_registrations = _require_preparation_authority(plan_path, stage_state_paths)
    settings, recorded_environment = execution_settings(batch_size, preprocess_workers), environment()
    output, source = Path(output).resolve(), Path(source).resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("CheXficient output must be outside its incumbent source experiment")
    incumbent_plan = read_json(source / "plan.json")
    incumbent_state = read_json(source / "experiment.json")
    if (incumbent_state.get("status") != "complete"
            or incumbent_state.get("candidate_promoted") is not True
            or incumbent_state.get("test_csv_exported") is not True):
        raise ValueError("named CXR-LT incumbent must be complete and promoted")
    incumbent_recipe = read_json(source / "recipe.json")
    if incumbent_recipe.get("head_sha256") != sha256(source / "final_head.npz"):
        raise ValueError("named CXR-LT incumbent recipe does not match its final head")
    data_root = Path(incumbent_plan["data_root"]).resolve()
    output = safe_output(output, data_root, inputs=(source,))
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"run directory is not empty: {output}")
    snapshot, source_snapshot = output / "snapshot", output / "source"
    snapshot.mkdir()
    source_snapshot.mkdir()
    input_locks, snapshot_locks = {}, {}

    # The incumbent plan also registers a historical test CSV; never open or copy it here.
    locked_input_copies = {}
    for index, (name, expected_digest) in enumerate(incumbent_plan["input_locks"].items()):
        original = Path(name).resolve()
        old_test = incumbent_plan.get("old_test")
        if old_test and original == Path(old_test).resolve():
            continue
        if sha256(original) != expected_digest:
            raise ValueError(f"incumbent input changed: {original}")
        copied = _copy_snapshot_file(original, snapshot / _relative_copy_name(index, original),
                                     input_locks, snapshot_locks, snapshot)
        locked_input_copies[str(original)] = str(copied)

    incumbent_copy = snapshot / "incumbent"
    for name in INCUMBENT_ARTIFACTS:
        _copy_snapshot_file(source / name, incumbent_copy / name,
                            input_locks, snapshot_locks, snapshot)
    train_cache = source / "train_features"
    for name in SPLIT_CACHE_FILES:
        _copy_snapshot_file(train_cache / name, incumbent_copy / "train_features" / name,
                            input_locks, snapshot_locks, snapshot)
    train_progress = read_json(train_cache / "progress.json")
    if (train_progress.get("status") != "complete"
            or sha256(train_cache / "image_features.npy") != train_progress.get("sha256")):
        raise ValueError("incumbent train feature cache is incomplete or changed")

    # Register valid/test cache metadata and existing matrix digests without opening their matrices.
    deferred_cxrlt = {}
    for split in ("valid", "test"):
        cache = source / f"{split}_features"
        input_path, progress_path = cache / "input.json", cache / "progress.json"
        manifest, progress = read_json(input_path), read_json(progress_path)
        if progress.get("status") != "complete" or not isinstance(progress.get("sha256"), str):
            raise ValueError(f"incumbent {split} feature cache metadata is incomplete")
        for name, path in (("input.json", input_path), ("progress.json", progress_path)):
            _copy_snapshot_file(path, incumbent_copy / f"{split}_features" / name,
                                input_locks, snapshot_locks, snapshot)
        deferred_cxrlt[split] = {
            "cache_dir": str(cache), "input_sha256": sha256(input_path),
            "progress_sha256": sha256(progress_path), "matrix_sha256": progress["sha256"],
            "images_sha256": _json_hash(manifest.get("images", [])),
        }

    source_code_locks = {}
    for relative in SOURCE_FILES:
        _copy_snapshot_file(ROOT / relative, source_snapshot / relative,
                            input_locks, source_code_locks, source_snapshot)

    incumbent_source_locks = {}
    for relative, expected_digest in incumbent_plan.get("source_locks", {}).items():
        original = source / "source" / relative
        if not original.is_file() or sha256(original) != expected_digest:
            raise ValueError(f"incumbent source snapshot changed: {relative}")
        _copy_snapshot_file(original, snapshot / "incumbent_source" / relative,
                            input_locks, snapshot_locks, snapshot)
        incumbent_source_locks[relative] = expected_digest

    manifest = read_json(MANIFEST_PATH)
    if manifest.get("revision") != CHEXFICIENT_REVISION:
        raise ValueError("CheXficient source manifest revision differs from the fixed protocol")
    model_snapshot = snapshot / "chexficient_model"
    model_files = manifest.get("files", {})
    if set(model_files) != set(CHEXFICIENT_MODEL_FILES):
        raise ValueError("CheXficient source manifest file set differs from the fixed protocol")
    for relative in CHEXFICIENT_MODEL_FILES:
        expected = model_files.get(relative, {}).get("sha256")
        original = MODEL_DIR / relative
        if not expected or not original.is_file() or sha256(original) != expected:
            raise ValueError(f"CheXficient source asset differs from pinned manifest: {relative}")
        _copy_snapshot_file(original, model_snapshot / relative,
                            input_locks, snapshot_locks, snapshot)
    if model_files.get("model.safetensors", {}).get("sha256") != CHEXFICIENT_WEIGHT_SHA256:
        raise ValueError("CheXficient checkpoint hash differs from the fixed protocol")
    if model_files["model.safetensors"].get("bytes") != CHEXFICIENT_WEIGHT_BYTES:
        raise ValueError("CheXficient checkpoint size differs from the fixed protocol")
    cpu_evidence = read_json(ENGINEERING_EVIDENCE[0])
    gpu_evidence = read_json(ENGINEERING_EVIDENCE[1])
    if (cpu_evidence.get("status") != "passed"
            or cpu_evidence.get("revision") != CHEXFICIENT_REVISION
            or cpu_evidence.get("adapter_sha256") != CHEXFICIENT_ADAPTER_SHA256
            or cpu_evidence.get("weights_sha256") != CHEXFICIENT_WEIGHT_SHA256):
        raise ValueError("CheXficient CPU engineering evidence differs from the fixed protocol")
    if (gpu_evidence.get("status") != "passed"
            or gpu_evidence.get("adapter_sha256") != CHEXFICIENT_ADAPTER_SHA256
            or gpu_evidence.get("weights_sha256") != CHEXFICIENT_WEIGHT_SHA256):
        raise ValueError("CheXficient GPU engineering evidence differs from the fixed encoder/checkpoint")
    if (gpu_evidence.get("batch_size") != CHEXFICIENT_BATCH_SIZE
            or gpu_evidence.get("preprocess_workers") != CHEXFICIENT_PREPROCESS_WORKERS
            or gpu_evidence.get("amp") != "cuda float16"
            or gpu_evidence.get("repeat_max_abs_error") != 0.0):
        raise ValueError("CheXficient GPU engineering evidence does not match fixed extraction settings")
    evidence_locks = {}
    for evidence in ENGINEERING_EVIDENCE:
        copied = _copy_snapshot_file(evidence, snapshot / "engineering" / evidence.name,
                                     input_locks, snapshot_locks, snapshot)
        evidence_locks[evidence.name] = sha256(copied)

    partition_copy = _copy_snapshot_file(
        Path(incumbent_plan["partition"]).resolve(), snapshot / "partition.json",
        input_locks, snapshot_locks, snapshot)
    legacy_folder = Path(incumbent_plan["old_oof"]).resolve()
    legacy_copy = snapshot / "legacy_oof"
    for name in ("oof_component_probabilities.npy", "oof_study_ids.json", "plan.json"):
        _copy_snapshot_file(legacy_folder / name, legacy_copy / name,
                            input_locks, snapshot_locks, snapshot)
    legacy_valid_copy = _copy_snapshot_file(
        Path(incumbent_plan["old_valid"]).resolve(), snapshot / "legacy_valid.csv",
        input_locks, snapshot_locks, snapshot)

    smoothing_root = Path(stage_registrations[1]["root"])
    smoothing_plan = read_json(smoothing_root / "plan.json")
    eva_audit = read_json(INDEPENDENT_EVIDENCE[0])
    smoothing_audit = read_json(INDEPENDENT_EVIDENCE[1])
    if (eva_audit.get("status") != "verified_oof_only"
            or eva_audit.get("runner_status") != "complete"
            or eva_audit.get("selected_oof_candidate") is not None
            or eva_audit.get("candidate_promoted") is not False
            or eva_audit.get("coverage", {}).get("oof_heads_and_gates") != "verified"):
        raise ValueError("independent EVA-X OOF verification did not pass the fixed prerequisite")
    smoothing_checks = smoothing_audit.get("checks", {})
    required_audits = ("canonical_plan_sha256", "83_input_locks", "12_source_locks",
                       "7_artifacts", "deferred_test_unopened", "split_identity_isolation",
                       "22_legacy_targets", "OOF_replay_and_order", "OOF_metrics",
                       "valid_gate_failure")
    if (smoothing_audit.get("result") != "passed"
            or any(not smoothing_checks.get(name, {}).get("passed") for name in required_audits)):
        raise ValueError("independent patient-smoothing numerical/identity verification failed")
    independent_evidence_locks = {}
    for report_path in INDEPENDENT_EVIDENCE:
        copied = _copy_snapshot_file(report_path, snapshot / "independent" / report_path.name,
                                     input_locks, snapshot_locks, snapshot)
        independent_evidence_locks[report_path.name] = sha256(copied)
    legacy_target_sources = []
    for index, source_row in enumerate(smoothing_plan.get("legacy_target_sources", [])):
        ids_rel, targets_rel = source_row["ids"], source_row["targets"]
        expected_ids = smoothing_plan.get("input_locks", {}).get(ids_rel)
        expected_targets = smoothing_plan.get("input_locks", {}).get(targets_rel)
        if (not expected_ids or not expected_targets
                or sha256(smoothing_root / ids_rel) != expected_ids
                or sha256(smoothing_root / targets_rel) != expected_targets):
            raise ValueError(f"legacy target source differs from patient-smoothing input locks: {index}")
        ids_copy = _copy_snapshot_file(smoothing_root / ids_rel,
            snapshot / "legacy_targets" / f"{index:02d}" / "study_ids.json",
            input_locks, snapshot_locks, snapshot)
        targets_copy = _copy_snapshot_file(smoothing_root / targets_rel,
            snapshot / "legacy_targets" / f"{index:02d}" / "study_targets.npy",
            input_locks, snapshot_locks, snapshot)
        legacy_target_sources.append({
            "ids": str(ids_copy.relative_to(snapshot)),
            "targets": str(targets_copy.relative_to(snapshot)),
        })
    if len(legacy_target_sources) != 22:
        raise ValueError("fixed legacy training-target evidence must contain 22 sources")

    stage_snapshot = snapshot / "prerequisite_stages"
    for index, registration in enumerate(stage_registrations):
        stage_root = Path(registration["root"])
        folder = stage_snapshot / f"{index:02d}-{stage_root.name}"
        stage_names = ["experiment.json", "plan.json", "artifacts.json"]
        if (stage_root / "plan_lock.json").is_file():
            stage_names.append("plan_lock.json")
        for name in stage_names:
            _copy_snapshot_file(stage_root / name, folder / name,
                                input_locks, snapshot_locks, snapshot)

    plan = {
        "created_utc": now(), "protocol": str(Path(plan_path).resolve()),
        "protocol_sha256": sha256(plan_path), "stage_registrations": stage_registrations,
        "incumbent_source_locks": incumbent_source_locks,
        "incumbent_source": str(source), "incumbent_snapshot": str(incumbent_copy),
        "data_root": str(data_root), "model_dir": str(model_snapshot),
        "partition": str(partition_copy), "legacy_oof": str(legacy_copy),
        "legacy_valid": str(legacy_valid_copy),
        "legacy_target_sources": legacy_target_sources,
        "current_valid": str(incumbent_copy / "candidate_valid.csv"),
        "locked_input_copies": locked_input_copies, "source_locks": source_code_locks,
        "snapshot_locks": snapshot_locks, "input_locks": input_locks,
        "environment": recorded_environment, "model": "StanfordAIMI/CheXficient",
        "revision": CHEXFICIENT_REVISION, "weight_sha256": CHEXFICIENT_WEIGHT_SHA256,
        "manifest_sha256": sha256(MANIFEST_PATH), "engineering_evidence": evidence_locks,
        "independent_evidence": independent_evidence_locks,
        "candidates": list(CANDIDATES), "feature_widths": FEATURE_WIDTHS,
        "fit": {"l2": L2, "max_iter": MAX_ITER, "seed": SEED,
                "class_balance": CLASS_BALANCE}, "folds": list(FOLDS), **settings,
        "device": incumbent_plan["device"],
        "incumbent_plan_sha256": sha256(source / "plan.json"),
        "incumbent_recipe_sha256": sha256(source / "recipe.json"),
        "incumbent_head_sha256": sha256(source / "final_head.npz"),
        "legacy_valid_sha256": sha256(legacy_valid_copy),
        "deferred_cxrlt": deferred_cxrlt, "test_extracted": False,
    }
    atomic_json(output / "plan.json", plan)
    atomic_json(output / "plan_lock.json", {"sha256": sha256(output / "plan.json")})
    atomic_json(output / "experiment.json", {
        "status": "prepared", "phase": None, "created_utc": now(),
        "test_extracted": False,
    })
    return output


def verify_plan(output, plan):
    output = Path(output)
    if environment() != plan["environment"]:
        raise ValueError("software environment changed")
    for path, digest in plan["input_locks"].items():
        if sha256(path) != digest:
            raise ValueError(f"locked input changed: {path}")
    for relative, digest in plan["source_locks"].items():
        if sha256(output / "source" / relative) != digest:
            raise ValueError(f"source snapshot changed: {relative}")
    for relative, digest in plan["snapshot_locks"].items():
        if sha256(output / "snapshot" / relative) != digest:
            raise ValueError(f"registered snapshot changed: {relative}")
    if sha256(plan["protocol"]) != plan["protocol_sha256"]:
        raise ValueError("fixed CheXficient protocol changed")
    plan_path = output / "plan.json"
    if not (output / "plan_lock.json").is_file() or sha256(plan_path) != read_json(output / "plan_lock.json").get("sha256"):
        raise ValueError("frozen CheXficient plan changed or plan_lock.json is missing")
    stage_paths = [Path(row["root"]) / "experiment.json" for row in plan["stage_registrations"]]
    current_stages = _require_preparation_authority(plan["protocol"], stage_paths)
    if current_stages != plan["stage_registrations"]:
        raise ValueError("prerequisite stage registrations changed")
    for relative, digest in plan["incumbent_source_locks"].items():
        if sha256(output / "snapshot" / "incumbent_source" / relative) != digest:
            raise ValueError(f"incumbent source snapshot changed: {relative}")


def load_registered_plan(output):
    output = Path(output).resolve()
    plan = read_json(output / "plan.json")
    verify_plan(output, plan)
    plan["plan_sha256"] = sha256(output / "plan.json")
    plan["output_dir"] = str(output)
    return plan


def _validate_matrix(values, rows, width, name, *, chunk_rows=512):
    if values.shape != (rows, width) or values.dtype != np.float32:
        raise ValueError(f"{name} must be finite float32 with shape ({rows}, {width})")
    for start in range(0, rows, chunk_rows):
        if not np.isfinite(values[start:start + chunk_rows]).all():
            raise ValueError(f"{name} contains non-finite rows")


def _identity_rows(identity):
    rows = []
    seen, seen_paths = set(), set()
    for item in identity:
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            raise ValueError("image identity rows must be [dicom_id, path, size, mtime_ns]")
        dicom, path, size, mtime = item
        if (not isinstance(dicom, str) or not dicom or dicom != dicom.strip()
                or not isinstance(path, str) or not path):
            raise ValueError("image identity DICOM IDs must be nonempty exact strings")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("image identity sizes must be positive integers")
        if isinstance(mtime, bool) or not isinstance(mtime, int) or mtime < 0:
            raise ValueError("image identity mtimes must be nonnegative integers")
        resolved = str(Path(path).resolve())
        if not resolved or dicom in seen or resolved in seen_paths:
            raise ValueError(f"duplicate or conflicting image identity: {dicom}")
        seen.add(dicom)
        seen_paths.add(resolved)
        rows.append([dicom, resolved, int(size), int(mtime)])
    return rows


def require_exact_image_identity(expected, actual):
    expected_rows, actual_rows = _identity_rows(expected), _identity_rows(actual)
    if expected_rows != actual_rows:
        raise ValueError("image feature caches have different ordered image identities")
    return expected_rows


def _split_inputs(plan, split):
    from cxr.data import read_records
    from scripts.run_frozen_study_probe import image_identity, records_with_views

    source = Path(plan["incumbent_snapshot"])
    cache = source / f"{split}_features"
    if split == "train":
        saved = read_json(source / "train_records.json")
        records, views = saved["records"], saved["views"]
    else:
        records, views = records_with_views(Path(plan["data_root"]), split)
    manifest = read_json(cache / "input.json")
    identities = image_identity(records)
    require_exact_image_identity(manifest.get("images", []), identities)
    progress = read_json(cache / "progress.json")
    if progress.get("status") != "complete" or progress.get("completed") != len(records):
        raise ValueError(f"incumbent {split} feature cache is incomplete")
    matrix_path = cache / "image_features.npy"
    if sha256(matrix_path) != progress.get("sha256"):
        raise ValueError(f"incumbent {split} feature matrix hash mismatch")
    values = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    _validate_matrix(values, len(records), 1054, f"incumbent {split} features")
    return records, views, identities, values


def _materialize_deferred_cxrlt(plan, split):
    if split not in ("valid", "test"):
        raise ValueError("only valid/test CXR-LT caches may be deferred")
    registration = plan["deferred_cxrlt"][split]
    source = Path(registration["cache_dir"])
    destination = Path(plan["incumbent_snapshot"]) / f"{split}_features"
    for name, expected in (("input.json", registration["input_sha256"]),
                           ("progress.json", registration["progress_sha256"])):
        if sha256(source / name) != expected:
            raise ValueError(f"deferred CXR-LT {split} {name} changed")
    progress = read_json(source / "progress.json")
    if progress.get("status") != "complete" or progress.get("sha256") != registration["matrix_sha256"]:
        raise ValueError(f"deferred CXR-LT {split} cache registration changed")
    matrix = source / "image_features.npy"
    if sha256(matrix) != registration["matrix_sha256"]:
        raise ValueError(f"deferred CXR-LT {split} feature matrix hash mismatch")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "image_features.npy"
    if target.exists():
        if sha256(target) != registration["matrix_sha256"]:
            raise ValueError(f"materialized CXR-LT {split} cache changed")
    else:
        shutil.copy2(matrix, target)
        if sha256(target) != registration["matrix_sha256"]:
            raise ValueError(f"materialized CXR-LT {split} matrix hash mismatch")


def _bind_chexficient_cache(plan, split, identities):
    cache = Path(plan["output_dir"]) / "chexficient_features" / split
    cache.mkdir(parents=True, exist_ok=True)
    binding = {
        "plan_sha256": plan["plan_sha256"],
        "model_revision": plan["revision"],
        "weight_sha256": plan["weight_sha256"],
        "adapter_sha256": plan["source_locks"]["cxr/chexficient_transfer.py"],
        "manifest_sha256": plan["manifest_sha256"],
        "image_identity_sha256": _json_hash(_identity_rows(identities)),
        "feature_size": 512,
        "input_size": 378,
        "feature_pooling": "dinov2_vitb14_normalized_cls_projected_512_l2_normalized",
        "batch_size": plan["batch_size"],
        "preprocess_workers": plan["preprocess_workers"],
        "device": plan["device"],
    }
    path = cache / "binding.json"
    if path.exists():
        if read_json(path) != binding:
            raise ValueError(f"CheXficient feature cache binding mismatch: {split}")
    else:
        payload = [item for item in cache.iterdir() if item.name != "binding.json"]
        if payload:
            raise ValueError(f"CheXficient {split} cache payload exists without its binding")
        atomic_json(path, binding)
    return cache


def _json_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _extract_chexficient_split(plan, records, identities, split, model_bundle):
    from scripts.run_frozen_study_probe import extract_features

    cache = _bind_chexficient_cache(plan, split, identities)
    model, processor, device = model_bundle
    values = extract_features(
        model, processor, records, cache,
        batch_size=plan["batch_size"], device=device,
        preprocess_workers=plan["preprocess_workers"],
    )
    progress = read_json(cache / "progress.json")
    matrix_path = cache / "image_features.npy"
    if (progress.get("status") != "complete"
            or sha256(matrix_path) != progress.get("sha256")):
        raise ValueError(f"CheXficient {split} feature cache is incomplete or changed")
    _validate_matrix(values, len(records), 512, f"CheXficient {split} features")
    _register_chexficient_cache(cache, plan["plan_sha256"])
    return values


def _register_chexficient_cache(cache, plan_sha256):
    cache = Path(cache)
    binding_path, input_path = cache / "binding.json", cache / "input.json"
    progress_path, matrix_path = cache / "progress.json", cache / "image_features.npy"
    for path in (binding_path, input_path, progress_path, matrix_path):
        if not path.is_file():
            raise ValueError(f"CheXficient feature cache registration is missing {path.name}")
    progress = read_json(progress_path)
    matrix_sha = sha256(matrix_path)
    if progress.get("status") != "complete" or progress.get("sha256") != matrix_sha:
        raise ValueError("CheXficient feature cache cannot be registered before a complete matrix hash")
    registration = {
        "plan_sha256": plan_sha256,
        "binding_sha256": sha256(binding_path),
        "input_sha256": sha256(input_path),
        "progress_sha256": sha256(progress_path),
        "matrix_sha256": matrix_sha,
    }
    path = cache / "registration.json"
    if path.exists():
        if read_json(path) != registration:
            raise ValueError("registered CheXficient feature cache hashes changed")
    else:
        atomic_json(path, registration)
    return registration


def _study_key_rows(studies):
    rows, seen, owners = [], set(), {}
    for row in studies:
        subject, study = row.get("subject_id"), row.get("study_id")
        if any(not isinstance(value, str) or not value or value != value.strip()
               for value in (subject, study)):
            raise ValueError("study identities must be nonempty exact strings")
        if (subject, study) in seen or study in owners and owners[study] != subject:
            raise ValueError("study IDs are duplicate or have conflicting patient identities")
        seen.add((subject, study))
        owners[study] = subject
        rows.append([subject, study])
    return rows


def _validate_legacy_targets(output, plan, train_studies, targets):
    expected = {tuple(key): targets[index] for index, key in enumerate(_study_key_rows(train_studies))}
    covered = set()
    for source in plan["legacy_target_sources"]:
        ids = read_json(Path(output) / "snapshot" / source["ids"])
        keys, key_set, owners = [], set(), {}
        for key in ids:
            if (not isinstance(key, (list, tuple)) or len(key) != 2
                    or any(not isinstance(value, str) or not value or value != value.strip()
                           for value in key)):
                raise ValueError("legacy target IDs must be exact subject/study pairs")
            pair = tuple(key)
            if pair in key_set or pair[1] in owners and owners[pair[1]] != pair[0]:
                raise ValueError("legacy target IDs contain duplicates or conflicting subjects")
            keys.append(pair)
            key_set.add(pair)
            owners[pair[1]] = pair[0]
        if not keys or not set(keys) <= set(expected):
            raise ValueError("legacy target evidence contains unknown training studies")
        values = np.load(Path(output) / "snapshot" / source["targets"], allow_pickle=False)
        if values.shape != (len(keys), 10) or not np.array_equal(values, [expected[key] for key in keys]):
            raise ValueError("legacy targets do not exactly match local training labels")
        covered.update(keys)
    if covered != set(expected):
        raise ValueError("legacy target evidence does not cover complete training data")
    return {"sources": len(plan["legacy_target_sources"]), "covered_studies": len(covered)}


def _validate_split_disjointness(first, second, label):
    def identities(records):
        owners, keys = {}, set()
        for row in records:
            subject, study = row.get("subject_id"), row.get("study_id")
            if any(not isinstance(value, str) or not value or value != value.strip()
                   for value in (subject, study)):
                raise ValueError("record study identities must be nonempty exact strings")
            if study in owners and owners[study] != subject:
                raise ValueError(f"conflicting patient identity for study {study}")
            owners[study] = subject
            keys.add((subject, study))
        return keys

    first_keys, second_keys = identities(first), identities(second)
    first_subjects = {row[0] for row in first_keys}
    second_subjects = {row[0] for row in second_keys}
    first_studies = {row[1] for row in first_keys}
    second_studies = {row[1] for row in second_keys}
    if (first_keys & second_keys or first_subjects & second_subjects
            or first_studies & second_studies):
        raise ValueError(f"patient or study overlap across train and {label} split")


def pool_candidates(records, views, cxrlt_images, chexficient_images, cxrlt_ids, chexficient_ids):
    """Pool aligned encoders separately, then form study-level features."""
    require_exact_image_identity(cxrlt_ids, chexficient_ids)
    if len(cxrlt_images) != len(records) or len(chexficient_images) != len(records):
        raise ValueError("feature rows must match the shared record order")
    if cxrlt_images.shape[1] != 1054 or chexficient_images.shape[1] != 512:
        raise ValueError("image feature width differs from the fixed source contracts")
    studies_c, cxrlt, targets_c = pool_study_features(records, cxrlt_images, views, "mean")
    studies_e, chexficient, targets_e = pool_study_features(records, chexficient_images, views, "mean")
    if _study_key_rows(studies_c) != _study_key_rows(studies_e):
        raise ValueError("encoder study pooling produced different ordered studies")
    if ((targets_c is None) != (targets_e is None)
            or (targets_c is not None and not np.array_equal(targets_c, targets_e))):
        raise ValueError("encoder study pooling produced different targets")
    if cxrlt.shape[1] != 1054 or chexficient.shape[1] != 512:
        raise ValueError("pooled feature width differs from the fixed contract")
    return studies_c, {"chexficient": chexficient,
                       "joint": np.concatenate((cxrlt, chexficient), axis=1)}, targets_c


def _score(targets, probabilities):
    report = evaluate(targets, probabilities)
    return {
        "macro_auc": report["macro_auc"],
        "macro_auc_defined_classes": report["macro_auc_defined_classes"],
        "per_class_auc": [row["auc"] for row in report["per_class"]],
    }


def _validate_probabilities(values, shape, name):
    values = np.asarray(values)
    if (values.shape != shape or not np.isfinite(values).all()
            or np.any((values < 0) | (values > 1))):
        raise ValueError(f"{name} must be a finite probability matrix within [0, 1]")
    return values


def candidate_gate(targets, incumbent, legacy, candidate, folds):
    gate = training_gate(targets, incumbent, candidate, folds, legacy)
    legacy_summary = _score(targets, legacy)
    gate["passed"] = bool(gate["passed"] and legacy_summary["macro_auc_defined_classes"] == 10)
    gate["legacy_full_class_summary"] = legacy_summary
    return gate


def select_candidate(rows):
    eligible = [row for row in rows if row["gate"]["passed"]]
    if not eligible:
        return None
    order = {name: index for index, name in enumerate(CANDIDATES)}
    return sorted(eligible, key=lambda row: (
        -row["gate"]["candidate"]["macro_auc"], order[row["candidate"]],
    ))[0]


class FoldHeadCache:
    """Register exactly replayable patient-held-out heads bound to the plan/features."""

    def __init__(self, folder, features, targets, folds, subjects,
                 plan_sha256, feature_identity):
        self.folder = Path(folder)
        self.x, self.y = features, targets
        self.folds, self.subjects = np.asarray(folds, dtype=object), np.asarray(subjects)
        self.plan_sha256, self.feature_identity = plan_sha256, feature_identity

    def predict(self, candidate, held_fold):
        if candidate not in CANDIDATES or held_fold not in FOLDS:
            raise ValueError("unsupported candidate or held fold")
        held = self.folds == held_fold
        training_folds = tuple(fold for fold in FOLDS if fold != held_fold)
        train = np.isin(self.folds, np.asarray(training_folds, dtype=object))
        held_indices = np.flatnonzero(held)
        if not held_indices.size or not train.any():
            raise ValueError("fold requires nonempty training and held rows")
        if np.any(train[held_indices]) or set(self.subjects[train]) & set(self.subjects[held]):
            raise ValueError("patient leakage in requested fold fit")
        identity = {
            "plan_sha256": self.plan_sha256,
            "feature_identity": self.feature_identity,
            "candidate": candidate,
            "width": FEATURE_WIDTHS[candidate],
            "training_folds": list(training_folds),
            "held_fold": held_fold,
            "training_patients": sorted(set(map(str, self.subjects[train]))),
            "l2": L2, "max_iter": MAX_ITER, "seed": SEED,
            "class_balance": CLASS_BALANCE,
        }
        folder = self.folder / candidate / str(held_fold)
        head_path, metadata_path = folder / "head.npz", folder / "fit.json"
        if metadata_path.exists():
            metadata = read_json(metadata_path)
            if (metadata.get("identity") != identity or not head_path.is_file()
                    or sha256(head_path) != metadata.get("head_sha256")):
                raise ValueError("cached fold head binding or hash mismatch")
            payload = _load_head(head_path)
        else:
            if head_path.exists():
                head_path.unlink()
            payload = fit_linear_probe(
                self.x[train], self.y[train], l2=L2, max_iter=MAX_ITER,
                seed=SEED, class_balance=CLASS_BALANCE,
            )
            folder.mkdir(parents=True, exist_ok=True)
            _save_head(head_path, payload)
            reloaded = _load_head(head_path)
            expected = predict_linear_probe(payload, self.x[held])
            replay = predict_linear_probe(reloaded, self.x[held])
            if not np.array_equal(expected, replay):
                raise ValueError("saved fold head does not replay exactly")
            metadata = {"identity": identity, "head_sha256": sha256(head_path),
                        "held_prediction_sha256": _array_hash(replay)}
            atomic_json(metadata_path, metadata)
        probabilities = predict_linear_probe(payload, self.x[held])
        if metadata.get("held_prediction_sha256") != _array_hash(probabilities):
            raise ValueError("cached fold head prediction replay mismatch")
        if not np.isfinite(probabilities).all():
            raise ValueError("fold head produced non-finite probabilities")
        return probabilities


def _feature_identity(candidate, source_sha, chexficient_sha, study_keys):
    return _json_hash({
        "candidate": candidate,
        "width": FEATURE_WIDTHS[candidate],
        "source_cxrlt_sha256": source_sha,
        "chexficient_features_sha256": chexficient_sha,
        "study_keys_sha256": _json_hash(study_keys),
    })


def _fit_final_head(output, plan, candidate, features, targets, feature_identity):
    head_path, recipe_path = output / "final_head.npz", output / "recipe.json"
    plan_sha = sha256(output / "plan.json")
    if recipe_path.exists():
        recipe = read_json(recipe_path)
        if (recipe.get("plan_sha256") != plan_sha
                or recipe.get("feature_identity") != feature_identity
                or recipe.get("candidate") != candidate
                or not head_path.is_file()
                or sha256(head_path) != recipe.get("head_sha256")):
            raise ValueError("saved final head is bound to another plan, candidate, or feature cache")
        payload = _load_head(head_path)
    else:
        if head_path.exists():
            head_path.unlink()
        payload = fit_linear_probe(
            features, targets, l2=L2, max_iter=MAX_ITER,
            seed=SEED, class_balance=CLASS_BALANCE,
        )
        _save_head(head_path, payload)
        reloaded = _load_head(head_path)
        if not np.array_equal(predict_linear_probe(payload, features),
                              predict_linear_probe(reloaded, features)):
            raise ValueError("saved final head does not replay exactly")
        recipe = {
            "candidate": candidate,
            "width": FEATURE_WIDTHS[candidate],
            "plan_sha256": plan_sha,
            "feature_identity": feature_identity,
            "source_plan_sha256": plan["incumbent_plan_sha256"],
            "l2": L2, "max_iter": MAX_ITER, "seed": SEED,
            "class_balance": CLASS_BALANCE,
            "head_file": head_path.name,
            "head_sha256": sha256(head_path),
        }
        atomic_json(recipe_path, recipe)
    return recipe, payload


def _gate_then_export(comparisons, export_test):
    if not validation_gate(comparisons):
        return False
    export_test()
    return True


def _artifact_manifest(output):
    return {
        str(path.relative_to(output)): sha256(path)
        for path in output.rglob("*")
        if path.is_file() and "source" not in path.relative_to(output).parts
        and "snapshot" not in path.relative_to(output).parts
        and path.name not in {"artifacts.json", "experiment.json", "launcher.json"}
        and not path.name.endswith(".log")
    }


def run(output):
    from scripts.compare_runs import compare_arrays, load_submission
    from cxr.submission import validate_submission, write_submission
    from scripts.run_frozen_study_probe import records_with_views
    from cxr.chexficient_transfer import load_encoder

    output = Path(output).resolve()
    plan = load_registered_plan(output)
    state = read_json(output / "experiment.json")
    torch.set_num_threads(4)
    plan_sha = sha256(output / "plan.json")
    if state.get("status") == "complete":
        for name, digest in read_json(output / "artifacts.json").items():
            if sha256(output / name) != digest:
                raise ValueError(f"completed artifact changed: {name}")
        return

    def stage(phase, **extra):
        state.update(status="running", phase=phase, updated_utc=now(),
                     pid=os.getpid(), error=None, **extra)
        atomic_json(output / "experiment.json", state)

    def finish(decision, *, promoted=False, test_exported=False):
        verify_plan(output, read_json(output / "plan.json"))
        atomic_json(output / "artifacts.json", _artifact_manifest(output))
        state.update(status="complete", phase=None, updated_utc=now(),
                     decision=decision, candidate_promoted=promoted,
                     test_csv_exported=test_exported, error=None)
        atomic_json(output / "experiment.json", state)

    try:
        stage("load_and_validate_train_cache")
        records, views, cxr_ids, cxr_images = _split_inputs(plan, "train")
        incumbent = Path(plan["incumbent_snapshot"])
        source_oof_ids = read_json(incumbent / "oof_study_ids.json")
        source_targets = np.load(incumbent / "oof_targets.npy", allow_pickle=False)
        source_baseline = np.load(incumbent / "oof_mean_probabilities.npy", allow_pickle=False)
        studies_c, cxr_study_x, targets = pool_study_features(
            records, cxr_images, views, "mean")
        if targets is None or _study_key_rows(studies_c) != source_oof_ids:
            raise ValueError("training studies differ from incumbent OOF study IDs")
        if not np.array_equal(targets, source_targets):
            raise ValueError("training targets differ elementwise from incumbent OOF targets")
        legacy_target_audit = _validate_legacy_targets(output, plan, studies_c, targets)
        _validate_probabilities(source_baseline, targets.shape, "incumbent OOF probabilities")
        legacy = legacy_oof({"legacy_oof": plan["legacy_oof"]}, studies_c)
        _validate_probabilities(legacy, targets.shape, "legacy OOF probabilities")
        del cxr_study_x
        assignment = read_json(plan["partition"])["subject_partition"]
        folds = _fold_memberships(studies_c, assignment)
        subjects = [str(row["subject_id"]) for row in studies_c]
        model, processor = load_encoder(plan["model_dir"], plan["device"])
        model.requires_grad_(False)
        model.eval()
        device = torch.device(plan["device"])
        model = model.to(device).eval()
        model_bundle = (model, processor, device)
        stage("extract_chexficient_train")
        chexficient_images = _extract_chexficient_split(
            plan, records, cxr_ids, "train", model_bundle)
        chexficient_cache_path = output / "chexficient_features/train/image_features.npy"
        chexficient_sha = sha256(chexficient_cache_path)
        chexficient_ids = read_json(output / "chexficient_features/train/input.json")["images"]
        require_exact_image_identity(cxr_ids, chexficient_ids)
        studies, train_features, pooled_targets = pool_candidates(
            records, views, cxr_images, chexficient_images, cxr_ids, chexficient_ids)
        if _study_key_rows(studies) != source_oof_ids or not np.array_equal(pooled_targets, source_targets):
            raise ValueError("pooled joint study IDs or targets differ from incumbent OOF")
        if not np.array_equal(pooled_targets, targets):
            raise ValueError("candidate target vectors differ")
        del cxr_images, chexficient_images, model, processor, model_bundle
        if device.type == "cuda":
            torch.cuda.empty_cache()

        candidate_rows = []
        oof_predictions = {}
        feature_ids = {}
        for candidate in CANDIDATES:
            features = train_features[candidate]
            if features.shape != (len(targets), FEATURE_WIDTHS[candidate]) or not np.isfinite(features).all():
                raise ValueError(f"invalid pooled {candidate} training features")
            feature_ids[candidate] = _feature_identity(
                candidate,
                sha256(incumbent / "train_features/image_features.npy"),
                chexficient_sha,
                source_oof_ids,
            )
            fits = FoldHeadCache(output / "fits", features, targets, folds, subjects,
                                 plan_sha, feature_ids[candidate])
            predictions = np.full_like(targets, np.nan, dtype=np.float64)
            for fold in FOLDS:
                held = folds == fold
                predictions[held] = fits.predict(candidate, fold)
            if not np.isfinite(predictions).all():
                raise ValueError(f"{candidate} OOF predictions do not cover all studies")
            gate = candidate_gate(targets, source_baseline, legacy, predictions, folds)
            candidate_rows.append({"candidate": candidate, "gate": gate,
                                   "feature_identity": feature_ids[candidate]})
            oof_predictions[candidate] = predictions
            np.save(output / f"oof_{candidate}_probabilities.npy", predictions, allow_pickle=False)
            del features, fits, predictions

        atomic_json(output / "oof_study_ids.json", source_oof_ids)
        np.save(output / "oof_targets.npy", source_targets, allow_pickle=False)
        selected = select_candidate(candidate_rows)
        atomic_json(output / "oof_report.json", {
            "candidates": candidate_rows,
            "legacy_target_audit": legacy_target_audit,
            "selected": None if selected is None else selected["candidate"],
            "selection": "highest eligible OOF Macro-AUC; exact tie prefers chexficient",
            "eligibility": {"all_ten_classes": True, "gain_vs_incumbent_min": 0.001,
                            "gain_vs_legacy_min": 0.001, "positive_incumbent_groups_min": 3},
            "interpretation": "finite-candidate patient OOF used for selection; not a nested unbiased estimate",
        })
        if selected is None:
            finish("retain incumbent; neither candidate passed the patient OOF gate")
            return

        candidate = selected["candidate"]
        feature_id = feature_ids[candidate]
        selected_features = train_features[candidate]
        stage("fit_and_register_full_train_head", selected_candidate=candidate)
        recipe, head = _fit_final_head(output, plan, candidate, selected_features,
                                       targets, feature_id)
        recipe_sha, head_sha = sha256(output / "recipe.json"), sha256(output / "final_head.npz")
        del train_features, selected_features, head, oof_predictions

        stage("extract_chexficient_valid", selected_candidate=candidate)
        _materialize_deferred_cxrlt(plan, "valid")
        valid_records, valid_views, valid_ids, valid_cxr = _split_inputs(plan, "valid")
        _validate_split_disjointness(records, valid_records, "valid")
        model, processor = load_encoder(plan["model_dir"], plan["device"])
        model.requires_grad_(False)
        model = model.to(device).eval()
        valid_chexficient = _extract_chexficient_split(
            plan, valid_records, valid_ids, "valid", (model, processor, device))
        valid_chexficient_sha = sha256(output / "chexficient_features/valid/image_features.npy")
        valid_chexficient_ids = read_json(output / "chexficient_features/valid/input.json")["images"]
        studies_v, valid_features, valid_targets = pool_candidates(
            valid_records, valid_views, valid_cxr, valid_chexficient, valid_ids, valid_chexficient_ids)
        if valid_targets is None:
            raise ValueError("official validation records must include labels")
        valid_x = valid_features[candidate]
        if valid_x.shape[1] != FEATURE_WIDTHS[candidate]:
            raise ValueError("validation feature width differs from the registered recipe")
        old_current, _, _ = load_submission(plan["current_valid"], studies_v)
        old_legacy, _, _ = load_submission(plan["legacy_valid"], studies_v)
        if recipe_sha != sha256(output / "recipe.json") or head_sha != sha256(output / "final_head.npz"):
            raise ValueError("registered recipe or final head changed before validation")
        if recipe.get("feature_identity") != feature_id:
            raise ValueError("validation candidate feature identity differs from final head")
        valid_prediction = predict_linear_probe(_load_head(output / "final_head.npz"), valid_x)
        _validate_probabilities(valid_prediction, valid_targets.shape, "validation probabilities")
        valid_csv = output / "candidate_valid.csv"
        report_path = output / "validation_report.json"
        if report_path.exists():
            registered = read_json(report_path)
            if (registered.get("candidate") != candidate
                    or registered.get("recipe_sha256") != recipe_sha
                    or registered.get("head_sha256") != head_sha
                    or not valid_csv.is_file()
                    or registered.get("candidate_valid_csv_sha256") != sha256(valid_csv)):
                raise ValueError("registered validation result or CSV changed")
        else:
            from cxr.submission import write_submission
            write_submission(valid_csv, studies_v, valid_prediction)
        replay, _, _ = load_submission(valid_csv, studies_v)
        if not np.array_equal(valid_prediction, replay):
            raise ValueError("candidate validation CSV does not replay exactly")
        comparisons = {
            "incumbent": compare_arrays([row["subject_id"] for row in studies_v],
                valid_targets, old_current, valid_prediction, bootstrap=1000, seed=SEED),
            "legacy": compare_arrays([row["subject_id"] for row in studies_v],
                valid_targets, old_legacy, valid_prediction, bootstrap=1000, seed=SEED),
        }
        atomic_json(report_path, {
            "candidate": candidate,
            "comparisons": comparisons,
            "promoted": validation_gate(comparisons),
            "recipe_sha256": recipe_sha,
            "head_sha256": head_sha,
            "candidate_valid_csv_sha256": sha256(valid_csv),
            "valid_chexficient_features_sha256": valid_chexficient_sha,
            "new_online_score": None,
            "bootstrap": {"replicates": 1000, "seed": SEED},
        })
        promoted = validation_gate(comparisons)
        del valid_cxr, valid_chexficient, valid_features, valid_x, model, processor
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if not promoted:
            finish("retain incumbent; fixed official validation gate failed")
            return

        def export_test():
            stage("extract_chexficient_test_and_export", selected_candidate=candidate)
            _materialize_deferred_cxrlt(plan, "test")
            test_records, test_views, test_ids, test_cxr = _split_inputs(plan, "test")
            _validate_split_disjointness(records, test_records, "test")
            _validate_split_disjointness(valid_records, test_records, "test")
            test_model, test_processor = load_encoder(plan["model_dir"], plan["device"])
            test_model.requires_grad_(False)
            test_model = test_model.to(device).eval()
            test_chexficient = _extract_chexficient_split(
                plan, test_records, test_ids, "test", (test_model, test_processor, device))
            test_chexficient_ids = read_json(output / "chexficient_features/test/input.json")["images"]
            test_studies, test_features, test_targets = pool_candidates(
                test_records, test_views, test_cxr, test_chexficient, test_ids, test_chexficient_ids)
            if test_targets is not None:
                raise ValueError("test records unexpectedly contain labels")
            if len(test_studies) != 1414 or len(test_studies) * 10 != 14140:
                raise ValueError("test manifest coverage differs from the registered dataset contract")
            prediction = predict_linear_probe(
                _load_head(output / "final_head.npz"), test_features[candidate])
            if prediction.shape != (1414, 10) or not np.isfinite(prediction).all() or np.any(
                    (prediction < 0) | (prediction > 1)):
                raise ValueError("test predictions are invalid")
            csv_path = output / "submission_test_1414_chexficient_study_20261108.csv"
            write_submission(csv_path, test_studies, prediction)
            verification = validate_submission(csv_path, test_studies)
            replay, _, _ = load_submission(csv_path, test_studies)
            if not np.array_equal(prediction, replay):
                raise ValueError("saved test CSV does not replay exactly")
            atomic_json(output / "submission_verification.json", {
                **verification, "sha256": sha256(csv_path), "online_score": None,
                "replay_max_error": 0.0, "recipe_sha256": recipe_sha,
                "head_sha256": head_sha,
            })
            state["test_extracted"] = True

        test_exported = _gate_then_export(comparisons, export_test)
        finish("promote locally; both fixed validation comparisons passed" if test_exported
               else "retain incumbent; validation gate failed",
               promoted=test_exported, test_exported=test_exported)
    except Exception as error:
        state.update(status="failed", updated_utc=now(), error=repr(error),
                     traceback=traceback.format_exc())
        atomic_json(output / "experiment.json", state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-experiment", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--run", action="store_true", help="prepare then execute the frozen snapshot")
    parser.add_argument("--resume", action="store_true", help="execute an existing prepared snapshot")
    parser.add_argument("--batch-size", type=int, default=CHEXFICIENT_BATCH_SIZE)
    parser.add_argument("--preprocess-workers", type=int, default=CHEXFICIENT_PREPROCESS_WORKERS)
    args = parser.parse_args(argv)
    if args.run and args.resume:
        parser.error("choose either --run or --resume")
    output = safe_output(args.output)
    if args.resume:
        if not (output / "plan.json").is_file():
            parser.error("--resume requires a prepared run directory")
    else:
        output = prepare(output, args.source_experiment, batch_size=args.batch_size,
                         preprocess_workers=args.preprocess_workers)
    if not (args.run or args.resume):
        print(json.dumps({"status": "prepared", "output": str(output)}), flush=True)
        return
    snapshot = output / "source/scripts/run_chexficient_study_probe.py"
    if Path(__file__).resolve() != snapshot.resolve():
        raise SystemExit(subprocess.call([sys.executable, "-u", str(snapshot),
                                         "--output", str(output), "--resume"]))
    run(output)


if __name__ == "__main__":
    main()
