"""Prepare or run the fixed two-candidate EVA-X study probe."""
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
from cxr.safety import fresh_run_directory, safe_output
from cxr.study_probe import fit_linear_probe, pool_study_features, predict_linear_probe
from scripts.run_bottleneck_study_probe import legacy_oof, training_gate, validation_gate
from scripts.evaluate_frozen_study_probe import (
    FOLDS, _fold_memberships, _json_write, _load_head, _save_head, _sha256,
)


SEED = 20261108
MAX_ITER = 200
L2 = 0.01
CLASS_BALANCE = False
EVA_BATCH_SIZE = 16
EVA_PREPROCESS_WORKERS = 2
FEATURE_WIDTHS = {"eva": 389, "joint": 1443}
CANDIDATES = ("eva", "joint")
PLAN_PATH = ROOT / "docs/EVA_STUDY_PROBE_PLAN_20261108.md"
STAGE_STATE_PATH = ROOT / "outputs/cxrlt-stage-finetune-20261108/experiment.json"
DEFAULT_SOURCE = ROOT / "outputs/cxrlt-supervised-study-probe-20261108"
DEFAULT_OUTPUT = ROOT / "outputs/eva-study-probe-20261108"

SOURCE_FILES = (
    "scripts/run_eva_study_probe.py",
    "scripts/run_frozen_study_probe.py",
    "scripts/evaluate_frozen_study_probe.py",
    "scripts/run_bottleneck_study_probe.py",
    "scripts/compare_runs.py",
    "cxr/__init__.py", "cxr/data.py", "cxr/metrics.py", "cxr/safety.py",
    "cxr/submission.py", "cxr/study_probe.py", "cxr/bottleneck_probe.py",
    "cxr/eva_x_transfer.py",
    "docs/EVA_STUDY_PROBE_PLAN_20261108.md",
)
EVA_MODEL_FILES = (
    "eva_x_small_patch16_merged520k_mim_chexpert_ft.pth",
    "eva_x.py", "models_eva.py", "models_vit.py", "rope.py",
    "provenance.json", "huggingface-README.md", "upstream-README.md",
    "upstream-LICENSE.txt", "upstream-requirements.txt",
    "upstream/classification/train_files/eva_x/chexpert/vit_s.sh",
    "upstream/classification/utils/datasets.py",
    "upstream/classification/utils/dataloader_med.py",
)
INCUMBENT_ARTIFACTS = (
    "plan.json", "experiment.json", "recipe.json", "final_head.npz",
    "report.json", "validation_report.json", "train_records.json",
    "oof_mean_probabilities.npy", "oof_targets.npy", "oof_study_ids.json",
    "candidate_valid.csv", "submission_test_1414_cxrlt_study_20261108.csv",
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


def execution_settings(batch_size=EVA_BATCH_SIZE,
                       preprocess_workers=EVA_PREPROCESS_WORKERS):
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("EVA-X batch size must be a positive integer")
    if not isinstance(preprocess_workers, int) or preprocess_workers < 0:
        raise ValueError("EVA-X preprocessing workers must be a nonnegative integer")
    return {"batch_size": batch_size, "preprocess_workers": preprocess_workers}


def _require_preparation_authority(plan_path=PLAN_PATH, stage_state_path=STAGE_STATE_PATH):
    if not Path(plan_path).is_file():
        raise FileNotFoundError(f"fixed EVA-X protocol is required before preparation: {plan_path}")
    stage_state_path = Path(stage_state_path).resolve()
    stage_root = stage_state_path.parent
    required = ("experiment.json", "plan.json", "plan_lock.json", "artifacts.json")
    missing = [name for name in required if not (stage_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"terminal stage registration is incomplete: {', '.join(missing)}")
    stage = read_json(stage_root / "experiment.json")
    if stage.get("status") != "complete":
        raise ValueError("EVA-X preparation is blocked until the CXR-LT stage is complete")
    if stage.get("candidate_promoted") is not False:
        raise ValueError("CXR-LT stage promoted a new incumbent; revise the EVA-X comparison protocol first")
    stage_plan = stage_root / "plan.json"
    if sha256(stage_plan) != read_json(stage_root / "plan_lock.json").get("sha256"):
        raise ValueError("CXR-LT stage plan hash does not match plan_lock.json")
    artifacts = read_json(stage_root / "artifacts.json")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("CXR-LT stage artifacts registration is empty or invalid")
    for relative, digest in artifacts.items():
        path = stage_root / relative
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"CXR-LT stage artifact changed or is missing: {relative}")
    recipe_path = stage_root / "recipe.json"
    if recipe_path.exists():
        recipe = read_json(recipe_path)
        model_relative = recipe.get("model_file")
        model_path = stage_root / model_relative if model_relative else None
        if (not model_path or not model_path.is_file()
                or sha256(model_path) != recipe.get("model_sha256")):
            raise ValueError("CXR-LT final recipe/model registration is incomplete or changed")
        if artifacts.get("recipe.json") != sha256(recipe_path) or artifacts.get(model_relative) != sha256(model_path):
            raise ValueError("CXR-LT final recipe/model are not registered in artifacts.json")
    return stage


def _relative_copy_name(index, path):
    path = Path(path)
    safe_name = "".join(char if char.isalnum() or char in "._-" else "_" for char in path.name)
    return Path("locked_inputs") / f"{index:03d}-{safe_name}"


def prepare(output, source=DEFAULT_SOURCE, *, plan_path=PLAN_PATH,
            stage_state_path=STAGE_STATE_PATH, batch_size=EVA_BATCH_SIZE,
            preprocess_workers=EVA_PREPROCESS_WORKERS):
    """Register a fully copied, immutable source snapshot without running it."""
    _require_preparation_authority(plan_path, stage_state_path)
    settings = execution_settings(batch_size, preprocess_workers)
    recorded_environment = environment()
    output = Path(output).resolve()
    source = Path(source).resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("EVA-X output must be outside its incumbent source experiment")
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
    snapshot = output / "snapshot"
    source_snapshot = output / "source"
    snapshot.mkdir()
    source_snapshot.mkdir()
    input_locks = {}
    snapshot_locks = {}

    # Preserve every registered source input as well as the source experiment's outputs.
    locked_input_copies = {}
    for index, (name, expected_digest) in enumerate(incumbent_plan["input_locks"].items()):
        original = Path(name).resolve()
        if sha256(original) != expected_digest:
            raise ValueError(f"incumbent input changed: {original}")
        relative = _relative_copy_name(index, original)
        copied = _copy_snapshot_file(original, snapshot / relative, input_locks,
                                     snapshot_locks, snapshot)
        locked_input_copies[str(original)] = str(copied)

    incumbent_copy = snapshot / "incumbent"
    for name in INCUMBENT_ARTIFACTS:
        _copy_snapshot_file(source / name, incumbent_copy / name,
                            input_locks, snapshot_locks, snapshot)
    for split in ("train", "valid", "test"):
        for name in SPLIT_CACHE_FILES:
            source_path = source / f"{split}_features" / name
            _copy_snapshot_file(source_path, incumbent_copy / f"{split}_features" / name,
                                input_locks, snapshot_locks, snapshot)
        cache = source / f"{split}_features"
        progress = read_json(cache / "progress.json")
        if (progress.get("status") != "complete"
                or sha256(cache / "image_features.npy") != progress.get("sha256")):
            raise ValueError(f"incumbent {split} feature cache is incomplete or changed")

    source_code_locks = {}
    for relative in SOURCE_FILES:
        source_file = ROOT / relative
        copied = source_snapshot / relative
        _copy_snapshot_file(source_file, copied, input_locks,
                            source_code_locks, source_snapshot)

    # The incumbent's own source snapshot is part of the registered evidence.
    incumbent_source_locks = {}
    for relative, expected_digest in incumbent_plan.get("source_locks", {}).items():
        original = source / "source" / relative
        if not original.is_file() or sha256(original) != expected_digest:
            raise ValueError(f"incumbent source snapshot changed: {relative}")
        copied = _copy_snapshot_file(original, snapshot / "incumbent_source" / relative,
                                     input_locks, snapshot_locks, snapshot)
        incumbent_source_locks[relative] = expected_digest

    eva_model_dir = ROOT / "weights/medical/eva-x-small-chexpert"
    model_snapshot = snapshot / "eva_model"
    provenance = read_json(eva_model_dir / "provenance.json")
    if (provenance.get("model_revision") != "35ddcd6dab6ca99bbdb6cb45c8d1b093aefbd0ee"
            or provenance.get("source_revision") != "eedd642c85339b25682ceb574b644cb9afe88e62"):
        raise ValueError("EVA-X provenance does not match the fixed public revisions")
    provenance_files = {item["file"]: item["sha256"] for item in provenance.get("artifacts", [])}
    for relative in EVA_MODEL_FILES:
        original = eva_model_dir / relative
        expected_digest = provenance_files.get(relative)
        if relative == "eva_x_small_patch16_merged520k_mim_chexpert_ft.pth":
            expected_digest = "1a5b63570ca4094c1118018e44d3012a95b904132942a8edb2fc0e5dc8aa0e97"
        elif relative == "eva_x.py":
            expected_digest = "5c21eb02cd977c35d4b8cf2984081b88b73db49953c1b4c2235f30df29df8219"
        if relative != "provenance.json" and (expected_digest is None or sha256(original) != expected_digest):
            raise ValueError(f"EVA-X model/source file differs from pinned provenance: {relative}")
        _copy_snapshot_file(original, model_snapshot / relative, input_locks, snapshot_locks, snapshot)

    partition_original = Path(incumbent_plan["partition"]).resolve()
    partition_copy = _copy_snapshot_file(
        partition_original, snapshot / "partition.json", input_locks,
        snapshot_locks, snapshot,
    )
    legacy_folder = Path(incumbent_plan["old_oof"]).resolve()
    legacy_copy = snapshot / "legacy_oof"
    for name in ("oof_component_probabilities.npy", "oof_study_ids.json", "plan.json"):
        _copy_snapshot_file(legacy_folder / name, legacy_copy / name,
                            input_locks, snapshot_locks, snapshot)
    legacy_valid_original = Path(incumbent_plan["old_valid"]).resolve()
    legacy_valid_copy = _copy_snapshot_file(
        legacy_valid_original, snapshot / "legacy_valid.csv", input_locks,
        snapshot_locks, snapshot,
    )
    stage_root = Path(stage_state_path).resolve().parent
    stage_registration = {}
    stage_names = ["experiment.json", "plan.json", "plan_lock.json", "artifacts.json"]
    stage_recipe = stage_root / "recipe.json"
    stage_artifacts = read_json(stage_root / "artifacts.json")
    if stage_recipe.exists():
        stage_names.append("recipe.json")
        model_relative = read_json(stage_recipe).get("model_file")
        if model_relative:
            stage_names.append(model_relative)
    elif "fits/full_train/model.pt" in stage_artifacts:
        stage_names.append("fits/full_train/model.pt")
    for name in stage_names:
        copied = _copy_snapshot_file(stage_root / name, snapshot / "stage" / name,
                                     input_locks, snapshot_locks, snapshot)
        stage_registration[name] = sha256(copied)

    # Retain copy mappings needed by the executor, without making it read mutable run outputs.
    plan = {
        "created_utc": now(),
        "protocol": str(Path(plan_path).resolve()),
        "protocol_sha256": sha256(plan_path),
        "stage_state_snapshot": str(snapshot / "stage/experiment.json"),
        "stage_registration_dir": str(stage_root),
        "stage_registration": stage_registration,
        "incumbent_source_locks": incumbent_source_locks,
        "incumbent_source": str(source),
        "incumbent_snapshot": str(incumbent_copy),
        "data_root": str(data_root),
        "model_dir": str(model_snapshot),
        "partition": str(partition_copy),
        "legacy_oof": str(legacy_copy),
        "legacy_valid": str(legacy_valid_copy),
        "current_valid": str(incumbent_copy / "candidate_valid.csv"),
        "locked_input_copies": locked_input_copies,
        "source_locks": source_code_locks,
        "snapshot_locks": snapshot_locks,
        "input_locks": input_locks,
        "environment": recorded_environment,
        "model": "MapleF/eva_x",
        "revision": "35ddcd6dab6ca99bbdb6cb45c8d1b093aefbd0ee",
        "weight_sha256": "1a5b63570ca4094c1118018e44d3012a95b904132942a8edb2fc0e5dc8aa0e97",
        "source_revision": "eedd642c85339b25682ceb574b644cb9afe88e62",
        "candidates": list(CANDIDATES),
        "feature_widths": FEATURE_WIDTHS,
        "fit": {"l2": L2, "max_iter": MAX_ITER, "seed": SEED,
                 "class_balance": CLASS_BALANCE},
        "folds": list(FOLDS),
        **settings,
        "device": incumbent_plan["device"],
        "incumbent_plan_sha256": sha256(source / "plan.json"),
        "incumbent_recipe_sha256": sha256(source / "recipe.json"),
        "incumbent_head_sha256": sha256(source / "final_head.npz"),
        "legacy_valid_sha256": sha256(legacy_valid_copy),
        "stage_candidate_promoted": False,
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
        raise ValueError("fixed EVA-X protocol changed")
    plan_path = output / "plan.json"
    if not (output / "plan_lock.json").is_file() or sha256(plan_path) != read_json(output / "plan_lock.json").get("sha256"):
        raise ValueError("frozen EVA-X plan changed or plan_lock.json is missing")
    _require_preparation_authority(plan["protocol"], Path(plan["stage_registration_dir"]) / "experiment.json")
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
    seen = set()
    for item in identity:
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            raise ValueError("image identity rows must be [dicom_id, path, size, mtime_ns]")
        dicom, path, size, mtime = item
        dicom = str(dicom)
        resolved = str(Path(path).resolve())
        if dicom in seen:
            raise ValueError(f"duplicate image identity: {dicom}")
        seen.add(dicom)
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


def _bind_eva_cache(plan, split, identities):
    cache = Path(plan["output_dir"]) / "eva_features" / split
    cache.mkdir(parents=True, exist_ok=True)
    binding = {
        "plan_sha256": plan["plan_sha256"],
        "model_revision": plan["revision"],
        "weight_sha256": plan["weight_sha256"],
        "eva_source_sha256": plan["snapshot_locks"]["eva_model/eva_x.py"],
        "image_identity_sha256": _json_hash(_identity_rows(identities)),
        "feature_size": 389,
        "input_size": 224,
        "feature_pooling": "eva_x_mean_pooled_patch_features_384_plus_raw_logits_5",
        "batch_size": plan["batch_size"],
        "preprocess_workers": plan["preprocess_workers"],
        "device": plan["device"],
    }
    path = cache / "binding.json"
    if path.exists():
        if read_json(path) != binding:
            raise ValueError(f"EVA-X feature cache binding mismatch: {split}")
    else:
        payload = [item for item in cache.iterdir() if item.name != "binding.json"]
        if payload:
            raise ValueError(f"EVA-X {split} cache payload exists without its binding")
        atomic_json(path, binding)
    return cache


def _json_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _extract_eva_split(plan, records, identities, split, model_bundle):
    from scripts.run_frozen_study_probe import extract_features

    cache = _bind_eva_cache(plan, split, identities)
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
        raise ValueError(f"EVA-X {split} feature cache is incomplete or changed")
    _validate_matrix(values, len(records), 389, f"EVA-X {split} features")
    _register_eva_cache(cache, plan["plan_sha256"])
    return values


def _register_eva_cache(cache, plan_sha256):
    cache = Path(cache)
    binding_path, input_path = cache / "binding.json", cache / "input.json"
    progress_path, matrix_path = cache / "progress.json", cache / "image_features.npy"
    for path in (binding_path, input_path, progress_path, matrix_path):
        if not path.is_file():
            raise ValueError(f"EVA-X feature cache registration is missing {path.name}")
    progress = read_json(progress_path)
    matrix_sha = sha256(matrix_path)
    if progress.get("status") != "complete" or progress.get("sha256") != matrix_sha:
        raise ValueError("EVA-X feature cache cannot be registered before a complete matrix hash")
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
            raise ValueError("registered EVA-X feature cache hashes changed")
    else:
        atomic_json(path, registration)
    return registration


def _study_key_rows(studies):
    return [[str(row["subject_id"]), str(row["study_id"])] for row in studies]


def pool_candidates(records, views, cxrlt_images, eva_images, cxrlt_ids, eva_ids):
    """Pool aligned encoders separately, then form equivalent study-level features."""
    require_exact_image_identity(cxrlt_ids, eva_ids)
    if len(cxrlt_images) != len(records) or len(eva_images) != len(records):
        raise ValueError("feature rows must match the shared record order")
    if cxrlt_images.shape[1] != 1054 or eva_images.shape[1] != 389:
        raise ValueError("image feature width differs from the fixed source contracts")
    studies_c, cxrlt, targets_c = pool_study_features(records, cxrlt_images, views, "mean")
    studies_e, eva, targets_e = pool_study_features(records, eva_images, views, "mean")
    if _study_key_rows(studies_c) != _study_key_rows(studies_e):
        raise ValueError("encoder study pooling produced different ordered studies")
    if ((targets_c is None) != (targets_e is None)
            or (targets_c is not None and not np.array_equal(targets_c, targets_e))):
        raise ValueError("encoder study pooling produced different targets")
    if cxrlt.shape[1] != 1054 or eva.shape[1] != 389:
        raise ValueError("pooled feature width differs from the fixed contract")
    return studies_c, {"eva": eva, "joint": np.concatenate((cxrlt, eva), axis=1)}, targets_c


def _score(targets, probabilities):
    report = evaluate(targets, probabilities)
    return {
        "macro_auc": report["macro_auc"],
        "macro_auc_defined_classes": report["macro_auc_defined_classes"],
        "per_class_auc": [row["auc"] for row in report["per_class"]],
    }


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


def _feature_identity(candidate, source_sha, eva_sha, study_keys):
    return _json_hash({
        "candidate": candidate,
        "width": FEATURE_WIDTHS[candidate],
        "source_cxrlt_sha256": source_sha,
        "eva_features_sha256": eva_sha,
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
    from cxr.eva_x_transfer import load_encoder

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
        if source_baseline.shape != targets.shape or not np.isfinite(source_baseline).all():
            raise ValueError("incumbent OOF probabilities have invalid shape or values")
        legacy = legacy_oof({"legacy_oof": plan["legacy_oof"]}, studies_c)
        if legacy.shape != targets.shape or not np.isfinite(legacy).all():
            raise ValueError("legacy OOF probabilities have invalid shape or values")
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
        stage("extract_eva_train")
        eva_images = _extract_eva_split(plan, records, cxr_ids, "train", model_bundle)
        eva_cache_path = output / "eva_features/train/image_features.npy"
        eva_sha = sha256(eva_cache_path)
        eva_ids = read_json(output / "eva_features/train/input.json")["images"]
        require_exact_image_identity(cxr_ids, eva_ids)
        studies, train_features, pooled_targets = pool_candidates(
            records, views, cxr_images, eva_images, cxr_ids, eva_ids)
        if _study_key_rows(studies) != source_oof_ids or not np.array_equal(pooled_targets, source_targets):
            raise ValueError("pooled joint study IDs or targets differ from incumbent OOF")
        if not np.array_equal(pooled_targets, targets):
            raise ValueError("candidate target vectors differ")
        del cxr_images, eva_images, model, processor, model_bundle
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
                eva_sha,
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
            "selected": None if selected is None else selected["candidate"],
            "selection": "highest eligible OOF Macro-AUC; exact tie prefers eva",
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

        stage("extract_eva_valid", selected_candidate=candidate)
        valid_records, valid_views, valid_ids, valid_cxr = _split_inputs(plan, "valid")
        model, processor = load_encoder(plan["model_dir"], plan["device"])
        model.requires_grad_(False)
        model = model.to(device).eval()
        valid_eva = _extract_eva_split(
            plan, valid_records, valid_ids, "valid", (model, processor, device))
        valid_eva_sha = sha256(output / "eva_features/valid/image_features.npy")
        valid_eva_ids = read_json(output / "eva_features/valid/input.json")["images"]
        studies_v, valid_features, valid_targets = pool_candidates(
            valid_records, valid_views, valid_cxr, valid_eva, valid_ids, valid_eva_ids)
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
            "valid_eva_features_sha256": valid_eva_sha,
            "new_online_score": None,
            "bootstrap": {"replicates": 1000, "seed": SEED},
        })
        promoted = validation_gate(comparisons)
        del valid_cxr, valid_eva, valid_features, valid_x, model, processor
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if not promoted:
            finish("retain incumbent; fixed official validation gate failed")
            return

        def export_test():
            stage("extract_eva_test_and_export", selected_candidate=candidate)
            test_records, test_views, test_ids, test_cxr = _split_inputs(plan, "test")
            test_model, test_processor = load_encoder(plan["model_dir"], plan["device"])
            test_model.requires_grad_(False)
            test_model = test_model.to(device).eval()
            test_eva = _extract_eva_split(
                plan, test_records, test_ids, "test", (test_model, test_processor, device))
            test_eva_ids = read_json(output / "eva_features/test/input.json")["images"]
            test_studies, test_features, test_targets = pool_candidates(
                test_records, test_views, test_cxr, test_eva, test_ids, test_eva_ids)
            if test_targets is not None:
                raise ValueError("test records unexpectedly contain labels")
            if len(test_studies) != 1414 or len(test_studies) * 10 != 14140:
                raise ValueError("test manifest coverage differs from the registered dataset contract")
            prediction = predict_linear_probe(
                _load_head(output / "final_head.npz"), test_features[candidate])
            if prediction.shape != (1414, 10) or not np.isfinite(prediction).all() or np.any(
                    (prediction < 0) | (prediction > 1)):
                raise ValueError("test predictions are invalid")
            csv_path = output / "submission_test_1414_eva_study_20261108.csv"
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
    parser.add_argument("--batch-size", type=int, default=EVA_BATCH_SIZE)
    parser.add_argument("--preprocess-workers", type=int, default=EVA_PREPROCESS_WORKERS)
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
    snapshot = output / "source/scripts/run_eva_study_probe.py"
    if Path(__file__).resolve() != snapshot.resolve():
        raise SystemExit(subprocess.call([sys.executable, "-u", str(snapshot),
                                         "--output", str(output), "--resume"]))
    run(output)


if __name__ == "__main__":
    main()
