"""Immutable input registration for the fixed BioViL-T study probe."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import stat
import numpy as np

from cxr.safety import safe_output
from scripts.evaluate_frozen_study_probe import _json_write, _sha256

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "outputs/cxrlt-supervised-study-probe-20261108"
DEFAULT_OUTPUT = ROOT / "outputs/biovil-t-study-probe-20261108"
CHEX_RUN = ROOT / "outputs/chexficient-study-probe-20261108"
PLAN_PATH = ROOT / "docs/BIOVIL_T_STUDY_PROBE_PLAN_20261108.md"
BIOVIL_DIR = ROOT / "weights/medical/biovil-t"
BIOVIL_REVISION = "301f526e823b805d3fe712d0cf06a66f042789c5"
BIOVIL_WEIGHT_SHA256 = "b2399d73dc2a68b9f3a1950e864ae0ecd24093fb07aa459d7e65807ebdc0fb77"
BIOVIL_WEIGHT_BYTES = 109745561
BIOVIL_ADAPTER_SHA256 = "e6ad1ce0325d507c363db9de60f0bc6f42d12dd9fd1d0cbe5e42f59c727befb8"
CHEX_REVISION = "34bb5ea905c4cb504d2b73c9a6687a5d5082f52a"
CHEX_WEIGHT_SHA256 = "f691262c3f77c3c850bebff420180602cf4ca5d5214449e377366f3205548336"
CHEX_ADAPTER_SHA256 = "f45aca9b42d21953aa459e43e5b665df1f22f0228e6457f5583bfd1d1ee3e6c3"
CHEX_TRAIN_SHA256 = "373aef3b1c79c8815b12ebda51e358973ebd0e8628596101ee1dd1755738ded2"
ENGINEERING = (
    ROOT / "outputs/biovil-t-preparation-20261108/cpu_engineering_check.json",
    ROOT / "outputs/biovil-t-preparation-20261108/gpu_engineering_check.json",
    ROOT / "outputs/biovil-t-preparation-20261108/extraction_integration_check.json",
)
CHEX_AUDITS = (
    ROOT / "outputs/chexficient-study-probe-20261108-independent-oof-verification.json",
    ROOT / "outputs/chexficient-study-probe-20261108-independent-final-verification.json",
)
SOURCE_FILES = (
    "scripts/run_biovil_t_study_probe.py", "scripts/biovil_t_probe_inputs.py",
    "scripts/run_chexficient_study_probe.py", "scripts/run_frozen_study_probe.py",
    "scripts/evaluate_frozen_study_probe.py", "scripts/run_bottleneck_study_probe.py",
    "scripts/compare_runs.py", "cxr/__init__.py", "cxr/data.py", "cxr/metrics.py",
    "cxr/safety.py", "cxr/submission.py", "cxr/study_probe.py", "cxr/bottleneck_probe.py",
    "cxr/cxrlt_transfer.py", "cxr/chexficient_transfer.py", "cxr/biovil_t_transfer.py",
    "docs/BIOVIL_T_STUDY_PROBE_PLAN_20261108.md", "docs/biovil_t_source_manifest_20261108.json",
    "docs/chexficient_source_manifest_20261108.json",
)
INCUMBENT_FILES = (
    "plan.json", "experiment.json", "recipe.json", "final_head.npz", "report.json",
    "validation_report.json", "submission_verification.json", "train_records.json",
    "oof_mean_probabilities.npy", "oof_targets.npy", "oof_study_ids.json", "candidate_valid.csv",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return _sha256(Path(path))


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write(path, value):
    _json_write(Path(path), value)


def _copy(source, destination, inputs, snapshots, snapshot_root):
    source = Path(source).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = sha256(source)
    inputs[str(source)] = digest
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != digest:
            raise ValueError(f"snapshot destination already contains different bytes: {destination}")
    else:
        shutil.copy2(source, destination)
    if sha256(destination) != digest:
        raise ValueError(f"snapshot copy hash mismatch: {source}")
    destination.chmod(destination.stat().st_mode & ~stat.S_IWRITE)
    snapshots[destination.relative_to(snapshot_root).as_posix()] = digest
    return destination


def _environment():
    names = ("numpy", "torch", "torchvision", "Pillow", "timm", "transformers")
    return {"python": __import__("sys").version,
            **{name: importlib.metadata.version(name) for name in names}}


def _validate_chex_registration(root=CHEX_RUN, audits=CHEX_AUDITS):
    root = Path(root).resolve()
    plan_path, lock_path = root / "plan.json", root / "plan_lock.json"
    if sha256(plan_path) != read_json(lock_path).get("sha256"):
        raise ValueError("CheXficient prerequisite plan lock mismatch")
    plan, state = read_json(plan_path), read_json(root / "experiment.json")
    if (state.get("status") != "complete" or state.get("candidate_promoted") is not False
            or state.get("test_extracted") is not False or state.get("test_csv_exported") is not False):
        raise ValueError("CheXficient prerequisite is not a completed, non-promoted, no-test run")
    registration = read_json(root / "artifacts.json")
    if not isinstance(registration, dict) or not registration:
        raise ValueError("CheXficient artifact registration is missing")
    # The valid matrix is intentionally deferred. Do not hash or open it here.
    deferred_key = "chexficient_features\\valid\\image_features.npy"
    deferred_sha = registration.get(deferred_key)
    if not isinstance(deferred_sha, str):
        raise ValueError("CheXficient valid matrix hash is not registered")
    for relative, digest in registration.items():
        normalized = relative.replace("/", "\\")
        if normalized == deferred_key:
            continue
        path = root / Path(relative.replace("\\", "/"))
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"CheXficient registered artifact changed: {relative}")
    for audit in audits:
        evidence = read_json(audit)
        if Path(audit).name == Path(CHEX_AUDITS[0]).name:
            if evidence.get("status") != "verified_oof" or evidence.get("runner_status") != "running":
                raise ValueError("independent CheXficient OOF verification failed")
        elif (evidence.get("status") != "verified_terminal_validation_decision"
              or evidence.get("runner_status") != "complete"
              or evidence.get("candidate_promoted") is not False):
            raise ValueError("independent CheXficient terminal verification failed")
    train = root / "chexficient_features/train"
    binding, cache_registration = read_json(train / "binding.json"), read_json(train / "registration.json")
    if (binding.get("plan_sha256") != read_json(lock_path).get("sha256")
            or binding.get("model_revision") != CHEX_REVISION
            or binding.get("weight_sha256") != CHEX_WEIGHT_SHA256
            or binding.get("adapter_sha256") != CHEX_ADAPTER_SHA256
            or cache_registration.get("matrix_sha256") != CHEX_TRAIN_SHA256
            or cache_registration.get("plan_sha256") != binding.get("plan_sha256")):
        raise ValueError("registered CheXficient train cache binding differs from fixed protocol")
    for filename, key in (("binding.json", "binding_sha256"), ("input.json", "input_sha256"),
                          ("progress.json", "progress_sha256"), ("image_features.npy", "matrix_sha256")):
        path = train / filename
        expected = cache_registration.get(key)
        if not expected or sha256(path) != expected:
            raise ValueError(f"registered CheXficient train cache changed: {filename}")
    progress = read_json(train / "progress.json")
    if progress.get("status") != "complete" or progress.get("sha256") != CHEX_TRAIN_SHA256:
        raise ValueError("CheXficient train feature matrix is incomplete")
    return {"root": str(root.resolve()), "plan_sha256": sha256(plan_path),
            "artifact_manifest_sha256": sha256(root / "artifacts.json"),
            "deferred_valid_matrix_sha256": deferred_sha,
            "train_registration": cache_registration,
            "audit_paths": [str(Path(a).resolve()) for a in audits],
            "audit_sha256": {Path(a).name: sha256(a) for a in audits}}


def _validate_biovil_assets():
    manifest_path = ROOT / "docs/biovil_t_source_manifest_20261108.json"
    manifest = read_json(manifest_path)
    if manifest.get("revision") != BIOVIL_REVISION:
        raise ValueError("BioViL-T revision differs from fixed protocol")
    files = manifest.get("files", {})
    provenance = read_json(BIOVIL_DIR / "provenance.json")
    if provenance.get("revision") != BIOVIL_REVISION:
        raise ValueError("BioViL-T local provenance revision mismatch")
    if provenance.get("code_revision") != "b67c1d27c6b17d8e8ff01f8c507f3cabdb307388":
        raise ValueError("BioViL-T author source revision mismatch")
    for rel, item in files.items():
        if rel.startswith("source/"):
            path = BIOVIL_DIR / Path(rel)
        else:
            path = BIOVIL_DIR / rel
        if not path.is_file() or sha256(path) != item.get("sha256"):
            raise ValueError(f"BioViL-T pinned asset changed: {rel}")
    if (files.get("biovil_t_image_model_proj_size_128.pt", {}).get("sha256") != BIOVIL_WEIGHT_SHA256
            or files["biovil_t_image_model_proj_size_128.pt"].get("bytes") != BIOVIL_WEIGHT_BYTES):
        raise ValueError("BioViL-T checkpoint differs from fixed protocol")
    return manifest, provenance


def prepare(output=DEFAULT_OUTPUT, source=DEFAULT_SOURCE):
    output, source = Path(output).resolve(), Path(source).resolve()
    if output == source or output.is_relative_to(source) or output.exists() and any(output.iterdir()):
        raise ValueError("BioViL-T output must be a new directory outside the incumbent")
    if not PLAN_PATH.is_file():
        raise FileNotFoundError(f"fixed BioViL-T protocol missing: {PLAN_PATH}")
    chex = _validate_chex_registration()
    incumbent_plan = read_json(source / "plan.json")
    state = read_json(source / "experiment.json")
    if state.get("status") != "complete" or state.get("candidate_promoted") is not True:
        raise ValueError("CXR-LT incumbent is not complete and promoted")
    incumbent_recipe = read_json(source / "recipe.json")
    if incumbent_recipe.get("head_sha256") != sha256(source / "final_head.npz"):
        raise ValueError("CXR-LT incumbent final head does not match its recipe")
    data_root = Path(incumbent_plan["data_root"]).resolve()
    output = safe_output(output, data_root, inputs=(source, CHEX_RUN))
    output.mkdir(parents=True, exist_ok=True)
    snapshot, source_root = output / "snapshot", output / "source"
    snapshot.mkdir(); source_root.mkdir()
    inputs, snapshots, source_locks = {}, {}, {}
    incumbent = snapshot / "incumbent"
    for name in INCUMBENT_FILES:
        _copy(source / name, incumbent / name, inputs, snapshots, snapshot)
    # Historical test CSVs are deliberately excluded from the input lock traversal.
    locked = {}
    for index, (name, digest) in enumerate(incumbent_plan.get("input_locks", {}).items()):
        original = Path(name).resolve()
        if incumbent_plan.get("old_test") and original == Path(incumbent_plan["old_test"]).resolve():
            continue
        if sha256(original) != digest:
            raise ValueError(f"incumbent input changed: {original}")
        copy = _copy(original, snapshot / "locked_inputs" / f"{index:03d}-{original.name}", inputs, snapshots, snapshot)
        locked[str(original)] = str(copy)
    for filename in ("input.json", "progress.json", "image_features.npy"):
        _copy(source / "train_features" / filename, incumbent / "train_features" / filename, inputs, snapshots, snapshot)
    train_progress = read_json(source / "train_features/progress.json")
    if train_progress.get("status") != "complete" or sha256(source / "train_features/image_features.npy") != train_progress.get("sha256"):
        raise ValueError("CXR-LT train feature cache is incomplete")
    deferred_cxrlt = {}
    for split in ("valid", "test"):
        folder = source / f"{split}_features"
        manifest, progress = read_json(folder / "input.json"), read_json(folder / "progress.json")
        if progress.get("status") != "complete" or not isinstance(progress.get("sha256"), str):
            raise ValueError(f"CXR-LT {split} cache metadata is incomplete")
        for filename in ("input.json", "progress.json"):
            _copy(folder / filename, incumbent / f"{split}_features" / filename, inputs, snapshots, snapshot)
        deferred_cxrlt[split] = {"cache_dir": str(folder), "input_sha256": sha256(folder / "input.json"),
            "progress_sha256": sha256(folder / "progress.json"), "matrix_sha256": progress["sha256"],
            "images_sha256": _json_hash(manifest.get("images", []))}

    # Keep only the old CheX train cache; valid stays source-backed and unread until triple is chosen.
    chex_root = Path(chex["root"])
    chex_train = chex_root / "chexficient_features/train"
    chex_snapshot = snapshot / "chexficient_train"
    for filename in ("binding.json", "registration.json", "input.json", "progress.json", "image_features.npy"):
        _copy(chex_train / filename, chex_snapshot / filename, inputs, snapshots, snapshot)
    chex_plan = read_json(chex_root / "plan.json")
    chex_model_source = Path(chex_plan["model_dir"])
    chex_model_snapshot = snapshot / "chexficient_model"
    for model_file in sorted(path for path in chex_model_source.rglob("*") if path.is_file()):
        _copy(model_file, chex_model_snapshot / model_file.relative_to(chex_model_source),
              inputs, snapshots, snapshot)
    chex_valid = chex_root / "chexficient_features/valid"
    chex_valid_registration = read_json(chex_valid / "registration.json")
    reused_chex = {"train_cache_dir": str(chex_snapshot), "train_registration": chex["train_registration"],
        "valid_cache_dir": str(chex_valid), "valid_registration": chex_valid_registration,
        "valid_matrix_sha256": chex["deferred_valid_matrix_sha256"]}

    manifest, provenance = _validate_biovil_assets()
    model_snapshot = snapshot / "biovil_model"
    for rel in manifest["files"]:
        _copy(BIOVIL_DIR / rel, model_snapshot / rel, inputs, snapshots, snapshot)
    _copy(BIOVIL_DIR / "provenance.json", model_snapshot / "provenance.json", inputs, snapshots, snapshot)
    for path in ENGINEERING:
        evidence = read_json(path)
        if evidence.get("status") != "passed":
            raise ValueError(f"BioViL-T engineering evidence failed: {path.name}")
        _copy(path, snapshot / "engineering" / path.name, inputs, snapshots, snapshot)
    cpu, gpu, extraction = [read_json(path) for path in ENGINEERING]
    if (cpu.get("model_revision") != BIOVIL_REVISION or cpu.get("checkpoint_sha256") != BIOVIL_WEIGHT_SHA256
            or cpu.get("feature_size") != 384 or cpu.get("processor_max_abs_error") != 0.0
            or gpu.get("model_revision") != BIOVIL_REVISION or gpu.get("weights_sha256") != BIOVIL_WEIGHT_SHA256
            or gpu.get("adapter_sha256") != BIOVIL_ADAPTER_SHA256 or gpu.get("batch_size") != 2
            or gpu.get("amp") != "cuda float16" or extraction.get("batch_size") != 2
            or extraction.get("preprocess_workers") != 2 or extraction.get("resume_vs_fresh_max_abs_error") != 0.0):
        raise ValueError("BioViL-T engineering evidence does not match pinned extraction protocol")
    for relative in SOURCE_FILES:
        _copy(ROOT / relative, source_root / relative, inputs, source_locks, source_root)

    # Freeze the prerequisite's exact registrations and independent audit evidence.
    prereq_snapshot = snapshot / "chexficient_prerequisite"
    for filename in ("plan.json", "plan_lock.json", "experiment.json", "artifacts.json"):
        _copy(chex_root / filename, prereq_snapshot / filename, inputs, snapshots, snapshot)
    for audit in CHEX_AUDITS:
        _copy(audit, prereq_snapshot / "independent" / Path(audit).name, inputs, snapshots, snapshot)
    for relative, digest in chex_plan.get("source_locks", {}).items():
        original = chex_root / "source" / relative
        if sha256(original) != digest:
            raise ValueError(f"CheXficient frozen source changed: {relative}")
        _copy(original, prereq_snapshot / "source" / relative, inputs, snapshots, snapshot)
    for relative, digest in chex_plan.get("incumbent_source_locks", {}).items():
        original = chex_root / "snapshot" / "incumbent_source" / relative
        if sha256(original) != digest:
            raise ValueError(f"CXR-LT incumbent source in CheXficient snapshot changed: {relative}")
        _copy(original, prereq_snapshot / "incumbent_source" / relative, inputs, snapshots, snapshot)
    legacy_sources = []
    chex_snapshot_root = chex_root / "snapshot"
    for index, row in enumerate(chex_plan["legacy_target_sources"]):
        ids = _copy(chex_snapshot_root / row["ids"], snapshot / "legacy_targets" / f"{index:02d}/study_ids.json", inputs, snapshots, snapshot)
        labels = _copy(chex_snapshot_root / row["targets"], snapshot / "legacy_targets" / f"{index:02d}/study_targets.npy", inputs, snapshots, snapshot)
        legacy_sources.append({"ids": str(ids.relative_to(snapshot)), "targets": str(labels.relative_to(snapshot))})
    partition = _copy(Path(incumbent_plan["partition"]), snapshot / "partition.json", inputs, snapshots, snapshot)
    old_oof = Path(incumbent_plan["old_oof"])
    legacy = snapshot / "legacy_oof"
    for filename in ("oof_component_probabilities.npy", "oof_study_ids.json", "plan.json"):
        _copy(old_oof / filename, legacy / filename, inputs, snapshots, snapshot)
    legacy_valid = _copy(Path(incumbent_plan["old_valid"]), snapshot / "legacy_valid.csv", inputs, snapshots, snapshot)

    packages = ("numpy", "torch", "torchvision", "Pillow", "timm", "transformers")
    plan = {"protocol": str(PLAN_PATH.resolve()), "protocol_sha256": sha256(PLAN_PATH),
        "chex_registration": chex, "incumbent_source": str(source), "incumbent_snapshot": str(incumbent),
        "incumbent_plan_sha256": sha256(source / "plan.json"), "incumbent_recipe_sha256": sha256(source / "recipe.json"),
        "incumbent_head_sha256": sha256(source / "final_head.npz"), "data_root": str(data_root),
        "partition": str(partition), "legacy_oof": str(legacy), "legacy_valid": str(legacy_valid),
        "legacy_target_sources": legacy_sources, "current_valid": str(incumbent / "candidate_valid.csv"),
        "model_dir": str(model_snapshot), "chex_model_dir": str(chex_model_snapshot),
        "reused_chex": reused_chex, "deferred_cxrlt": deferred_cxrlt,
        "source_locks": source_locks, "snapshot_locks": snapshots, "input_locks": inputs,
        "locked_input_copies": locked, "engineering_evidence": [p.name for p in ENGINEERING],
        "environment": _environment(), "device": incumbent_plan["device"],
        "batch_size": 2, "preprocess_workers": 2, "test_extracted": False,
        "model_revision": BIOVIL_REVISION, "weight_sha256": BIOVIL_WEIGHT_SHA256,
        "manifest_sha256": sha256(ROOT / "docs/biovil_t_source_manifest_20261108.json"),
        "chex_valid_deferred": True, "candidates": ["joint", "triple"],
        "feature_widths": {"joint": 1438, "triple": 1950},
        "fit": {"l2": 0.01, "max_iter": 200, "seed": 20261108, "class_balance": False},
        "folds": [0, 1, 2, "audit"]}
    _write(output / "plan.json", plan)
    _write(output / "plan_lock.json", {"sha256": sha256(output / "plan.json")})
    _write(output / "experiment.json", {"status": "prepared", "phase": None, "test_extracted": False,
        "created_utc": datetime.now(timezone.utc).isoformat()})
    return output


def verify_plan(output, plan):
    output = Path(output).resolve()
    if _environment() != plan["environment"]:
        raise ValueError("software environment changed")
    for path, digest in plan["input_locks"].items():
        if sha256(path) != digest:
            raise ValueError(f"locked input changed: {path}")
    for rel, digest in plan["source_locks"].items():
        if sha256(output / "source" / rel) != digest:
            raise ValueError(f"source snapshot changed: {rel}")
    for rel, digest in plan["snapshot_locks"].items():
        if sha256(output / "snapshot" / rel) != digest:
            raise ValueError(f"snapshot changed: {rel}")
    if sha256(plan["protocol"]) != plan["protocol_sha256"]:
        raise ValueError("fixed BioViL-T protocol changed")
    if sha256(output / "plan.json") != read_json(output / "plan_lock.json").get("sha256"):
        raise ValueError("frozen BioViL-T plan lock mismatch")
    current = _validate_chex_registration(plan["chex_registration"]["root"],
                                          plan["chex_registration"]["audit_paths"])
    if current != plan["chex_registration"]:
        raise ValueError("CheXficient prerequisite registration changed")


def load_registered_plan(output):
    output = Path(output).resolve()
    plan = read_json(output / "plan.json")
    verify_plan(output, plan)
    plan["plan_sha256"] = sha256(output / "plan.json")
    plan["output_dir"] = str(output)
    return plan


def _load_reused_chex(plan, split, identities):
    if split not in ("train", "valid"):
        raise ValueError("only registered CheXficient train/valid caches may be reused")
    if split == "valid" and not plan.get("chex_valid_unlocked"):
        raise ValueError("CheXficient valid cache is deferred until triple is selected")
    registration = plan["reused_chex"]["train_registration" if split == "train" else "valid_registration"]
    cache = Path(plan["reused_chex"]["train_cache_dir" if split == "train" else "valid_cache_dir"])
    for filename, key in (("binding.json", "binding_sha256"), ("input.json", "input_sha256"),
                          ("progress.json", "progress_sha256"), ("image_features.npy", "matrix_sha256")):
        if sha256(cache / filename) != registration.get(key):
            raise ValueError(f"registered CheXficient {split} cache changed: {filename}")
    binding, manifest, progress = (read_json(cache / "binding.json"), read_json(cache / "input.json"),
                                   read_json(cache / "progress.json"))
    expected_sha = plan["reused_chex"]["valid_matrix_sha256"] if split == "valid" else CHEX_TRAIN_SHA256
    if (binding.get("plan_sha256") != registration.get("plan_sha256")
            or binding.get("model_revision") != CHEX_REVISION
            or binding.get("weight_sha256") != CHEX_WEIGHT_SHA256
            or binding.get("adapter_sha256") != CHEX_ADAPTER_SHA256
            or progress.get("status") != "complete" or progress.get("sha256") != expected_sha
            or registration.get("matrix_sha256") != expected_sha):
        raise ValueError(f"CheXficient {split} cache binding/progress mismatch")
    expected_ids = manifest.get("images", [])
    if expected_ids != identities:
        raise ValueError(f"CheXficient {split} image identity/order mismatch")
    values = np.load(cache / "image_features.npy", mmap_mode="r", allow_pickle=False)
    if values.shape != (len(identities), 512) or values.dtype != np.float32:
        raise ValueError(f"CheXficient {split} matrix has invalid shape or dtype")
    for start in range(0, len(values), 512):
        if not np.isfinite(values[start:start + 512]).all():
            raise ValueError(f"CheXficient {split} matrix contains non-finite rows")
    return values
