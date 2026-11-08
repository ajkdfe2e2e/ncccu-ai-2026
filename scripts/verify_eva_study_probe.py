"""Read-only NumPy verification for a completed EVA-X study-probe run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FOLDS = (0, 1, 2, "audit")
CANDIDATES = ("eva", "joint")
WIDTHS = {"eva": 389, "joint": 1443}
NUM_CLASSES = 10
ATOL = 1e-10
CHUNK_ROWS = 2048
REQUIRED_OOF = (
    "oof_eva_probabilities.npy", "oof_joint_probabilities.npy",
    "oof_targets.npy", "oof_study_ids.json", "oof_report.json",
)


class IncompleteRun(Exception):
    """The runner has not produced a complete, verifiable OOF result."""


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def _normalize_artifact_path(relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError("registered artifact path must be a nonempty string")
    windows = PureWindowsPath(relative)
    portable = relative.replace("\\", "/")
    if windows.is_absolute() or windows.drive or portable.startswith("/"):
        raise ValueError(f"registered artifact path must be relative: {relative}")
    parts = PurePosixPath(portable).parts
    if not parts or any(part in {"..", ""} for part in parts):
        raise ValueError(f"registered artifact path escapes the experiment directory: {relative}")
    normalized = "/".join(parts)
    if normalized in {"", "."}:
        raise ValueError(f"registered artifact path is invalid: {relative}")
    return normalized


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _metrics(targets, probabilities):
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if (targets.ndim != 2 or targets.shape[1] != NUM_CLASSES
            or probabilities.shape != targets.shape):
        raise ValueError("OOF targets and probabilities must both have shape (N, 10)")
    if not np.isfinite(targets).all() or not np.isin(targets, (0, 1)).all():
        raise ValueError("OOF targets must be finite binary labels")
    if (not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))):
        raise ValueError("OOF probabilities must be finite and in [0, 1]")
    aucs = []
    for column in range(NUM_CLASSES):
        y, score = targets[:, column], probabilities[:, column]
        positive, negative = int(y.sum()), int(len(y) - y.sum())
        if positive == 0 or negative == 0:
            aucs.append(None)
            continue
        order = np.argsort(score, kind="mergesort")
        ordered = score[order]
        starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
        ends = np.r_[starts[1:], len(y)]
        ranks = np.empty(len(y), dtype=np.float64)
        ranks[order] = np.repeat((starts + ends + 1) / 2.0, ends - starts)
        positive_rank_sum = ranks[y == 1].sum()
        aucs.append(float((positive_rank_sum - positive * (positive + 1) / 2.0)
                          / (positive * negative)))
    defined = [value for value in aucs if value is not None]
    return {"macro_auc": float(np.mean(defined)) if defined else None,
            "macro_auc_defined_classes": len(defined), "per_class_auc": aucs}


def _gate(targets, baseline, legacy, candidate, folds):
    old, new, older = (_metrics(targets, values)
                       for values in (baseline, candidate, legacy))
    held_rows = []
    for fold in FOLDS:
        mask = folds == fold
        before, after = _metrics(targets[mask], baseline[mask]), _metrics(
            targets[mask], candidate[mask])
        gain = (None if before["macro_auc"] is None or after["macro_auc"] is None
                else after["macro_auc"] - before["macro_auc"])
        held_rows.append({"fold": fold, "baseline_macro_auc": before["macro_auc"],
                          "candidate_macro_auc": after["macro_auc"],
                          "defined_classes": after["macro_auc_defined_classes"],
                          "gain": gain})
    gain = None if old["macro_auc"] is None or new["macro_auc"] is None else new["macro_auc"] - old["macro_auc"]
    legacy_gain = None if older["macro_auc"] is None or new["macro_auc"] is None else new["macro_auc"] - older["macro_auc"]
    positive = sum(row["gain"] is not None and row["gain"] > 0 for row in held_rows)
    passed = bool(old["macro_auc_defined_classes"] == new["macro_auc_defined_classes"] == 10
                  and all(row["defined_classes"] == 10 for row in held_rows)
                  and _metrics(targets, legacy)["macro_auc_defined_classes"] == 10
                  and gain is not None and gain >= 0.001 and positive >= 3
                  and legacy_gain is not None and legacy_gain >= 0.001)
    return {"passed": passed, "baseline": old, "candidate": new, "legacy": older,
            "gain": gain, "legacy_gain": legacy_gain, "positive_held_groups": positive,
            "held_groups": held_rows}


def _compare_recorded(actual, expected, name="report"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f"{name} must be an object")
        for key, value in expected.items():
            if key not in actual:
                raise ValueError(f"{name} is missing {key}")
            _compare_recorded(actual[key], value, f"{name}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{name} differs in length")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _compare_recorded(left, right, f"{name}[{index}]")
    elif isinstance(expected, (float, np.floating)):
        if actual is None or not math.isfinite(float(actual)) or abs(float(actual) - float(expected)) > ATOL:
            raise ValueError(f"{name} differs from independent recomputation")
    elif actual != expected:
        raise ValueError(f"{name} differs from independent recomputation")


def _verify_manifest(output, plan, checks):
    plan_path, lock_path = output / "plan.json", output / "plan_lock.json"
    if not plan_path.is_file() or not lock_path.is_file():
        raise IncompleteRun("plan.json or plan_lock.json is missing")
    plan_sha = _sha256(plan_path)
    if _json(lock_path).get("sha256") != plan_sha:
        raise ValueError("plan hash differs from plan_lock.json")
    if plan.get("plan_sha256") not in (None, plan_sha):
        raise ValueError("plan self-binding hash differs")
    checks.append({"name": "plan_lock", "sha256": plan_sha})
    if plan.get("protocol"):
        protocol = Path(plan["protocol"])
        if not protocol.is_file() or _sha256(protocol) != plan.get("protocol_sha256"):
            raise ValueError("fixed study protocol is missing or changed")
        checks.append({"name": "fixed_protocol", "sha256": plan["protocol_sha256"]})
    for path, expected in plan.get("input_locks", {}).items():
        if not Path(path).is_file() or _sha256(path) != expected:
            raise ValueError(f"locked input is missing or changed: {path}")
    checks.append({"name": "input_locks", "count": len(plan.get("input_locks", {}))})
    for relative, expected in plan.get("source_locks", {}).items():
        path = output / "source" / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"source snapshot is missing or changed: {relative}")
    checks.append({"name": "source_locks", "count": len(plan.get("source_locks", {}))})
    for relative, expected in plan.get("snapshot_locks", {}).items():
        path = output / "snapshot" / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"snapshot is missing or changed: {relative}")
    checks.append({"name": "snapshot_locks", "count": len(plan.get("snapshot_locks", {}))})
    for relative, expected in plan.get("incumbent_source_locks", {}).items():
        path = output / "snapshot" / "incumbent_source" / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"incumbent source snapshot is missing or changed: {relative}")

    artifact_path = output / "artifacts.json"
    if not artifact_path.is_file():
        raise IncompleteRun("artifacts.json is missing")
    artifacts = _json(artifact_path)
    if not isinstance(artifacts, dict) or not artifacts:
        raise IncompleteRun("artifacts.json is empty or invalid")
    normalized_artifacts = {}
    for raw_relative, expected in artifacts.items():
        relative = _normalize_artifact_path(raw_relative)
        if relative in normalized_artifacts:
            raise ValueError(f"duplicate normalized artifact path: {relative}")
        normalized_artifacts[relative] = expected
    for relative, expected in normalized_artifacts.items():
        candidate = (output / relative).resolve()
        if not _inside(candidate, output):
            raise ValueError(f"registered artifact escapes the experiment directory: {relative}")
        if not candidate.is_file():
            raise IncompleteRun(f"registered artifact is missing: {relative}")
        if _sha256(candidate) != expected:
            raise ValueError(f"registered artifact hash changed: {relative}")
    missing = [name for name in REQUIRED_OOF if name not in normalized_artifacts]
    missing += [f"fits/{candidate}/{fold}/head.npz" for candidate in CANDIDATES
                for fold in FOLDS if f"fits/{candidate}/{fold}/head.npz" not in normalized_artifacts]
    missing += [f"fits/{candidate}/{fold}/fit.json" for candidate in CANDIDATES
                for fold in FOLDS if f"fits/{candidate}/{fold}/fit.json" not in normalized_artifacts]
    if missing:
        raise IncompleteRun("required OOF artifacts are not registered: " + ", ".join(missing))
    checks.append({"name": "artifact_hashes", "count": len(normalized_artifacts)})
    return plan_sha, normalized_artifacts


def _folds_for(studies, plan):
    partition_path = Path(plan["partition"])
    partition = _json(partition_path).get("subject_partition")
    if not isinstance(partition, dict):
        raise ValueError("partition subject_partition is missing")
    subjects = [str(row[0]) for row in studies]
    missing = set(subjects) - set(map(str, partition))
    if missing:
        raise ValueError(f"partition omits {len(missing)} training subjects")
    assignments = {str(key): value for key, value in partition.items()}
    folds = np.asarray([assignments[subject] for subject in subjects], dtype=object)
    if any(fold not in FOLDS for fold in folds) or set(folds.tolist()) != set(FOLDS):
        raise ValueError("partition must populate only folds 0, 1, 2, and audit")
    _assert_fold_isolation(folds, subjects)
    return folds


def _assert_fold_isolation(folds, subjects):
    membership = {}
    for fold, subject in zip(folds, subjects):
        membership.setdefault(str(subject), set()).add(fold)
    leaked = [subject for subject, values in membership.items() if len(values) != 1]
    if leaked:
        raise ValueError(f"patient leakage across folds: {leaked[:3]}")
    for fold in FOLDS:
        held_patients = {str(subject) for subject, assigned in zip(subjects, folds) if assigned == fold}
        training_patients = {str(subject) for subject, assigned in zip(subjects, folds) if assigned != fold}
        if held_patients & training_patients:
            raise ValueError(f"patient leakage in held fold {fold}")


def _validate_cache(output, plan, records, checks):
    incumbent = Path(plan["incumbent_snapshot"])
    cxr_cache = incumbent / "train_features"
    eva_cache = output / "eva_features" / "train"
    cxr_path, eva_path = cxr_cache / "image_features.npy", eva_cache / "image_features.npy"
    for cache, matrix_path, width in ((cxr_cache, cxr_path, 1054), (eva_cache, eva_path, 389)):
        for name in ("input.json", "progress.json", "image_features.npy"):
            if not (cache / name).is_file():
                raise IncompleteRun(f"training {cache.name} cache is missing {name}")
        progress = _json(cache / "progress.json")
        matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
        if (progress.get("status") != "complete"
                or progress.get("completed") != len(records)
                or progress.get("sha256") != _sha256(matrix_path)
                or matrix.shape != (len(records), width)
                or matrix.dtype != np.float32):
            raise ValueError(f"training {cache.name} feature cache is incomplete or invalid")
    cxr_input = _json(cxr_cache / "input.json")
    eva_input = _json(eva_cache / "input.json")
    identities = cxr_input.get("images")
    if not isinstance(identities, list) or len(identities) != len(records):
        raise ValueError("CXR training image identities do not align with records")
    if identities != eva_input.get("images"):
        raise ValueError("CXR and EVA image identities differ")
    for record, identity in zip(records, identities):
        if len(identity) != 4 or str(record.get("dicom_id")) != str(identity[0]):
            raise ValueError("training cache image identity order differs from records")
    registration_path = eva_cache / "registration.json"
    binding_path = eva_cache / "binding.json"
    if not registration_path.is_file() or not binding_path.is_file():
        raise IncompleteRun("EVA training cache registration or binding is missing")
    registration = _json(registration_path)
    for key, path in (("binding_sha256", binding_path), ("input_sha256", eva_cache / "input.json"),
                      ("progress_sha256", eva_cache / "progress.json"),
                      ("matrix_sha256", eva_path)):
        if registration.get(key) != _sha256(path):
            raise ValueError(f"EVA training cache registration differs for {key}")
    if registration.get("plan_sha256") != _sha256(output / "plan.json"):
        raise ValueError("EVA training cache belongs to another frozen plan")
    binding = _json(binding_path)
    if binding.get("plan_sha256") != registration["plan_sha256"]:
        raise ValueError("EVA training cache binding belongs to another frozen plan")
    checks.append({"name": "training_cache_registration", "images": len(records)})
    return np.load(cxr_path, mmap_mode="r", allow_pickle=False), np.load(
        eva_path, mmap_mode="r", allow_pickle=False), identities


def _pool_candidate(records, views, cxr_images, eva_images, candidate, study_ids, temp_root):
    if len(views) != len(records) or len(cxr_images) != len(records) or len(eva_images) != len(records):
        raise ValueError("training records, views, and image feature rows differ")
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown candidate {candidate}")
    keys, positions, labels, subjects = [], {}, [], []
    owners = {}
    image_studies = np.empty(len(records), dtype=np.int64)
    sums = np.memmap(Path(temp_root) / f"{candidate}-sum.dat", dtype=np.float64,
                     mode="w+", shape=(len(study_ids), WIDTHS[candidate]))
    sums[:] = 0
    counts = np.zeros(len(study_ids), dtype=np.int64)
    for index, row in enumerate(records):
        key = [str(row["subject_id"]), str(row["study_id"])]
        prior_owner = owners.setdefault(key[1], key[0])
        if prior_owner != key[0]:
            raise ValueError("one study ID is assigned to more than one patient")
        study = positions.get(tuple(key))
        if study is None:
            study = len(keys)
            if study >= len(study_ids) or study_ids[study] != key:
                raise ValueError("training study order differs from registered OOF IDs")
            positions[tuple(key)] = study
            keys.append(key)
            subjects.append(key[0])
            target = row.get("labels")
            if target is None or len(target) != NUM_CLASSES:
                raise ValueError("training records need ten labels per study")
            labels.append(target)
        elif subjects[study] != key[0]:
            raise ValueError("a study is assigned to more than one patient")
        elif not np.array_equal(np.asarray(labels[study]), np.asarray(row.get("labels"))):
            raise ValueError("a study has inconsistent image-level targets")
        image_studies[index] = study
    if keys != study_ids:
        raise ValueError("registered OOF study IDs differ from pooled training records")
    for start in range(0, len(records), CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, len(records))
        left = np.asarray(cxr_images[start:stop], dtype=np.float64)
        right = np.asarray(eva_images[start:stop], dtype=np.float64)
        chunk = right if candidate == "eva" else np.concatenate((left, right), axis=1)
        np.add.at(sums, image_studies[start:stop], chunk)
        counts += np.bincount(image_studies[start:stop], minlength=len(study_ids))
    if np.any(counts == 0):
        raise ValueError("pooled training study has no image rows")
    sums[:] /= counts[:, None]
    sums.flush()
    return sums, np.asarray(labels, dtype=np.float64), np.asarray(subjects, dtype=str)


def _head(path):
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key].copy() for key in archive.files}
    required = ("mean", "std", "weight", "bias")
    if any(key not in payload for key in required):
        raise ValueError(f"saved head is incomplete: {path}")
    for key in required:
        payload[key] = np.asarray(payload[key], dtype=np.float64)
    return payload


def _moments(features, indices):
    width = features.shape[1]
    total = np.zeros(width, dtype=np.float64)
    count = 0
    for start in range(0, len(indices), CHUNK_ROWS):
        rows = np.asarray(features[indices[start:start + CHUNK_ROWS]], dtype=np.float64)
        total += rows.sum(axis=0)
        count += len(rows)
    mean = total / count
    squared = np.zeros(width, dtype=np.float64)
    for start in range(0, len(indices), CHUNK_ROWS):
        rows = np.asarray(features[indices[start:start + CHUNK_ROWS]], dtype=np.float64)
        delta = rows - mean
        squared += np.einsum("ij,ij->j", delta, delta)
    std = np.sqrt(squared / count)
    std[std == 0] = 1.0
    return mean, std


def _predict_head(payload, features):
    logits = ((features - payload["mean"]) / payload["std"]) @ payload["weight"].T + payload["bias"]
    result = np.empty_like(logits, dtype=np.float64)
    positive = logits >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    result[~positive] = exp_logits / (1.0 + exp_logits)
    return result


def _verify_candidate(output, plan, candidate, plan_sha, study_ids, targets, folds,
                      subjects, features, predicted, checks):
    feature_identity = _json_hash({
        "candidate": candidate, "width": WIDTHS[candidate],
        "source_cxrlt_sha256": _sha256(Path(plan["incumbent_snapshot"]) / "train_features/image_features.npy"),
        "eva_features_sha256": _sha256(output / "eva_features/train/image_features.npy"),
        "study_keys_sha256": _json_hash(study_ids),
    })
    max_error = 0.0
    for fold in FOLDS:
        held = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        train_patients = sorted(set(subjects[train].tolist()))
        held_patients = set(subjects[held].tolist())
        if held_patients & set(train_patients):
            raise ValueError(f"patient leakage for {candidate} fold {fold}")
        folder = output / "fits" / candidate / str(fold)
        metadata = _json(folder / "fit.json")
        identity = metadata.get("identity", {})
        expected = {"plan_sha256": plan_sha, "feature_identity": feature_identity,
                    "candidate": candidate, "width": WIDTHS[candidate],
                    "training_folds": [item for item in FOLDS if item != fold],
                    "held_fold": fold, "training_patients": train_patients,
                    "l2": 0.01, "max_iter": 200, "seed": 20261108,
                    "class_balance": False}
        if identity != expected:
            raise ValueError(f"{candidate} fold {fold} fit identity differs")
        head_path = folder / "head.npz"
        if metadata.get("head_sha256") != _sha256(head_path):
            raise ValueError(f"{candidate} fold {fold} head hash differs")
        payload = _head(head_path)
        if (payload["mean"].shape != (WIDTHS[candidate],)
                or payload["std"].shape != (WIDTHS[candidate],)
                or payload["weight"].shape != (NUM_CLASSES, WIDTHS[candidate])
                or payload["bias"].shape != (NUM_CLASSES,)
                or not all(np.isfinite(payload[key]).all() for key in ("mean", "std", "weight", "bias"))
                or np.any(payload["std"] <= 0)):
            raise ValueError(f"{candidate} fold {fold} head arrays are invalid")
        expected_mean, expected_std = _moments(features, train)
        mean_error = float(np.max(np.abs(payload["mean"] - expected_mean)))
        std_error = float(np.max(np.abs(payload["std"] - expected_std)))
        if max(mean_error, std_error) > ATOL:
            raise ValueError(f"{candidate} fold {fold} standardization is not train-only")
        for start in range(0, len(held), CHUNK_ROWS):
            rows = held[start:start + CHUNK_ROWS]
            actual = _predict_head(payload, np.asarray(features[rows], dtype=np.float64))
            if not np.isfinite(actual).all():
                raise ValueError(f"{candidate} fold {fold} replay is non-finite")
            error = float(np.max(np.abs(actual - predicted[rows])))
            max_error = max(max_error, error)
            if error > ATOL:
                raise ValueError(f"{candidate} fold {fold} head replay differs by {error:g}")
        checks.append({"name": f"{candidate}_fold_{fold}_head_replay",
                       "mean_max_abs_error": mean_error,
                       "std_max_abs_error": std_error})
    return feature_identity, max_error


def verify_run(run_path):
    output = Path(run_path).resolve()
    state_path = output / "experiment.json"
    if not state_path.is_file():
        raise IncompleteRun("experiment.json is missing")
    state = _json(state_path)
    if state.get("status") != "complete":
        raise IncompleteRun(f"runner status is {state.get('status', 'missing')!r}")
    checks = []
    plan = _json(output / "plan.json") if (output / "plan.json").is_file() else {}
    plan_sha, artifacts = _verify_manifest(output, plan, checks)

    ids_path, targets_path = output / "oof_study_ids.json", output / "oof_targets.npy"
    study_ids = _json(ids_path)
    if (not isinstance(study_ids, list) or not study_ids
            or any(not isinstance(row, list) or len(row) != 2 for row in study_ids)
            or len({tuple(map(str, row)) for row in study_ids}) != len(study_ids)):
        raise ValueError("OOF study IDs must be unique [subject_id, study_id] pairs")
    incumbent = Path(plan["incumbent_snapshot"])
    if _json(incumbent / "oof_study_ids.json") != study_ids:
        raise ValueError("OOF study IDs differ from the locked incumbent study order")
    targets = np.load(targets_path, mmap_mode="r", allow_pickle=False)
    if targets.shape != (len(study_ids), NUM_CLASSES) or not np.isin(targets, (0, 1)).all():
        raise ValueError("OOF targets do not align to ten binary labels per study")
    predictions = {}
    for candidate in CANDIDATES:
        matrix = np.load(output / f"oof_{candidate}_probabilities.npy", mmap_mode="r", allow_pickle=False)
        _metrics(targets, matrix)
        predictions[candidate] = matrix
    checks.append({"name": "oof_alignment", "studies": len(study_ids), "classes": NUM_CLASSES})

    train_records_payload = _json(incumbent / "train_records.json")
    records, views = train_records_payload["records"], train_records_payload["views"]
    if len(records) != len(views):
        raise ValueError("saved training records and views differ in length")
    cxr_images, eva_images, _ = _validate_cache(output, plan, records, checks)
    folds = _folds_for(study_ids, plan)
    subjects = np.asarray([str(row[0]) for row in study_ids], dtype=str)
    source_targets = np.load(incumbent / "oof_targets.npy", mmap_mode="r", allow_pickle=False)
    baseline = np.load(incumbent / "oof_mean_probabilities.npy", mmap_mode="r", allow_pickle=False)
    if source_targets.shape != targets.shape or not np.array_equal(source_targets, targets):
        raise ValueError("OOF targets differ from locked incumbent targets")
    legacy_plan = _json(Path(plan["legacy_oof"]) / "plan.json")
    components = np.load(Path(plan["legacy_oof"]) / "oof_component_probabilities.npy",
                         mmap_mode="r", allow_pickle=False)
    prior = np.asarray(legacy_plan["prior"], dtype=np.float64)
    if components.shape != (len(study_ids), 8, NUM_CLASSES) or prior.shape != (8, NUM_CLASSES):
        raise ValueError("legacy OOF dimensions differ from fixed contract")
    legacy = np.zeros((len(study_ids), NUM_CLASSES), dtype=np.float64)
    for start in range(0, len(study_ids), CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, len(study_ids))
        block = np.asarray(components[start:stop], dtype=np.float64)
        if np.any(~np.isfinite(block) & (prior[None] != 0)):
            raise ValueError("legacy OOF has unavailable active components")
        legacy[start:stop] = (np.where(np.isfinite(block), block, 0) * prior[None]).sum(axis=1)

    gate_rows = []
    errors = {}
    with tempfile.TemporaryDirectory(prefix="eva-oof-verify-") as temp_root:
        for candidate in CANDIDATES:
            features, pooled_targets, pooled_subjects = _pool_candidate(
                records, views, cxr_images, eva_images, candidate, study_ids, temp_root)
            if not np.array_equal(pooled_targets, targets):
                raise ValueError(f"rebuilt {candidate} targets differ from locked OOF targets")
            if not np.array_equal(pooled_subjects, subjects):
                raise ValueError(f"rebuilt {candidate} patient IDs differ from OOF study IDs")
            feature_identity, replay_error = _verify_candidate(
                output, plan, candidate, plan_sha, study_ids, targets, folds, subjects,
                features, predictions[candidate], checks)
            errors[candidate] = replay_error
            gate = _gate(targets, baseline, legacy, predictions[candidate], folds)
            gate_rows.append({"candidate": candidate, "gate": gate,
                              "feature_identity": feature_identity})
            del features
            (Path(temp_root) / f"{candidate}-sum.dat").unlink(missing_ok=True)

    eligible = [row for row in gate_rows if row["gate"]["passed"]]
    selected = (sorted(eligible, key=lambda row: (-row["gate"]["candidate"]["macro_auc"],
                                                    CANDIDATES.index(row["candidate"])))[0]
                if eligible else None)
    selected_name = None if selected is None else selected["candidate"]
    recorded = _json(output / "oof_report.json")
    _compare_recorded(recorded.get("candidates"), gate_rows, "oof_report.candidates")
    if recorded.get("selected") != selected_name:
        raise ValueError("recorded candidate selection differs from independent OOF gates")
    if selected_name is None:
        if state.get("candidate_promoted") is not False or state.get("test_csv_exported") is not False:
            raise ValueError("terminal promotion flags conflict with failed OOF gates")
        expected_decision = "retain incumbent; neither candidate passed the patient OOF gate"
        if state.get("decision") != expected_decision:
            raise ValueError("terminal decision differs from the failed OOF selection")
    checks.append({"name": "independent_oof_gates", "selected": selected_name,
                   "all_ten_classes": all(row["gate"]["candidate"]["macro_auc_defined_classes"] == 10
                                           for row in gate_rows)})
    downstream = any(name in artifacts for name in ("validation_report.json", "submission_verification.json"))
    return {"status": "verified_oof_only", "run": str(output),
            "runner_status": state.get("status"), "selected_oof_candidate": selected_name,
            "candidate_promoted": state.get("candidate_promoted"),
            "coverage": {"oof_heads_and_gates": "verified",
                         "validation_bootstrap_and_test_export": "not independently verified",
                         "downstream_artifacts_present": downstream},
            "numerical_tolerance": ATOL,
            "head_replay_max_abs_error": errors,
            "note": "NumPy replay within tolerance; this does not claim bitwise PyTorch equivalence.",
            "checks": checks}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="completed EVA study-probe directory")
    parser.add_argument("--output", type=Path,
                        help="JSON report path (default: sibling of the experiment directory)")
    args = parser.parse_args(argv)
    run_path = args.run.resolve()
    report_path = (args.output.resolve() if args.output else
                   run_path.parent / f"{run_path.name}-verification.json")
    if _inside(report_path, run_path):
        parser.error("verification report must be outside the experiment directory")
    if report_path.exists():
        parser.error(f"refusing to overwrite existing report: {report_path}")
    try:
        report = verify_run(run_path)
        exit_code = 0
    except IncompleteRun as error:
        report = {"status": "incomplete", "run": str(run_path), "reason": str(error)}
        exit_code = 2
    except Exception as error:
        report = {"status": "failed", "run": str(run_path),
                  "reason": f"{type(error).__name__}: {error}"}
        exit_code = 1
    _atomic_json(report_path, report)
    print(json.dumps({"status": report["status"], "report": str(report_path)}, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
