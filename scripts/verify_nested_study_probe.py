"""Read-only replay verification for a completed nested study-probe run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cxr.data import read_records
from cxr.study_probe import pool_study_features, predict_linear_probe
from scripts.compare_runs import load_submission
from scripts.evaluate_frozen_study_probe import FOLDS, _load_head


TEST_CSV = "submission_test_1414_cxrlt_nested_head_20261108.csv"
NUM_CLASSES = 10
ATOL = 1e-12


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _same_config(left, right):
    return (left.get("class_balance") is right.get("class_balance")
            and float(left.get("l2")) == float(right.get("l2")))


def _config_name(config):
    return f"{'balanced' if config['class_balance'] else 'bce'}_l2_{config['l2']:g}"


def _max_error(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        return float("inf")
    return float(np.max(np.abs(left - right))) if left.size else 0.0


def _require_close(left, right, name, errors, atol=ATOL):
    error = _max_error(left, right)
    errors[name] = error
    if not np.isfinite(error) or error > atol:
        raise ValueError(f"{name} mismatch: max absolute error {error:g} exceeds {atol:g}")


def _require_optional_vector(actual, expected, name, errors, atol=ATOL):
    actual, expected = list(actual), list(expected)
    if len(actual) != len(expected):
        raise ValueError(f"{name} length differs")
    actual_defined = [value is not None for value in actual]
    expected_defined = [value is not None and np.isfinite(value) for value in expected]
    if actual_defined != expected_defined:
        raise ValueError(f"{name} defined-value mask differs")
    indices = [index for index, defined in enumerate(actual_defined) if defined]
    _require_close([actual[index] for index in indices], [expected[index] for index in indices],
                   name, errors, atol)


def _auc(target, probability):
    target = np.asarray(target, dtype=bool)
    probability = np.asarray(probability, dtype=np.float64)
    positive = probability[target]
    negative = np.sort(probability[~target])
    if not len(positive) or not len(negative):
        return None
    lower = np.searchsorted(negative, positive, side="left")
    upper = np.searchsorted(negative, positive, side="right")
    return float(np.mean((lower + upper) / (2.0 * len(negative))))


def _metrics(targets, probabilities, gates=None):
    targets = np.asarray(targets, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if targets.shape != probabilities.shape or targets.ndim != 2 or targets.shape[1] != NUM_CLASSES:
        raise ValueError("metric arrays must both have shape (N, 10)")
    if not np.isfinite(targets).all() or not np.isin(targets, (0, 1)).all():
        raise ValueError("metric targets must be finite and binary")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("metric probabilities must be finite and in [0, 1]")
    if gates is None:
        gates = np.full_like(probabilities, 0.5)
    else:
        gates = np.asarray(gates, dtype=np.float64)
    if gates.shape != probabilities.shape or not np.isfinite(gates).all():
        raise ValueError("metric gates must be finite and have shape (N, 10)")
    per_class = []
    for index in range(NUM_CLASSES):
        target = targets[:, index]
        truth = target == 1
        auc = _auc(target, probabilities[:, index])
        values = []
        for threshold in (0.5, gates[:, index]):
            predicted = probabilities[:, index] >= threshold
            denominator = int(truth.sum() + predicted.sum())
            values.append(float(2 * np.count_nonzero(truth & predicted) / denominator)
                          if denominator else 0.0)
        per_class.append((auc, *values))
    matrix = np.asarray([[np.nan if value is None else value for value in row]
                         for row in per_class], dtype=np.float64)
    auc_defined = int(np.isfinite(matrix[:, 0]).sum())
    macro = []
    for column in range(3):
        available = matrix[:, column]
        finite = np.isfinite(available)
        macro.append(float(available[finite].mean()) if finite.any() else None)
    return {"defined_auc": auc_defined, "per_class": per_class,
            "macro": {"auc": macro[0], "f1_fixed_0_5": macro[1],
                      "f1_declared_gate": macro[2]}}


def _rank_score(targets, probabilities):
    result = _metrics(targets, probabilities)
    return {"macro_auc": result["macro"]["auc"],
            "macro_auc_defined_classes": result["defined_auc"],
            "macro_f1": result["macro"]["f1_fixed_0_5"],
            "per_class_auc": [row[0] for row in result["per_class"]]}


def _select(rows):
    valid = [row for row in rows if row["macro_auc_defined_classes"] == NUM_CLASSES
             and row["macro_auc"] is not None and np.isfinite(row["macro_auc"])]
    if len(valid) != len(rows) or not valid:
        raise ValueError("configuration selection lacks ten defined AUC classes")
    return sorted(valid, key=lambda row: (-row["macro_auc"],
                                         row["config"]["class_balance"],
                                         -row["config"]["l2"]))[0]


def _verify_hashes(output, plan, checks):
    for path, expected in plan["input_locks"].items():
        if _sha256(path) != expected:
            raise ValueError(f"input hash mismatch: {path}")
    checks.append({"name": "input_hashes", "count": len(plan["input_locks"])})
    for name, expected in plan["source_locks"].items():
        path = output / "source" / name
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"source snapshot hash mismatch: {name}")
    checks.append({"name": "source_snapshot_hashes", "count": len(plan["source_locks"])})
    artifacts = _json(output / "artifacts.json")
    for name, expected in artifacts.items():
        path = (output / name).resolve()
        if not path.is_relative_to(output.resolve()) or not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"registered artifact hash mismatch: {name}")
    checks.append({"name": "completed_artifact_hashes", "count": len(artifacts)})
    return artifacts


def _load_split(plan, split, checks):
    source = Path(plan["source_experiment"])
    data_root = Path(plan["data_root"])
    records = (_json(source / "train_records.json")["records"] if split == "train"
               else read_records(data_root / f"{split}.csv", data_root / split))
    cache = source / f"{split}_features"
    identities = _json(cache / "input.json")["images"]
    if len(records) != len(identities) or [row["dicom_id"] for row in records] != [row[0] for row in identities]:
        raise ValueError(f"{split} feature rows do not match manifest order")
    for record, identity in zip(records, identities):
        dicom, path, size, mtime = identity
        stat = Path(path).stat()
        if (str(Path(record["path"]).resolve()) != str(Path(path).resolve())
                or stat.st_size != size or stat.st_mtime_ns != mtime):
            raise ValueError(f"{split} image identity changed: {dicom}")
    progress = _json(cache / "progress.json")
    if progress.get("status") != "complete" or progress.get("completed") != len(records):
        raise ValueError(f"{split} feature cache is incomplete")
    features = np.load(cache / "image_features.npy", mmap_mode="r", allow_pickle=False)
    if features.shape != (len(records), 1054) or features.dtype != np.float32:
        raise ValueError(f"{split} feature matrix shape or dtype mismatch")
    studies, pooled, targets = pool_study_features(records, features, [""] * len(records), "mean")
    checks.append({"name": f"{split}_cache_alignment", "images": len(records),
                   "studies": len(studies), "feature_shape": list(features.shape)})
    return studies, pooled, targets


def _load_legacy(plan, studies):
    folder = Path(plan["legacy_oof"])
    keys = [[row["subject_id"], row["study_id"]] for row in studies]
    if _json(folder / "oof_study_ids.json") != keys:
        raise ValueError("legacy OOF study order differs")
    components = np.load(folder / "oof_component_probabilities.npy", allow_pickle=False)
    prior = np.asarray(_json(folder / "plan.json")["prior"], dtype=np.float64)
    if components.shape != (len(studies), 8, NUM_CLASSES) or prior.shape != (8, NUM_CLASSES):
        raise ValueError("legacy OOF dimensions differ")
    if np.any(~np.isfinite(components) & (prior[None] != 0)):
        raise ValueError("legacy OOF requires unavailable component output")
    return (np.where(np.isfinite(components), components, 0) * prior[None]).sum(axis=1)


def _folds_for(studies, plan):
    partition = _json(plan["partition"])["subject_partition"]
    missing = {row["subject_id"] for row in studies} - set(partition)
    if missing:
        raise ValueError(f"partition omits {len(missing)} training subjects")
    folds = np.asarray([partition[row["subject_id"]] for row in studies], dtype=object)
    if any(value not in FOLDS for value in folds):
        raise ValueError("partition contains unsupported fold values")
    return folds


def _head_for(output, plan, features, targets, folds, subjects, training_folds, config):
    training_folds = tuple(fold for fold in FOLDS if fold in training_folds)
    train = np.isin(folds, np.asarray(training_folds, dtype=object))
    identity = {"plan_sha256": _sha256(output / "plan.json"),
                "training_folds": list(training_folds), "config": dict(config),
                "max_iter": plan["max_iter"], "seed": plan["seed"],
                "training_studies": int(train.sum()),
                "training_patients": len(set(np.asarray(subjects)[train]))}
    folder = output / "fits" / "_".join(map(str, training_folds)) / _config_name(config)
    metadata = _json(folder / "fit.json")
    if metadata.get("identity") != identity:
        raise ValueError(f"fit identity mismatch: {folder}")
    head_path = folder / "head.npz"
    if _sha256(head_path) != metadata.get("head_sha256"):
        raise ValueError(f"fit head hash mismatch: {head_path}")
    return _load_head(head_path)


def _metric_error(actual, expected, name, errors):
    if actual.get("macro_auc_defined_classes") != expected["macro_auc_defined_classes"]:
        raise ValueError(f"{name} macro_auc_defined_classes differs")
    for key in ("macro_auc", "macro_f1"):
        left, right = actual.get(key), expected[key]
        if left is None or right is None:
            if left is not right:
                raise ValueError(f"{name} {key} differs")
        else:
            _require_close([left], [right], f"{name}.{key}", errors)
    _require_optional_vector(actual.get("per_class_auc"), expected["per_class_auc"],
                             f"{name}.per_class_auc", errors)


def _verify_nested(output, plan, report, studies, features, targets, folds, checks, errors):
    keys = [[row["subject_id"], row["study_id"]] for row in studies]
    if _json(output / "oof_study_ids.json") != keys:
        raise ValueError("nested OOF study identity/order mismatch")
    if _json(Path(plan["source_experiment"]) / "oof_study_ids.json") != keys:
        raise ValueError("incumbent OOF study order mismatch")
    nested_probs = np.load(output / "nested_probabilities.npy", allow_pickle=False)
    if nested_probs.shape != targets.shape or not np.isfinite(nested_probs).all():
        raise ValueError("nested predictions have invalid shape or values")
    if np.any((nested_probs < 0) | (nested_probs > 1)):
        raise ValueError("nested probabilities are outside [0, 1]")
    rows_by_outer = {row["outer_fold"]: row for row in report["outer_selection"]}
    if set(rows_by_outer) != set(FOLDS) or len(rows_by_outer) != len(FOLDS):
        raise ValueError("outer report does not cover every fold exactly once")
    subjects = [row["subject_id"] for row in studies]
    replay_count = 0
    for outer in FOLDS:
        outer_report = rows_by_outer[outer]
        remaining = tuple(fold for fold in FOLDS if fold != outer)
        inner_mask = folds != outer
        scored_rows = []
        for row in outer_report["inner_selection"]:
            config = row["config"]
            replay = np.full_like(targets, np.nan, dtype=np.float64)
            for inner in remaining:
                training = tuple(fold for fold in remaining if fold != inner)
                held = np.flatnonzero(folds == inner)
                if outer in training or inner in training:
                    raise ValueError("inner fit contains its held fold")
                head = _head_for(output, plan, features, targets, folds, subjects, training, config)
                replay[held] = predict_linear_probe(head, features[held])
                replay_count += 1
            fresh = _rank_score(targets[inner_mask], replay[inner_mask])
            _metric_error(row, fresh, f"outer_{outer}_inner_{_config_name(config)}", errors)
            scored_rows.append(row)
        if len(scored_rows) != len(plan["configs"]):
            raise ValueError(f"outer {outer!r} inner rows do not cover all configurations")
        chosen = _select(scored_rows)
        if not _same_config(chosen["config"], outer_report["selected_config"]):
            raise ValueError(f"outer {outer!r} selected configuration mismatch")
        held = np.flatnonzero(folds == outer)
        outer_head = _head_for(output, plan, features, targets, folds, subjects,
                               remaining, outer_report["selected_config"])
        replayed_outer = predict_linear_probe(outer_head, features[held])
        _require_close(nested_probs[held], replayed_outer, f"outer_{outer}_predictions", errors)
        if outer_report.get("held_studies") != len(held):
            raise ValueError(f"outer {outer!r} held-study count differs")
    checks.append({"name": "nested_inner_and_outer_replay", "outer_folds": len(FOLDS),
                   "inner_head_replays": replay_count, "outer_study_predictions": len(studies)})
    return nested_probs


def _training_gate(targets, baseline, candidate, legacy, folds):
    old, new, older = (_rank_score(targets, probs) for probs in (baseline, candidate, legacy))
    groups = []
    for fold in FOLDS:
        held = folds == fold
        before = _rank_score(targets[held], baseline[held])
        after = _rank_score(targets[held], candidate[held])
        gain = (None if before["macro_auc"] is None or after["macro_auc"] is None
                else after["macro_auc"] - before["macro_auc"])
        groups.append({"fold": fold, "defined_classes": after["macro_auc_defined_classes"], "gain": gain})
    gain = None if old["macro_auc"] is None or new["macro_auc"] is None else new["macro_auc"] - old["macro_auc"]
    legacy_gain = None if older["macro_auc"] is None or new["macro_auc"] is None else new["macro_auc"] - older["macro_auc"]
    positive = sum(item["gain"] is not None and item["gain"] > 0 for item in groups)
    passed = bool(old["macro_auc_defined_classes"] == new["macro_auc_defined_classes"] == NUM_CLASSES
                  and all(row["defined_classes"] == NUM_CLASSES for row in groups)
                  and gain is not None and gain >= 0.001 and positive >= 3
                  and legacy_gain is not None and legacy_gain >= 0.001)
    return {"passed": passed, "baseline": old, "candidate": new, "legacy": older,
            "gain": gain, "legacy_gain": legacy_gain, "positive_held_groups": positive,
            "held_groups": groups}


def _verify_gate_record(recorded, fresh, errors, name):
    if recorded.get("passed") != fresh["passed"]:
        raise ValueError(f"{name} pass/fail differs")
    if recorded.get("positive_held_groups") != fresh["positive_held_groups"]:
        raise ValueError(f"{name} positive group count differs")
    for key in ("gain", "legacy_gain"):
        if recorded.get(key) is None or fresh[key] is None:
            if recorded.get(key) is not fresh[key]:
                raise ValueError(f"{name} {key} differs")
        else:
            _require_close([recorded[key]], [fresh[key]], f"{name}.{key}", errors)
    for scope in ("baseline", "candidate", "legacy"):
        _metric_error(recorded[scope], fresh[scope], f"{name}.{scope}", errors)
    recorded_groups = recorded.get("held_groups", [])
    if len(recorded_groups) != len(fresh["held_groups"]):
        raise ValueError(f"{name} held group count differs")
    for old, new in zip(recorded_groups, fresh["held_groups"]):
        if old.get("fold") != new["fold"] or old.get("defined_classes") != new["defined_classes"]:
            raise ValueError(f"{name} held group identity differs")
        if old.get("gain") is None or new["gain"] is None:
            if old.get("gain") is not new["gain"]:
                raise ValueError(f"{name} held group gain differs")
        else:
            _require_close([old["gain"]], [new["gain"]], f"{name}.held_group_gain", errors)


def _assert_no_promotion_artifacts(output, checks):
    names = ["final_selection.json", "final_head.npz", "recipe.json", "candidate_valid.csv",
             "validation_report.json", TEST_CSV, "submission_verification.json"]
    found = [name for name in names if (output / name).exists()]
    found += [path.name for path in output.glob("final_cv_*.npy")]
    if found:
        raise ValueError(f"artifacts exist despite failed training gate: {found}")
    checks.append({"name": "training_gate_failure_stops_before_validation", "valid_labels_read": False})


def _verify_final_cv(output, plan, selection, features, targets, folds, subjects,
                    checks, errors):
    rows = selection["candidates"]
    if len(rows) == 0:
        raise ValueError("final selection has no candidates")
    for row in rows:
        config = row["config"]
        predictions = np.load(output / f"final_cv_{_config_name(config)}.npy", allow_pickle=False)
        if predictions.shape != targets.shape or not np.isfinite(predictions).all():
            raise ValueError("final CV prediction shape or values invalid")
        replay = np.full_like(targets, np.nan, dtype=np.float64)
        for fold in FOLDS:
            held = np.flatnonzero(folds == fold)
            training_folds = tuple(other for other in FOLDS if other != fold)
            if not len(held) or fold in training_folds:
                raise ValueError(f"final CV held fold {fold!r} is invalid")
            head = _head_for(output, plan, features, targets, folds, subjects,
                             training_folds, config)
            replay[held] = predict_linear_probe(head, features[held])
        _require_close(predictions, replay,
                       f"final_cv_{_config_name(config)}_head_replay", errors)
        fresh = _rank_score(targets, predictions)
        _metric_error(row, fresh, f"final_cv_{_config_name(config)}", errors)
    selected = _select(rows)
    if not _same_config(selected["config"], selection["selected"]["config"]):
        raise ValueError("final selected configuration differs from tie rule")
    _metric_error(selection["selected"], _rank_score(
        targets, np.load(output / f"final_cv_{_config_name(selected['config'])}.npy", allow_pickle=False)),
        "final_selected", errors)
    checks.append({"name": "final_cv_replay", "configurations": len(rows),
                   "held_folds": len(FOLDS)})
    return selected


def _read_point_metric(comparison, expected, name, errors):
    mapping = {"auc": "auc", "f1_fixed_0_5": "f1_fixed_0_5",
               "f1_declared_gate": "f1_declared_gate"}
    for stored_metric, fresh_metric in mapping.items():
        stored = comparison["macro"][stored_metric]
        current = expected["macro"][fresh_metric]
        for endpoint in ("baseline", "candidate", "delta"):
            value = stored.get(endpoint)
            fresh = current[endpoint]
            if value is None or fresh is None:
                if value is not fresh:
                    raise ValueError(f"{name} macro {stored_metric}.{endpoint} differs")
            else:
                _require_close([value], [fresh], f"{name}.{stored_metric}.{endpoint}", errors, atol=1e-10)
    for index, row in enumerate(comparison["per_class"]):
        for stored_metric, fresh_metric in mapping.items():
            stored = row["metrics"][stored_metric]
            current = expected["per_class"][index][stored_metric]
            for endpoint in ("baseline", "candidate", "delta"):
                value, fresh = stored.get(endpoint), current.get(endpoint)
                if value is None or fresh is None:
                    if value is not fresh:
                        raise ValueError(f"{name} class {index} {stored_metric}.{endpoint} differs")
                else:
                    _require_close([value], [fresh], f"{name}.class_{index}_{stored_metric}_{endpoint}",
                                   errors, atol=1e-10)


def _independent_comparison(targets, baseline, candidate, baseline_gates, candidate_gates):
    base = _metrics(targets, baseline, baseline_gates)
    cand = _metrics(targets, candidate, candidate_gates)
    per_class, macro = [], {}
    names = ("auc", "f1_fixed_0_5", "f1_declared_gate")
    for index in range(NUM_CLASSES):
        metrics = {}
        for column, name in enumerate(names):
            before, after = base["per_class"][index][column], cand["per_class"][index][column]
            metrics[name] = {"baseline": before, "candidate": after,
                             "delta": None if before is None or after is None else after - before}
        per_class.append(metrics)
    for name, key in zip(names, ("auc", "f1_fixed_0_5", "f1_declared_gate")):
        before, after = base["macro"][key], cand["macro"][key]
        macro[name] = {"baseline": before, "candidate": after,
                       "delta": None if before is None or after is None else after - before}
    return {"macro": macro, "per_class": per_class,
            "macro_auc_defined_classes": base["defined_auc"]}


def _valid_gate(comparisons):
    outcomes = []
    for name in ("incumbent", "legacy"):
        comparison = comparisons[name]
        interval = comparison["macro"]["auc"].get("ci95_percentile")
        delta = comparison["macro"]["auc"].get("delta")
        outcomes.append(bool(
            comparison["macro_auc_defined_classes"] == NUM_CLASSES
            and all(row["metrics"]["auc"]["baseline"] is not None
                    and row["metrics"]["auc"]["candidate"] is not None
                    for row in comparison["per_class"])
            and delta is not None and delta >= 0.001
            and interval is not None and interval[0] > 0
            and comparison["bootstrap"]["auc_defined_classes_histogram"].get("10", 0) >= 950
        ))
    return all(outcomes)


def _verify_validation_and_test(output, plan, source, head, valid_report,
                                checks, errors):
    valid_studies, valid_features, valid_targets = _load_split(plan, "valid", checks)
    if valid_targets is None:
        raise ValueError("validation split has no labels")
    candidate, candidate_gates, _ = load_submission(output / "candidate_valid.csv", valid_studies)
    incumbent, incumbent_gates, _ = load_submission(source / "candidate_valid.csv", valid_studies)
    legacy, legacy_gates, _ = load_submission(plan["legacy_valid"], valid_studies)
    replay = predict_linear_probe(head, valid_features)
    _require_close(candidate, replay, "validation_csv_vs_head", errors, atol=1e-10)
    if _sha256(output / "candidate_valid.csv") != valid_report["candidate_valid_csv_sha256"]:
        raise ValueError("validation candidate CSV hash differs")
    for name, baseline, baseline_gates in (
        ("incumbent", incumbent, incumbent_gates), ("legacy", legacy, legacy_gates)
    ):
        fresh = _independent_comparison(valid_targets, baseline, candidate,
                                        baseline_gates, candidate_gates)
        stored = valid_report["comparisons"][name]
        if stored["macro_auc_defined_classes"] != fresh["macro_auc_defined_classes"]:
            raise ValueError(f"{name} valid AUC class count differs")
        _read_point_metric(stored, fresh, name, errors)
    gate = _valid_gate(valid_report["comparisons"])
    if bool(valid_report["promoted"]) != gate:
        raise ValueError("stored validation promotion does not match fixed valid gate")
    checks.append({"name": "validation_point_metrics_and_gate", "studies": len(valid_studies),
                   "labels_read": True, "valid_gate": gate})

    if bool(valid_report["promoted"]):
        test_studies, test_features, test_targets = _load_split(plan, "test", checks)
        if test_targets is not None:
            raise ValueError("test split unexpectedly contains labels")
        path = output / TEST_CSV
        test_probabilities, _, verification = load_submission(path, test_studies)
        predicted = predict_linear_probe(head, test_features)
        _require_close(test_probabilities, predicted, "test_csv_vs_head", errors, atol=1e-10)
        submission_check = _json(output / "submission_verification.json")
        if submission_check.get("sha256") != _sha256(path):
            raise ValueError("test CSV verification hash mismatch")
        if verification.get("valid") is not True:
            raise ValueError("test CSV failed submission validation")
        checks.append({"name": "test_csv_replay", "studies": len(test_studies),
                       "test_labels_read": False})
    else:
        unexpected = [name for name in (TEST_CSV, "submission_verification.json")
                      if (output / name).exists()]
        if unexpected:
            raise ValueError(f"test submission artifacts exist despite failed validation gate: {unexpected}")
    return gate


def verify(output):
    output = Path(output).resolve()
    state = _json(output / "experiment.json")
    if state.get("status") != "complete":
        raise ValueError("nested experiment must have status complete")
    plan = _json(output / "plan.json")
    checks, errors = [], {}
    artifacts = _verify_hashes(output, plan, checks)
    studies, pooled_features, targets = _load_split(plan, "train", checks)
    if targets is None:
        raise ValueError("training split has no targets")
    source = Path(plan["source_experiment"])
    expected_targets = np.load(source / "oof_targets.npy", allow_pickle=False)
    _require_close(targets, expected_targets, "training_targets", errors, atol=0)
    keys = [[row["subject_id"], row["study_id"]] for row in studies]
    if keys != _json(source / "oof_study_ids.json"):
        raise ValueError("reconstructed training study order differs from incumbent")
    if keys != _json(output / "oof_study_ids.json"):
        raise ValueError("nested output study order differs from reconstructed studies")
    checks.append({"name": "training_repool_targets_order", "studies": len(studies),
                   "feature_width": int(pooled_features.shape[1])})

    folds = _folds_for(studies, plan)
    counts = {str(fold): int(np.count_nonzero(folds == fold)) for fold in FOLDS}
    if any(count == 0 for count in counts.values()):
        raise ValueError("partition contains an empty outer fold")
    checks.append({"name": "partition_alignment", "fold_studies": counts})

    nested_report = _json(output / "nested_report.json")
    nested_probs = _verify_nested(output, plan, nested_report, studies, pooled_features,
                                  targets, folds, checks, errors)
    baseline = np.load(source / "oof_mean_probabilities.npy", allow_pickle=False)
    if baseline.shape != targets.shape:
        raise ValueError("incumbent baseline OOF shape differs")
    legacy = _load_legacy(plan, studies)
    fresh_training_gate = _training_gate(targets, baseline, nested_probs, legacy, folds)
    _verify_gate_record(nested_report["gate"], fresh_training_gate, errors, "training_gate")
    checks.append({"name": "independent_training_gate", "passed": fresh_training_gate["passed"],
                   "positive_held_groups": fresh_training_gate["positive_held_groups"]})

    promoted = bool(state.get("candidate_promoted"))
    validation_gate = None
    if not fresh_training_gate["passed"]:
        _assert_no_promotion_artifacts(output, checks)
        if promoted or state.get("test_csv_exported"):
            raise ValueError("experiment promoted despite failed training gate")
        labels_read = False
    else:
        selection_path = output / "final_selection.json"
        if not selection_path.is_file():
            raise ValueError("training gate passed but final selection is missing")
        selection = _json(selection_path)
        subjects = [row["subject_id"] for row in studies]
        chosen = _verify_final_cv(output, plan, selection, pooled_features, targets,
                                  folds, subjects, checks, errors)
        selected_gain = chosen["macro_auc"] - fresh_training_gate["baseline"]["macro_auc"]
        if selected_gain < 0.001:
            if any((output / name).exists() for name in ("final_head.npz", "recipe.json", "candidate_valid.csv", TEST_CSV)):
                raise ValueError("final CV gate failed but downstream artifacts exist")
            if promoted:
                raise ValueError("experiment promoted despite final CV gate failure")
            labels_read = False
        else:
            head_path = output / "final_head.npz"
            recipe_path = output / "recipe.json"
            if not head_path.is_file() or not recipe_path.is_file():
                raise ValueError("selected final candidate is missing its head or recipe")
            recipe = _json(recipe_path)
            head_sha = _sha256(head_path)
            if (recipe.get("head_file") != head_path.name or recipe.get("head_sha256") != head_sha
                    or recipe.get("source_plan_sha256") != _sha256(output / "plan.json")
                    or not _same_config(recipe.get("config", {}), chosen["config"])):
                raise ValueError("final recipe/head fingerprint or selected config differs")
            head = _load_head(head_path)
            valid_report_path = output / "validation_report.json"
            if not valid_report_path.is_file():
                raise ValueError("final head exists but validation report is missing")
            valid_report = _json(valid_report_path)
            if valid_report.get("head_sha256") != head_sha:
                raise ValueError("validation report head hash differs")
            if valid_report.get("recipe_sha256") != _sha256(recipe_path):
                raise ValueError("validation report recipe hash differs")
            validation_gate = _verify_validation_and_test(output, plan, source,
                                                          head, valid_report, checks, errors)
            labels_read = True
            if validation_gate != promoted or bool(state.get("test_csv_exported")) != promoted:
                raise ValueError("experiment promotion state differs from valid gate")

    checks.append({"name": "bootstrap_replay_scope", "recomputed": False,
                   "reason": "The saved 1000-replicate patient bootstrap was hash-verified and its gate thresholds checked; replicates were not rerun."})
    return {
        "schema_version": 1,
        "status": "verified",
        "experiment_status": state["status"],
        "decision": state.get("decision"),
        "candidate_promoted": promoted,
        "validation_gate": validation_gate,
        "valid_labels_read": labels_read,
        "checks": checks,
        "counts": {"input_hashes": len(plan["input_locks"]),
                   "source_hashes": len(plan["source_locks"]),
                   "artifacts": len(artifacts), "studies": len(studies),
                   "outer_folds": len(FOLDS)},
        "max_absolute_errors": errors,
        "limitations": ["Bootstrap confidence intervals were not regenerated; the saved report was hash-verified and gate criteria were rechecked."],
    }


def _validate_report_path(path, output, plan, artifacts):
    path = Path(path).resolve()
    if path == output or path.is_relative_to(output):
        raise ValueError("--report must be outside the experiment directory")
    if path.exists() and path.is_dir():
        raise ValueError("--report must name a JSON file")
    locked_paths = {Path(name).resolve() for name in plan["input_locks"]}
    locked_paths.update((output / "source" / name).resolve() for name in plan["source_locks"])
    locked_paths.update((output / name).resolve() for name in artifacts)
    if path in locked_paths:
        raise ValueError("--report must not overwrite a locked input, source, or artifact")
    return path


def _write_report(path, report):
    path = Path(path).resolve()
    if path.exists() and path.is_dir():
        raise ValueError("--report must name a JSON file")
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="completed experiment directory")
    parser.add_argument("--report", type=Path, help="optional JSON report path outside the experiment")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    report_path = None
    if args.report:
        plan = _json(output / "plan.json")
        artifacts = _json(output / "artifacts.json")
        report_path = _validate_report_path(args.report, output, plan, artifacts)
    try:
        report = verify(output)
    except Exception as error:
        failure = {"schema_version": 1, "status": "failed", "error": f"{type(error).__name__}: {error}"}
        if report_path:
            _write_report(report_path, failure)
        print(json.dumps(failure, allow_nan=False), file=sys.stderr)
        return 1
    if report_path:
        _write_report(report_path, report)
    print(json.dumps({"status": report["status"], "decision": report["decision"],
                      "candidate_promoted": report["candidate_promoted"],
                      "checks": len(report["checks"]), "studies": report["counts"]["studies"]},
                     allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
