"""Run the single registered patient-probability smoothing experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cxr.data import _read_metadata
from cxr.patient_smoothing import ALPHA, smooth_patient_probabilities
from cxr.safety import safe_output
from cxr.submission import validate_submission, write_submission
from scripts.compare_runs import compare_arrays, load_submission
from scripts.evaluate_frozen_study_probe import FOLDS, _fold_memberships, _sha256
from scripts.run_bottleneck_study_probe import legacy_oof, training_gate, validation_gate

OUTPUT = ROOT / "outputs/cxrlt-patient-smoothing-20261108"
INCUMBENT = ROOT / "outputs/cxrlt-supervised-study-probe-20261108"
EVA = ROOT / "outputs/eva-study-probe-20261108-v2"
EVA_REPORT = ROOT / "outputs/eva-study-probe-20261108-v2-independent-oof-verification-v2.json"
SOURCE_FILES = (
    "scripts/run_patient_smoothing.py", "scripts/run_bottleneck_study_probe.py",
    "scripts/evaluate_frozen_study_probe.py", "scripts/compare_runs.py",
    "cxr/__init__.py", "cxr/data.py", "cxr/metrics.py", "cxr/safety.py",
    "cxr/submission.py", "cxr/patient_smoothing.py", "cxr/study_probe.py",
    "cxr/bottleneck_probe.py",
)
SPLITS = ("train", "valid", "test")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _plan_digest(plan):
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _copy_lock(source, target):
    source, target = Path(source), Path(target)
    if not source.is_file():
        raise ValueError(f"required source file missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = _sha256(source)
    shutil.copy2(source, target)
    if _sha256(target) != digest:
        raise ValueError(f"snapshot copy differs from source: {source}")
    return digest


def _keys(values, name):
    result = []
    for row in values:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError(f"{name} IDs must be subject/study pairs")
        if any(not isinstance(value, str) or not value or value != value.strip() for value in row):
            raise ValueError(f"{name} IDs must be nonempty strings")
        result.append(tuple(row))
    if len(set(result)) != len(result) or len({study for _, study in result}) != len(result):
        raise ValueError(f"{name} contains duplicate or conflicting study IDs")
    return result


def _legacy_target_sources(folder):
    """Locate label evidence already locked by the original legacy experiment."""
    folder = Path(folder)
    plan, state = _json(folder / "plan.json"), _json(folder / "experiment.json")
    if state.get("status") != "complete":
        raise ValueError("legacy OOF experiment must be complete")
    sources = []
    for target_name, target_hash in sorted(plan["input_locks"].items()):
        target = Path(target_name)
        if target.name != "study_targets.npy":
            continue
        ids = target.with_name("study_ids.json")
        ids_hash = plan["input_locks"].get(str(ids))
        if not ids_hash:
            raise ValueError("legacy target evidence lacks its original ID lock")
        sources.append((ids, ids_hash, target, target_hash))
    for stage in state.get("stages", {}).values():
        if stage.get("status") != "complete":
            raise ValueError("legacy local OOF stage is incomplete")
        path = Path(stage["output"])
        hashes = stage["artifacts"]
        sources.append((path / "study_ids.json", hashes["study_ids.json"],
                        path / "study_targets.npy", hashes["study_targets.npy"]))
    if not sources:
        raise ValueError("legacy OOF target evidence is missing")
    return sources


def _validate_legacy_targets(output, plan, train_ids, targets):
    expected = {key: targets[i] for i, key in enumerate(train_ids)}
    expected_keys = set(expected)
    covered = set()
    for source in plan["legacy_target_sources"]:
        ids = _keys(_json(output / source["ids"]), "legacy target evidence")
        if not ids or not set(ids) <= expected_keys:
            raise ValueError("legacy target evidence contains unknown train studies")
        y = np.load(output / source["targets"], allow_pickle=False)
        if y.shape != (len(ids), 10) or not np.array_equal(y, [expected[key] for key in ids]):
            raise ValueError("legacy targets do not align with train manifest")
        covered.update(ids)
    if covered != expected_keys:
        raise ValueError("legacy target evidence does not cover complete train")
    return {"sources": len(plan["legacy_target_sources"]), "covered_studies": len(covered)}


def _identities(path):
    rows = _read_metadata(path)
    studies = {}
    owners = {}
    for row in rows:
        key = (row["subject_id"], row["study_id"])
        if row["study_id"] in owners and owners[row["study_id"]] != row["subject_id"]:
            raise ValueError(f"conflicting study identity in {path}: {row['study_id']}")
        owners[row["study_id"]] = row["subject_id"]
        if key in studies and not np.array_equal(studies[key], row["labels"]):
            raise ValueError(f"conflicting study labels in {path}: {key}")
        studies[key] = row["labels"]
    return list(studies), studies, {key[0] for key in studies}


def prepare(output=OUTPUT):
    incumbent, eva = INCUMBENT, EVA
    inc_plan = _json(incumbent / "plan.json")
    output = safe_output(output, data_root=inc_plan["data_root"])
    if output == Path(incumbent).resolve() or output == Path(eva).resolve() or output.exists() and any(output.iterdir()):
        raise ValueError("output must be a fresh directory separate from registered source runs")
    inc_state = _json(incumbent / "experiment.json")
    eva_state, eva_report = _json(eva / "experiment.json"), _json(EVA_REPORT)
    if inc_state.get("status") != "complete" or not inc_state.get("candidate_promoted") or not inc_state.get("test_csv_exported"):
        raise ValueError("incumbent must be complete, promoted, and test-exported")
    if eva_state.get("status") != "complete" or eva_state.get("candidate_promoted") or eva_state.get("test_csv_exported") or eva_state.get("test_extracted"):
        raise ValueError("EVA run must be complete, unpromoted, and test-unextracted")
    if (eva_report.get("status") != "verified_oof_only" or eva_report.get("runner_status") != "complete"
            or eva_report.get("selected_oof_candidate") is not None or eva_report.get("candidate_promoted")
            or eva_report.get("coverage", {}).get("oof_heads_and_gates") != "verified"):
        raise ValueError("independent EVA OOF-only verification is not acceptable")

    output.mkdir(parents=True, exist_ok=True)
    inputs = {}
    def lock(key, source, expected=None):
        rel = Path("inputs") / key
        digest = _copy_lock(source, output / rel)
        if expected is not None and digest != expected:
            raise ValueError(f"source differs from original experiment lock: {source}")
        inputs[rel.as_posix()] = digest
        return output / rel

    # Freeze all prediction, recipe, state, manifest, partition, legacy and EVA evidence.
    incumbent_files = ["plan.json", "experiment.json", "recipe.json", "final_head.npz", "report.json",
                       "validation_report.json", "oof_mean_probabilities.npy", "oof_targets.npy",
                       "oof_study_ids.json", "candidate_valid.csv", "submission_verification.json"]
    for name in incumbent_files:
        lock(f"incumbent/{name}", incumbent / name)
    legacy = Path(inc_plan["old_oof"])
    for name in ("plan.json", "experiment.json", "oof_study_ids.json", "oof_component_probabilities.npy"):
        lock(f"legacy/{name}", legacy / name)
    legacy_targets = []
    for index, (ids, ids_hash, targets, targets_hash) in enumerate(_legacy_target_sources(legacy)):
        ids_key = f"legacy_targets/{index:02d}/study_ids.json"
        target_key = f"legacy_targets/{index:02d}/study_targets.npy"
        lock(ids_key, ids, ids_hash)
        lock(target_key, targets, targets_hash)
        legacy_targets.append({"ids": f"inputs/{ids_key}", "targets": f"inputs/{target_key}",
                               "source_ids": str(ids), "source_targets": str(targets)})
    lock("legacy/valid.csv", inc_plan["old_valid"])
    for split in SPLITS:
        lock(f"manifests/{split}.csv", Path(inc_plan["data_root"]) / f"{split}.csv")
    lock("partition.json", inc_plan["partition"])
    for name in ("plan.json", "plan_lock.json", "experiment.json", "oof_report.json", "artifacts.json"):
        lock(f"eva/{name}", eva / name)
    lock("eva_independent_verification.json", EVA_REPORT)
    lock("protocol/PATIENT_SMOOTHING_PLAN_20261108.md", ROOT / "docs/PATIENT_SMOOTHING_PLAN_20261108.md")
    for name in SOURCE_FILES:
        lock(f"source/{name}", ROOT / name)

    manifests = {split: _identities(output / "inputs" / "manifests" / f"{split}.csv") for split in SPLITS}
    split_subjects = {split: item[2] for split, item in manifests.items()}
    if any(split_subjects[a] & split_subjects[b] for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]):
        raise ValueError("patient identities overlap across official splits")
    partition = _json(output / "inputs/partition.json").get("subject_partition", {})
    if any(subject not in partition for subject in split_subjects["train"]):
        raise ValueError("train manifest patient missing from fixed patient partition")
    all_owners = {}
    for split, (ids, _, _) in manifests.items():
        for subject, study in ids:
            if study in all_owners:
                raise ValueError("study identities overlap across official splits")
            all_owners[study] = subject
    legacy_plan = _json(output / "inputs/legacy/plan.json")
    for key, current_path in ((str(Path(legacy_plan["data_root"]) / "train.csv"),
                               output / "inputs/manifests/train.csv"),
                              (legacy_plan["partition"], output / "inputs/partition.json")):
        if legacy_plan["input_locks"].get(key) != _sha256(current_path):
            raise ValueError("legacy train manifest or patient partition differs")
    test_verification = _json(output / "inputs/incumbent/submission_verification.json")
    test_digest = test_verification.get("sha256", "")
    if (not test_verification.get("valid") or test_verification.get("studies") != 1414
            or len(test_digest) != 64 or any(char not in "0123456789abcdef" for char in test_digest)):
        raise ValueError("incumbent test export lacks a valid prior SHA-256 record")
    plan = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "alpha": ALPHA, "seed": 20261108, "bootstrap": 1000,
            "output": str(output), "incumbent": str(incumbent), "legacy_oof": str(legacy),
            "legacy_valid": str(inc_plan["old_valid"]), "data_root": inc_plan["data_root"],
            "source_locks": {name: inputs[f"inputs/source/{name}"] for name in SOURCE_FILES},
            "input_locks": inputs, "train_studies": len(manifests["train"][0]),
            "legacy_target_sources": legacy_targets,
            "deferred_test_input": {
                "source": str(incumbent / "submission_test_1414_cxrlt_study_20261108.csv"),
                "snapshot": "deferred_inputs/incumbent_test.csv", "sha256": test_digest},
            "valid_studies": len(manifests["valid"][0]), "test_studies": len(manifests["test"][0]),
            "protocol": "fixed q=0.75*p+0.25*within-split patient mean; singleton identity"}
    plan["plan_sha256"] = _plan_digest(plan)
    _write(output / "plan.json", plan)
    _write(output / "experiment.json", {"status": "prepared", "phase": None,
           "created_utc": plan["created_utc"], "error": None,
           "candidate_promoted": False, "valid_evaluated": False,
           "test_read": False, "test_csv_exported": False, "online_score": None})
    return output


def _verify(output, plan):
    for rel, digest in plan["input_locks"].items():
        if _sha256(output / rel) != digest:
            raise ValueError(f"locked input changed: {rel}")
    if _plan_digest(plan) != plan["plan_sha256"]:
        raise ValueError("plan lock mismatch")
    deferred = plan.get("deferred_test_input")
    if deferred and (output / deferred["snapshot"]).exists():
        if _sha256(output / deferred["snapshot"]) != deferred["sha256"]:
            raise ValueError("deferred test snapshot differs from registered source hash")


def _align(ids, targets, expected_ids, expected_targets, name):
    source, expected = _keys(ids, name), _keys(expected_ids, "train manifest")
    if set(source) != set(expected):
        raise ValueError(f"{name} study ID set does not exactly match train manifest")
    positions = {key: i for i, key in enumerate(source)}
    order = np.asarray([positions[key] for key in expected], dtype=np.int64)
    y = np.asarray(targets)
    if y.shape != np.asarray(expected_targets).shape or not np.array_equal(y[order], expected_targets):
        raise ValueError(f"{name} labels do not align with train manifest")
    return order


def run(output):
    output = Path(output).resolve()
    plan = _json(output / "plan.json")
    state = _json(output / "experiment.json")
    _verify(output, plan)
    if state["status"] == "complete":
        for rel, digest in _json(output / "artifacts.json").items():
            if _sha256(output / rel) != digest:
                raise ValueError(f"completed artifact changed: {rel}")
        return state
    try:
        state.update(status="running", phase="oof", candidate_promoted=False,
                     test_csv_exported=False, valid_evaluated=False, test_read=False, error=None)
        _write(output / "experiment.json", state)
        inc = output / "inputs/incumbent"
        train_ids, train_labels, _ = _identities(output / "inputs/manifests/train.csv")
        ids = _json(inc / "oof_study_ids.json")
        targets = np.load(inc / "oof_targets.npy", allow_pickle=False)
        order = _align(ids, targets, train_ids, np.asarray([train_labels[k] for k in train_ids]), "incumbent OOF")
        baseline = np.load(inc / "oof_mean_probabilities.npy", allow_pickle=False)[order]
        if baseline.shape != targets.shape or not np.isfinite(baseline).all():
            raise ValueError("incumbent OOF probabilities have invalid shape or values")
        legacy_ids = _keys(_json(output / "inputs/legacy/oof_study_ids.json"), "legacy OOF")
        if set(map(tuple, legacy_ids)) != set(train_ids) or len(legacy_ids) != len(train_ids):
            raise ValueError("legacy OOF study IDs do not exactly match train manifest")
        legacy_studies = [{"subject_id": a, "study_id": b} for a, b in legacy_ids]
        legacy_plan = {"legacy_oof": str(output / "inputs/legacy")}
        legacy_values = legacy_oof(legacy_plan, legacy_studies)
        legacy_map = {tuple(key): row for key, row in zip(legacy_ids, legacy_values)}
        legacy = np.asarray([legacy_map[tuple(key)] for key in train_ids])
        targets = targets[order]
        alignment = _validate_legacy_targets(output, plan, train_ids, targets)
        _write(output / "alignment_report.json", alignment)
        subjects = [key[0] for key in train_ids]
        folds = _fold_memberships([{"subject_id": a} for a in subjects],
                                  _json(output / "inputs/partition.json")["subject_partition"])
        candidate = smooth_patient_probabilities(baseline, subjects, [key[1] for key in train_ids])
        gate = training_gate(targets, baseline, candidate, folds, legacy)
        np.save(output / "oof_candidate_probabilities.npy", candidate, allow_pickle=False)
        _write(output / "oof_study_ids.json", train_ids)
        _write(output / "oof_report.json", {"gate": gate, "alpha": ALPHA, "folds": list(FOLDS)})
        if not gate["passed"]:
            state.update(status="complete", phase=None, decision="retain incumbent; patient OOF gate failed",
                         candidate_promoted=False, test_csv_exported=False)
            _finish(output, plan, state)
            return state

        state.update(status="running", phase="valid", valid_evaluated=True)
        _write(output / "experiment.json", state)
        valid_path = output / "inputs/manifests/valid.csv"
        valid_ids, valid_labels, _ = _identities(valid_path)
        valid_studies = [{"subject_id": a, "study_id": b} for a, b in valid_ids]
        current, _, _ = load_submission(inc / "candidate_valid.csv", valid_studies)
        old, _, _ = load_submission(output / "inputs/legacy/valid.csv", valid_studies)
        valid_candidate = smooth_patient_probabilities(current, [x[0] for x in valid_ids], [x[1] for x in valid_ids])
        yvalid = np.asarray([valid_labels[k] for k in valid_ids])
        comparisons = {name: compare_arrays([x[0] for x in valid_ids], yvalid, base, valid_candidate,
                                            bootstrap=1000, seed=20261108)
                       for name, base in (("incumbent", current), ("legacy", old))}
        passed = validation_gate(comparisons)
        write_submission(output / "candidate_valid.csv", valid_studies, valid_candidate)
        _write(output / "valid_report.json", {"comparisons": comparisons, "passed": passed})
        if not passed:
            state.update(status="complete", phase=None, decision="retain incumbent; valid gate failed",
                         candidate_promoted=False, test_csv_exported=False)
            _finish(output, plan, state)
            return state

        state.update(phase="test", test_read=True)
        _write(output / "experiment.json", state)
        test_studies, _, _ = _identities(output / "inputs/manifests/test.csv")
        if len(test_studies) != plan["test_studies"] or len(test_studies) != 1414:
            raise ValueError("test export requires exactly the registered 1,414 studies")
        test_records = [{"subject_id": a, "study_id": b} for a, b in test_studies]
        deferred = plan["deferred_test_input"]
        test_snapshot = output / deferred["snapshot"]
        if _copy_lock(deferred["source"], test_snapshot) != deferred["sha256"]:
            raise ValueError("incumbent test CSV differs from its pre-registered hash")
        current_test, _, _ = load_submission(test_snapshot, test_records)
        test_candidate = smooth_patient_probabilities(current_test, [x[0] for x in test_studies], [x[1] for x in test_studies])
        csv_path = output / "submission_test_1414_cxrlt_patient025_20261108.csv"
        write_submission(csv_path, test_records, test_candidate)
        check = validate_submission(csv_path, test_records)
        if check["rows"] != 14140 or check["default_threshold_rows"] != 14140:
            raise ValueError("test export requires 14,140 rows with Gate=0.5")
        replay, _, _ = load_submission(csv_path, test_records)
        if not np.array_equal(replay, test_candidate):
            raise ValueError("test output replay mismatch")
        _write(output / "submission_verification.json", {**check, "sha256": _sha256(csv_path), "online_score": None})
        state.update(status="complete", phase=None, decision="fixed valid gate passed", candidate_promoted=True,
                     test_csv_exported=True)
        _finish(output, plan, state)
        return state
    except Exception as exc:
        state.update(status="failed", phase=state.get("phase"), error=repr(exc), traceback=traceback.format_exc())
        _write(output / "experiment.json", state)
        raise


def _finish(output, plan, state):
    _verify(output, plan)
    artifacts = {p.relative_to(output).as_posix(): _sha256(p) for p in output.rglob("*")
                 if p.is_file() and "inputs" not in p.relative_to(output).parts
                 and p.name not in {"artifacts.json", "experiment.json"}}
    _write(output / "artifacts.json", artifacts)
    state["updated_utc"] = datetime.now(timezone.utc).isoformat()
    _write(output / "experiment.json", state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", "--run", dest="resume", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not args.resume:
        output = prepare(output)
        if args.prepare_only:
            print(json.dumps({"status": "prepared", "output": str(output)}))
            return 0
    snapshot_script = output / "inputs/source/scripts/run_patient_smoothing.py"
    if Path(__file__).resolve() != snapshot_script.resolve():
        _verify(output, _json(output / "plan.json"))
        return subprocess.call([sys.executable, str(snapshot_script),
                                "--output", str(output), "--resume"])
    run(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
