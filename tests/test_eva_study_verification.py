"""Synthetic boundary tests for the read-only EVA OOF verifier."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import verify_eva_study_probe as verifier


def _manifest_fixture(root):
    root = Path(root)
    plan = {"input_locks": {}, "source_locks": {}, "snapshot_locks": {},
            "incumbent_source_locks": {}}
    plan_path = root / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    (root / "plan_lock.json").write_text(
        json.dumps({"sha256": verifier._sha256(plan_path)}), encoding="utf-8")
    names = list(verifier.REQUIRED_OOF)
    names += [f"fits/{candidate}/{fold}/{name}"
              for candidate in verifier.CANDIDATES for fold in verifier.FOLDS
              for name in ("head.npz", "fit.json")]
    artifacts = {}
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic", encoding="utf-8")
        artifacts[name] = verifier._sha256(path)
    (root / "artifacts.json").write_text(json.dumps(artifacts), encoding="utf-8")
    return plan, artifacts


def _complete_synthetic_run(root):
    root = Path(root)
    incumbent = root / "snapshot" / "incumbent"
    cxr_cache = incumbent / "train_features"
    eva_cache = root / "eva_features" / "train"
    legacy = root / "snapshot" / "legacy_oof"
    for path in (cxr_cache, eva_cache, legacy):
        path.mkdir(parents=True, exist_ok=True)

    records, views, ids, targets = [], [], [], []
    cxr_rows, eva_rows = [], []
    fold_values = (0, 0, 1, 1, 2, 2, "audit", "audit")
    for index, fold in enumerate(fold_values):
        subject, study, dicom = f"p{index}", f"s{index}", f"d{index}"
        path = str((root / f"image-{index}.jpg").resolve())
        identity = [dicom, path, 10 + index, 100 + index]
        ids.append(identity)
        labels = [index % 2] * 10
        targets.append(labels)
        records.append({"subject_id": subject, "study_id": study,
                        "dicom_id": dicom, "path": path, "labels": labels})
        views.append("AP")
        cxr_rows.append(np.full(1054, index + 1, dtype=np.float32))
        eva_rows.append(np.full(389, (index + 1) / 10, dtype=np.float32))
    cxr_values = np.stack(cxr_rows)
    eva_values = np.stack(eva_rows)
    targets = np.asarray(targets, dtype=np.uint8)
    study_ids = [[row["subject_id"], row["study_id"]] for row in records]
    (incumbent / "oof_study_ids.json").write_text(json.dumps(study_ids), encoding="utf-8")
    (incumbent / "train_records.json").write_text(
        json.dumps({"records": records, "views": views}), encoding="utf-8")
    (cxr_cache / "input.json").write_text(json.dumps({"images": ids}), encoding="utf-8")
    (eva_cache / "input.json").write_text(json.dumps({"images": ids}), encoding="utf-8")
    np.save(cxr_cache / "image_features.npy", cxr_values, allow_pickle=False)
    np.save(eva_cache / "image_features.npy", eva_values, allow_pickle=False)
    for cache, matrix in ((cxr_cache, cxr_cache / "image_features.npy"),
                          (eva_cache, eva_cache / "image_features.npy")):
        (cache / "progress.json").write_text(json.dumps({
            "status": "complete", "completed": len(records),
            "sha256": verifier._sha256(matrix),
        }), encoding="utf-8")
    (legacy / "plan.json").write_text(json.dumps({"prior": [[0.125] * 10] * 8}),
                                       encoding="utf-8")
    np.save(legacy / "oof_component_probabilities.npy",
            np.full((len(records), 8, 10), 0.5), allow_pickle=False)
    np.save(incumbent / "oof_targets.npy", targets, allow_pickle=False)
    np.save(incumbent / "oof_mean_probabilities.npy",
            np.full((len(records), 10), 0.5), allow_pickle=False)

    partition = root / "snapshot" / "partition.json"
    partition.parent.mkdir(parents=True, exist_ok=True)
    partition.write_text(json.dumps({"subject_partition": {
        f"p{index}": fold for index, fold in enumerate(fold_values)}}), encoding="utf-8")
    plan = {"incumbent_snapshot": str(incumbent), "legacy_oof": str(legacy),
            "partition": str(partition), "input_locks": {}, "source_locks": {},
            "snapshot_locks": {}, "incumbent_source_locks": {}}
    plan_path = root / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    plan_sha = verifier._sha256(plan_path)
    (root / "plan_lock.json").write_text(json.dumps({"sha256": plan_sha}), encoding="utf-8")
    (eva_cache / "binding.json").write_text(json.dumps({"plan_sha256": plan_sha}),
                                              encoding="utf-8")
    registration = {
        "plan_sha256": plan_sha,
        "binding_sha256": verifier._sha256(eva_cache / "binding.json"),
        "input_sha256": verifier._sha256(eva_cache / "input.json"),
        "progress_sha256": verifier._sha256(eva_cache / "progress.json"),
        "matrix_sha256": verifier._sha256(eva_cache / "image_features.npy"),
    }
    (eva_cache / "registration.json").write_text(json.dumps(registration), encoding="utf-8")

    for candidate, features in (("eva", eva_values.astype(np.float64)),
                                ("joint", np.concatenate((cxr_values, eva_values), axis=1).astype(np.float64))):
        # One image per study in this fixture, so these are also pooled study rows.
        identity = verifier._json_hash({
            "candidate": candidate, "width": verifier.WIDTHS[candidate],
            "source_cxrlt_sha256": verifier._sha256(cxr_cache / "image_features.npy"),
            "eva_features_sha256": verifier._sha256(eva_cache / "image_features.npy"),
            "study_keys_sha256": verifier._json_hash(study_ids),
        })
        for fold in verifier.FOLDS:
            train = np.flatnonzero(np.asarray(fold_values, dtype=object) != fold)
            mean, std = features[train].mean(axis=0), features[train].std(axis=0)
            std[std == 0] = 1.0
            folder = root / "fits" / candidate / str(fold)
            folder.mkdir(parents=True, exist_ok=True)
            head_path = folder / "head.npz"
            np.savez(head_path, mean=mean, std=std,
                     weight=np.zeros((10, features.shape[1])), bias=np.zeros(10))
            training_patients = sorted(f"p{index}" for index, value in enumerate(fold_values)
                                       if value != fold)
            fit_identity = {"plan_sha256": plan_sha, "feature_identity": identity,
                            "candidate": candidate, "width": verifier.WIDTHS[candidate],
                            "training_folds": [item for item in verifier.FOLDS if item != fold],
                            "held_fold": fold, "training_patients": training_patients,
                            "l2": 0.01, "max_iter": 200, "seed": 20261108,
                            "class_balance": False}
            (folder / "fit.json").write_text(json.dumps({
                "identity": fit_identity, "head_sha256": verifier._sha256(head_path),
            }), encoding="utf-8")

    np.save(root / "oof_targets.npy", targets, allow_pickle=False)
    (root / "oof_study_ids.json").write_text(json.dumps(study_ids), encoding="utf-8")
    probabilities = np.full((len(records), 10), 0.5)
    np.save(root / "oof_eva_probabilities.npy", probabilities, allow_pickle=False)
    np.save(root / "oof_joint_probabilities.npy", probabilities, allow_pickle=False)
    folds = np.asarray(fold_values, dtype=object)
    gates = []
    for candidate in verifier.CANDIDATES:
        gates.append({"candidate": candidate,
                      "gate": verifier._gate(targets, np.full_like(probabilities, 0.5),
                                             np.full_like(probabilities, 0.5),
                                             probabilities, folds),
                      "feature_identity": verifier._json_hash({
                          "candidate": candidate, "width": verifier.WIDTHS[candidate],
                          "source_cxrlt_sha256": verifier._sha256(cxr_cache / "image_features.npy"),
                          "eva_features_sha256": verifier._sha256(eva_cache / "image_features.npy"),
                          "study_keys_sha256": verifier._json_hash(study_ids),
                      })})
    (root / "oof_report.json").write_text(json.dumps({"candidates": gates, "selected": None}),
                                          encoding="utf-8")
    (root / "experiment.json").write_text(json.dumps({
        "status": "complete", "candidate_promoted": False, "test_csv_exported": False,
        "decision": "retain incumbent; neither candidate passed the patient OOF gate",
    }), encoding="utf-8")
    names = list(verifier.REQUIRED_OOF)
    names += [f"fits/{candidate}/{fold}/{name}"
              for candidate in verifier.CANDIDATES for fold in verifier.FOLDS
              for name in ("head.npz", "fit.json")]
    names += [str(path.relative_to(root)) for path in (root / "eva_features").rglob("*")
              if path.is_file()]
    artifact_map = {name: verifier._sha256(root / name) for name in names}
    (root / "artifacts.json").write_text(json.dumps(artifact_map), encoding="utf-8")


class EvaStudyVerificationTests(unittest.TestCase):
    def test_running_experiment_is_reported_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experiment.json").write_text(
                json.dumps({"status": "running"}), encoding="utf-8")
            with self.assertRaisesRegex(verifier.IncompleteRun, "running"):
                verifier.verify_run(root)

    def test_unregistered_required_oof_file_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, artifacts = _manifest_fixture(root)
            del artifacts["oof_targets.npy"]
            (root / "artifacts.json").write_text(json.dumps(artifacts), encoding="utf-8")
            with self.assertRaisesRegex(verifier.IncompleteRun, "not registered"):
                verifier._verify_manifest(root, plan, [])

    def test_changed_registered_oof_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, _ = _manifest_fixture(root)
            (root / "oof_targets.npy").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash changed"):
                verifier._verify_manifest(root, plan, [])

    def test_windows_artifact_separators_normalize_to_required_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, artifacts = _manifest_fixture(root)
            windows_artifacts = {name.replace("/", "\\"): digest
                                 for name, digest in artifacts.items()}
            (root / "artifacts.json").write_text(json.dumps(windows_artifacts), encoding="utf-8")
            _, normalized = verifier._verify_manifest(root, plan, [])
            self.assertIn("fits/eva/0/head.npz", normalized)
            self.assertIn("oof_targets.npy", normalized)

    def test_duplicate_or_escaping_artifact_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, artifacts = _manifest_fixture(root)
            first_name = "fits/eva/0/head.npz"
            artifacts[first_name.replace("/", "\\")] = artifacts[first_name]
            (root / "artifacts.json").write_text(json.dumps(artifacts), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate normalized"):
                verifier._verify_manifest(root, plan, [])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, artifacts = _manifest_fixture(root)
            artifacts["..\\outside.txt"] = "0" * 64
            (root / "artifacts.json").write_text(json.dumps(artifacts), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escapes"):
                verifier._verify_manifest(root, plan, [])

    def test_missing_target_class_prevents_candidate_gate(self):
        targets = np.tile(np.asarray([[0] * 10, [1] * 10], dtype=np.uint8), (4, 1))
        targets[:, 0] = 0
        probabilities = np.tile(np.linspace(0.1, 0.9, 8)[:, None], (1, 10))
        folds = np.repeat(np.asarray([0, 1, 2, "audit"], dtype=object), 2)
        gate = verifier._gate(targets, probabilities, probabilities, probabilities, folds)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["candidate"]["macro_auc_defined_classes"], 9)

    def test_patient_cannot_appear_in_multiple_outer_folds(self):
        folds = np.asarray([0, 0, 1, 1], dtype=object)
        subjects = ["patient-a", "patient-b", "patient-a", "patient-c"]
        with self.assertRaisesRegex(ValueError, "patient leakage"):
            verifier._assert_fold_isolation(folds, subjects)

    def test_complete_synthetic_oof_run_replays_without_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            _complete_synthetic_run(Path(directory))
            report = verifier.verify_run(directory)
            self.assertEqual(report["status"], "verified_oof_only")
            self.assertIsNone(report["selected_oof_candidate"])
            self.assertEqual(report["coverage"]["oof_heads_and_gates"], "verified")


if __name__ == "__main__":
    unittest.main()
