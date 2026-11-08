"""Synthetic isolation, recovery and stage-boundary tests; no private inputs."""

import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from scripts import run_chexficient_study_probe as runner


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value), encoding="utf-8")
    return path


def identities(n):
    return [[f"d{i}", str(Path(f"synthetic-{i}.jpg").resolve()), i + 1, i]
            for i in range(n)]


def head(width=512):
    weight = np.zeros((10, width))
    weight[:, 0] = np.linspace(0.1, 1, 10)
    return {"mean": np.zeros(width), "std": np.ones(width),
            "weight": weight, "bias": np.linspace(-0.5, 0.5, 10),
            "l2": 0.01, "iterations": 1, "closure_calls": 1,
            "loss": 0.5, "seed": runner.SEED}


def cv_data():
    folds = np.array([fold for fold in runner.FOLDS for _ in range(4)], dtype=object)
    subjects = [f"p{i}" for i in range(len(folds))]
    features = np.zeros((len(folds), 512))
    features[:, 0] = np.arange(len(folds)) / 10
    targets = np.tile([[0, 1] * 5, [1, 0] * 5], (len(folds) // 2, 1))
    return features, targets, folds, subjects


class ChexficientStudyRunnerTests(unittest.TestCase):
    def test_fixed_candidate_widths_and_execution_defaults(self):
        self.assertEqual(runner.CANDIDATES, ("chexficient", "joint"))
        self.assertEqual(runner.FEATURE_WIDTHS, {"chexficient": 512, "joint": 1566})
        self.assertEqual(runner.execution_settings(), {"batch_size": 4, "preprocess_workers": 2})

    def test_pooling_uses_each_image_once_and_preserves_study_order(self):
        records = [{"subject_id": "p1", "study_id": "s1", "dicom_id": "d0", "labels": [0] * 10},
                   {"subject_id": "p2", "study_id": "s2", "dicom_id": "d1", "labels": [1] * 10},
                   {"subject_id": "p1", "study_id": "s1", "dicom_id": "d2", "labels": [0] * 10}]
        cxrlt = np.repeat(np.array([[2], [10], [6]], dtype=np.float32), 1054, axis=1)
        chex = np.repeat(np.array([[1], [8], [3]], dtype=np.float32), 512, axis=1)
        ids = identities(3)
        studies, features, targets = runner.pool_candidates(
            records, ["AP", "AP", "LL"], cxrlt, chex, ids, ids)
        self.assertEqual([s["study_id"] for s in studies], ["s1", "s2"])
        np.testing.assert_array_equal(features["chexficient"][:, 0], [2, 8])
        np.testing.assert_array_equal(features["joint"][:, 0], [4, 10])
        np.testing.assert_array_equal(features["joint"][:, 1054], [2, 8])
        np.testing.assert_array_equal(targets, [[0] * 10, [1] * 10])
        with self.assertRaises(ValueError):
            runner.pool_candidates(records, ["AP"] * 3, cxrlt, chex, ids, ids[::-1])

    def test_image_identity_rejects_numeric_duplicate_and_reordered_ids(self):
        valid = identities(2)
        runner.require_exact_image_identity(valid, valid)
        malformed = copy.deepcopy(valid)
        malformed[0][0] = 123
        for value in (malformed, [valid[0], valid[0]], valid[::-1]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.require_exact_image_identity(valid, value)

    def test_cache_binds_encoder_plan_and_exact_order(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = {"output_dir": directory, "plan_sha256": "plan", "revision": "rev",
                    "weight_sha256": "weights", "manifest_sha256": "manifest",
                    "source_locks": {"cxr/chexficient_transfer.py": "adapter"},
                    "device": "cpu", **runner.execution_settings()}
            ids = identities(2)
            folder = runner._bind_chexficient_cache(plan, "train", ids)
            self.assertTrue((folder / "binding.json").is_file())
            self.assertEqual(folder, runner._bind_chexficient_cache(plan, "train", ids))
            for key in ("plan_sha256", "revision", "weight_sha256", "manifest_sha256"):
                changed = {**plan, key: "different"}
                with self.subTest(key=key), self.assertRaises(ValueError):
                    runner._bind_chexficient_cache(changed, "train", ids)
            changed = copy.deepcopy(plan)
            changed["source_locks"]["cxr/chexficient_transfer.py"] = "other-adapter"
            with self.assertRaises(ValueError):
                runner._bind_chexficient_cache(changed, "train", ids)
            with self.assertRaises(ValueError):
                runner._bind_chexficient_cache(plan, "train", ids[::-1])

    def test_unbound_feature_payload_cannot_be_adopted(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = {"output_dir": directory, "plan_sha256": "plan", "revision": "rev",
                    "weight_sha256": "weights", "manifest_sha256": "manifest",
                    "source_locks": {"cxr/chexficient_transfer.py": "adapter"},
                    "device": "cpu", **runner.execution_settings()}
            put(Path(directory) / "chexficient_features/train/progress.json", {"completed": 4})
            with self.assertRaisesRegex(ValueError, "without its binding"):
                runner._bind_chexficient_cache(plan, "train", identities(2))

    def test_completed_feature_registration_rejects_later_matrix_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            put(root / "binding.json", {})
            put(root / "input.json", {"images": identities(2)})
            matrix = put(root / "image_features.npy", b"synthetic matrix")
            put(root / "progress.json", {"status": "complete", "sha256": runner.sha256(matrix)})
            registered = runner._register_chexficient_cache(root, "plan")
            self.assertEqual(registered, runner._register_chexficient_cache(root, "plan"))
            matrix.write_bytes(b"changed matrix")
            with self.assertRaises(ValueError):
                runner._register_chexficient_cache(root, "plan")

    def test_deferred_cache_materializes_only_original_registered_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original"
            matrix = put(source / "image_features.npy", b"synthetic deferred features")
            input_path = put(source / "input.json", {"images": identities(2)})
            progress = put(source / "progress.json", {"status": "complete", "sha256": runner.sha256(matrix)})
            plan = {"incumbent_snapshot": str(root / "snapshot"), "deferred_cxrlt": {
                "valid": {"cache_dir": str(source), "input_sha256": runner.sha256(input_path),
                          "progress_sha256": runner.sha256(progress), "matrix_sha256": runner.sha256(matrix)}}}
            runner._materialize_deferred_cxrlt(plan, "valid")
            target = root / "snapshot/valid_features/image_features.npy"
            self.assertEqual(target.read_bytes(), matrix.read_bytes())
            matrix.write_bytes(b"different features")
            with self.assertRaises(ValueError):
                runner._materialize_deferred_cxrlt(plan, "valid")

    def test_fold_head_fits_only_other_patients_and_replays_saved_head(self):
        x, y, folds, subjects = cv_data()
        with tempfile.TemporaryDirectory() as directory, patch.object(
                runner, "fit_linear_probe", return_value=head()) as fit:
            cache = runner.FoldHeadCache(directory, x, y, folds, subjects, "plan", "features")
            first = cache.predict("chexficient", 0)
            second = cache.predict("chexficient", 0)
            np.testing.assert_array_equal(first, second)
            self.assertEqual(fit.call_count, 1)
            np.testing.assert_array_equal(fit.call_args.args[0], x[folds != 0])
            np.testing.assert_array_equal(fit.call_args.args[1], y[folds != 0])
            metadata = runner.read_json(Path(directory) / "chexficient/0/fit.json")
            self.assertEqual(metadata["identity"]["training_patients"], sorted(subjects[4:]))
            self.assertFalse(fit.call_args.kwargs["class_balance"])
            self.assertEqual(fit.call_args.kwargs["max_iter"], 200)

    def test_fold_cache_rejects_changed_plan_features_head_and_held_predictions(self):
        x, y, folds, subjects = cv_data()
        with tempfile.TemporaryDirectory() as directory, patch.object(
                runner, "fit_linear_probe", return_value=head()):
            cache = runner.FoldHeadCache(directory, x, y, folds, subjects, "plan", "features")
            cache.predict("chexficient", 0)
            for plan_id, feature_id in (("other-plan", "features"), ("plan", "other-features")):
                other = runner.FoldHeadCache(directory, x, y, folds, subjects, plan_id, feature_id)
                with self.assertRaises(ValueError):
                    other.predict("chexficient", 0)
            changed = x.copy()
            changed[0, 0] += 1
            other = runner.FoldHeadCache(directory, changed, y, folds, subjects, "plan", "features")
            with self.assertRaises(ValueError):
                other.predict("chexficient", 0)
            path = Path(directory) / "chexficient/0/head.npz"
            path.write_bytes(path.read_bytes() + b"tampered")
            with self.assertRaises(ValueError):
                cache.predict("chexficient", 0)

    def test_shared_patient_across_train_and_held_is_rejected_before_fit(self):
        x, y, folds, subjects = cv_data()
        subjects[4] = subjects[0]
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "fit_linear_probe") as fit:
            cache = runner.FoldHeadCache(directory, x, y, folds, subjects, "plan", "features")
            with self.assertRaisesRegex(ValueError, "patient leakage"):
                cache.predict("chexficient", 0)
            fit.assert_not_called()

    def test_interrupted_head_write_can_refit_without_committing_bad_metadata(self):
        x, y, folds, subjects = cv_data()
        with tempfile.TemporaryDirectory() as directory, patch.object(
                runner, "fit_linear_probe", return_value=head()) as fit:
            cache = runner.FoldHeadCache(directory, x, y, folds, subjects, "plan", "features")
            with patch.object(runner, "atomic_json", side_effect=RuntimeError("stop")):
                with self.assertRaisesRegex(RuntimeError, "stop"):
                    cache.predict("chexficient", 0)
            self.assertFalse((Path(directory) / "chexficient/0/fit.json").exists())
            self.assertEqual(cache.predict("chexficient", 0).shape, (4, 10))
            self.assertEqual(fit.call_count, 2)

    def test_full_head_recipe_rejects_a_different_feature_identity(self):
        x, y, _, _ = cv_data()
        with tempfile.TemporaryDirectory() as directory, patch.object(
                runner, "fit_linear_probe", return_value=head()) as fit:
            output = Path(directory)
            put(output / "plan.json", {})
            plan = {"incumbent_plan_sha256": "incumbent-plan"}
            runner._fit_final_head(output, plan, "chexficient", x, y, "features")
            runner._fit_final_head(output, plan, "chexficient", x, y, "features")
            self.assertEqual(fit.call_count, 1)
            with self.assertRaises(ValueError):
                runner._fit_final_head(output, plan, "chexficient", x, y, "other-features")

    def test_oof_gate_requires_improvement_in_at_least_three_patient_groups(self):
        _, y, folds, _ = cv_data()
        perfect = 0.1 + 0.8 * y
        baseline = perfect.copy()
        for fold in runner.FOLDS:
            indices = np.flatnonzero(folds == fold)
            baseline[indices[:2]] = baseline[indices[:2][::-1]]
        gate = runner.candidate_gate(y, baseline, baseline, perfect, folds)
        self.assertTrue(gate["passed"])
        mixed = perfect.copy()
        for fold in runner.FOLDS[:2]:
            mixed[folds == fold] = baseline[folds == fold]
        gate = runner.candidate_gate(y, baseline, baseline, mixed, folds)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["positive_held_groups"], 2)

    def test_selection_requires_gate_and_exact_ties_prefer_512_dimensions(self):
        rows = [{"candidate": name, "gate": {"passed": True, "candidate": {"macro_auc": 0.84}}}
                for name in reversed(runner.CANDIDATES)]
        self.assertEqual(runner.select_candidate(rows)["candidate"], "chexficient")
        rows[0]["gate"]["passed"] = rows[1]["gate"]["passed"] = False
        self.assertIsNone(runner.select_candidate(rows))

    def test_failed_validation_cannot_invoke_test_export(self):
        export = Mock()
        with patch.object(runner, "validation_gate", return_value=False):
            self.assertFalse(runner._gate_then_export({}, export))
        export.assert_not_called()

    def test_prepare_requires_authority_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not-created"
            with patch.object(runner, "_require_preparation_authority", side_effect=ValueError("incomplete")):
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    runner.prepare(output, Path(directory) / "missing")
            self.assertFalse(output.exists())

    def test_prerequisites_accept_both_real_plan_lock_schemas_and_reject_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = put(root / "protocol.md", b"fixed protocol")
            states = []
            for index in range(2):
                stage = root / f"stage-{index}"
                plan = {"fixed": index}
                if index:
                    plan["plan_sha256"] = runner._json_hash(plan)
                plan_path = put(stage / "plan.json", plan)
                if not index:
                    put(stage / "plan_lock.json", {"sha256": runner.sha256(plan_path)})
                artifact = put(stage / "report.json", {"complete": True})
                put(stage / "artifacts.json", {"report.json": runner.sha256(artifact)})
                states.append(put(stage / "experiment.json", {
                    "status": "complete", "candidate_promoted": False,
                    "test_csv_exported": False, "test_read": False, "test_extracted": False}))
            registrations = runner._require_preparation_authority(protocol, states)
            self.assertEqual(len(registrations), 2)
            second_plan = states[1].parent / "plan.json"
            put(second_plan, {**runner.read_json(second_plan), "fixed": "changed"})
            with self.assertRaises(ValueError):
                runner._require_preparation_authority(protocol, states)

    def test_split_isolation_allows_multiple_images_but_rejects_shared_patient_or_study(self):
        train = [{"subject_id": "p0", "study_id": "s0", "dicom_id": "d0"},
                 {"subject_id": "p0", "study_id": "s0", "dicom_id": "d1"}]
        valid = [{"subject_id": "p1", "study_id": "s1", "dicom_id": "d2"},
                 {"subject_id": "p1", "study_id": "s1", "dicom_id": "d3"}]
        runner._validate_split_disjointness(train, valid, "valid")
        for subject, study in (("p0", "s1"), ("p1", "s0")):
            with self.subTest(subject=subject, study=study), self.assertRaises(ValueError):
                runner._validate_split_disjointness(train, [{"subject_id": subject, "study_id": study}], "valid")
        conflict = train + [{"subject_id": "p-other", "study_id": "s0"}]
        with self.assertRaises(ValueError):
            runner._validate_split_disjointness(conflict, valid, "valid")

    def test_legacy_labels_require_exact_string_ids_complete_coverage_and_matching_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = [["p0", "s0"], ["p1", "s1"]]
            studies = [{"subject_id": p, "study_id": s} for p, s in ids]
            targets = np.array([[0, 1] * 5, [1, 0] * 5], dtype=float)
            id_path = put(root / "snapshot/legacy/ids.json", ids)
            target_path = root / "snapshot/legacy/targets.npy"
            np.save(target_path, targets, allow_pickle=False)
            plan = {"legacy_target_sources": [{"ids": "legacy/ids.json", "targets": "legacy/targets.npy"}]}
            audit = runner._validate_legacy_targets(root, plan, studies, targets)
            self.assertEqual(audit["covered_studies"], 2)
            for malformed in ([[0, "s0"], ["p1", "s1"]], [ids[0], ids[0]], [ids[0]]):
                put(id_path, malformed)
                with self.subTest(ids=malformed), self.assertRaises(ValueError):
                    runner._validate_legacy_targets(root, plan, studies, targets)
            put(id_path, ids)
            np.save(target_path, 1 - targets, allow_pickle=False)
            with self.assertRaises(ValueError):
                runner._validate_legacy_targets(root, plan, studies, targets)

    def test_prepare_and_plan_replay_do_not_open_missing_deferred_inputs(self):
        """A real prepare snapshot must succeed with inaccessible test predictions."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, source, data = root / "repo", root / "incumbent", root / "data"
            output = root / "run"
            for relative in runner.SOURCE_FILES:
                put(repo / relative, b"# synthetic source asset\n")
            protocol = repo / "docs/CHEXFICIENT_STUDY_PROBE_PLAN_20261108.md"
            model = repo / "weights/model"
            model_files = {}
            for relative in runner.CHEXFICIENT_MODEL_FILES:
                path = put(model / relative, f"synthetic asset {relative}".encode())
                model_files[relative] = {"sha256": runner.sha256(path), "bytes": path.stat().st_size}
            weight_hash = model_files["model.safetensors"]["sha256"]
            manifest_path = put(repo / "docs/chexficient_source_manifest_20261108.json", {
                "repository": "StanfordAIMI/CheXficient", "revision": runner.CHEXFICIENT_REVISION,
                "files": model_files})
            adapter_hash = runner.sha256(repo / "cxr/chexficient_transfer.py")
            evidence = []
            for name in ("cpu_engineering_check-v2.json", "gpu_engineering_check.json"):
                evidence.append(put(root / "evidence" / name, {
                    "status": "passed", "adapter_sha256": adapter_hash, "weights_sha256": weight_hash,
                    "batch_size": 4, "preprocess_workers": 2, "amp": "cuda float16",
                    "model": "StanfordAIMI/CheXficient", "revision": runner.CHEXFICIENT_REVISION,
                    "input_shape": [4, 3, 378, 378], "embedding_shape": [4, 512],
                    "repeat_max_abs_error": 0.0, "frozen": True, "eval": True}))
            for name in runner.INCUMBENT_ARTIFACTS:
                put(source / name, {})
            head_path = put(source / "final_head.npz", b"synthetic incumbent head")
            put(source / "recipe.json", {"head_sha256": runner.sha256(head_path)})
            put(source / "experiment.json", {"status": "complete", "candidate_promoted": True,
                                             "test_csv_exported": True})
            put(source / "submission_verification.json", {"valid": True, "studies": 1414,
                                                         "rows": 14140, "sha256": "a" * 64})
            partition = put(source / "partition.json", {"subject_partition": {"p0": 0}})
            old_valid = put(source / "legacy_valid.csv", b"synthetic valid prediction")
            old_oof = source / "legacy"
            for name in ("plan.json", "experiment.json", "oof_study_ids.json"):
                put(old_oof / name, {})
            put(old_oof / "oof_component_probabilities.npy", b"synthetic legacy OOF")
            for split in ("train", "valid", "test"):
                header = "Dicom_id,Subject_id,Study_id" + (",Predict_class" if split != "test" else "")
                line = f"d-{split},p-{split},s-{split}" + (",1" if split != "test" else "")
                put(data / f"{split}.csv", (header + "\n" + line + "\n").encode())
                cache = source / f"{split}_features"
                put(cache / "input.json", {"images": []})
                digest = "b" * 64
                if split == "train":
                    digest = runner.sha256(put(cache / "image_features.npy", b"synthetic train cache"))
                put(cache / "progress.json", {"status": "complete", "sha256": digest})
            # These files deliberately do not exist; even a hash/copy attempt fails.
            missing_test_csv = root / "arbitrary-name-private-prediction.csv"
            input_locks = {str(data / f"{split}.csv"): runner.sha256(data / f"{split}.csv")
                           for split in ("train", "valid", "test")}
            input_locks[str(missing_test_csv)] = "c" * 64
            put(source / "plan.json", {"data_root": str(data), "input_locks": input_locks,
                "source_locks": {}, "partition": str(partition), "old_oof": str(old_oof),
                "old_valid": str(old_valid), "old_test": str(missing_test_csv), "device": "cpu"})
            stages = []
            for index in range(2):
                stage = root / f"stage-{index}"
                target_sources, locks = [], {}
                if index == 1:
                    for i in range(22):
                        ids_rel = f"inputs/legacy_targets/{i:02d}/study_ids.json"
                        target_rel = f"inputs/legacy_targets/{i:02d}/study_targets.npy"
                        ids = put(stage / ids_rel, [["p0", "s0"]])
                        target = stage / target_rel
                        np.save(target, np.zeros((1, 10)), allow_pickle=False)
                        locks[ids_rel], locks[target_rel] = runner.sha256(ids), runner.sha256(target)
                        target_sources.append({"ids": ids_rel, "targets": target_rel})
                stage_plan = {"legacy_target_sources": target_sources, "input_locks": locks}
                stage_plan["plan_sha256"] = runner._json_hash(stage_plan)
                put(stage / "plan.json", stage_plan)
                artifact = put(stage / "report.json", {})
                put(stage / "artifacts.json", {"report.json": runner.sha256(artifact)})
                state = {"status": "complete", "candidate_promoted": False, "test_csv_exported": False}
                put(stage / "experiment.json", state)
                stages.append({"root": str(stage), "state": state,
                               "plan_sha256": runner.sha256(stage / "plan.json"),
                               "artifacts_sha256": runner.sha256(stage / "artifacts.json")})
            eva_report = put(root / "independent/eva.json", {
                "status": "verified_oof_only", "run": stages[0]["root"],
                "runner_status": "complete", "selected_oof_candidate": None,
                "candidate_promoted": False, "coverage": {"oof_heads_and_gates": "verified"}})
            audit_names = ("canonical_plan_sha256", "83_input_locks", "12_source_locks",
                           "7_artifacts", "deferred_test_unopened", "split_identity_isolation",
                           "22_legacy_targets", "OOF_replay_and_order", "OOF_metrics",
                           "valid_gate_failure", "valid_metrics_and_patient_bootstrap", "terminal_gate_state")
            smoothing_report = put(root / "independent/smoothing.json", {
                "result": "passed", "experiment": stages[1]["root"], "failed_checks": [],
                "plan_sha256": runner.read_json(Path(stages[1]["root"]) / "plan.json")["plan_sha256"],
                "checks": {name: {"passed": True} for name in audit_names}})
            with patch.object(runner, "ROOT", repo), patch.object(runner, "MODEL_DIR", model), \
                    patch.object(runner, "MANIFEST_PATH", manifest_path), \
                    patch.object(runner, "CHEXFICIENT_WEIGHT_SHA256", weight_hash), \
                    patch.object(runner, "CHEXFICIENT_WEIGHT_BYTES", model_files["model.safetensors"]["bytes"]), \
                    patch.object(runner, "CHEXFICIENT_ADAPTER_SHA256", adapter_hash), \
                    patch.object(runner, "ENGINEERING_EVIDENCE", tuple(evidence)), \
                    patch.object(runner, "INDEPENDENT_EVIDENCE", (eva_report, smoothing_report)), \
                    patch.object(runner, "environment", return_value={"synthetic": "test"}), \
                    patch.object(runner, "_require_preparation_authority", return_value=stages):
                runner.prepare(output, source, plan_path=protocol)
                plan = runner.load_registered_plan(output)
                self.assertNotIn(str(missing_test_csv), plan["input_locks"])
                for split in ("valid", "test"):
                    self.assertFalse((Path(plan["incumbent_snapshot"]) / f"{split}_features/image_features.npy").exists())
                    self.assertEqual(plan["deferred_cxrlt"][split]["matrix_sha256"], "b" * 64)
                self.assertEqual(len(plan["legacy_target_sources"]), 22)
                put(output / "plan.json", {**runner.read_json(output / "plan.json"), "changed": True})
                with self.assertRaises(ValueError):
                    runner.load_registered_plan(output)

    def test_snapshot_contains_all_local_python_imports(self):
        files = set(runner.SOURCE_FILES)
        for relative in sorted(files):
            if not relative.endswith(".py"):
                continue
            tree = ast.parse((runner.ROOT / relative).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                elif isinstance(node, ast.Import):
                    modules = [item.name for item in node.names]
                else:
                    continue
                for module in modules:
                    path = Path(*module.split(".")).with_suffix(".py")
                    if path.parts[0] in {"cxr", "scripts"} and (runner.ROOT / path).is_file():
                        self.assertIn(path.as_posix(), files, f"{relative} imports unsnapshotted {path}")


if __name__ == "__main__":
    unittest.main()
