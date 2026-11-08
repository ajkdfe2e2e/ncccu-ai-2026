"""Synthetic contract tests for the fixed EVA-X study runner."""
import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import run_eva_study_probe as runner


def fake_head(width):
    return {"mean": np.zeros(width), "std": np.ones(width),
            "weight": np.zeros((10, width)), "bias": np.zeros(10),
            "l2": 0.01, "iterations": 1, "closure_calls": 1,
            "loss": 0.5, "seed": runner.SEED}


def image_identity(rows):
    return [[f"d{i}", str(Path(f"image-{i}.jpg").resolve()), 100 + i, 1000 + i]
            for i in range(rows)]


class EvaStudyRunnerTests(unittest.TestCase):
    def test_fixed_eva_extraction_settings_are_plan_ready(self):
        self.assertEqual(runner.execution_settings(), {
            "batch_size": 16, "preprocess_workers": 2,
        })
        self.assertEqual(runner.execution_settings(8, 1), {
            "batch_size": 8, "preprocess_workers": 1,
        })
        for settings in ((0, 2), (16, -1)):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                runner.execution_settings(*settings)

    def test_eva_cache_binding_only_initializes_empty_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = {
                "output_dir": directory, "plan_sha256": "plan", "revision": "revision",
                "weight_sha256": "weight", "snapshot_locks": {"eva_model/eva_x.py": "source"},
                **runner.execution_settings(), "device": "cpu",
            }
            identities = image_identity(1)
            empty = runner._bind_eva_cache(plan, "empty", identities)
            self.assertTrue((empty / "binding.json").is_file())
            for split, completed in (("partial", False), ("complete", True)):
                cache = Path(directory) / "eva_features" / split
                cache.mkdir(parents=True)
                if completed:
                    matrix = cache / "image_features.npy"
                    matrix.write_bytes(b"complete but unbound")
                    (cache / "input.json").write_text("{}", encoding="utf-8")
                    (cache / "progress.json").write_text(json.dumps({
                        "status": "complete", "sha256": runner.sha256(matrix),
                    }), encoding="utf-8")
                else:
                    (cache / "progress.json").write_text("{}", encoding="utf-8")
                with self.subTest(split=split), self.assertRaisesRegex(ValueError, "without its binding"):
                    runner._bind_eva_cache(plan, split, identities)

    def test_eva_cache_registration_locks_binding_input_progress_and_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            (cache / "binding.json").write_text("{}", encoding="utf-8")
            (cache / "input.json").write_text("{\"images\": []}", encoding="utf-8")
            matrix = cache / "image_features.npy"
            matrix.write_bytes(b"finite-feature-matrix-placeholder")
            progress = cache / "progress.json"
            progress.write_text(json.dumps({"status": "complete", "sha256": runner.sha256(matrix)}),
                                encoding="utf-8")
            registration = runner._register_eva_cache(cache, "plan-sha")
            self.assertEqual(registration["plan_sha256"], "plan-sha")
            self.assertEqual(registration["binding_sha256"], runner.sha256(cache / "binding.json"))
            self.assertEqual(registration["input_sha256"], runner.sha256(cache / "input.json"))
            self.assertEqual(registration["progress_sha256"], runner.sha256(progress))
            self.assertEqual(registration["matrix_sha256"], runner.sha256(matrix))
            self.assertEqual(runner._register_eva_cache(cache, "plan-sha"), registration)
            matrix.write_bytes(b"matrix changed")
            with self.assertRaisesRegex(ValueError, "complete matrix hash"):
                runner._register_eva_cache(cache, "plan-sha")

    def test_image_identity_is_ordered_and_unique(self):
        expected = image_identity(3)
        self.assertEqual(runner.require_exact_image_identity(expected, expected), expected)
        with self.assertRaisesRegex(ValueError, "ordered image identities"):
            runner.require_exact_image_identity(expected, [expected[1], expected[0], expected[2]])
        with self.assertRaisesRegex(ValueError, "duplicate image identity"):
            runner.require_exact_image_identity(expected, [expected[0], expected[0], expected[2]])

    def test_pool_candidates_enforces_source_width_and_aligned_rows(self):
        records = [{"subject_id": "p1", "study_id": "s1", "dicom_id": "d1",
                    "labels": [0] * 10}]
        ids = image_identity(1)
        studies, features, targets = runner.pool_candidates(
            records, ["AP"], np.zeros((1, 1054), dtype=np.float32),
            np.ones((1, 389), dtype=np.float32), ids, ids)
        self.assertEqual([features[name].shape for name in runner.CANDIDATES], [(1, 389), (1, 1443)])
        self.assertEqual(targets.shape, (1, 10))
        self.assertEqual(studies[0]["study_id"], "s1")
        with self.assertRaisesRegex(ValueError, "width"):
            runner.pool_candidates(records, ["AP"], np.zeros((1, 1053)),
                                   np.ones((1, 389)), ids, ids)

    def test_matrix_contract_requires_finite_float32_and_exact_shape(self):
        runner._validate_matrix(np.zeros((2, 389), dtype=np.float32), 2, 389, "eva")
        for matrix in (np.zeros((2, 388), dtype=np.float32),
                       np.zeros((2, 389), dtype=np.float64),
                       np.full((2, 389), np.nan, dtype=np.float32)):
            with self.subTest(dtype=matrix.dtype, shape=matrix.shape):
                with self.assertRaises(ValueError):
                    runner._validate_matrix(matrix, 2, 389, "eva")

    def test_candidate_gate_requires_legacy_full_class_summary(self):
        mocked_gate = {"passed": True, "candidate": {"macro_auc": 0.8}}
        with patch.object(runner, "training_gate", return_value=mocked_gate), \
                patch.object(runner, "_score", return_value={
                    "macro_auc": 0.7, "macro_auc_defined_classes": 9, "per_class_auc": []}):
            result = runner.candidate_gate(np.zeros((2, 10)), np.zeros((2, 10)),
                                           np.zeros((2, 10)), np.zeros((2, 10)), np.zeros(2))
        self.assertFalse(result["passed"])

    def test_exact_tie_prefers_eva(self):
        rows = [{"candidate": name, "gate": {"passed": True,
                "candidate": {"macro_auc": 0.75}}} for name in reversed(runner.CANDIDATES)]
        self.assertEqual(runner.select_candidate(rows)["candidate"], "eva")
        self.assertIsNone(runner.select_candidate([
            {"candidate": "eva", "gate": {"passed": False, "candidate": {"macro_auc": 1.0}}}
        ]))

    def test_fold_head_cache_replays_and_rejects_binding_or_head_tampering(self):
        folds = np.asarray([*([fold] * 2 for fold in runner.FOLDS)], dtype=object).reshape(-1)
        subjects = [f"patient-{index}" for index in range(len(folds))]
        features = np.zeros((len(folds), 389), dtype=np.float64)
        targets = np.tile(np.array([[0, 1] * 5, [1, 0] * 5], dtype=float),
                          (len(folds) // 2, 1))
        payload = fake_head(389)
        with tempfile.TemporaryDirectory() as directory, patch.object(
                runner, "fit_linear_probe", return_value=payload) as fit:
            cache = runner.FoldHeadCache(directory, features, targets, folds, subjects,
                                         "plan-sha", "feature-sha")
            first = cache.predict("eva", runner.FOLDS[0])
            second = cache.predict("eva", runner.FOLDS[0])
            np.testing.assert_array_equal(first, second)
            self.assertEqual(fit.call_count, 1)
            head_path = Path(directory) / "eva" / str(runner.FOLDS[0]) / "head.npz"
            changed_plan = runner.FoldHeadCache(directory, features, targets, folds, subjects,
                                                "different-plan", "feature-sha")
            with self.assertRaisesRegex(ValueError, "binding or hash"):
                changed_plan.predict("eva", runner.FOLDS[0])
            head_path.write_bytes(head_path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "binding or hash"):
                cache.predict("eva", runner.FOLDS[0])

    def test_fold_cache_rejects_patient_leakage(self):
        folds = np.asarray([*([fold] * 2 for fold in runner.FOLDS)], dtype=object).reshape(-1)
        subjects = [f"patient-{index}" for index in range(len(folds))]
        subjects[2] = subjects[0]
        cache = runner.FoldHeadCache(tempfile.gettempdir(), np.zeros((len(folds), 389)),
                                     np.zeros((len(folds), 10)), folds, subjects,
                                     "plan-sha", "feature-sha")
        with self.assertRaisesRegex(ValueError, "patient leakage"):
            cache.predict("eva", runner.FOLDS[0])

    def test_fold_cache_recovers_head_written_before_metadata_commit(self):
        folds = np.asarray([*([fold] * 2 for fold in runner.FOLDS)], dtype=object).reshape(-1)
        subjects = [f"patient-{index}" for index in range(len(folds))]
        features = np.zeros((len(folds), 389), dtype=np.float64)
        targets = np.tile(np.array([[0, 1] * 5, [1, 0] * 5], dtype=float),
                          (len(folds) // 2, 1))
        cache_payload = fake_head(389)
        with tempfile.TemporaryDirectory() as directory, patch.object(
                runner, "fit_linear_probe", return_value=cache_payload) as fit:
            cache = runner.FoldHeadCache(directory, features, targets, folds, subjects,
                                         "plan-sha", "feature-sha")
            original_atomic_json = runner.atomic_json
            with patch.object(runner, "atomic_json", side_effect=RuntimeError("simulated stop")):
                with self.assertRaisesRegex(RuntimeError, "simulated stop"):
                    cache.predict("eva", runner.FOLDS[0])
            head_path = Path(directory) / "eva" / str(runner.FOLDS[0]) / "head.npz"
            self.assertTrue(head_path.is_file())
            self.assertFalse((head_path.parent / "fit.json").exists())
            with patch.object(runner, "atomic_json", side_effect=original_atomic_json):
                replay = cache.predict("eva", runner.FOLDS[0])
            self.assertEqual(fit.call_count, 2)
            self.assertEqual(replay.shape, (2, 10))

    def test_final_head_replay_failure_does_not_commit_recipe(self):
        features = np.zeros((3, 389), dtype=np.float64)
        targets = np.tile(np.array([[0, 1] * 5], dtype=float), (3, 1))
        payload = fake_head(389)
        with tempfile.TemporaryDirectory() as directory, patch.object(
                runner, "fit_linear_probe", return_value=payload) as fit:
            output = Path(directory)
            (output / "plan.json").write_text("{}", encoding="utf-8")
            plan = {"incumbent_plan_sha256": "source-plan"}
            with patch.object(runner, "_load_head", return_value=fake_head(388)):
                with self.assertRaises(ValueError):
                    runner._fit_final_head(output, plan, "eva", features, targets, "feature")
            self.assertTrue((output / "final_head.npz").is_file())
            self.assertFalse((output / "recipe.json").exists())
            recipe, _ = runner._fit_final_head(output, plan, "eva", features, targets, "feature")
            self.assertEqual(recipe["feature_identity"], "feature")
            self.assertEqual(fit.call_count, 2)

    def test_failed_validation_gate_never_exports_test(self):
        export = unittest.mock.Mock()
        with patch.object(runner, "validation_gate", return_value=False):
            self.assertFalse(runner._gate_then_export({}, export))
        export.assert_not_called()

    def test_stage_registration_requires_terminal_locked_unpromoted_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.md"
            protocol.write_text("fixed protocol", encoding="utf-8")
            (root / "plan.json").write_text("{}", encoding="utf-8")
            stage_plan_hash = runner.sha256(root / "plan.json")
            (root / "plan_lock.json").write_text(
                json.dumps({"sha256": stage_plan_hash}), encoding="utf-8")
            report = root / "report.json"
            report.write_text("{}", encoding="utf-8")
            (root / "artifacts.json").write_text(
                json.dumps({"report.json": runner.sha256(report)}), encoding="utf-8")
            state = root / "experiment.json"
            state.write_text(json.dumps({"status": "complete", "candidate_promoted": False}),
                             encoding="utf-8")
            self.assertFalse(runner._require_preparation_authority(protocol, state)["candidate_promoted"])

            state.write_text(json.dumps({"status": "running", "candidate_promoted": False}),
                             encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "complete"):
                runner._require_preparation_authority(protocol, state)
            state.write_text(json.dumps({"status": "complete", "candidate_promoted": True}),
                             encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "promoted"):
                runner._require_preparation_authority(protocol, state)

    def test_prepare_entry_stops_before_source_or_output_while_stage_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.md"
            protocol.write_text("fixed protocol", encoding="utf-8")
            stage = root / "stage"
            stage.mkdir()
            (stage / "experiment.json").write_text(
                json.dumps({"status": "running", "candidate_promoted": False}), encoding="utf-8")
            (stage / "plan.json").write_text("{}", encoding="utf-8")
            (stage / "plan_lock.json").write_text("{}", encoding="utf-8")
            (stage / "artifacts.json").write_text("{}", encoding="utf-8")
            output = root / "should-not-be-created"
            with self.assertRaisesRegex(ValueError, "complete"):
                runner.prepare(output, root / "missing-source", plan_path=protocol,
                               stage_state_path=stage / "experiment.json")
            self.assertFalse(output.exists())

    def test_stage_registration_rejects_changed_plan_or_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.md"
            protocol.write_text("fixed protocol", encoding="utf-8")
            stage_plan = root / "plan.json"
            stage_plan.write_text("{}", encoding="utf-8")
            (root / "plan_lock.json").write_text(
                json.dumps({"sha256": runner.sha256(stage_plan)}), encoding="utf-8")
            artifact = root / "report.json"
            artifact.write_text("registered", encoding="utf-8")
            (root / "artifacts.json").write_text(
                json.dumps({"report.json": runner.sha256(artifact)}), encoding="utf-8")
            (root / "experiment.json").write_text(
                json.dumps({"status": "complete", "candidate_promoted": False}), encoding="utf-8")
            artifact.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                runner._require_preparation_authority(protocol, root / "experiment.json")

    @patch.object(runner, "environment", return_value={"synthetic": "test"})
    def test_registered_plan_hash_is_bound_into_eva_cache(self, _environment):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "run"
            output.mkdir()
            (output / "snapshot/eva_model").mkdir(parents=True)
            eva_source = output / "snapshot/eva_model/eva_x.py"
            eva_source.write_text("pinned", encoding="utf-8")
            stage = root / "stage"
            stage.mkdir()
            protocol = root / "protocol.md"
            protocol.write_text("fixed", encoding="utf-8")
            (stage / "plan.json").write_text("{}", encoding="utf-8")
            (stage / "plan_lock.json").write_text(
                json.dumps({"sha256": runner.sha256(stage / "plan.json")}), encoding="utf-8")
            report = stage / "report.json"
            report.write_text("{}", encoding="utf-8")
            (stage / "artifacts.json").write_text(
                json.dumps({"report.json": runner.sha256(report)}), encoding="utf-8")
            (stage / "experiment.json").write_text(
                json.dumps({"status": "complete", "candidate_promoted": False}), encoding="utf-8")
            plan = {
                "environment": runner.environment(), "input_locks": {}, "source_locks": {},
                "snapshot_locks": {"eva_model/eva_x.py": runner.sha256(eva_source)},
                "incumbent_source_locks": {}, "protocol": str(protocol),
                "protocol_sha256": runner.sha256(protocol),
                "stage_registration_dir": str(stage),
                "revision": "revision", "weight_sha256": "weight-sha",
                **runner.execution_settings(), "device": "cpu",
            }
            (output / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
            (output / "plan_lock.json").write_text(
                json.dumps({"sha256": runner.sha256(output / "plan.json")}), encoding="utf-8")
            registered = runner.load_registered_plan(output)
            self.assertEqual(registered["plan_sha256"], runner.sha256(output / "plan.json"))
            cache = runner._bind_eva_cache(registered, "train", image_identity(1))
            binding = json.loads((cache / "binding.json").read_text(encoding="utf-8"))
            self.assertEqual(binding["plan_sha256"], registered["plan_sha256"])
            self.assertEqual(binding["batch_size"], 16)
            self.assertEqual(binding["preprocess_workers"], 2)
            (output / "plan.json").write_text(json.dumps({**plan, "changed": True}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen EVA-X plan"):
                runner.load_registered_plan(output)

    @patch.object(runner, "environment", return_value={"synthetic": "test"})
    def test_prepare_registers_complete_tiny_snapshot_and_locked_plan(self, _environment):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            source = root / "incumbent"
            data_root = root / "dataset"
            output = root / "eva-run"
            stage = root / "stage"
            for path in (repo / "docs", source, data_root, stage):
                path.mkdir(parents=True, exist_ok=True)

            protocol = repo / "docs/EVA_STUDY_PROBE_PLAN_20261108.md"
            protocol.write_text("synthetic fixed protocol\n", encoding="utf-8")
            for relative in runner.SOURCE_FILES:
                target = repo / relative
                if target == protocol:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(f"synthetic source: {relative}\n", encoding="utf-8")

            model_dir = repo / "weights/medical/eva-x-small-chexpert"
            provenance_rows = []
            for relative in runner.EVA_MODEL_FILES:
                if relative == "provenance.json":
                    continue
                target = model_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(f"small synthetic model asset: {relative}".encode())
                digest = runner._sha256(target)
                if relative == "eva_x_small_patch16_merged520k_mim_chexpert_ft.pth":
                    digest = "1a5b63570ca4094c1118018e44d3012a95b904132942a8edb2fc0e5dc8aa0e97"
                elif relative == "eva_x.py":
                    digest = "5c21eb02cd977c35d4b8cf2984081b88b73db49953c1b4c2235f30df29df8219"
                provenance_rows.append({"file": relative, "sha256": digest})
            (model_dir / "provenance.json").write_text(json.dumps({
                "model_revision": "35ddcd6dab6ca99bbdb6cb45c8d1b093aefbd0ee",
                "source_revision": "eedd642c85339b25682ceb574b644cb9afe88e62",
                "artifacts": provenance_rows,
            }), encoding="utf-8")

            def put(path, content):
                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(content, bytes):
                    path.write_bytes(content)
                else:
                    path.write_text(content, encoding="utf-8")
                return path

            incumbent_head = put(source / "final_head.npz", b"tiny incumbent head")
            old_oof = source / "legacy_oof"
            put(old_oof / "oof_component_probabilities.npy", b"legacy matrix")
            put(old_oof / "oof_study_ids.json", "[]")
            put(old_oof / "plan.json", "{}")
            partition = put(source / "partition.json", "{}")
            old_valid = put(source / "legacy_valid.csv", "id,score\n")
            incumbent_plan = {
                "data_root": str(data_root), "input_locks": {}, "source_locks": {},
                "partition": str(partition), "old_oof": str(old_oof),
                "old_valid": str(old_valid), "device": "cpu",
            }
            put(source / "plan.json", json.dumps(incumbent_plan))
            put(source / "experiment.json", json.dumps({
                "status": "complete", "candidate_promoted": True, "test_csv_exported": True,
            }))
            put(source / "recipe.json", json.dumps({"head_sha256": runner._sha256(incumbent_head)}))
            for name in runner.INCUMBENT_ARTIFACTS:
                path = source / name
                if not path.exists():
                    put(path, "{}")
            for split in ("train", "valid", "test"):
                cache = source / f"{split}_features"
                matrix = put(cache / "image_features.npy", b"tiny source cache")
                put(cache / "input.json", "{\"images\": []}")
                put(cache / "progress.json", json.dumps({
                    "status": "complete", "sha256": runner._sha256(matrix),
                }))

            stage_plan = put(stage / "plan.json", "{}")
            stage_report = put(stage / "report.json", "{}")
            put(stage / "plan_lock.json", json.dumps({"sha256": runner._sha256(stage_plan)}))
            put(stage / "artifacts.json", json.dumps({
                "report.json": runner._sha256(stage_report),
            }))
            stage_state = put(stage / "experiment.json", json.dumps({
                "status": "complete", "candidate_promoted": False,
            }))

            real_sha = runner.sha256

            def fixture_sha(path):
                path = Path(path)
                if path.name == "eva_x_small_patch16_merged520k_mim_chexpert_ft.pth":
                    return "1a5b63570ca4094c1118018e44d3012a95b904132942a8edb2fc0e5dc8aa0e97"
                if path.name == "eva_x.py":
                    return "5c21eb02cd977c35d4b8cf2984081b88b73db49953c1b4c2235f30df29df8219"
                return real_sha(path)

            with patch.object(runner, "ROOT", repo), patch.object(
                    runner, "sha256", side_effect=fixture_sha):
                prepared = runner.prepare(output, source, plan_path=protocol,
                                          stage_state_path=stage_state)
                plan = runner.load_registered_plan(prepared)
                lock = runner.read_json(prepared / "plan_lock.json")
                self.assertEqual(lock["sha256"], fixture_sha(prepared / "plan.json"))
                self.assertEqual(plan["batch_size"], 16)
                self.assertEqual(plan["preprocess_workers"], 2)
                self.assertEqual(plan["stage_candidate_promoted"], False)
                self.assertIn("eva_model/eva_x.py", plan["snapshot_locks"])
                cache = runner._bind_eva_cache(plan, "train", image_identity(1))
                self.assertEqual(runner.read_json(cache / "binding.json")["eva_source_sha256"],
                                 plan["snapshot_locks"]["eva_model/eva_x.py"])
                for relative in plan["snapshot_locks"]:
                    snapshot_file = prepared / "snapshot" / relative
                    self.assertTrue(snapshot_file.is_file(), relative)
                    self.assertEqual(snapshot_file.stat().st_mode & stat.S_IWRITE, 0, relative)
                self.assertEqual(
                    (prepared / "source/docs/EVA_STUDY_PROBE_PLAN_20261108.md").read_text(
                        encoding="utf-8"), protocol.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
