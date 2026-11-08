"""Synthetic feature order, recovery and selected-only split execution tests."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import run_biovil_t_study_probe as runner


def identities(n):
    return [[f"d{i}", str(Path(f"synthetic-{i}.png").resolve()), i + 1, i] for i in range(n)]


def records(n, prefix="p"):
    return [{"subject_id": f"{prefix}{i}", "study_id": f"{prefix}s{i}",
             "dicom_id": f"d{i}", "labels": [i % 2, 1 - i % 2] * 5}
            for i in range(n)]


def head(width):
    return {"mean": np.zeros(width), "std": np.ones(width), "weight": np.zeros((10, width)),
            "bias": np.zeros(10), "l2": .01, "iterations": 1, "closure_calls": 1,
            "loss": .5, "seed": runner.SEED}


class BioViLExecutionTests(unittest.TestCase):
    def test_pooling_order_is_cxr_chex_biovil_and_joint_omits_chex(self):
        rows = records(3)
        rows[2] = {**rows[0], "dicom_id": "d2"}
        ids = identities(3)
        cxr = np.repeat(np.array([[2], [10], [6]], np.float32), 1054, axis=1)
        bio = np.repeat(np.array([[1], [8], [3]], np.float32), 384, axis=1)
        chex = np.repeat(np.array([[20], [80], [40]], np.float32), 512, axis=1)
        _, features, _ = runner.pool_candidates(rows, ["AP"] * 3, cxr, bio, ids, ids, chex, ids)
        self.assertEqual(features["joint"].shape, (2, 1438))
        self.assertEqual(features["triple"].shape, (2, 1950))
        np.testing.assert_array_equal(features["joint"][:, [0, 1054]], [[4, 2], [10, 8]])
        np.testing.assert_array_equal(features["triple"][:, [0, 1054, 1566]], [[4, 30, 2], [10, 80, 8]])
        _, two, _ = runner.pool_candidates(rows, ["AP"] * 3, cxr, bio, ids, ids)
        self.assertEqual(set(two), {"joint"})
        with self.assertRaises(ValueError):
            runner.pool_candidates(rows, ["AP"] * 3, cxr, bio, ids, ids[::-1])

    def test_selection_and_feature_identity_keep_candidate_sources_separate(self):
        rows = [{"candidate": name, "gate": {"passed": True, "candidate": {"macro_auc": .84}}}
                for name in ("triple", "joint")]
        self.assertEqual(runner.select_candidate(rows)["candidate"], "joint")
        base = runner._feature_identity("joint", "c", "b", "e", [])
        self.assertEqual(base, runner._feature_identity("joint", "c", "b", "different", []))
        self.assertNotEqual(runner._feature_identity("triple", "c", "b", "e", []),
                            runner._feature_identity("triple", "c", "b", "different", []))

    def test_new_cache_requires_model_binding_and_preserves_committed_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = {"output_dir": directory, "plan_sha256": "plan", "device": "cpu", "source_locks": {
                "cxr/biovil_t_transfer.py": "bio-adapter", "cxr/chexficient_transfer.py": "chex-adapter",
                "docs/biovil_t_source_manifest_20261108.json": "bio-manifest",
                "docs/chexficient_source_manifest_20261108.json": "chex-manifest"}}
            ids, rows = identities(2), records(2)
            cache = Path(directory) / "biovil_features/train"

            def extract(model, processor, source, folder, **kwargs):
                runner.atomic_json(folder / "input.json", {"images": ids})
                values = np.ones((2, 384), np.float32)
                np.save(folder / "image_features.npy", values)
                runner.atomic_json(folder / "progress.json", {"status": "complete", "completed": 2,
                    "sha256": runner.sha256(folder / "image_features.npy")})
                return values

            with patch("scripts.run_frozen_study_probe.extract_features", side_effect=extract):
                runner._extract_split(plan, rows, ids, "train", "biovil", (None, None, torch.device("cpu")))
                changed = {**plan, "plan_sha256": "changed"}
                with self.assertRaisesRegex(ValueError, "binding mismatch"):
                    runner._extract_split(changed, rows, ids, "train", "biovil", (None, None, torch.device("cpu")))
                registration = runner.read_json(cache / "registration.json")
                registration["matrix_sha256"] = "tampered"
                runner.atomic_json(cache / "registration.json", registration)
                with self.assertRaisesRegex(ValueError, "registration changed"):
                    runner._extract_split(plan, rows, ids, "train", "biovil", (None, None, torch.device("cpu")))
            with self.assertRaises(ValueError):
                runner._binding(plan, "train", "chexficient", ids)
            binding = runner._binding(plan, "test", "chexficient", ids)
            self.assertEqual((binding["feature_size"], binding["input_size"], binding["batch_size"]), (512, 378, 4))

    def test_triple_cv_head_uses_only_other_patients_and_replays_without_refit(self):
        folds = np.array([fold for fold in runner.FOLDS for _ in range(2)], dtype=object)
        x = np.zeros((8, 1950))
        y = np.tile([[0, 1] * 5, [1, 0] * 5], (4, 1))
        subjects = [f"p{i}" for i in range(8)]
        with tempfile.TemporaryDirectory() as directory, patch.object(
                runner, "fit_linear_probe", return_value=head(1950)) as fit:
            cache = runner.FoldHeadCache(directory, x, y, folds, subjects, "plan", "features")
            first, second = cache.predict("triple", 0), cache.predict("triple", 0)
            np.testing.assert_array_equal(first, second)
            np.testing.assert_array_equal(fit.call_args.args[0], x[folds != 0])
            self.assertEqual(fit.call_count, 1)
            altered = runner.FoldHeadCache(directory, x, y, folds, subjects, "other", "features")
            with self.assertRaises(ValueError):
                altered.predict("triple", 0)
            subjects[2] = subjects[0]
            leaking = runner.FoldHeadCache(directory, x, y, folds, subjects, "plan", "features")
            with self.assertRaisesRegex(ValueError, "patient leakage"):
                leaking.predict("triple", 0)

    def _run_until_gate(self, selected, promoted=False):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            incumbent = output / "incumbent"
            incumbent.mkdir()
            train_rows, valid_rows, ids = records(8), records(2, "v"), identities(8)
            test_rows = [{**row, "labels": None} for row in records(1414, "t")]
            target = np.array([row["labels"] for row in train_rows], float)
            runner.atomic_json(incumbent / "oof_study_ids.json", [[r["subject_id"], r["study_id"]] for r in train_rows])
            np.save(incumbent / "oof_targets.npy", target)
            np.save(incumbent / "oof_mean_probabilities.npy", np.full((8, 10), .4))
            (incumbent / "train_features").mkdir()
            np.save(incumbent / "train_features/image_features.npy", np.zeros((8, 1054), np.float32))
            folds = [fold for fold in runner.FOLDS for _ in range(2)]
            runner.atomic_json(output / "partition.json", {"subject_partition": {
                row["subject_id"]: fold for row, fold in zip(train_rows, folds)}})
            plan = {"output_dir": directory, "plan_sha256": "plan", "incumbent_snapshot": str(incumbent),
                    "partition": str(output / "partition.json"), "legacy_oof": "unused", "device": "cpu",
                    "current_valid": "current", "legacy_valid": "legacy", "incumbent_plan_sha256": "source",
                    "candidates": list(runner.CANDIDATES), "feature_widths": runner.FEATURE_WIDTHS,
                    "fit": {"l2": .01, "max_iter": 200, "seed": runner.SEED, "class_balance": False},
                    "folds": list(runner.FOLDS), "batch_size": 2, "preprocess_workers": 2,
                    "reused_chex": {"train_registration": {"matrix_sha256": "registered-chex-matrix"}}}
            runner.atomic_json(output / "plan.json", plan)
            runner.atomic_json(output / "experiment.json", {"status": "prepared", "test_extracted": False})
            accessed, reused, extracted = [], [], []

            def split_inputs(p, split):
                accessed.append(split)
                if split == "test" and not promoted:
                    raise AssertionError("test accessed before fixed validation gate")
                rows = train_rows if split == "train" else valid_rows if split == "valid" else test_rows
                return rows, ["AP"] * len(rows), identities(len(rows)), np.zeros((len(rows), 1054), np.float32)

            def extract(p, rows, order, split, encoder="biovil"):
                extracted.append((encoder, split))
                width = 384 if encoder == "biovil" else 512
                folder = output / f"{encoder}_features/{split}"
                folder.mkdir(parents=True)
                np.save(folder / "image_features.npy", np.ones((len(rows), width), np.float32))
                runner.atomic_json(folder / "input.json", {"images": order})
                return np.ones((len(rows), width), np.float32)

            def reuse(p, split, order):
                reused.append(split)
                if split != "train" and selected != "triple":
                    raise AssertionError("unused CheX valid/test matrix accessed")
                return np.ones((len(order), 512), np.float32)

            inputs = types.ModuleType("scripts.biovil_t_probe_inputs")
            inputs.load_registered_plan = lambda path: plan
            inputs.verify_plan = lambda path, p: None
            inputs._load_reused_chex = reuse
            gates = iter([{"passed": selected == name, "candidate": {"macro_auc": .84}}
                          for name in runner.CANDIDATES])
            original_load = __import__("scripts.compare_runs", fromlist=["load_submission"]).load_submission

            def load(path, studies):
                if str(path) in ("current", "legacy"):
                    return np.full((len(studies), 10), .4), None, None
                return original_load(path, studies)

            comparison = {"macro_auc_defined_classes": 10,
                "macro": {"auc": {"delta": .01, "ci95_percentile": [.001 if promoted else -.01, .02]}},
                "per_class": [{"metrics": {"auc": {"baseline": .7, "candidate": .71}}} for _ in range(10)],
                "bootstrap": {"auc_defined_classes_histogram": {"10": 1000}}}

            with patch.dict("sys.modules", {"scripts.biovil_t_probe_inputs": inputs}), \
                    patch.object(runner, "_split_inputs", side_effect=split_inputs), \
                    patch.object(runner, "_extract_encoder", side_effect=extract), \
                    patch.object(runner, "_validate_legacy_targets", return_value={"sources": 22}), \
                    patch.object(runner, "legacy_oof", return_value=np.full((8, 10), .4)), \
                    patch.object(runner, "candidate_gate", side_effect=lambda *a: next(gates)), \
                    patch.object(runner.FoldHeadCache, "predict", return_value=np.full((2, 10), .5)), \
                    patch.object(runner, "fit_linear_probe", side_effect=lambda x, y, **kw: head(x.shape[1])), \
                    patch.object(runner, "_materialize_deferred_cxrlt"), \
                    patch("scripts.compare_runs.load_submission", side_effect=load), \
                    patch("scripts.compare_runs.compare_arrays", return_value=comparison):
                runner.run(output)
            state = runner.read_json(output / "experiment.json")
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["candidate_promoted"], promoted)
            self.assertEqual(state["test_csv_exported"], promoted)
            self.assertEqual(accessed, ["train"] if selected is None else
                             ["train", "valid", "test"] if promoted else ["train", "valid"])
            self.assertEqual(reused, ["train", "valid"] if selected == "triple" else ["train"])
            if promoted:
                self.assertTrue(state["test_extracted"])
                expected = [("biovil", "train"), ("biovil", "valid"), ("biovil", "test")]
                if selected == "triple":
                    expected.append(("chexficient", "test"))
                self.assertEqual(extracted, expected)
                verification = runner.read_json(output / "submission_verification.json")
                self.assertEqual(verification["replay_max_error"], 0)

    def test_oof_failure_never_accesses_validation_or_test(self):
        self._run_until_gate(None)

    def test_joint_selection_never_reads_chex_validation_or_test(self):
        self._run_until_gate("joint")

    def test_triple_reads_chex_validation_only_after_selection_and_never_test_on_failure(self):
        self._run_until_gate("triple")

    def test_joint_promotion_exports_exact_1414_studies_without_chex_test(self):
        self._run_until_gate("joint", promoted=True)

    def test_triple_promotion_extracts_chex_test_sequentially_and_exports(self):
        self._run_until_gate("triple", promoted=True)


if __name__ == "__main__":
    unittest.main()
