"""Synthetic tests for BioViL-T input locks and deferred CheXficient cache access."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import biovil_t_probe_inputs as inputs


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value), encoding="utf-8")
    return path


def identity_rows(count=2):
    return [[f"dicom-{i}", str(Path(f"synthetic-{i}.png").resolve()), i + 1, i]
            for i in range(count)]


class BioViLInputTests(unittest.TestCase):
    def test_fixed_plan_contract_and_candidate_widths(self):
        self.assertEqual(inputs.BIOVIL_REVISION, "301f526e823b805d3fe712d0cf06a66f042789c5")
        self.assertEqual(inputs.CHEX_TRAIN_SHA256,
                         "373aef3b1c79c8815b12ebda51e358973ebd0e8628596101ee1dd1755738ded2")

    def test_reused_train_cache_requires_all_registered_hashes_and_exact_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "train"
            identities = identity_rows()
            matrix_path = cache / "image_features.npy"
            matrix_path.parent.mkdir(parents=True)
            np.save(matrix_path, np.ones((2, 512), dtype=np.float32), allow_pickle=False)
            binding_path = put(cache / "binding.json", {
                "plan_sha256": "old-chex-plan", "model_revision": inputs.CHEX_REVISION,
                "weight_sha256": inputs.CHEX_WEIGHT_SHA256,
                "adapter_sha256": inputs.CHEX_ADAPTER_SHA256,
            })
            input_path = put(cache / "input.json", {"images": identities})
            progress_path = put(cache / "progress.json", {
                "status": "complete", "sha256": inputs.sha256(matrix_path)})
            registration = {
                "plan_sha256": "old-chex-plan",
                "binding_sha256": inputs.sha256(binding_path),
                "input_sha256": inputs.sha256(input_path),
                "progress_sha256": inputs.sha256(progress_path),
                "matrix_sha256": inputs.CHEX_TRAIN_SHA256,
            }
            # Use synthetic bytes while retaining the cache contract under test.
            registration["matrix_sha256"] = inputs.sha256(matrix_path)
            plan = {"reused_chex": {
                "train_cache_dir": str(cache), "train_registration": registration,
                "valid_cache_dir": str(cache), "valid_registration": registration,
                "valid_matrix_sha256": "deferred-valid-sha",
            }}
            with patch.object(inputs, "CHEX_TRAIN_SHA256", inputs.sha256(matrix_path)):
                values = inputs._load_reused_chex(plan, "train", identities)
            self.assertEqual(values.shape, (2, 512))
            self.assertFalse(values.flags.writeable)
            del values
            with self.assertRaisesRegex(ValueError, "identity/order mismatch"):
                with patch.object(inputs, "CHEX_TRAIN_SHA256", inputs.sha256(matrix_path)):
                    inputs._load_reused_chex(plan, "train", identities[::-1])

    def test_valid_and_test_cache_are_unread_until_triple_selection(self):
        plan = {"reused_chex": {"valid_cache_dir": "unopened", "valid_registration": {},
                                "valid_matrix_sha256": "expected"}}
        with patch.object(inputs, "sha256", side_effect=AssertionError("cache was touched")):
            with self.assertRaisesRegex(ValueError, "deferred until triple"):
                inputs._load_reused_chex(plan, "valid", identity_rows())
            with self.assertRaisesRegex(ValueError, "train/valid"):
                inputs._load_reused_chex(plan, "test", identity_rows())

    def test_registered_frozen_plan_loads_and_rejects_changed_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            protocol = put(output / "protocol.md", "fixed synthetic protocol")
            plan = {
                "environment": {"synthetic": "env"},
                "input_locks": {}, "source_locks": {}, "snapshot_locks": {},
                "protocol": str(protocol), "protocol_sha256": inputs.sha256(protocol),
                "chex_registration": {"root": str(output / "chex"), "audit_paths": []},
            }
            put(output / "plan.json", plan)
            lock = {"sha256": inputs.sha256(output / "plan.json")}
            put(output / "plan_lock.json", lock)
            registered = {"root": str(output / "chex"), "audit_paths": []}
            with patch.object(inputs, "_environment", return_value={"synthetic": "env"}), \
                    patch.object(inputs, "_validate_chex_registration", return_value=registered):
                loaded = inputs.load_registered_plan(output)
                self.assertEqual(loaded["plan_sha256"], lock["sha256"])
                self.assertEqual(loaded["output_dir"], str(output.resolve()))
                put(output / "plan.json", {**plan, "batch_size": 99})
                with self.assertRaisesRegex(ValueError, "plan lock mismatch"):
                    inputs.load_registered_plan(output)

    def test_synthetic_prepare_snapshots_and_reloads_the_frozen_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source, output, raw = base / "incumbent", base / "prepared", base / "raw"
            source.mkdir(); raw.mkdir()
            (base / "docs").mkdir()
            protocol = put(base / "protocol.md", "synthetic fixed protocol")
            manifest_file = put(base / "docs/biovil_t_source_manifest_20261108.json", "manifest bytes")
            bio = base / "bio"
            put(bio / "weight.bin", b"synthetic bio weights")
            put(bio / "provenance.json", {"synthetic": True})
            bio_manifest = {"files": {"weight.bin": {"sha256": "synthetic", "bytes": 22}}}
            reports = [put(base / f"engineering-{i}.json", {"status": "passed"}) for i in range(3)]
            reports_data = [
                {"status": "passed", "model_revision": inputs.BIOVIL_REVISION,
                 "checkpoint_sha256": inputs.BIOVIL_WEIGHT_SHA256, "feature_size": 384,
                 "processor_max_abs_error": 0.0},
                {"status": "passed", "model_revision": inputs.BIOVIL_REVISION,
                 "weights_sha256": inputs.BIOVIL_WEIGHT_SHA256,
                 "adapter_sha256": inputs.BIOVIL_ADAPTER_SHA256, "batch_size": 2,
                 "amp": "cuda float16"},
                {"status": "passed", "batch_size": 2, "preprocess_workers": 2,
                 "resume_vs_fresh_max_abs_error": 0.0},
            ]
            for path, value in zip(reports, reports_data):
                put(path, value)
            audits = [put(base / "audit-oof.json", {}), put(base / "audit-final.json", {})]

            head_file = put(source / "final_head.npz", b"synthetic head")
            incumbent_plan = {
                "data_root": str(raw), "input_locks": {}, "old_test": None,
                "device": "cuda", "partition": str(put(base / "partition.json", {})),
                "old_oof": str(base / "old-oof"), "old_valid": str(put(base / "legacy-valid.csv", "id,p")),
            }
            old_oof = Path(incumbent_plan["old_oof"])
            put(old_oof / "oof_component_probabilities.npy", b"oof")
            put(old_oof / "oof_study_ids.json", [])
            put(old_oof / "plan.json", {})
            put(source / "plan.json", incumbent_plan)
            put(source / "experiment.json", {"status": "complete", "candidate_promoted": True})
            put(source / "recipe.json", {"head_sha256": inputs.sha256(head_file)})
            for name in ("report.json", "validation_report.json", "submission_verification.json",
                         "train_records.json", "oof_mean_probabilities.npy", "oof_targets.npy",
                         "oof_study_ids.json", "candidate_valid.csv"):
                put(source / name, b"placeholder")
            train = source / "train_features"
            train_matrix = put(train / "image_features.npy", b"CXR train")
            put(train / "input.json", {"images": []})
            put(train / "progress.json", {"status": "complete", "sha256": inputs.sha256(train_matrix)})
            for split in ("valid", "test"):
                folder = source / f"{split}_features"
                put(folder / "input.json", {"images": []})
                put(folder / "progress.json", {"status": "complete", "sha256": f"deferred-{split}"})

            chex_root = base / "chex-run"
            chex_model = chex_root / "snapshot/chexficient_model"
            put(chex_model / "model.bin", b"chex model")
            chex_plan = {"model_dir": str(chex_model), "source_locks": {},
                         "incumbent_source_locks": {},
                         "legacy_target_sources": [{"ids": "legacy_targets/00/study_ids.json",
                                                    "targets": "legacy_targets/00/study_targets.npy"}]}
            put(chex_root / "plan.json", chex_plan)
            for name, value in (("plan_lock.json", {"sha256": "lock"}),
                                ("experiment.json", {"status": "complete"}),
                                ("artifacts.json", {})):
                put(chex_root / name, value)
            old_target_root = chex_root / "snapshot/legacy_targets/00"
            put(old_target_root / "study_ids.json", [])
            put(old_target_root / "study_targets.npy", b"targets")
            chex_train = chex_root / "chexficient_features/train"
            for name, value in (("binding.json", {}), ("registration.json", {}),
                                ("input.json", {"images": []}), ("progress.json", {}),
                                ("image_features.npy", b"registered train")):
                put(chex_train / name, value)
            put(chex_root / "chexficient_features/valid/registration.json", {"matrix_sha256": "deferred"})

            registration = {"root": str(chex_root.resolve()), "train_registration": {},
                            "deferred_valid_matrix_sha256": "deferred", "audit_paths": [str(x) for x in audits],
                            "audit_sha256": {x.name: inputs.sha256(x) for x in audits}}
            with patch.multiple(inputs, ROOT=base, PLAN_PATH=protocol, BIOVIL_DIR=bio,
                                CHEX_RUN=chex_root, ENGINEERING=tuple(reports), CHEX_AUDITS=tuple(audits),
                                INCUMBENT_FILES=("plan.json", "experiment.json", "recipe.json", "final_head.npz"),
                                SOURCE_FILES=()), \
                    patch.object(inputs, "_validate_chex_registration", return_value=registration), \
                    patch.object(inputs, "_validate_biovil_assets", return_value=(bio_manifest, {})):
                prepared = inputs.prepare(output, source)
                self.assertEqual(prepared, output.resolve())
                loaded = inputs.load_registered_plan(prepared)
                self.assertEqual(loaded["candidates"], ["joint", "triple"])
                self.assertEqual(loaded["feature_widths"], {"joint": 1438, "triple": 1950})
                self.assertTrue((prepared / "snapshot/chexficient_model/model.bin").is_file())
                self.assertTrue((prepared / "snapshot/legacy_targets/00/study_ids.json").is_file())
                self.assertEqual(loaded["deferred_cxrlt"]["test"]["matrix_sha256"], "deferred-test")
                self.assertFalse((prepared / "snapshot/incumbent/test_features/image_features.npy").exists())


if __name__ == "__main__":
    unittest.main()
