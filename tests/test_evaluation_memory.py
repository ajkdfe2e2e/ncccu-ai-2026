from __future__ import annotations

import json
import pickle
from pathlib import Path
import random
import tempfile
import types
import unittest

import numpy as np
import torch

from cxr import evaluation_memory
from scripts import run_grouped_memory_guard as guard


ROOT = Path(__file__).resolve().parents[1]


class FakeIterator:
    def __init__(self):
        self.shutdowns = 0

    def _shutdown_workers(self):
        self.shutdowns += 1


class FakeLoader:
    def __init__(self):
        self._iterator = FakeIterator()


class FakeModel:
    def __init__(self, name):
        self.name = name

    def state_dict(self):
        return {"name": self.name}


class FakeEMA:
    def __init__(self, model, decay):
        self.module = FakeModel("ema")
        self.model = model
        self.decay = decay


class MemoryGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.guard_path = ROOT / "scripts/run_grouped_memory_guard.py"
        self.helper_path = ROOT / "cxr/evaluation_memory.py"
        self.output = self.root / "experiment" / "runs" / "cv_A_fold1"
        self.output.mkdir(parents=True)
        amendment_root = self.output.parents[1]
        (amendment_root / "experiment.json").write_text(
            json.dumps({"audit_opened": False}), encoding="utf-8",
        )
        self.amendment_path = amendment_root / "runtime_amendment.json"
        self.write_amendment()

    def tearDown(self):
        self.temp.cleanup()

    def write_amendment(self, **extra):
        payload = {
            "schema_version": 1, "policy": evaluation_memory.POLICY,
            "guard_sha256": evaluation_memory.sha256(self.guard_path),
            "helper_sha256": evaluation_memory.sha256(self.helper_path),
            "audit_opened_at_registration": False,
            **extra,
        }
        self.amendment_path.write_text(json.dumps(payload), encoding="utf-8")

    def make_trainer(self, saved_calls):
        trainer = types.SimpleNamespace()

        class Dataset:
            def __init__(self, training):
                self.training = training

        loader_calls = []

        def data_loader(dataset, **kwargs):
            loader_calls.append((dataset.training, dict(kwargs)))
            return FakeLoader()

        def initialize_model(*args, **kwargs):
            return "processor", FakeModel("raw")

        def save(*args, **kwargs):
            with Path(args[1]).open("wb") as stream:
                pickle.dump(args[0], stream)

        def predict(model, records, processor, device, batch_size, workers):
            saved_calls.append(("predict", workers, list(records)))
            return list(records)

        def save_prediction(output, variant, model, records, processor, device, cfg, provenance):
            trainer.predict_records(model, records, processor, device,
                                    cfg.batch_size, cfg.workers)
            raise MemoryError("simulated evaluation failure")

        trainer.DataLoader = data_loader
        trainer.ModelEMA = FakeEMA
        trainer.initialize_model = initialize_model
        trainer.predict_records = predict
        trainer.save_prediction = save_prediction
        trainer.atomic_json = lambda path, data: Path(path).write_text(json.dumps(data), encoding="utf-8")
        trainer.torch = types.SimpleNamespace(save=save)
        trainer.DataSet = Dataset
        trainer.loader_calls = loader_calls
        return trainer

    def test_training_loader_is_unchanged_eval_uses_zero_and_snapshots_precede_failure(self):
        calls = []
        trainer = self.make_trainer(calls)
        cfg = types.SimpleNamespace(batch_size=2, workers=4, model_dir=Path("model"), epochs=3)
        python_rng_before = random.getstate()
        numpy_rng_before = np.random.get_state()
        torch_rng_before = torch.random.get_rng_state().clone()
        evaluation_memory.install(
            trainer, cfg, self.output, {}, self.amendment_path,
            self.guard_path, self.helper_path,
        )
        original_parameters = {
            "batch_size": 2, "shuffle": True, "num_workers": 4,
            "pin_memory": True, "persistent_workers": True,
        }
        loader = trainer.DataLoader(trainer.DataSet(True), **original_parameters)
        self.assertEqual(trainer.loader_calls, [(True, original_parameters)])
        self.assertEqual(loader._iterator.shutdowns, 0)
        trainer.initialize_model()
        ema = trainer.ModelEMA(FakeModel("raw"), decay=0.999)
        self.assertEqual(ema.decay, 0.999)
        self.assertEqual(random.getstate(), python_rng_before)
        numpy_rng_after = np.random.get_state()
        self.assertEqual(numpy_rng_after[0], numpy_rng_before[0])
        self.assertTrue(np.array_equal(numpy_rng_after[1], numpy_rng_before[1]))
        self.assertEqual(numpy_rng_after[2:], numpy_rng_before[2:])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_rng_before))
        report = {"status": "evaluating_fixed_final_weights", "history": [{"epoch": 3}]}
        (self.output / "report.json").write_text(json.dumps(report), encoding="utf-8")

        iterator = loader._iterator
        with self.assertRaisesRegex(MemoryError, "simulated"):
            trainer.save_prediction(
                self.output, "raw", FakeModel("raw"), ["first", "second"],
                "processor", types.SimpleNamespace(type="cpu"), cfg,
                {"phase": "cv", "fold": 1},
            )
        recovery = self.output / "runtime_final"
        manifest = json.loads((recovery / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(set(manifest["checkpoint_sha256"]), {"raw.pt", "ema.pt"})
        for filename, expected in manifest["checkpoint_sha256"].items():
            self.assertEqual(evaluation_memory.sha256(recovery / filename), expected)
            with (recovery / filename).open("rb") as stream:
                checkpoint = pickle.load(stream)
            self.assertEqual(checkpoint["kind"], "reliability_snapshot_not_selected_epoch")
            self.assertEqual(checkpoint["provenance"], {"phase": "cv", "fold": 1})
            self.assertEqual(checkpoint["training_report"]["history"], [{"epoch": 3}])
            self.assertEqual(checkpoint["runtime_amendment_sha256"],
                             evaluation_memory.sha256(self.amendment_path))
        self.assertEqual(iterator.shutdowns, 1)
        self.assertIsNone(loader._iterator)
        self.assertEqual(calls, [("predict", 0, ["first", "second"])])

    def test_amendment_hash_mismatch_fails_closed(self):
        self.write_amendment(guard_sha256="0" * 64)
        with self.assertRaisesRegex(RuntimeError, "guard_sha256"):
            evaluation_memory.verify_amendment(
                self.amendment_path, self.guard_path, self.helper_path,
            )

    def test_only_snapshotted_trainer_command_is_rewritten(self):
        command = ["python", "-u", "E:/run/source/scripts/train_rad_dino_grouped.py",
                   "--workers", "4", "--output", "run"]
        rewritten = guard.rewrite_trainer_command(command, self.guard_path)
        self.assertEqual(rewritten[:5], ["python", "-u", str(self.guard_path), "--trainer",
                                         "E:/run/source/scripts/train_rad_dino_grouped.py"])
        self.assertEqual(rewritten[5:], command[3:])
        infer = ["python", "-u", "E:/run/source/scripts/infer_public.py", "--branch", "x"]
        self.assertEqual(guard.rewrite_trainer_command(infer, self.guard_path), infer)

    def test_source_and_fusion_delegate_with_original_arguments(self):
        for fusion in (False, True):
            with self.subTest(fusion=fusion):
                output = self.root / ("fusion" if fusion else "source")
                snapshot = output / "source/scripts"
                snapshot.mkdir(parents=True)
                source_output = self.root / "source-registration" if fusion else output
                source_output.mkdir(parents=True, exist_ok=True)
                source_plan_path = source_output / "plan.json"
                if fusion:
                    source_plan_path.write_text(json.dumps({"registered": "source"}), encoding="utf-8")
                source_partition = source_output / "partition.json"
                source_partition.write_text("{}", encoding="utf-8")
                source_inventory = source_output / "image_inventory.json"
                source_inventory.write_text("{}", encoding="utf-8")
                runner_name = "run_oof_fusion.py" if fusion else "run_grouped_generalization.py"
                runner_path = snapshot / runner_name
                log_path = self.root / f"delegation-{fusion}.json"
                trainer_path = str(snapshot / "train_rad_dino_grouped.py")
                if fusion:
                    plan = {"source_experiment": str(source_output)}
                    runner_code = (
                        "import json\n"
                        "ARGS = None\n"
                        "def atomic_json(path, data):\n"
                        "    with open(path, 'w') as stream: stream.write(json.dumps(data))\n"
                        "def run_stage(output, state, name, command, folder, artifacts):\n"
                        f"    with open({str(log_path)!r}, 'w') as stream: stream.write(json.dumps([ARGS, command]))\n"
                        "def main(argv=None):\n"
                        "    global ARGS\n"
                        "    ARGS = argv\n"
                        f"    atomic_json({str(output / 'report.json')!r}, {{'status':'complete'}})\n"
                        f"    run_stage(None, None, 'x', ['python','-u',{trainer_path!r},'--x'], None, [])\n"
                    )
                else:
                    plan = {"source_experiment": None}
                    runner_code = (
                        "import json\n"
                        "def atomic_json(path, data):\n"
                        "    with open(path, 'w') as stream: stream.write(json.dumps(data))\n"
                        "def stage_command(*args, **kwargs):\n"
                        f"    return 'x', ['python','-u',{trainer_path!r},'--x']\n"
                        "def main(argv=None):\n"
                        "    name, command = stage_command()\n"
                        f"    atomic_json({str(output / 'report.json')!r}, {{'status':'complete'}})\n"
                        f"    with open({str(log_path)!r}, 'w') as stream: stream.write(json.dumps([argv, command]))\n"
                    )
                    plan = {}
                runner_path.write_text(runner_code, encoding="utf-8")
                (output / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
                (output / "experiment.json").write_text(
                    json.dumps({"audit_opened": False}), encoding="utf-8",
                )
                amendment = {
                    "schema_version": 1, "policy": evaluation_memory.POLICY,
                    "guard_sha256": evaluation_memory.sha256(self.guard_path),
                    "helper_sha256": evaluation_memory.sha256(self.helper_path),
                    "audit_opened_at_registration": False,
                    "experiment_output": str(output.resolve()),
                    "experiment_plan_sha256": evaluation_memory.sha256(output / "plan.json"),
                    "source_plan_sha256": evaluation_memory.sha256(source_plan_path),
                    "fusion_plan_sha256": (evaluation_memory.sha256(output / "plan.json")
                                           if fusion else "registered-fusion-plan"),
                    "partition_sha256": evaluation_memory.sha256(source_partition),
                    "inventory_sha256": evaluation_memory.sha256(source_inventory),
                }
                (output / evaluation_memory.AMENDMENT_NAME).write_text(
                    json.dumps(amendment), encoding="utf-8",
                )
                args = ["--resume", "--output", str(output)]
                guard.runner_entry(output, args)
                delegated = json.loads(log_path.read_text(encoding="utf-8"))
                self.assertEqual(delegated[0], args)
                rewritten = delegated[1] if fusion else delegated[1]
                self.assertEqual(rewritten[2], str(self.guard_path))
                self.assertEqual(rewritten[3:5], ["--trainer", trainer_path])
                self.assertEqual(rewritten[5:], ["--x"])
                root_report = json.loads((output / "report.json").read_text(encoding="utf-8"))
                self.assertEqual(root_report["runtime_amendment"]["path"],
                                 str((output / evaluation_memory.AMENDMENT_NAME).resolve()))


if __name__ == "__main__":
    unittest.main()
