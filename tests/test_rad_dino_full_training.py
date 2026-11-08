import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from cxr.optimization import ModelEMA
from cxr.training_state import safe_load
from scripts import train_rad_dino_grouped as trainer


class TinyRadDino(nn.Module):
    def __init__(self):
        super().__init__()
        self.dinov2 = nn.Module()
        self.dinov2.encoder = nn.Module()
        self.dinov2.encoder.layer = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
        self.dinov2.layernorm = nn.LayerNorm(2)
        self.classifier = nn.Linear(2, 10)

    def forward(self, pixel_values):
        for layer in self.dinov2.encoder.layer:
            pixel_values = layer(pixel_values).tanh()
        return self.classifier(self.dinov2.layernorm(pixel_values))


class TinyImages(torch.utils.data.Dataset):
    def __init__(self, records, processor, training=False):
        self.records = records
        self.training = training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return torch.tensor([index / 4, 1.0]), torch.tensor(
            self.records[index]["labels"], dtype=torch.float32)


class FullTrainingTests(unittest.TestCase):
    def test_full_refit_requires_exact_locked_patient_coverage_and_no_heldout(self):
        records = [{"subject_id": "development"}, {"subject_id": "audit"}]
        partition = {"subject_partition": {"development": 0, "audit": "audit"}}
        train, held = trainer.select_records(records, partition, "full", None)
        self.assertEqual(train, records)
        self.assertEqual(held, [])
        with self.assertRaisesRegex(ValueError, "all locked"):
            trainer.select_records(records[:1], partition, "full", None)
        with self.assertRaisesRegex(ValueError, "phase"):
            trainer.select_records(records, partition, "full", 0)

    def test_resume_is_confined_to_full_refits_with_single_process_loading(self):
        base = ["--data-root", "data", "--model-dir", "model", "--partition", "partition",
                "--output", "output", "--head-lr", "0.001"]
        cfg = trainer.parser().parse_args([*base, "--phase", "audit", "--resume"])
        with self.assertRaisesRegex(ValueError, "only for a full"):
            trainer.validate_config(cfg)
        cfg = trainer.parser().parse_args([*base, "--phase", "full"])
        with self.assertRaisesRegex(ValueError, "workers=0"):
            trainer.validate_config(cfg)
        cfg.workers = 0
        trainer.validate_config(cfg)

    def test_epoch_resume_preserves_optimizer_ema_scheduler_and_random_streams(self):
        config = {"epochs": 3, "workers": 0, "seed": 17}

        def setup():
            random.seed(17)
            np.random.seed(17)
            torch.manual_seed(17)
            model = nn.Linear(3, 2)
            ema = ModelEMA(model, decay=0.9)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            generator = torch.Generator().manual_seed(17)
            return model, ema, optimizer, scheduler, scaler, generator

        def epoch(objects):
            model, ema, optimizer, scheduler, _, generator = objects
            order = torch.randperm(5, generator=generator)
            for index in order:
                inputs = torch.randn(1, 3) + random.random() + float(np.random.rand())
                optimizer.zero_grad(set_to_none=True)
                (model(inputs) - index.float()).square().mean().backward()
                optimizer.step()
                ema.update(model)
            scheduler.step()

        uninterrupted = setup()
        epoch(uninterrupted)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pt"
            trainer.save_full_refit(path, config, *uninterrupted,
                                    [{"epoch": 1, "loss": 1.0}], 5, 0, 10.0)
            epoch(uninterrupted)
            expected_model = {k: v.detach().clone() for k, v in uninterrupted[0].state_dict().items()}
            expected_ema = {k: v.detach().clone() for k, v in uninterrupted[1].module.state_dict().items()}
            resumed = setup()
            completed, history, updates, skipped, elapsed = trainer.restore_full_refit(
                path, config, *resumed)
            self.assertEqual((completed, len(history), updates, skipped, elapsed), (1, 1, 5, 0, 10.0))
            epoch(resumed)
            for key, expected in expected_model.items():
                self.assertTrue(torch.equal(expected, resumed[0].state_dict()[key]))
            for key, expected in expected_ema.items():
                self.assertTrue(torch.equal(expected, resumed[1].module.state_dict()[key]))
            self.assertEqual(uninterrupted[2].param_groups[0]["lr"], resumed[2].param_groups[0]["lr"])
            with self.assertRaisesRegex(ValueError, "configuration"):
                trainer.restore_full_refit(path, {**config, "epochs": 4}, *setup())

    def test_full_training_saves_reloadable_weights_without_evaluating_or_reading_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, model_dir = root / "data", root / "public"
            data.mkdir()
            model_dir.mkdir()
            (data / "train.csv").write_text("locked train metadata")
            (model_dir / "pytorch_model.bin").write_bytes(b"public initialization")
            records, inventory = [], {}
            for index in range(4):
                path = data / f"image{index}.jpg"
                path.write_bytes(b"locked input")
                stat = path.stat()
                records.append({"subject_id": f"p{index}", "study_id": f"s{index}",
                                "dicom_id": f"d{index}", "path": str(path),
                                "labels": [index % 2] * 10})
                inventory[f"d{index}"] = {"path": str(path), "bytes": stat.st_size,
                                          "mtime_ns": stat.st_mtime_ns}
            inventory_path = root / "inventory.json"
            inventory_path.write_text(json.dumps(inventory))
            partition = {"subject_partition": {f"p{i}": "audit" if i == 0 else 0 for i in range(4)},
                         "train_metadata_sha256": trainer.sha256(data / "train.csv"),
                         "public_checkpoint_sha256": trainer.sha256(model_dir / "pytorch_model.bin"),
                         "image_inventory_path": str(inventory_path),
                         "image_inventory_sha256": trainer.sha256(inventory_path)}
            partition_path = root / "partition.json"
            partition_path.write_text(json.dumps(partition))
            output = root / "run"
            argv = ["--data-root", str(data), "--model-dir", str(model_dir),
                    "--partition", str(partition_path), "--output", str(output),
                    "--phase", "full", "--head-lr", "0.001", "--epochs", "2",
                    "--workers", "0", "--device", "cpu"]
            with patch.object(trainer, "read_records", return_value=records) as read, \
                    patch.object(trainer, "initialize_model", side_effect=lambda *a: (None, TinyRadDino())), \
                    patch.object(trainer, "ImageRecords", TinyImages), \
                    patch.object(trainer, "save_prediction", side_effect=AssertionError("evaluation forbidden")):
                trainer.main(argv)
                self.assertEqual(read.call_args.args[0], data / "train.csv")
                self.assertEqual(read.call_count, 1)
                checkpoint = safe_load(output / "ema/last.pt")
                self.assertIsNone(checkpoint["metrics"])
                self.assertEqual(checkpoint["epoch"], 2)
                self.assertFalse(checkpoint["provenance"]["eligible_for_oof"])
                report = json.loads((output / "report.json").read_text())
                self.assertEqual((report["status"], report["evaluation_images"]), ("complete", 0))
                initial = {key: value.clone() for key, value in checkpoint["model"].items()}
                trainer.main([*argv, "--resume"])
                resumed = safe_load(output / "ema/last.pt")
                for key, value in initial.items():
                    self.assertTrue(torch.equal(value, resumed["model"][key]))
                restored_report = json.loads((output / "report.json").read_text())
                self.assertEqual(len(restored_report["history"]), 2)


if __name__ == "__main__":
    unittest.main()
