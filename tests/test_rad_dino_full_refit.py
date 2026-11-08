import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import run_rad_dino_full_refit as runner


class FullRefitRunnerTests(unittest.TestCase):
    def test_training_and_eval_commands_lock_expected_arguments(self):
        binding = {
            "plan": {}, "data_root": r"G:\cxr_first(1)\cxr_first",
            "model_dir": r"G:\models\rad-dino",
        }
        output = Path("run-output")
        source = output / "source"
        command = runner.training_command(Path("python.exe"), source, binding, output)
        self.assertEqual(command[command.index("--phase") + 1], "full")
        self.assertEqual(command[command.index("--workers") + 1], "0")
        self.assertEqual(command[command.index("--head-lr") + 1], "0.001")
        self.assertEqual(command[command.index("--epochs") + 1], "3")
        self.assertIn("ema", command)
        self.assertNotIn("--resume", command)
        resumed = runner.training_command(Path("python.exe"), source, binding, output, resume=True)
        self.assertEqual(resumed[-1], "--resume")
        evaluation = runner.evaluation_command(Path("python.exe"), source, binding, output)
        self.assertEqual(evaluation[evaluation.index("--split") + 1], "test")
        self.assertEqual(evaluation[evaluation.index("--batch-size") + 1], "2")
        self.assertEqual(evaluation[evaluation.index("--device") + 1], "cuda")
        self.assertTrue(evaluation[evaluation.index("--checkpoint") + 1].endswith("train\\ema\\last.pt"))

    def test_training_verification_requires_complete_metric_free_both_variants(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            files = [output / "train/ema/last.pt", output / "train/raw/last.pt"]
            for path in files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(path.parent.name.encode("ascii"))
            report = {
                "status": "complete", "phase": "full", "train_images": 30344,
                "train_patients": 11055, "independent_validation_metrics": None,
                "eligible_for_oof": False, "includes_former_audit_patients": True,
                "epochs": 3, "head_lr": 0.001, "batch_size": 2,
                "accumulation_steps": 8, "seed": 2026, "ema_decay": 0.999,
                "evaluation_images": 0,
                "history": [{"epoch": 1}, {"epoch": 2}, {"epoch": 3}],
                "variants": {},
            }
            for variant, path in zip(("ema", "raw"), files):
                report["variants"][variant] = {
                    "metrics": None, "checkpoint_sha256": runner.sha256(path),
                }
            (output / "train/report.json").write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(set(runner._verify_training(output)["checkpoint_hashes"]),
                             {"train/ema/last.pt", "train/raw/last.pt"})
            report["variants"]["ema"]["metrics"] = {"auc": 0.5}
            (output / "train/report.json").write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpectedly has metrics"):
                runner._verify_training(output)

    def test_nonzero_stage_exit_is_recorded_and_raises(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "source").mkdir()
            state = {"stages": {}}
            process = mock.Mock(pid=456, wait=mock.Mock(return_value=17))
            with mock.patch.object(runner.subprocess, "Popen", return_value=process):
                with self.assertRaisesRegex(RuntimeError, "exited with code 17"):
                    runner._record_stage(output, state, "training", ["python", "train.py"])
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["stages"]["training"]["pid"], 456)
            self.assertEqual(state["stages"]["training"]["returncode"], 17)
            self.assertTrue((output / "logs/training.log").is_file())

    def test_failed_inference_output_is_renamed_without_data_loss(self):
        with tempfile.TemporaryDirectory() as temporary:
            test_output = Path(temporary) / "test"
            test_output.mkdir()
            artifact = test_output / "partial.bin"
            artifact.write_bytes(b"preserve me")
            quarantined = Path(runner._quarantine_test_output(test_output))
            self.assertFalse(test_output.exists())
            self.assertEqual((quarantined / "partial.bin").read_bytes(), b"preserve me")
            self.assertRegex(quarantined.name, r"^test\.failed-\d{8}-\d{6}-\d+$")

    def test_submission_exports_source_order_exact_float_roundtrip_and_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, data_root = root / "run", root / "data"
            test_dir = output / "test"
            test_dir.mkdir(parents=True)
            data_root.mkdir()
            csv_path = root / "submission.csv"
            records = []
            studies = []
            with (data_root / "test.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("Dicom_id", "Subject_id", "Study_id"))
                for study_index in range(1414):
                    subject, study = f"subject_{study_index:04d}", f"study_{study_index:04d}"
                    studies.append([subject, study])
                    image_count = 2 if study_index < 804 else 1
                    for image_index in range(image_count):
                        dicom = f"image_{len(records):04d}"
                        records.append((dicom, subject, study))
                        writer.writerow((dicom, subject, study))
            self.assertEqual(len(records), 2218)
            probabilities = np.linspace(0.0003, 0.9997, 14140, dtype=np.float64).reshape(1414, 10)
            (test_dir / "report.json").write_text(json.dumps({
                "split": "test", "test_images": 2218, "test_studies": 1414,
            }), encoding="utf-8")
            (test_dir / "study_ids.json").write_text(json.dumps(studies), encoding="utf-8")
            np.save(test_dir / "study_probabilities.npy", probabilities)
            result = runner.export_submission(output, csv_path, data_root)
            self.assertEqual(result["validation"]["rows"], 14140)
            self.assertEqual(result["validation"]["default_threshold_rows"], 14140)
            with csv_path.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 14140)
            self.assertEqual((rows[0]["Subject_id"], rows[0]["Study_id"], rows[0]["Gate"]),
                             (studies[0][0], studies[0][1], "0.5"))
            self.assertEqual((rows[-1]["Subject_id"], rows[-1]["Study_id"], rows[-1]["Gate"]),
                             (studies[-1][0], studies[-1][1], "0.5"))


if __name__ == "__main__":
    unittest.main()
