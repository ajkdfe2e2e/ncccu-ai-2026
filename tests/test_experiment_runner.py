"""Lightweight orchestration tests; subprocesses and data access stay mocked."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import run_aggressive_experiment as runner


class ExperimentRunnerTests(unittest.TestCase):
    def test_stage_artifacts_share_training_directory_and_snapshot_scripts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "experiment"
            source = output / "source"
            args = SimpleNamespace(
                data_root=Path(directory) / "data", baseline=Path(directory) / "baseline.csv",
                batch_size=4, accumulation_steps=4, gradient_checkpointing=True,
                image_cache=Path(directory) / "cache", ema_device="cpu",
            )
            stages = runner.build_stages(args, output, source)

        self.assertEqual([name for name, _ in stages], [
            "train", "predict_valid", "compare_runs", "inference_manifest", "benchmark_valid",
        ])
        training = output / "training"
        commands = [command for _, command in stages]
        self.assertEqual(Path(commands[0][1]), source / "train.py")
        self.assertEqual(commands[0][commands[0].index("--output") + 1], str(training))
        self.assertIn("--gradient-checkpointing", commands[0])
        self.assertEqual(commands[0][commands[0].index("--image-cache") + 1], str(args.image_cache))
        self.assertEqual(Path(commands[1][1]), source / "predict.py")
        self.assertEqual(commands[1][commands[1].index("--output") + 1], str(training / "valid.csv"))
        self.assertEqual(Path(commands[2][1]), source / "scripts" / "compare_runs.py")
        self.assertEqual(commands[2][commands[2].index("--candidate") + 1], str(training / "valid.csv"))
        self.assertEqual(commands[2][commands[2].index("--output") + 1], str(training / "comparison.json"))
        self.assertEqual(commands[2][commands[2].index("--bootstrap") + 1], "500")
        self.assertEqual(Path(commands[3][1]), source / "scripts" / "inference_manifest.py")
        self.assertEqual(commands[3][2], str(training / "best.pt"))
        self.assertEqual(commands[3][commands[3].index("--output") + 1],
                         str(training / "inference_manifest.json"))
        self.assertTrue(all(str(source) in item for item in commands[3][3:commands[3].index("--output")]))
        self.assertEqual(Path(commands[4][1]), source / "scripts" / "benchmark_valid.py")
        self.assertEqual(commands[4][commands[4].index("--run") + 1], str(training))

    def _inputs(self, directory):
        root = Path(directory)
        data = root / "data"
        data.mkdir()
        (data / "train.csv").write_text("placeholder\n", encoding="utf-8")
        (data / "valid.csv").write_text("placeholder\n", encoding="utf-8")
        baseline = root / "baseline.csv"
        baseline.write_text("read-only\n", encoding="utf-8")
        weights = root / "weights.pth"
        weights.write_bytes(b"mock")
        return data, baseline, weights

    def test_failed_child_persists_status_return_code_and_logs(self):
        class FailedProcess:
            pid = 4312

            def wait(self):
                return 7

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, baseline, weights = self._inputs(root)
            output = root / "experiment"

            def fake_snapshot(destination):
                Path(destination).mkdir(parents=True)
                return {}

            def fail_first_stage(command, *, cwd, stdout, stderr):
                self.assertEqual(Path(cwd), output.resolve() / "source")
                stdout.write("child stdout\n")
                stderr.write("child stderr\n")
                return FailedProcess()

            argv = ["--data-root", str(data), "--output", str(output), "--baseline", str(baseline)]
            with patch.object(runner, "PRETRAINED", weights), \
                 patch.object(runner, "file_sha256", return_value="983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d"), \
                 patch.object(runner, "snapshot_sources", side_effect=fake_snapshot), \
                 patch.object(runner.subprocess, "Popen", side_effect=fail_first_stage) as popen:
                self.assertEqual(runner.main(argv), 7)

            popen.assert_called_once()
            state = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["stage"], "train")
            self.assertEqual(state["returncode"], 7)
            self.assertEqual(state["stages"][-1]["child_pid"], 4312)
            self.assertEqual(state["stages"][-1]["status"], "failed")
            self.assertIn("child stdout", (output / "logs" / "train.stdout.log").read_text(encoding="utf-8"))
            self.assertIn("child stderr", (output / "logs" / "train.stderr.log").read_text(encoding="utf-8"))
            self.assertEqual(baseline.read_text(encoding="utf-8"), "read-only\n")

    def test_nonempty_output_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, baseline, weights = self._inputs(root)
            output = root / "experiment"
            output.mkdir()
            marker = output / "preserve.txt"
            marker.write_text("keep", encoding="utf-8")
            argv = ["--data-root", str(data), "--output", str(output), "--baseline", str(baseline)]
            with patch.object(runner, "PRETRAINED", weights), patch.object(
                    runner, "file_sha256",
                    return_value="983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d"):
                with self.assertRaises(SystemExit) as raised:
                    runner.main(argv)
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
            self.assertFalse((output / "experiment.json").exists())


if __name__ == "__main__":
    unittest.main()
