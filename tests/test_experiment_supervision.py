import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from scripts.supervise_grouped_experiments import supervise


def _write_runner(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "import json, os, sys, time\n"
        "from pathlib import Path\n"
        "args = sys.argv\n"
        "output = Path(args[args.index('--output') + 1])\n"
        "state_path = output / 'experiment.json'\n"
        + body,
        encoding="utf-8",
    )


class ExperimentSupervisionTests(unittest.TestCase):
    def _layout(self, root, source_body, fusion_body):
        source = root / "source-output"
        fusion = root / "fusion-output"
        source.mkdir()
        fusion.mkdir()
        (source / "experiment.json").write_text(json.dumps({"status": "prepared", "runner_pid": None}))
        (fusion / "experiment.json").write_text(json.dumps({"status": "prepared", "runner_pid": None}))
        source_runner = source / "source" / "scripts" / "run_grouped_generalization.py"
        fusion_runner = fusion / "source" / "scripts" / "run_oof_fusion.py"
        _write_runner(source_runner, source_body)
        _write_runner(fusion_runner, fusion_body)
        return source, fusion, root / "supervisor", source_runner, fusion_runner

    def test_fusion_starts_only_after_source_runner_is_live_and_records_commands(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_body = (
                "state_path.write_text(json.dumps({'status': 'running', 'runner_pid': os.getpid()}))\n"
                "time.sleep(0.12)\n"
                "state_path.write_text(json.dumps({'status': 'complete', 'runner_pid': None}))\n"
            )
            fusion_body = (
                "source = Path(%r)\n" % str(root / "source-output" / "experiment.json")
                + "seen = json.loads(source.read_text())\n"
                + "assert seen['status'] in ('running', 'complete'), seen\n"
                + "state_path.write_text(json.dumps({'status': 'running', 'runner_pid': os.getpid()}))\n"
                + "state_path.write_text(json.dumps({'status': 'complete', 'runner_pid': None}))\n"
            )
            source, fusion, output, source_runner, fusion_runner = self._layout(root, source_body, fusion_body)
            result = supervise(source, fusion, output, poll_interval=0.02, startup_timeout=2)
            self.assertEqual(result["status"], "complete", result)
            saved = json.loads((output / "supervisor.json").read_text())
            self.assertEqual(saved["commands"]["source"], [
                sys.executable, str(source_runner.resolve()), "--resume", "--output", str(source.resolve())
            ])
            self.assertEqual(saved["commands"]["fusion"], [
                sys.executable, str(fusion_runner.resolve()), "--resume", "--output", str(fusion.resolve())
            ])
            self.assertEqual(saved["children"]["source"]["status"], "complete")
            self.assertEqual(saved["children"]["fusion"]["status"], "complete")
            self.assertTrue((output / "source.stdout.log").is_file())
            self.assertTrue((output / "fusion.stderr.log").is_file())

    def test_source_failure_does_not_start_fusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_body = (
                "state_path.write_text(json.dumps({'status': 'failed', 'runner_pid': None}))\n"
                "sys.exit(3)\n"
            )
            fusion_body = "raise AssertionError('fusion must not start')\n"
            source, fusion, output, _, _ = self._layout(root, source_body, fusion_body)
            result = supervise(source, fusion, output, poll_interval=0.02, startup_timeout=2)
            self.assertEqual(result["status"], "failed")
            self.assertIn("source", result["error"])
            self.assertEqual(result["children"]["fusion"]["status"], "pending")
            self.assertEqual(result["children"]["source"]["return_code"], 3)
            self.assertFalse((output / "fusion.stdout.log").exists())

    def test_explicit_alternate_entries_run_in_the_experiment_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, fusion, output, _, _ = self._layout(
                root, "raise AssertionError('original source must not run')\n",
                "raise AssertionError('original fusion must not run')\n",
            )
            source_entry = root / "alternate" / "source.py"
            fusion_entry = root / "alternate" / "fusion.py"
            body = (
                "assert Path.cwd() == output, (Path.cwd(), output)\n"
                "(output / 'alternate-used').write_text('yes')\n"
                "state_path.write_text(json.dumps({'status': 'complete', 'runner_pid': None}))\n"
            )
            _write_runner(source_entry, body)
            _write_runner(fusion_entry, body)
            result = supervise(source, fusion, output, poll_interval=0.02, startup_timeout=2,
                               source_entry=source_entry, fusion_entry=fusion_entry)
            self.assertEqual(result["status"], "complete", result)
            for name, entry, experiment in (("source", source_entry, source),
                                            ("fusion", fusion_entry, fusion)):
                self.assertTrue((experiment / "alternate-used").exists())
                self.assertEqual(result["commands"][name][1], str(entry.resolve()))

    def test_missing_alternate_entry_is_rejected_before_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, fusion, output, _, _ = self._layout(root, "", "")
            with self.assertRaises(FileNotFoundError):
                supervise(source, fusion, output, source_entry=root / "absent.py")
            self.assertFalse(output.exists())

    def test_fusion_failure_is_recorded_without_changing_source_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_body = "state_path.write_text(json.dumps({'status': 'complete', 'runner_pid': None}))\n"
            fusion_body = (
                "state_path.write_text(json.dumps({'status': 'failed', 'runner_pid': None}))\n"
                "sys.exit(4)\n"
            )
            source, fusion, output, _, _ = self._layout(root, source_body, fusion_body)
            result = supervise(source, fusion, output, poll_interval=0.02, startup_timeout=2)
            self.assertEqual(result["status"], "failed")
            self.assertIn("fusion", result["error"])
            self.assertEqual(result["children"]["fusion"]["return_code"], 4)
            self.assertEqual(result["children"]["source"]["status"], "complete")

    def test_refuses_existing_live_runner_pid_before_creating_supervisor_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, fusion, output, _, _ = self._layout(root, "", "")
            (source / "experiment.json").write_text(json.dumps({"status": "running", "runner_pid": os.getpid()}))
            with self.assertRaisesRegex(RuntimeError, "already active"):
                supervise(source, fusion, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
