"""Supervise the locked grouped source experiment followed by its OOF fusion."""

from __future__ import annotations

import argparse
import csv
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ACTIVE_SOURCE_STATUSES = {"running", "audit_candidate_locked"}
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def process_alive(pid):
    """Check a recorded PID without risking termination on Windows."""
    if isinstance(pid, bool):
        return False
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, check=False,
        )
        return any(len(row) > 1 and row[1] == str(pid)
                   for row in csv.reader(result.stdout.splitlines()))
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _require_recorded_state(output, name):
    path = output / "experiment.json"
    if not path.is_file():
        raise FileNotFoundError(f"{name} experiment state is missing: {path}")
    state = read_json(path)
    if not isinstance(state, dict):
        raise ValueError(f"{name} experiment state must be a JSON object")
    pid = state.get("runner_pid")
    if pid and process_alive(pid):
        raise RuntimeError(f"{name} runner PID {pid} is already active")
    return state


def _execution_state(hold):
    if os.name != "nt":
        return
    kernel = ctypes.windll.kernel32
    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED if hold else ES_CONTINUOUS
    if not kernel.SetThreadExecutionState(flags):
        raise ctypes.WinError()


def _paths(source_output, fusion_output, supervisor_output, source_entry=None, fusion_entry=None):
    source_output = Path(source_output).resolve(strict=True)
    fusion_output = Path(fusion_output).resolve(strict=True)
    supervisor_output = Path(supervisor_output).resolve()
    source_runner = (source_output / "source" / "scripts" / "run_grouped_generalization.py").resolve(strict=True)
    fusion_runner = (fusion_output / "source" / "scripts" / "run_oof_fusion.py").resolve(strict=True)
    if not source_runner.is_file() or not fusion_runner.is_file():
        raise FileNotFoundError("both immutable snapshot runner files must exist")
    if source_entry is not None:
        source_runner = Path(source_entry).resolve(strict=True)
    if fusion_entry is not None:
        fusion_runner = Path(fusion_entry).resolve(strict=True)
    if not source_runner.is_file() or not fusion_runner.is_file():
        raise FileNotFoundError("both selected entry points must be files")
    return source_output, fusion_output, supervisor_output, source_runner, fusion_runner


def _source_ready(source_output, source_process, previous_pid):
    state_path = source_output / "experiment.json"
    if state_path.is_file():
        try:
            state = read_json(state_path)
        except (OSError, json.JSONDecodeError):
            state = None
        if isinstance(state, dict):
            if state.get("status") == "complete":
                return "complete", state
            if state.get("status") == "failed":
                return "failed", state
            runner_pid = state.get("runner_pid")
            if (runner_pid and runner_pid != previous_pid
                    and state.get("status") in ACTIVE_SOURCE_STATUSES
                    and process_alive(runner_pid)):
                return "active", state
    return None, None


def _spawn(command, log_dir, name, *, cwd):
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_file = (log_dir / f"{name}.stdout.log").open("ab")
    stderr_file = (log_dir / f"{name}.stderr.log").open("ab")
    kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": stdout_file,
        "stderr": stderr_file,
        "cwd": str(cwd),
        "close_fds": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        process = subprocess.Popen(command, **kwargs)
    finally:
        stdout_file.close()
        stderr_file.close()
    return process


def supervise(source_output, fusion_output, output, *, poll_interval=5.0, startup_timeout=120.0,
              source_entry=None, fusion_entry=None):
    if poll_interval <= 0 or startup_timeout <= 0:
        raise ValueError("poll interval and startup timeout must be positive")
    source_output, fusion_output, output, source_runner, fusion_runner = _paths(
        source_output, fusion_output, output, source_entry, fusion_entry
    )
    source_existing = _require_recorded_state(source_output, "source")
    fusion_existing = _require_recorded_state(fusion_output, "fusion")
    output.mkdir(parents=True, exist_ok=True)
    log_dir = output
    source_command = [sys.executable, str(source_runner), "--resume", "--output", str(source_output)]
    fusion_command = [sys.executable, str(fusion_runner), "--resume", "--output", str(fusion_output)]
    state = {
        "status": "starting_source",
        "supervisor_pid": os.getpid(),
        "execution_state_held": False,
        "started_utc": now(),
        "source_output": str(source_output),
        "fusion_output": str(fusion_output),
        "commands": {"source": source_command, "fusion": fusion_command},
        "children": {
            "source": {"status": "pending", "recorded_status_before_start": source_existing.get("status")},
            "fusion": {"status": "pending", "recorded_status_before_start": fusion_existing.get("status")},
        },
        "heartbeat_utc": now(),
    }
    state_path = output / "supervisor.json"
    _execution_state(True)
    state["execution_state_held"] = os.name == "nt"
    processes = {}
    failure = None
    try:
        source_process = _spawn(source_command, log_dir, "source", cwd=source_output)
        processes["source"] = source_process
        state["children"]["source"].update(pid=source_process.pid, status="starting")
        atomic_json(state_path, state)

        gate_started = time.monotonic()
        while True:
            ready, source_state = _source_ready(
                source_output, source_process, source_existing.get("runner_pid")
            )
            source_rc = source_process.poll()
            if ready == "failed":
                failure = "source experiment recorded failed status"
                break
            if ready == "complete":
                if source_rc not in (None, 0):
                    failure = f"source runner exited with code {source_rc} after recording complete"
                    break
                state["children"]["source"].update(status="complete", experiment_status="complete")
                break
            if ready == "active":
                state["children"]["source"].update(status="running", experiment_status=source_state.get("status"))
                break
            if source_rc is not None:
                if source_rc == 0:
                    try:
                        finished_state = read_json(source_output / "experiment.json")
                    except (OSError, json.JSONDecodeError):
                        finished_state = None
                    if isinstance(finished_state, dict) and finished_state.get("status") == "complete":
                        state["children"]["source"].update(
                            status="complete", experiment_status="complete", return_code=0
                        )
                        break
                failure = f"source runner exited with code {source_rc} before reporting active state"
                break
            if time.monotonic() - gate_started >= startup_timeout:
                failure = "source runner did not report its PID and active status before startup timeout"
                break
            state["status"] = "waiting_for_source_start"
            state["heartbeat_utc"] = now()
            atomic_json(state_path, state)
            time.sleep(poll_interval)

        if failure is None:
            fusion_process = _spawn(fusion_command, log_dir, "fusion", cwd=fusion_output)
            processes["fusion"] = fusion_process
            state["children"]["fusion"].update(pid=fusion_process.pid, status="running")
            state["status"] = "running"
            atomic_json(state_path, state)

        while failure is None and any(process.poll() is None for process in processes.values()):
            for name, process in processes.items():
                return_code = process.poll()
                if return_code is None:
                    continue
                child = state["children"][name]
                child["return_code"] = int(return_code)
                if return_code != 0:
                    child["status"] = "failed"
                    if failure is None:
                        failure = f"{name} runner exited with code {return_code}"
                elif child.get("status") != "complete":
                    exp_output = source_output if name == "source" else fusion_output
                    try:
                        experiment_status = read_json(exp_output / "experiment.json").get("status")
                    except (OSError, json.JSONDecodeError):
                        experiment_status = None
                    child["experiment_status"] = experiment_status
                    if experiment_status == "complete":
                        child["status"] = "complete"
                    else:
                        child["status"] = "failed"
                        if failure is None:
                            failure = f"{name} runner exited successfully without complete experiment state"
            if failure is not None:
                break
            state["heartbeat_utc"] = now()
            for name, process in processes.items():
                child = state["children"][name]
                if process.poll() is None and name == "source":
                    ready, source_state = _source_ready(
                        source_output, process, source_existing.get("runner_pid")
                    )
                    if ready == "failed":
                        child["experiment_status"] = "failed"
                        failure = "source experiment recorded failed status"
                        break
                    if source_state is not None:
                        child["experiment_status"] = source_state.get("status")
                if process.poll() is not None and "return_code" not in child:
                    continue
            atomic_json(state_path, state)
            if failure is not None:
                break
            time.sleep(poll_interval)

        for name, process in processes.items():
            if failure is not None and process.poll() is None:
                exp_output = source_output if name == "source" else fusion_output
                try:
                    recorded_failure = read_json(exp_output / "experiment.json").get("status") == "failed"
                except (OSError, json.JSONDecodeError):
                    recorded_failure = False
                if recorded_failure:
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
            return_code = process.poll()
            if return_code is not None:
                state["children"][name]["return_code"] = int(return_code)
                if return_code != 0:
                    state["children"][name]["status"] = "failed"

        if failure is not None:
            state["status"] = "failed"
            state["error"] = failure
        else:
            for name, process in processes.items():
                child = state["children"][name]
                if "return_code" not in child:
                    child["return_code"] = int(process.returncode)
                if child.get("status") != "complete":
                    exp_output = source_output if name == "source" else fusion_output
                    child["experiment_status"] = read_json(exp_output / "experiment.json").get("status")
                    if child["experiment_status"] != "complete":
                        failure = f"{name} experiment did not complete"
                        child["status"] = "failed"
                        break
                    child["status"] = "complete"
            state["status"] = "failed" if failure else "complete"
            if failure:
                state["error"] = failure
        state["ended_utc"] = now()
        state["heartbeat_utc"] = state["ended_utc"]
        state["execution_state_held"] = False
        atomic_json(state_path, state)
        return state
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = f"{type(error).__name__}: {error}"
        state["ended_utc"] = now()
        state["execution_state_held"] = False
        atomic_json(state_path, state)
        raise
    finally:
        _execution_state(False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--fusion-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--source-entry", type=Path)
    parser.add_argument("--fusion-entry", type=Path)
    args = parser.parse_args(argv)
    state = supervise(args.source_output, args.fusion_output, args.output,
                      poll_interval=args.poll_interval, startup_timeout=args.startup_timeout,
                      source_entry=args.source_entry, fusion_entry=args.fusion_entry)
    print(json.dumps({"status": state["status"], "state": str(args.output.resolve() / "supervisor.json")}))
    return 0 if state["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
