"""Run one fixed ConvNeXt experiment and its validation-only evaluation chain."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cxr.safety import fresh_run_directory

PRETRAINED = ROOT / "weights" / "convnext_tiny-983f1562.pth"
SNAPSHOT_FILES = (
    "train.py", "predict.py", "cost.py", "README.md", "requirements.txt",
    "requirements-lock.txt", "cxr/__init__.py", "cxr/data.py", "cxr/engine.py",
    "cxr/image_cache.py", "cxr/metrics.py", "cxr/models.py", "cxr/optimization.py", "cxr/safety.py",
    "cxr/submission.py", "cxr/training_state.py", "cxr/consistency.py",
    "scripts/compare_runs.py", "scripts/inference_manifest.py",
    "scripts/benchmark_valid.py", "scripts/run_aggressive_experiment.py",
    "AGGRESSIVE_OPTIMIZATION_PLAN.md",
)
INFERENCE_FILES = (
    "predict.py", "cxr/data.py", "cxr/engine.py", "cxr/metrics.py",
    "cxr/models.py", "cxr/safety.py", "cxr/submission.py",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def snapshot_sources(destination):
    destination.mkdir(parents=True)
    entries = {}
    for relative in SNAPSHOT_FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"Required source file is missing: {source}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        entries[relative] = {"bytes": target.stat().st_size, "sha256": file_sha256(target)}
    atomic_json(destination / "manifest.json", entries)
    return entries


def build_stages(args, output, source=None):
    source = Path(source) if source is not None else output / "source"
    training = output / "training"
    checkpoint = training / "best.pt"
    candidate = training / "valid.csv"
    comparison = training / "comparison.json"
    stages = []

    train = [
        sys.executable, str(source / "train.py"), "--data-root", str(args.data_root),
        "--output", str(training), "--pretrained", str(PRETRAINED),
        "--architecture", "convnext_tiny", "--epochs", "20",
        "--batch-size", str(args.batch_size), "--accumulation-steps", str(args.accumulation_steps),
        "--image-size", "384", "--lr", "0.0001", "--weight-decay", "0.01",
        "--scheduler", "cosine", "--warmup-epochs", "2", "--min-lr", "0.000001",
        "--ema-decay", "0.999", "--ema-device", args.ema_device,
        "--workers", "0", "--device", "cuda", "--seed", "2026",
    ]
    if args.gradient_checkpointing:
        train.append("--gradient-checkpointing")
    if args.image_cache is not None:
        train.extend(("--image-cache", str(args.image_cache)))
    stages.append(("train", train))

    stages.append(("predict_valid", [
        sys.executable, str(source / "predict.py"), "--data-root", str(args.data_root),
        "--split", "valid", "--checkpoint", str(checkpoint), "--output", str(candidate),
        "--device", "cuda",
    ]))
    stages.append(("compare_runs", [
        sys.executable, str(source / "scripts" / "compare_runs.py"),
        "--baseline", str(args.baseline), "--candidate", str(candidate),
        "--metadata", str(args.data_root / "valid.csv"), "--output", str(comparison),
        "--bootstrap", "500", "--seed", "2026",
    ]))
    manifest_files = [str(checkpoint)] + [str(source / item) for item in INFERENCE_FILES]
    stages.append(("inference_manifest", [
        sys.executable, str(source / "scripts" / "inference_manifest.py"),
        *manifest_files, "--output", str(training / "inference_manifest.json"),
    ]))
    stages.append(("benchmark_valid", [
        sys.executable, str(source / "scripts" / "benchmark_valid.py"),
        "--data-root", str(args.data_root), "--run", str(training), "--device", "cuda",
    ]))
    return stages


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New, empty experiment parent directory")
    parser.add_argument("--baseline", type=Path, required=True, help="Existing validation submission CSV; read-only")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--image-cache", type=Path)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--ema-device", choices=("model", "cpu"), default="model")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.accumulation_steps < 1:
        parser.error("--batch-size and --accumulation-steps must be positive")

    data_root = args.data_root.resolve(strict=True)
    baseline = args.baseline.resolve(strict=True)
    if not data_root.is_dir():
        parser.error("--data-root must be a directory")
    if not baseline.is_file():
        parser.error("--baseline must be an existing regular file")
    if not (data_root / "train.csv").is_file() or not (data_root / "valid.csv").is_file():
        parser.error("--data-root must contain train.csv and valid.csv")
    if not PRETRAINED.is_file():
        parser.error(f"Official ConvNeXt weights are missing: {PRETRAINED}")
    expected_weights_sha = "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d"
    actual_weights_sha = file_sha256(PRETRAINED)
    if actual_weights_sha != expected_weights_sha:
        parser.error("Official ConvNeXt weights do not match the recorded SHA-256 allowlist")
    args.data_root, args.baseline = data_root, baseline
    if args.image_cache is not None:
        args.image_cache = args.image_cache.resolve()

    try:
        output = fresh_run_directory(args.output, data_root)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    (output / "logs").mkdir()
    state_path = output / "experiment.json"
    state = {
        "status": "running", "stage": "source_snapshot", "command": ["internal", "snapshot source files"],
        "child_pid": None, "started_utc": utc_now(), "ended_utc": None, "returncode": None,
        "data_root": str(data_root), "baseline": str(baseline), "output": str(output),
        "pretrained_weights": {"path": str(PRETRAINED), "bytes": PRETRAINED.stat().st_size,
                                "sha256": actual_weights_sha},
        "stages": [],
    }
    atomic_json(state_path, state)
    print(f"Experiment directory: {output}", flush=True)

    active_process = None
    try:
        state["stage"] = "source_snapshot"
        print("Stage: source_snapshot", flush=True)
        (output / "logs" / "source_snapshot.stdout.log").write_text("", encoding="utf-8")
        (output / "logs" / "source_snapshot.stderr.log").write_text("", encoding="utf-8")
        state["source_files"] = snapshot_sources(output / "source")
        state["stages"].append({"name": "source_snapshot", "status": "complete",
                                "command": ["internal", "snapshot source files"],
                                "child_pid": None, "started_utc": state["started_utc"],
                                "ended_utc": utc_now(), "returncode": 0})
        state["source_manifest_sha256"] = file_sha256(output / "source" / "manifest.json")
        atomic_json(state_path, state)

        source = output / "source"
        for name, command in build_stages(args, output, source):
            stage = {"name": name, "status": "running", "command": command, "child_pid": None,
                     "started_utc": utc_now(), "ended_utc": None, "returncode": None,
                     "stdout_log": str(output / "logs" / f"{name}.stdout.log"),
                     "stderr_log": str(output / "logs" / f"{name}.stderr.log")}
            state["stage"], state["command"], state["child_pid"], state["returncode"] = name, command, None, None
            state["stages"].append(stage)
            atomic_json(state_path, state)
            print(f"Stage: {name}", flush=True)
            with Path(stage["stdout_log"]).open("w", encoding="utf-8", buffering=1) as stdout, \
                    Path(stage["stderr_log"]).open("w", encoding="utf-8", buffering=1) as stderr:
                active_process = subprocess.Popen(command, cwd=source, stdout=stdout, stderr=stderr)
                stage["child_pid"] = active_process.pid
                state["child_pid"] = active_process.pid
                atomic_json(state_path, state)
                print(f"  PID: {active_process.pid}", flush=True)
                returncode = active_process.wait()
                active_process = None
            stage["returncode"], stage["ended_utc"] = returncode, utc_now()
            stage["status"] = "complete" if returncode == 0 else "failed"
            state["returncode"] = returncode
            state["child_pid"] = None
            atomic_json(state_path, state)
            if returncode != 0:
                state["status"] = "failed"
                state["ended_utc"] = utc_now()
                state["error"] = f"Stage {name} failed with exit code {returncode}"
                atomic_json(state_path, state)
                print(state["error"], file=sys.stderr, flush=True)
                return returncode if returncode > 0 else 1

        report_path = output / "training" / "comparison.json"
        comparison = json.loads(report_path.read_text(encoding="utf-8"))
        model_path = output / "training" / "best.pt"
        state["result"] = {
            "macro_delta": {name: value.get("delta") for name, value in comparison["macro"].items()},
            "model": {"path": str(model_path), "bytes": model_path.stat().st_size,
                      "sha256": file_sha256(model_path)},
            "comparison": str(report_path),
        }
        state["status"], state["stage"], state["command"] = "complete", "complete", None
        state["returncode"], state["ended_utc"] = 0, utc_now()
        atomic_json(state_path, state)
        print(f"Experiment complete: {output}", flush=True)
        return 0
    except BaseException as exc:
        if active_process is not None and active_process.poll() is None:
            active_process.terminate()
            stopped_code = active_process.wait()
            if state.get("stages") and state["stages"][-1].get("ended_utc") is None:
                state["stages"][-1]["returncode"] = stopped_code
                state["stages"][-1]["ended_utc"] = utc_now()
                state["stages"][-1]["status"] = "failed"
            state["returncode"], state["child_pid"] = stopped_code, None
        state["status"], state["ended_utc"] = "failed", utc_now()
        state["error"] = f"{type(exc).__name__}: {exc}"
        if state.get("stages") and state["stages"][-1].get("ended_utc") is None:
            state["stages"][-1]["ended_utc"] = state["ended_utc"]
            state["stages"][-1]["status"] = "failed"
            state["stages"][-1]["error"] = state["error"]
        atomic_json(state_path, state)
        print(f"Experiment failed: {state['error']}", file=sys.stderr, flush=True)
        if isinstance(exc, KeyboardInterrupt):
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
