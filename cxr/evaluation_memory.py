"""Runtime-only evaluation memory guard for immutable grouped experiments."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


AMENDMENT_NAME = "runtime_amendment.json"
POLICY = "fixed_final_raw_and_ema_snapshots_then_single_worker_evaluation"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_amendment(output):
    output = Path(output).resolve()
    for directory in (output, *output.parents):
        candidate = directory / AMENDMENT_NAME
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"registered runtime amendment is absent above {output}")


def verify_amendment(path, guard_path, helper_path, *, require_audit_unopened=True):
    path = Path(path).resolve(strict=True)
    amendment = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "policy": POLICY,
        "guard_sha256": sha256(guard_path),
        "helper_sha256": sha256(helper_path),
        "audit_opened_at_registration": False,
    }
    for key, value in expected.items():
        if amendment.get(key) != value:
            raise RuntimeError(f"runtime amendment mismatch for {key}")
    return amendment


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def install(trainer, cfg, output, provenance, amendment_path, guard_path, helper_path):
    """Install the explicit runtime wrapper on an already imported trainer module."""
    amendment_path = Path(amendment_path).resolve(strict=True)
    amendment = verify_amendment(amendment_path, guard_path, helper_path)
    original_loader = trainer.DataLoader
    original_predict = trainer.predict_records
    original_save_prediction = trainer.save_prediction
    original_atomic_json = trainer.atomic_json
    original_ema = trainer.ModelEMA
    original_initialize = trainer.initialize_model
    captured = {"training_loader": None, "model": None, "ema": None}

    def tracked_loader(*args, **kwargs):
        loader = original_loader(*args, **kwargs)
        dataset = args[0] if args else kwargs.get("dataset")
        if bool(getattr(dataset, "training", False)):
            if captured["training_loader"] is not None:
                raise RuntimeError("more than one training DataLoader was created")
            captured["training_loader"] = loader
        return loader

    def capture_initialize(*args, **kwargs):
        result = original_initialize(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("trainer initialize_model API changed")
        captured["model"] = result[1]
        return result

    def capture_ema(*args, **kwargs):
        ema = original_ema(*args, **kwargs)
        if captured["ema"] is not None:
            raise RuntimeError("more than one EMA object was created")
        captured["ema"] = ema
        return ema

    def close_training_workers():
        loader = captured["training_loader"]
        if loader is None:
            raise RuntimeError("training DataLoader was not captured")
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if not callable(shutdown):
                raise RuntimeError("training DataLoader iterator API changed")
            shutdown()
            loader._iterator = None

    def guarded_predict(model, records, processor, device, batch_size, workers):
        close_training_workers()
        if workers < 0:
            raise ValueError("evaluation workers must be nonnegative")
        return original_predict(model, records, processor, device, batch_size, 0)

    def save_recovery_artifacts(stage_output, stage_provenance):
        raw, ema = captured["model"], captured["ema"]
        if raw is None or ema is None:
            raise RuntimeError("final raw model or EMA object was not captured")
        try:
            report = json.loads((Path(stage_output) / "report.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("final training report/history is unavailable") from error
        if report.get("status") != "evaluating_fixed_final_weights":
            raise RuntimeError("final fixed-epoch training status is missing")
        recovery = Path(stage_output) / "runtime_final"
        recovery.mkdir(parents=True, exist_ok=True)
        binding = {
            "runtime_amendment": str(amendment_path),
            "runtime_amendment_sha256": sha256(amendment_path),
            "guard_sha256": sha256(guard_path), "helper_sha256": sha256(helper_path),
            "config": {key: str(value) if isinstance(value, Path) else value
                       for key, value in vars(cfg).items()},
            "provenance": stage_provenance, "training_report": report,
            "kind": "reliability_snapshot_not_selected_epoch",
        }
        snapshots = {}
        for name, model in (("raw", raw), ("ema", ema.module)):
            if not hasattr(model, "state_dict"):
                raise RuntimeError(f"{name} model state_dict API changed")
            checkpoint = recovery / f"{name}.pt"
            trainer.torch.save({"model": model.state_dict(), **binding, "variant": name}, checkpoint)
            snapshots[f"{name}.pt"] = sha256(checkpoint)
        atomic_json(recovery / "manifest.json", {**binding, "checkpoint_sha256": snapshots})

    def guarded_save_prediction(stage_output, *args, **kwargs):
        stage_provenance = args[6] if len(args) > 6 else kwargs.get("provenance")
        if not isinstance(stage_provenance, dict):
            raise RuntimeError("save_prediction provenance API changed")
        if not (Path(stage_output) / "runtime_final/manifest.json").exists():
            close_training_workers()
            save_recovery_artifacts(stage_output, stage_provenance)
        if len(args) >= 7:
            args = (*args[:6], {**stage_provenance,
                                "runtime_amendment": trainer.RUNTIME_AMENDMENT}, *args[7:])
        else:
            kwargs["provenance"] = {**stage_provenance,
                                    "runtime_amendment": trainer.RUNTIME_AMENDMENT}
        return original_save_prediction(stage_output, *args, **kwargs)

    def amendment_atomic_json(path, data):
        if Path(path).name == "report.json" and isinstance(data, dict):
            data = dict(data)
            data["runtime_amendment"] = trainer.RUNTIME_AMENDMENT
            if isinstance(data.get("provenance"), dict):
                data["provenance"] = {**data["provenance"],
                                      "runtime_amendment": trainer.RUNTIME_AMENDMENT}
        return original_atomic_json(path, data)

    trainer.DataLoader = tracked_loader
    trainer.initialize_model = capture_initialize
    trainer.ModelEMA = capture_ema
    trainer.predict_records = guarded_predict
    trainer.save_prediction = guarded_save_prediction
    trainer.atomic_json = amendment_atomic_json
    trainer.RUNTIME_AMENDMENT = {
        "path": str(amendment_path), "sha256": sha256(amendment_path),
        "policy": POLICY, "guard_sha256": sha256(guard_path),
        "helper_sha256": sha256(helper_path),
    }
    return {"originals": (original_loader, original_predict, original_save_prediction,
                          original_ema, original_initialize), "captured": captured}
