"""Frozen public features, patient-grouped study heads, then one locked comparison."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cxr.data import read_records
from cxr.safety import fresh_run_directory, safe_output
from cxr.study_probe import pool_study_features, predict_linear_probe
from cxr.submission import validate_submission, write_submission
from scripts.compare_runs import load_submission

REVISION = "f778e2ac267fdd651205e07b6fd1113754b146bb"
CXRLT_REVISION = "24bf8bdfd8d0262899167b030be16ab89f78ae20"
CXRLT_WEIGHT = "convnextv2_base_mimic-cxr_padchest_csra_dbcas.safetensors"
CXRLT_SHA256 = "6950f7b97422b807e16a821a954e5d62b6868f8fbfb1e841f6d090c91e4ca17b"
CXRLT_CODE_SHA256 = "b08fb08dc44dd936573aaa590244b502756b5335085e9b78cd84d939d4975ec0"
FILES = (
    "scripts/run_frozen_study_probe.py", "scripts/evaluate_frozen_study_probe.py",
    "scripts/compare_runs.py", "cxr/__init__.py", "cxr/data.py", "cxr/metrics.py",
    "cxr/safety.py", "cxr/submission.py", "cxr/study_probe.py",
    "docs/FROZEN_STUDY_PROBE_PLAN_20261108.md",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def environment():
    return {"python": sys.version, **{name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "numpy", "Pillow", "safetensors",
                         "timm", "opencv-python-headless")}}


def records_with_views(data_root, split):
    records = read_records(data_root / f"{split}.csv", data_root / split)
    with (data_root / f"{split}.csv").open(encoding="utf-8-sig", newline="") as stream:
        views = {row["Dicom_id"]: row.get("ViewPosition", "") for row in csv.DictReader(stream)}
    return records, [views[row["dicom_id"]] for row in records]


def image_identity(records):
    result = []
    for row in records:
        path = Path(row["path"])
        stat = path.stat()
        result.append([row["dicom_id"], str(path), stat.st_size, stat.st_mtime_ns])
    return result


def verify_images(identity):
    for dicom, name, size, mtime in identity:
        stat = Path(name).stat()
        if stat.st_size != size or stat.st_mtime_ns != mtime:
            raise ValueError(f"image changed since registration: {dicom}")


def pixel_batches(records, processor, start, batch_size, workers):
    def prepare_batch(offset):
        images = []
        try:
            for row in records[offset:offset + batch_size]:
                with Image.open(row["path"]) as source:
                    images.append(source.convert("RGB"))
            return processor(images=images, return_tensors="pt")["pixel_values"]
        finally:
            for image in images:
                image.close()
    offsets = iter(range(start, len(records), batch_size))
    if not workers:
        for offset in offsets:
            yield offset, prepare_batch(offset)
        return
    # Bound queued work: full-resolution radiographs must not accumulate in RAM.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = deque()
        for _ in range(workers):
            offset = next(offsets, None)
            if offset is not None:
                pending.append((offset, executor.submit(prepare_batch, offset)))
        while pending:
            offset, future = pending.popleft()
            pixels = future.result()
            following = next(offsets, None)
            if following is not None:
                pending.append((following, executor.submit(prepare_batch, following)))
            yield offset, pixels


def extract_features(model, processor, records, folder, *, batch_size, device, preprocess_workers=0):
    """Resume only committed rows of a fingerprinted, fixed-order feature cache."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    identities = image_identity(records)
    direct_features = hasattr(model, "feature_size")
    feature_size = int(model.feature_size) if direct_features else 2 * model.config.hidden_size
    input_size = int(model.input_size) if direct_features else 518
    expected = {
        "images": identities, "batch_size": batch_size, "device": str(device),
        "feature_size": feature_size, "amp": device.type == "cuda",
        "pooling": model.feature_pooling if direct_features else "normalized_final_cls_plus_mean_all_post_cls_including_registers",
    }
    if direct_features:
        expected["input_size"] = input_size
    spec = folder / "input.json"
    if spec.exists():
        if read_json(spec) != expected:
            raise ValueError(f"feature cache input mismatch: {folder}")
    else:
        atomic_json(spec, expected)
    progress_path = folder / "progress.json"
    progress = read_json(progress_path) if progress_path.exists() else {"completed": 0}
    shape = (len(records), expected["feature_size"])
    feature_path = folder / "image_features.npy"
    if feature_path.exists():
        values = np.load(feature_path, mmap_mode="r+", allow_pickle=False)
        if values.shape != shape or values.dtype != np.float32:
            raise ValueError("feature shape/dtype mismatch")
    else:
        if progress["completed"]:
            raise ValueError("cache claims committed rows without its matrix")
        values = np.lib.format.open_memmap(feature_path, mode="w+", dtype=np.float32, shape=shape)
    completed = progress["completed"]
    if not isinstance(completed, int) or not 0 <= completed <= len(records):
        raise ValueError("invalid committed feature cursor")
    if progress.get("status") == "complete":
        if completed != len(records) or sha256(feature_path) != progress["sha256"]:
            raise ValueError("completed feature cache hash mismatch")
        return values
    model.eval()
    started = time.monotonic()
    start_row = completed
    next_log = completed
    with torch.inference_mode():
        for start, pixels in pixel_batches(records, processor, completed, batch_size, preprocess_workers):
            end = min(start + batch_size, len(records))
            if pixels.shape[1:] != (3, input_size, input_size):
                raise ValueError(f"unexpected encoder input: {pixels.shape}")
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                result = model(pixel_values=pixels.to(device))
                if direct_features:
                    pooled = result
                else:
                    sequence = result.last_hidden_state
                    pooled = torch.cat((sequence[:, 0], sequence[:, 1:].mean(dim=1)), dim=1)
            batch = pooled.float().cpu().numpy()
            if batch.shape != (end - start, shape[1]) or not np.isfinite(batch).all():
                raise ValueError("invalid extracted features")
            values[start:end] = batch
            if end >= next_log or end == len(records):
                values.flush()
                elapsed = max(time.monotonic() - started, 1e-9)
                progress = {"status": "extracting", "completed": end, "total": len(records),
                            "updated_utc": now(), "current_segment_images_per_second": (end - start_row) / elapsed}
                atomic_json(progress_path, progress)
                print(json.dumps({"cache": folder.name, **progress}), flush=True)
                next_log = end + 100
    values.flush()
    verify_images(identities)
    progress.update(status="complete", completed=len(records), sha256=sha256(feature_path), updated_utc=now())
    atomic_json(progress_path, progress)
    return values


def load_encoder(plan):
    device = torch.device(plan["device"])
    if plan.get("encoder", "cxformer") == "cxrlt":
        from cxr.cxrlt_transfer import load_encoder as load_cxrlt
        model, processor = load_cxrlt(plan["model_dir"], device)
        return model, processor, device
    if plan.get("encoder", "cxformer") != "cxformer":
        raise ValueError("unsupported registered encoder")
    processor = AutoImageProcessor.from_pretrained(plan["model_dir"], trust_remote_code=True, local_files_only=True)
    if not getattr(processor, "do_hist_equalization", False):
        raise ValueError("CXformer published histogram equalization is required")
    model = AutoModel.from_pretrained(plan["model_dir"], local_files_only=True, attn_implementation="sdpa")
    model.requires_grad_(False)
    return model.to(device).eval(), processor, device


def prepare(args):
    output = fresh_run_directory(args.output, args.data_root)
    records, views = records_with_views(args.data_root, "train")
    if args.smoke_studies:
        ids = list(dict.fromkeys(row["study_id"] for row in records))[:args.smoke_studies]
        selected = [i for i, row in enumerate(records) if row["study_id"] in set(ids)]
        views, records = [views[i] for i in selected], [records[i] for i in selected]
    model_dir = args.model_dir.resolve()
    encoder = args.encoder
    source_files = list(FILES)
    if encoder == "cxrlt":
        provenance = read_json(model_dir / "provenance.json")
        if (provenance["revision"] != CXRLT_REVISION
                or sha256(model_dir / CXRLT_WEIGHT) != CXRLT_SHA256
                or sha256(model_dir / "model.py") != CXRLT_CODE_SHA256):
            raise ValueError("CXR-LT weight/code revision/hash mismatch")
        model_name, revision = "hieuphamha/cxrlt2026-task1-convnextv2", CXRLT_REVISION
        model_files = [model_dir / name for name in (CXRLT_WEIGHT, "model.py", "README.md", "provenance.json")]
        source_files += ["cxr/cxrlt_transfer.py", "docs/SUPERVISED_STUDY_PROBE_PLAN_20261108.md"]
    else:
        metadata = model_dir / ".cache/huggingface/download/model.safetensors.metadata"
        revision, expected_hash = metadata.read_text().splitlines()[:2]
        if revision != REVISION or sha256(model_dir / "model.safetensors") != expected_hash:
            raise ValueError("CXformer weight revision/hash mismatch")
        model_name = "m42-health/CXformer-base"
        model_files = [model_dir / name for name in ("model.safetensors", "config.json", "preprocessor_config.json", "custom_processor.py")]
    source_hashes = {}
    for relative in source_files:
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
        source_hashes[relative] = sha256(target)
    paths = [args.data_root / f"{split}.csv" for split in ("train", "valid", "test")]
    paths += [args.partition,
             args.old_oof / "oof_component_probabilities.npy", args.old_oof / "oof_study_ids.json",
             args.old_oof / "plan.json", args.old_valid, args.old_test] + model_files
    plan = {
        "created_utc": now(), "data_root": str(args.data_root.resolve()), "model_dir": str(model_dir),
        "encoder": encoder, "model": model_name, "revision": revision,
        "partition": str(args.partition.resolve()), "old_oof": str(args.old_oof.resolve()),
        "old_valid": str(args.old_valid.resolve()), "old_test": str(args.old_test.resolve()),
        "device": args.device, "batch_size": args.batch_size, "preprocess_workers": args.preprocess_workers,
        "smoke_studies": args.smoke_studies,
        "input_locks": {str(path.resolve()): sha256(path) for path in paths},
        "source_locks": source_hashes, "environment": environment(),
        "fit": {"l2": 0.01, "max_iter": 200, "seed": 20261108},
        "candidates": {"modes": ["mean", "views"], "alphas": [0.2, 1.0]},
        "train_images": len(records), "train_studies": len({row["study_id"] for row in records}),
        "train_patients": len({row["subject_id"] for row in records}),
        "caveat": "Historical validation selection; public-pretraining overlap unknown. OOF used to select candidates, not nested validation.",
    }
    atomic_json(output / "train_records.json", {"records": records, "views": views})
    atomic_json(output / "train_images.json", image_identity(records))
    plan["train_records_sha256"] = sha256(output / "train_records.json")
    plan["train_images_sha256"] = sha256(output / "train_images.json")
    atomic_json(output / "plan.json", plan)
    atomic_json(output / "experiment.json", {"status": "prepared", "phase": None, "created_utc": now()})
    return output


def verify_plan(output, plan):
    if environment() != plan["environment"]:
        raise ValueError("software environment changed")
    for name, digest in plan["input_locks"].items():
        if sha256(name) != digest:
            raise ValueError(f"registered input changed: {name}")
    for name, digest in plan["source_locks"].items():
        if sha256(output / "source" / name) != digest:
            raise ValueError(f"source snapshot changed: {name}")
    for name in ("train_records", "train_images"):
        if sha256(output / f"{name}.json") != plan[f"{name}_sha256"]:
            raise ValueError(f"registered {name} changed")
    verify_images(read_json(output / "train_images.json"))


def old_oof_predictions(plan, studies):
    directory = Path(plan["old_oof"])
    ids = read_json(directory / "oof_study_ids.json")
    expected = [[row["subject_id"], row["study_id"]] for row in studies]
    if ids != expected:
        raise ValueError("legacy OOF study order differs")
    components = np.load(directory / "oof_component_probabilities.npy", allow_pickle=False)
    prior = np.asarray(read_json(directory / "plan.json")["prior"])
    if components.shape != (len(studies), 8, 10) or prior.shape != (8, 10):
        raise ValueError("unexpected legacy OOF input shape")
    if np.any(~np.isfinite(components) & (prior[None] != 0)):
        raise ValueError("legacy prior depends on unavailable head")
    return (np.where(np.isfinite(components), components, 0) * prior[None]).sum(axis=1)


def run(output):
    from scripts.evaluate_frozen_study_probe import train_candidates, run_validation

    plan = read_json(output / "plan.json")
    state = read_json(output / "experiment.json")
    verify_plan(output, plan)
    if state["status"] == "complete":
        print(json.dumps(state), flush=True)
        return
    def stage(name):
        state.update(status="running", phase=name, updated_utc=now(), pid=os.getpid(), error=None)
        atomic_json(output / "experiment.json", state)
    torch.set_num_threads(4)
    rows = read_json(output / "train_records.json")
    records, views = rows["records"], rows["views"]
    try:
        stage("extract_train")
        model, processor, device = load_encoder(plan)
        features = extract_features(model, processor, records, output / "train_features",
                                    batch_size=plan["batch_size"], device=device,
                                    preprocess_workers=plan["preprocess_workers"])
        studies, _, _ = pool_study_features(records, features, views, "mean")
        if plan["smoke_studies"]:
            _, view_features, _ = pool_study_features(records, features, views, "views")
            # Independently reload features and require exact pooling reproduction.
            reloaded = np.load(output / "train_features/image_features.npy", allow_pickle=False)
            assert np.array_equal(features, reloaded)
            atomic_json(output / "smoke_verification.json", {
                "status": "passed", "images": len(records), "studies": len(studies),
                "image_shape": list(features.shape), "view_study_shape": list(view_features.shape),
                "frozen_encoder": not any(p.requires_grad for p in model.parameters()),
                "cache_reload_exact": True, "formal_training": False,
            })
            state.update(status="complete", phase=None, updated_utc=now(), smoke=True)
            atomic_json(output / "experiment.json", state)
            return
        del model, processor
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        stage("train_patient_grouped_heads")
        assignment = read_json(plan["partition"])["subject_partition"]
        recipe = train_candidates(output, records, features, views, assignment, old_oof_predictions(plan, studies))
        if recipe is None:
            state.update(status="complete", phase=None, updated_utc=now(), decision="retain old; train OOF gate failed", test_csv_exported=False)
            atomic_json(output / "experiment.json", state)
            return
        stage("validate_locked_candidate")
        model, processor, device = load_encoder(plan)
        valid, valid_views = records_with_views(Path(plan["data_root"]), "valid")
        valid_features = extract_features(model, processor, valid, output / "valid_features",
                                         batch_size=plan["batch_size"], device=device,
                                         preprocess_workers=plan["preprocess_workers"])
        valid_studies, _, _ = pool_study_features(valid, valid_features, valid_views, recipe["mode"])
        old_valid, _, _ = load_submission(plan["old_valid"], valid_studies)
        report = run_validation(output, valid, valid_features, valid_views, old_valid, recipe)
        promoted = bool(report.get("promoted", report.get("candidate_promoted", False)))
        if promoted:
            stage("predict_test")
            test, test_views = records_with_views(Path(plan["data_root"]), "test")
            test_features = extract_features(model, processor, test, output / "test_features",
                                            batch_size=plan["batch_size"], device=device,
                                            preprocess_workers=plan["preprocess_workers"])
            test_studies, test_x, _ = pool_study_features(test, test_features, test_views, recipe["mode"])
            old_test, _, _ = load_submission(plan["old_test"], test_studies)
            with np.load(output / "final_head.npz", allow_pickle=False) as saved:
                head = dict(saved)
            p = predict_linear_probe(head, test_x)
            probabilities = (1 - recipe["alpha"]) * old_test + recipe["alpha"] * p
            tag = plan.get("encoder", "cxformer")
            csv_path = output / f"submission_test_1414_{tag}_study_20261108.csv"
            write_submission(csv_path, test_studies, probabilities)
            verification = validate_submission(csv_path, test_studies)
            replay, _, _ = load_submission(csv_path, test_studies)
            if not np.array_equal(replay, probabilities):
                raise ValueError("test CSV replay differs")
            atomic_json(output / "submission_verification.json", {**verification, "sha256": sha256(csv_path), "online_score": None})
        verify_plan(output, plan)
        state.update(status="complete", phase=None, updated_utc=now(), candidate_promoted=promoted, test_csv_exported=promoted)
        atomic_json(output / "experiment.json", state)
    except Exception as error:
        state.update(status="failed", updated_utc=now(), error=repr(error), traceback=traceback.format_exc())
        atomic_json(output / "experiment.json", state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(r"G:\cxr_first(1)\cxr_first"))
    parser.add_argument("--encoder", choices=("cxformer", "cxrlt"), default="cxformer")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--partition", type=Path, default=ROOT / "outputs/generalization-rad-dino-20261105-v2/partition.json")
    parser.add_argument("--old-oof", type=Path, default=ROOT / "outputs/legacy-full-oof-20261107")
    parser.add_argument("--old-valid", type=Path, default=ROOT / "outputs/best-ensemble-improvement-20261105/valid.csv")
    parser.add_argument("--old-test", type=Path, default=ROOT / "outputs/submission_test_1414_best_20261105.csv")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--preprocess-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-studies", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.model_dir is None:
        folder = "cxrlt2026-convnextv2" if args.encoder == "cxrlt" else "cxformer-base"
        args.model_dir = ROOT / "weights/medical" / folder
    if args.batch_size < 1 or args.smoke_studies < 0 or not 0 <= args.preprocess_workers <= 4:
        parser.error("batch-size must be positive, smoke-studies nonnegative, preprocess-workers 0..4")
    output = safe_output(args.output, args.data_root)
    if not args.resume:
        output = prepare(args)
    if args.prepare_only:
        print(json.dumps({"status": "prepared", "output": str(output)}), flush=True)
        return
    # Run only the frozen source tree, so later workspace edits cannot change this experiment.
    snapshot = output / "source"
    if ROOT != snapshot:
        import subprocess
        command = [sys.executable, "-u", str(snapshot / "scripts/run_frozen_study_probe.py"), "--output", str(output), "--resume"]
        raise SystemExit(subprocess.call(command))
    run(output)


if __name__ == "__main__":
    main()
