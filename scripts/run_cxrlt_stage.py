"""One fixed final-stage adaptation, patient OOF gate, then one valid comparison."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cxr.cxrlt_transfer import load_encoder
from cxr.cxrlt_stage_adapter import split_official_model
from cxr.data import read_records
from cxr.safety import fresh_run_directory, safe_output
from cxr.spatial_cache import extract_prefix_cache
from cxr.stage_training import StageTrainingConfig, StudyBatch, train_one_epoch
from cxr.study_probe import pool_study_features
from cxr.submission import write_submission, validate_submission
from cxr.training_state import atomic_save, cpu_snapshot, safe_load
from scripts.compare_runs import compare_arrays, load_submission
from scripts.evaluate_frozen_study_probe import FOLDS, _fold_memberships, _load_head
from scripts.run_bottleneck_study_probe import training_gate, validation_gate, legacy_oof
from scripts.run_frozen_study_probe import (
    atomic_json, read_json, sha256, now, environment, image_identity, verify_images, pixel_batches,
)

SEED = 20261108
FILES = (
    "scripts/run_cxrlt_stage.py", "scripts/run_frozen_study_probe.py",
    "scripts/run_bottleneck_study_probe.py", "scripts/evaluate_frozen_study_probe.py",
    "scripts/compare_runs.py", "cxr/__init__.py", "cxr/cxrlt_transfer.py",
    "cxr/cxrlt_stage_adapter.py", "cxr/spatial_cache.py", "cxr/stage_training.py",
    "cxr/training_state.py", "cxr/study_probe.py", "cxr/bottleneck_probe.py",
    "cxr/data.py", "cxr/safety.py", "cxr/submission.py", "cxr/metrics.py",
    "docs/CXRLT_STAGE_FINETUNE_PLAN_20261108.md",
)


def study_layout(records):
    studies, _, targets = pool_study_features(records, np.zeros((len(records), 1)), [""] * len(records), "mean")
    positions = {(row["subject_id"], row["study_id"]): i for i, row in enumerate(studies)}
    groups = [[] for _ in studies]
    for i, row in enumerate(records):
        groups[positions[(row["subject_id"], row["study_id"])]].append(i)
    if targets is not None and np.any((targets != 0) & (targets != 1)):
        raise ValueError("study targets must be binary")
    return studies, groups, targets


def batch_reader(cache, groups, targets):
    def fetch(indices):
        indices = list(indices)
        image_rows = [row for study in indices for row in groups[study]]
        mapping = [local for local, study in enumerate(indices) for _ in groups[study]]
        x = np.asarray(cache[image_rows]).copy()
        y = np.zeros((len(indices), 10), dtype=np.float32) if targets is None else targets[indices].astype(np.float32)
        return StudyBatch(torch.from_numpy(x), torch.tensor(mapping, dtype=torch.long),
                          torch.from_numpy(y), torch.tensor(indices, dtype=torch.long))
    return fetch


def fold_indices(studies, partition, fold):
    memberships = _fold_memberships(studies, partition)
    held, train = np.flatnonzero(memberships == fold), np.flatnonzero(memberships != fold)
    if not len(held) or not len(train):
        raise ValueError("empty training or held patient group")
    train_subjects = {studies[i]["subject_id"] for i in train}
    held_subjects = {studies[i]["subject_id"] for i in held}
    if train_subjects & held_subjects:
        raise ValueError("training and held patients overlap")
    return train, held


def verify_plan(output, plan):
    if plan["environment"] != environment():
        raise ValueError("software environment differs from frozen plan")
    for path, digest in plan["input_locks"].items():
        if sha256(path) != digest:
            raise ValueError(f"input changed: {path}")
    for relative, digest in plan["source_locks"].items():
        if sha256(output / "source" / relative) != digest:
            raise ValueError(f"source snapshot changed: {relative}")
    if sha256(output / "plan.json") != read_json(output / "plan_lock.json")["sha256"]:
        raise ValueError("frozen stage plan changed")


def prepare(args):
    source = args.source.resolve()
    source_plan = read_json(source / "plan.json")
    source_state = read_json(source / "experiment.json")
    if not (source_state.get("status") == "complete" and source_state.get("candidate_promoted")):
        raise ValueError("source must be the completed promoted incumbent")
    smoke = read_json(args.smoke)
    if (smoke.get("status") != "complete" or smoke.get("real_candidate_fits") != 0
            or not smoke.get("old_image_cache_exact_equal") or not smoke.get("split_features_exact_equal")
            or smoke["adapter_sha256"] != sha256(ROOT / "cxr/cxrlt_stage_adapter.py")
            or smoke["source_head_sha256"] != sha256(source / "final_head.npz")
            or smoke["source_plan_sha256"] != sha256(source / "plan.json")):
        raise ValueError("engineering smoke provenance or parity check failed")
    data_root = Path(source_plan["data_root"])
    output = fresh_run_directory(args.output, data_root)
    cache_root = safe_output(args.cache_root, data_root).resolve()
    if cache_root.is_relative_to(output) or cache_root == output or cache_root.is_relative_to(source):
        raise ValueError("new spatial cache must have an independent directory")
    config = StageTrainingConfig(microbatch_studies=smoke["selected_microbatch_studies"], device="cuda", amp=True)
    config.validate()
    input_paths = [Path(name) for name in source_plan["input_locks"]]
    input_paths += [source / name for name in (
        "plan.json", "experiment.json", "report.json", "recipe.json", "final_head.npz", "train_records.json",
        "train_images.json", "oof_study_ids.json", "oof_targets.npy", "oof_mean_probabilities.npy", "candidate_valid.csv",
    )]
    input_paths += [source / "mean" / str(fold) / "head.npz" for fold in FOLDS]
    input_paths += [args.smoke.resolve()]
    for split in ("train", "valid", "test"):
        input_paths += [source / f"{split}_features" / name for name in ("input.json", "progress.json", "image_features.npy")]
    locks = {str(path.resolve()): sha256(path) for path in input_paths}
    for name, digest in source_plan["input_locks"].items():
        if locks[str(Path(name).resolve())] != digest:
            raise ValueError("incumbent registered input changed")
    source_report = read_json(source / "report.json")
    train_records = read_json(source / "train_records.json")["records"]
    cached_features = np.load(source / "train_features/image_features.npy", mmap_mode="r", allow_pickle=False)
    train_studies, pooled, _ = pool_study_features(train_records, cached_features, [""] * len(train_records), "mean")
    partition = read_json(source_plan["partition"])["subject_partition"]
    initialization_checks = {}
    for fold in FOLDS:
        path = source / "mean" / str(fold) / "head.npz"
        if sha256(path) != source_report["folds"]["mean"][str(fold)]["head_sha256"]:
            raise ValueError("CV head initialization differs from its source fold")
        train, held = fold_indices(train_studies, partition, fold)
        payload = _load_head(path)
        expected_mean, expected_std = pooled[train].mean(axis=0), pooled[train].std(axis=0)
        expected_std[expected_std == 0] = 1
        if not np.array_equal(payload["mean"], expected_mean) or not np.array_equal(payload["std"], expected_std):
            raise ValueError("CV head standardization is not derived from its training patients alone")
        initialization_checks[str(fold)] = {"head_sha256": sha256(path), "training_studies": len(train),
                                           "held_studies": len(held), "training_statistics_exact_equal": True,
                                           "patient_intersection": 0}
    del pooled, cached_features
    for split in ("train", "valid", "test"):
        progress = read_json(source / f"{split}_features/progress.json")
        if progress.get("status") != "complete" or locks[str((source / f"{split}_features/image_features.npy").resolve())] != progress["sha256"]:
            raise ValueError("incumbent image cache incomplete or changed")
    sources = {}
    for relative in FILES:
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
        sources[relative] = sha256(destination)
    plan = {
        "created_utc": now(), "source_experiment": str(source), "data_root": str(data_root),
        "model_dir": source_plan["model_dir"], "partition": source_plan["partition"],
        "legacy_oof": source_plan["old_oof"], "legacy_valid": source_plan["old_valid"],
        "cache_root": str(cache_root), "prefix_shape": smoke["prefix_shape"], "prefix_dtype": smoke["prefix_dtype"],
        "extraction_batch_size": 2, "preprocess_workers": 4, "device": "cuda", "seed": SEED,
        "fit": asdict(config), "environment": environment(), "input_locks": locks, "source_locks": sources,
        "selection": "one fixed final-stage candidate, four patient OOF models, max one full-train model",
        "caveat": "Historical adaptive fold/valid reuse; public-pretraining overlap unknown. No online score.",
    }
    atomic_json(output / "plan.json", plan)
    atomic_json(output / "plan_lock.json", {"sha256": sha256(output / "plan.json")})
    atomic_json(output / "initialization_checks.json", initialization_checks)
    atomic_json(output / "experiment.json", {"status": "prepared", "created_utc": now(), "candidate_promoted": False,
                                               "test_csv_exported": False})
    return output


def load_split(plan, split):
    source = Path(plan["source_experiment"])
    data = Path(plan["data_root"])
    records = read_json(source / "train_records.json")["records"] if split == "train" else read_records(data / f"{split}.csv", data / split)
    registered = read_json(source / f"{split}_features/input.json")["images"]
    identity = image_identity(records)
    if identity != registered:
        raise ValueError("image order or identity differs from incumbent cache")
    return records, identity, *study_layout(records)


def spatial_cache(plan, output, split, records, identity, notify):
    folder = Path(plan["cache_root"]) / split
    shape = [len(records), *plan["prefix_shape"]]
    spec = {"shape": shape, "dtype": plan["prefix_dtype"], "images": identity,
            "stage_plan_sha256": sha256(output / "plan.json"), "source_locks": plan["source_locks"],
            "environment": plan["environment"], "batch_size": plan["extraction_batch_size"],
            "amp": True, "model_dir": plan["model_dir"], "pooling": "stem_stages_0_1_2_raw_spatial"}
    adapter, processor = load_encoder(plan["model_dir"], "cpu")
    prefix, tail = split_official_model(adapter.model)
    prefix.to(plan["device"])
    del adapter, tail
    gc.collect()
    batches = lambda start: pixel_batches(records, processor, start, plan["extraction_batch_size"], plan["preprocess_workers"])
    values = extract_prefix_cache(prefix, batches, folder, spec, device=plan["device"], progress_callback=notify)
    verify_images(identity)
    registry_path = output / "cache_registry.json"
    registry = read_json(registry_path) if registry_path.exists() else {}
    item = {"folder": str(folder), "input_sha256": sha256(folder / "input.json"),
            "progress_sha256": sha256(folder / "progress.json"),
            "matrix_sha256": read_json(folder / "progress.json")["sha256"], "shape": shape, "dtype": plan["prefix_dtype"]}
    if split in registry and registry[split] != item:
        raise ValueError("completed spatial cache registration changed")
    registry[split] = item
    atomic_json(registry_path, registry)
    del prefix
    gc.collect()
    torch.cuda.empty_cache()
    return values


def load_tail(plan, head_path):
    torch.manual_seed(plan["seed"])
    random.seed(plan["seed"])
    np.random.seed(plan["seed"])
    adapter, _ = load_encoder(plan["model_dir"], "cpu")
    prefix, tail = split_official_model(adapter.model, _load_head(head_path))
    del prefix, adapter
    gc.collect()
    return tail.to(plan["device"]).eval()


def predict(tail, fetch, indices, device, microbatch):
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(indices), microbatch):
            batch = fetch(indices[start:start + microbatch])
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = tail.study_logits(batch.prefix_features.to(device), batch.image_to_study.to(device), len(batch.targets))
            probabilities.append(logits.float().sigmoid().cpu().numpy().astype(np.float64))
    result = np.concatenate(probabilities)
    if result.shape != (len(indices), 10) or not np.isfinite(result).all():
        raise ValueError("invalid fixed-stage predictions")
    return result


def fit_model(plan, output, name, head_path, indices, fetch, prefix_sha, notify):
    folder = output / "fits" / name
    folder.mkdir(parents=True, exist_ok=True)
    tail = load_tail(plan, head_path)
    config = StageTrainingConfig(**plan["fit"])
    binding = {"plan_sha256": sha256(output / "plan.json"), "initial_head_sha256": sha256(head_path),
               "prefix_cache_sha256": prefix_sha, "fold": name, "device": plan["device"],
               "torch_version": str(torch.__version__)}
    final_path, report_path = folder / "model.pt", folder / "fit.json"
    if report_path.exists():
        report = read_json(report_path)
        if report["binding"] != binding or report["model_sha256"] != sha256(final_path):
            raise ValueError("registered completed stage fit changed")
        final = safe_load(final_path)
        if final["binding"] != binding:
            raise ValueError("saved stage binding changed")
        tail.load_state_dict(final["model"], strict=True)
        return tail
    checkpoint = folder / "training.pt"
    summary = train_one_epoch(tail, indices, fetch, config=config, device=plan["device"], checkpoint_path=checkpoint,
                              resume=checkpoint.exists(), binding=binding, progress_callback=notify)
    if not summary["completed"] or summary["next_study"] != len(indices):
        raise ValueError("one-epoch fit did not finish")
    atomic_save({"model": cpu_snapshot(tail.state_dict()), "binding": binding}, final_path)
    replay = safe_load(final_path)
    if not all(torch.equal(value.cpu(), replay["model"][name]) for name, value in tail.state_dict().items()):
        raise ValueError("saved stage state differs")
    atomic_json(report_path, {"binding": binding, "summary": summary, "model_sha256": sha256(final_path),
                             "checkpoint_sha256": sha256(checkpoint), "training_studies": len(indices),
                             "initialized_from_corresponding_source_fold": name != "full_train"})
    return tail


def run(output):
    plan, state = read_json(output / "plan.json"), read_json(output / "experiment.json")
    verify_plan(output, plan)
    if state["status"] == "complete":
        for name, digest in read_json(output / "artifacts.json").items():
            if sha256(output / name) != digest:
                raise ValueError("completed stage artifact changed")
        print(json.dumps(state), flush=True)
        return

    def stage(phase, **extra):
        state.update(status="running", phase=phase, pid=os.getpid(), error=None, updated_utc=now(), **extra)
        atomic_json(output / "experiment.json", state)
        print(json.dumps({"phase": phase, **extra}), flush=True)

    def finish(decision, promoted=False):
        verify_plan(output, plan)
        artifacts = {str(path.relative_to(output)): sha256(path) for path in output.rglob("*")
                     if path.is_file() and "source" not in path.relative_to(output).parts
                     and path.name not in {"artifacts.json", "experiment.json", "launcher.json"}
                     and not path.name.endswith(".log")}
        atomic_json(output / "artifacts.json", artifacts)
        state.update(status="complete", phase=None, updated_utc=now(), decision=decision,
                     candidate_promoted=promoted, test_csv_exported=promoted, error=None)
        atomic_json(output / "experiment.json", state)

    torch.set_num_threads(4)
    try:
        source = Path(plan["source_experiment"])
        records, identity, studies, groups, targets = load_split(plan, "train")
        keys = [[row["subject_id"], row["study_id"]] for row in studies]
        if keys != read_json(source / "oof_study_ids.json") or not np.array_equal(targets, np.load(source / "oof_targets.npy")):
            raise ValueError("training study layout does not match incumbent")
        stage("extract_train_prefix")
        cache = spatial_cache(plan, output, "train", records, identity,
                              lambda progress: stage("extract_train_prefix", cache_progress={key: value for key, value in progress.items() if key != "blocks"}))
        prefix_sha = read_json(Path(plan["cache_root"]) / "train/progress.json")["sha256"]
        fetch = batch_reader(cache, groups, targets)
        baseline = np.load(source / "oof_mean_probabilities.npy", allow_pickle=False)
        legacy = legacy_oof(plan, studies)
        partition = read_json(plan["partition"])["subject_partition"]
        folds = _fold_memberships(studies, partition)
        probabilities = np.full_like(targets, np.nan)
        for fold in FOLDS:
            train, held = fold_indices(studies, partition, fold)
            stage("train_patient_fold", fold=fold, training_studies=len(train), held_studies=len(held))
            tail = fit_model(plan, output, f"fold_{fold}", source / "mean" / str(fold) / "head.npz", train, fetch, prefix_sha,
                             lambda progress: stage("train_patient_fold", fold=fold, training_progress=progress))
            probabilities[held] = predict(tail, fetch, held.tolist(), plan["device"], plan["fit"]["microbatch_studies"])
            np.save(output / f"held_fold_{fold}.npy", probabilities[held], allow_pickle=False)
            del tail
            gc.collect()
            torch.cuda.empty_cache()
        if not np.isfinite(probabilities).all():
            raise ValueError("OOF predictions do not cover every training study")
        np.save(output / "oof_probabilities.npy", probabilities, allow_pickle=False)
        atomic_json(output / "oof_study_ids.json", keys)
        gate = training_gate(targets, baseline, probabilities, folds, legacy)
        atomic_json(output / "oof_report.json", {"gate": gate, "caveat": plan["caveat"]})
        if not gate["passed"]:
            finish("retain incumbent; final-stage candidate failed fixed training gate")
            return
        stage("fit_full_train")
        tail = fit_model(plan, output, "full_train", source / "final_head.npz", np.arange(len(studies)), fetch, prefix_sha,
                         lambda progress: stage("fit_full_train", training_progress=progress))
        final_path = output / "fits/full_train/model.pt"
        recipe = {"model_sha256": sha256(final_path), "model_file": "fits/full_train/model.pt",
                  "plan_sha256": sha256(output / "plan.json"), "fit": plan["fit"], "study_pooling": "mean", "alpha": 1.0}
        recipe_path = output / "recipe.json"
        if recipe_path.exists() and read_json(recipe_path) != recipe:
            raise ValueError("frozen stage recipe changed")
        atomic_json(recipe_path, recipe)
        # Keep the final train model on CPU while extracting another frozen prefix.
        tail.cpu()
        torch.cuda.empty_cache()
        stage("extract_valid_prefix")
        vr, vi, valid_studies, vg, vy = load_split(plan, "valid")
        valid_cache = spatial_cache(plan, output, "valid", vr, vi,
                                    lambda p: stage("extract_valid_prefix", cache_progress={k: v for k, v in p.items() if k != "blocks"}))
        tail.to(plan["device"]).eval()
        candidate = predict(tail, batch_reader(valid_cache, vg, vy), list(range(len(valid_studies))), plan["device"], plan["fit"]["microbatch_studies"])
        csv_path = output / "candidate_valid.csv"
        write_submission(csv_path, valid_studies, candidate)
        replay, _, _ = load_submission(csv_path, valid_studies)
        if not np.array_equal(candidate, replay):
            raise ValueError("fixed validation CSV replay differs")
        current, _, _ = load_submission(source / "candidate_valid.csv", valid_studies)
        older, _, _ = load_submission(plan["legacy_valid"], valid_studies)
        stage("validate_fixed_candidate")
        comparisons = {name: compare_arrays([row["subject_id"] for row in valid_studies], vy, base, candidate,
                       bootstrap=1000, seed=SEED) for name, base in (("incumbent", current), ("legacy", older))}
        promoted = validation_gate(comparisons)
        atomic_json(output / "validation_report.json", {"comparisons": comparisons, "promoted": promoted,
                    "recipe_sha256": sha256(recipe_path), "candidate_valid_csv_sha256": sha256(csv_path), "online_score": None})
        if not promoted:
            finish("retain incumbent; final-stage candidate failed fixed validation gate")
            return
        tail.cpu()
        torch.cuda.empty_cache()
        stage("extract_test_prefix")
        tr, ti, test_studies, tg, _ = load_split(plan, "test")
        test_cache = spatial_cache(plan, output, "test", tr, ti,
                                   lambda p: stage("extract_test_prefix", cache_progress={k: v for k, v in p.items() if k != "blocks"}))
        tail.to(plan["device"]).eval()
        test = predict(tail, batch_reader(test_cache, tg, None), list(range(len(test_studies))), plan["device"], plan["fit"]["microbatch_studies"])
        test_path = output / "submission_test_1414_cxrlt_stage_20261108.csv"
        write_submission(test_path, test_studies, test)
        verification = validate_submission(test_path, test_studies)
        reloaded, _, _ = load_submission(test_path, test_studies)
        if not np.array_equal(test, reloaded):
            raise ValueError("test CSV does not replay exactly")
        atomic_json(output / "submission_verification.json", {**verification, "sha256": sha256(test_path), "online_score": None})
        finish("promote locally; fixed final-stage validation gate passed", True)
    except Exception as error:
        state.update(status="failed", updated_utc=now(), error=repr(error), traceback=traceback.format_exc())
        atomic_json(output / "experiment.json", state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/cxrlt-supervised-study-probe-20261108")
    parser.add_argument("--smoke", type=Path, default=ROOT / "outputs/cxrlt-stage-engineering-smoke-20261108.json")
    parser.add_argument("--cache-root", type=Path, default=Path("F:/cxr-training-cache/cxrlt-stage-20261108"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = safe_output(args.output)
    if not args.resume:
        output = prepare(args)
    if args.prepare_only:
        print(json.dumps({"status": "prepared", "output": str(output)}), flush=True)
        return
    snapshot = output / "source"
    if ROOT != snapshot:
        import subprocess
        raise SystemExit(subprocess.call([sys.executable, "-u", str(snapshot / "scripts/run_cxrlt_stage.py"), "--output", str(output), "--resume"]))
    run(output)


if __name__ == "__main__":
    main()
