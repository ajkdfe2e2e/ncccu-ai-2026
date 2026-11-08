"""Run a fixed three-epoch TorchXRayVision CheX transfer experiment.

The outputs are research artifacts and are deliberately incompatible with the
competition Predictor checkpoint format. Validation is used for model selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cxr.data import read_records
from cxr.engine import seed_everything
from cxr.medical_transfer import (TARGET_NAMES, WEIGHTS_NAME, SOURCE_INDICES,
                                  MedicalDataset, build_model, predict_records)
from cxr.metrics import _rank_auc, aggregate_studies, evaluate
from cxr.optimization import accumulation_window_images
from cxr.safety import fresh_run_directory, safe_output
from cxr.submission import write_submission


TOTAL_EPOCHS = 3
FROZEN_EPOCHS = 1
MICRO_BATCH_SIZE = 2
ACCUMULATION_STEPS = 8
CLASSIFIER_LR = 1e-3
FEATURES_LR = 1e-5
FINETUNE_CLASSIFIER_LR = 1e-4


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _atomic_json(path, value):
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


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_stage(model, frozen):
    """Set trainability and keep frozen DenseNet BatchNorm statistics fixed."""
    for parameter in model.features.parameters():
        parameter.requires_grad_(not frozen)
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)
    model.train()
    if frozen:
        model.features.eval()


def make_optimizer(model, frozen):
    if frozen:
        groups = [{"params": model.classifier.parameters(), "lr": CLASSIFIER_LR, "name": "classifier"}]
    else:
        groups = [
            {"params": model.features.parameters(), "lr": FEATURES_LR, "name": "features"},
            {"params": model.classifier.parameters(), "lr": FINETUNE_CLASSIFIER_LR, "name": "classifier"},
        ]
    return torch.optim.AdamW(groups, weight_decay=1e-4)


def align_and_fuse(studies, reference_studies, probabilities, reference_probabilities):
    """Align by (subject, study), then apply the fixed 0.5/0.5 probability mean."""
    studies, reference_studies = list(studies), list(reference_studies)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    reference_probabilities = np.asarray(reference_probabilities, dtype=np.float64)
    if probabilities.shape != (len(studies), 10) or reference_probabilities.shape != (len(reference_studies), 10):
        raise ValueError("both probability arrays must have one ten-class row per study")
    reference = {}
    for row, probability in zip(reference_studies, reference_probabilities):
        key = (str(row["subject_id"]), str(row["study_id"]))
        if key in reference:
            raise ValueError(f"duplicate reference study {key!r}")
        reference[key] = probability
    keys = [(str(row["subject_id"]), str(row["study_id"])) for row in studies]
    if len(keys) != len(set(keys)) or set(keys) != set(reference):
        raise ValueError("medical and reference studies must have identical unique IDs")
    aligned = np.stack([reference[key] for key in keys])
    return (probabilities + aligned) / 2.0


def zero_shot_report(targets, zero_shot, baseline, names=TARGET_NAMES):
    """Compare only the nine mapped CheX classes; No Finding is not reported."""
    targets = np.asarray(targets, dtype=np.float64)
    zero_shot = np.asarray(zero_shot, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 10 or zero_shot.shape != targets.shape or baseline.shape != targets.shape:
        raise ValueError("targets and probability arrays must have shape (N, 10)")
    classes = []
    defined = []
    for index in range(9):
        auc = _rank_auc(targets[:, index], zero_shot[:, index])
        baseline_auc = _rank_auc(targets[:, index], baseline[:, index])
        if auc is not None:
            defined.append(auc)
        classes.append({"name": names[index], "target_index": index,
                        "zero_shot_auc": auc, "baseline_auc": baseline_auc,
                        "delta": None if auc is None or baseline_auc is None else auc - baseline_auc})
    return {
        "classes_evaluated": 9,
        "excluded_class": names[9],
        "pretrained_source_indices": list(SOURCE_INDICES[:9]),
        "macro_auc": float(np.mean(defined)) if defined else None,
        "macro_auc_defined_classes": len(defined),
        "per_class": classes,
        "definition": "Validation study-level AUC for mapped CheX labels only; No Finding has no pretrained head.",
    }


class _IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, labels = self.dataset[index]
        return image, labels, index


def _study_subset(records, limit):
    if not limit:
        return records
    study_ids = list(dict.fromkeys(row["study_id"] for row in records))[:limit]
    selected = set(study_ids)
    return [row for row in records if row["study_id"] in selected]


def _class_weights(records):
    counts = Counter(row["study_id"] for row in records)
    image_weights = torch.tensor([1.0 / counts[row["study_id"]] for row in records], dtype=torch.float32)
    unique = {row["study_id"]: row for row in records}
    labels = np.asarray([row["labels"] for row in unique.values()], dtype=np.float64)
    positives = labels.sum(axis=0)
    pos_weight = np.clip((len(labels) - positives) / np.maximum(positives, 1), 1, 10)
    return image_weights, torch.as_tensor(pos_weight, dtype=torch.float32)


def _evaluate(model, records, device, batch_size):
    probabilities = predict_records(model, records, device, batch_size=batch_size)
    studies, study_probabilities, targets = aggregate_studies(records, probabilities)
    return studies, study_probabilities, targets, evaluate(targets, study_probabilities, thresholds=0.5)


def _save_torch(path, payload):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_baseline(path, expected_studies, targets):
    from scripts.compare_runs import load_submission

    baseline_studies, baseline_targets = expected_studies, targets
    probabilities, _, _ = load_submission(path, baseline_studies)
    if len(probabilities) != len(baseline_studies):
        raise ValueError("baseline validation rows do not match expected studies")
    return probabilities


def run(args):
    seed_everything(2026)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if not args.pretrained.is_file():
        raise FileNotFoundError(f"Pretrained XRV state_dict not found: {args.pretrained}")
    output = fresh_run_directory(args.output, args.data_root)
    state = {
        "status": "running", "stage": "initializing", "started_at": _utc_now(),
        "model": "TorchXRayVision DenseNet121 densenet121-res224-chex",
        "weights_name": WEIGHTS_NAME, "pretrained_source_indices": list(SOURCE_INDICES),
        "target_names": TARGET_NAMES, "pretrained_path": str(args.pretrained.resolve()),
        "pretrained_sha256": _sha256(args.pretrained), "research_only": True,
        "total_epochs": TOTAL_EPOCHS, "seed": 2026,
    }
    _atomic_json(output / "experiment.json", state)

    try:
        train_all = read_records(args.data_root / "train.csv", args.data_root / "train")
        valid_all = read_records(args.data_root / "valid.csv", args.data_root / "valid")
        if not train_all or not valid_all or any(row["labels"] is None for row in train_all + valid_all):
            raise ValueError("train and valid manifests must be nonempty and labeled")
        if {row["subject_id"] for row in train_all} & {row["subject_id"] for row in valid_all}:
            state["patient_overlap_warning"] = True
        train = _study_subset(train_all, args.smoke_studies)
        valid = _study_subset(valid_all, args.smoke_studies)
        if len(train) < 2 or not valid:
            raise ValueError("insufficient train or validation images after study selection")
        state.update(train_images=len(train), train_studies=len({row["study_id"] for row in train}),
                     valid_images=len(valid), valid_studies=len({row["study_id"] for row in valid}),
                     full_valid_images=len(valid_all),
                     full_valid_studies=len({row["study_id"] for row in valid_all}))
        _atomic_json(output / "experiment.json", state)

        reference_studies, _, full_targets = aggregate_studies(
            valid_all, np.zeros((len(valid_all), 10), dtype=np.float64))
        baseline_full = _load_baseline(args.baseline, reference_studies, full_targets)
        ref_position = {(str(row["subject_id"]), str(row["study_id"])): i
                        for i, row in enumerate(reference_studies)}
        valid_keys = list(dict.fromkeys((str(row["subject_id"]), str(row["study_id"])) for row in valid))
        select = np.asarray([ref_position[key] for key in valid_keys], dtype=np.int64)
        baseline = baseline_full[select]

        model = build_model(args.pretrained).to(device)
        state["stage"] = "zero_shot_valid"
        _atomic_json(output / "experiment.json", state)
        valid_studies, zero_shot, targets, zero_metrics = _evaluate(model, valid, device, args.batch_size)
        zero_report = zero_shot_report(targets, zero_shot, baseline)
        zero_report["macro_auc_all_ten_classes"] = None
        _atomic_json(output / "zero_shot_9class.json", zero_report)
        print(json.dumps({"stage": "zero_shot_valid", "report": zero_report}, allow_nan=False), flush=True)

        if args.zero_shot_only:
            state.update(status="complete", stage="zero_shot_valid", finished_at=_utc_now(),
                         zero_shot_report="zero_shot_9class.json")
            _atomic_json(output / "experiment.json", state)
            return 0

        image_weights, pos_weight = _class_weights(train)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device), reduction="none")
        indexed = _IndexedDataset(MedicalDataset(train, training=True, cache_root=args.image_cache))
        generator = torch.Generator().manual_seed(2026)
        loader = DataLoader(indexed, batch_size=args.batch_size, shuffle=True, num_workers=0,
                            generator=generator, pin_memory=device.type == "cuda", drop_last=False)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        best_auc = -float("inf")
        best_payload = None
        history = []
        stop_epoch = min(TOTAL_EPOCHS, args.stop_after_epochs or TOTAL_EPOCHS)
        for epoch in range(stop_epoch):
            frozen = epoch < FROZEN_EPOCHS
            state.update(stage="training_frozen_head" if frozen else f"training_finetune_epoch_{epoch + 1}",
                         current_epoch=epoch + 1)
            _atomic_json(output / "experiment.json", state)
            if epoch in (0, FROZEN_EPOCHS):
                configure_stage(model, frozen)
                optimizer = make_optimizer(model, frozen)
            else:
                model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            started = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for step, (images, labels, indices) in enumerate(loader, 1):
                model.train()
                if frozen:
                    model.features.eval()
                images, labels = images.to(device), labels.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = model(images)
                    per_image = loss_fn(logits, labels).mean(dim=1)
                    weights = image_weights[indices].to(device)
                    loss = (per_image * weights).mean() / image_weights.mean().to(device)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch + 1}, step {step}")
                window_images = accumulation_window_images(step - 1, args.batch_size, len(train), args.accumulation_steps)
                scaler.scale(loss * len(images) / window_images).backward()
                if step % args.accumulation_steps == 0 or step == len(loader):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    previous_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() < previous_scale:
                        state.setdefault("skipped_optimizer_steps", 0)
                        state["skipped_optimizer_steps"] += 1
                    optimizer.zero_grad(set_to_none=True)
                total_loss += float(loss.detach()) * len(images)
                if step % 100 == 0:
                    elapsed = time.perf_counter() - started
                    print(f"epoch={epoch + 1}/{TOTAL_EPOCHS} step={step}/{len(loader)} "
                          f"loss={total_loss / min(step * args.batch_size, len(train)):.6f} "
                          f"images_per_s={min(step * args.batch_size, len(train)) / elapsed:.2f} "
                          f"elapsed_s={elapsed:.1f}", flush=True)

            model.eval()
            evaluated_studies, probabilities, val_targets, metrics = _evaluate(model, valid, device, args.batch_size)
            score = metrics["macro_auc"]
            if score is None:
                raise ValueError("validation macro AUC is undefined for every class")
            row = {"epoch": epoch + 1, "stage": "frozen_head" if frozen else "finetune",
                   "train_loss": total_loss / len(train), "elapsed_s": time.perf_counter() - started,
                   "train_images": len(train), "valid_images": len(valid),
                   "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20
                   if device.type == "cuda" else None,
                   "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups}, **metrics}
            history.append(row)
            with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            if score > best_auc:
                best_auc = score
                best_payload = {
                    "format": "medical-transfer-research-only-v1", "model": "torchxrayvision-densenet121-res224-chex",
                    "weights_name": WEIGHTS_NAME, "class_names": TARGET_NAMES,
                    "epoch": epoch + 1, "validation_macro_auc": score,
                    "smoke": bool(args.smoke_studies),
                    "model_state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                }
                _save_torch(output / "best.pt", best_payload)

            rng_state = {
                "python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
            _save_torch(output / "last.pt", {
                "epoch": epoch + 1, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(), "rng_state": rng_state,
                "best_macro_auc": best_auc, "best_payload": best_payload, "history": history,
                "total_epochs": TOTAL_EPOCHS, "fixed_schedule": "1 frozen-head + 2 full-finetune",
            })
            state.update(stage="training", completed_epoch=epoch + 1, best_validation_macro_auc=best_auc)
            _atomic_json(output / "experiment.json", state)
            print(json.dumps(row, allow_nan=False), flush=True)

        if best_payload is None:
            raise RuntimeError("training completed without a validation-selected checkpoint")
        # Reload the selected on-disk checkpoint into a new model before its final validation pass.
        best_payload = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
        selected = build_model(None)
        selected.load_state_dict(best_payload["model_state"], strict=True)
        selected.to(device).eval()
        selected_studies, medical_probs, final_targets, final_metrics = _evaluate(selected, valid, device, args.batch_size)
        ensemble = align_and_fuse(selected_studies, valid_studies, medical_probs, zero_shot)
        write_submission(output / "valid.csv", selected_studies, ensemble, thresholds=0.5)
        _atomic_json(output / "valid_metrics.json", {
            "selection": "maximum ten-class validation macro AUC over completed fixed-schedule epochs",
            "best_epoch": best_payload["epoch"], "medical_model": final_metrics,
            "ensemble": evaluate(final_targets, ensemble, thresholds=0.5),
            "fusion": "0.5 * medical probability + 0.5 * existing ResNet18 probability",
            "study_count": len(selected_studies), "image_count": len(valid),
            "research_only": True, "zero_shot_9class": zero_report,
        })
        if args.smoke_studies:
            from scripts.compare_runs import compare_arrays
            subset_targets = np.asarray([row["labels"] for row in selected_studies], dtype=np.float64)
            comparison = compare_arrays([row["subject_id"] for row in selected_studies], subset_targets,
                                        baseline, ensemble, bootstrap=500, seed=2026)
            _atomic_json(output / "comparison.json", comparison)
        else:
            from scripts.compare_runs import compare_runs
            compare_runs(args.baseline, output / "valid.csv", args.data_root / "valid.csv",
                         output / "comparison.json", bootstrap=500, seed=2026)
        state.update(status="complete" if stop_epoch == TOTAL_EPOCHS else "stopped_early",
                     stage="complete", completed_epoch=stop_epoch, best_epoch=best_payload["epoch"],
                     finished_at=_utc_now(), artifacts={"best": "best.pt", "last": "last.pt",
                                                        "valid": "valid.csv", "metrics": "valid_metrics.json",
                                                        "comparison": "comparison.json"})
        _atomic_json(output / "experiment.json", state)
        return 0
    except Exception as exc:
        state.update(status="failed", failed_at=_utc_now(), error=f"{type(exc).__name__}: {exc}")
        _atomic_json(output / "experiment.json", state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True, help="Absolute TorchXRayVision CheX weights state_dict")
    parser.add_argument("--baseline", type=Path, required=True, help="Existing ten-class validation submission CSV")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=MICRO_BATCH_SIZE)
    parser.add_argument("--accumulation-steps", type=int, default=ACCUMULATION_STEPS)
    parser.add_argument("--image-cache", type=Path,
                        help="Optional lossless preprocessing cache, stored outside raw data")
    parser.add_argument("--smoke-studies", type=int, default=0)
    parser.add_argument("--stop-after-epochs", type=int, help="Early stop for smoke runs; total schedule remains three epochs")
    parser.add_argument("--zero-shot-only", action="store_true", help="Write nine-class zero-shot comparison and exit")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.accumulation_steps < 1 or args.smoke_studies < 0:
        parser.error("batch size and accumulation steps must be positive; smoke studies must be nonnegative")
    if args.stop_after_epochs is not None and not 1 <= args.stop_after_epochs <= TOTAL_EPOCHS:
        parser.error("--stop-after-epochs must be in 1..3")
    if args.pretrained.resolve() == args.output.resolve() or args.baseline.resolve() == args.output.resolve():
        parser.error("--output must not overwrite pretrained weights or baseline CSV")
    if args.image_cache is not None:
        args.image_cache = safe_output(args.image_cache, args.data_root)
    try:
        return run(args)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
