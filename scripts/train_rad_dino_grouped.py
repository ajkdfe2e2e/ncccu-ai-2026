"""Fixed-budget Rad-DINO training on explicit patient-disjoint train subsets.

Only train.csv is read. The partition determines an inner CV split, the final
development/audit split, or a deployment refit on all training patients. Full
refits have no validation score and cannot provide out-of-fold predictions.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cxr.data import read_records
from cxr.metrics import aggregate_studies, evaluate
from cxr.optimization import ModelEMA, accumulation_window_images
from cxr.patient_splits import audit_records, partition_records
from cxr.safety import fresh_run_directory, safe_output
from cxr.training_state import atomic_save, capture_rng, restore_rng, safe_load
from scripts.run_aggressive_experiment import atomic_json
from scripts.run_rad_dino_finetune import (
    ImageRecords, LOCAL_LABELS, collate, initialize_model, sha256,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--partition", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--phase", choices=("cv", "audit", "full"), default="cv")
    p.add_argument("--resume", action="store_true", help="resume a full refit at an epoch boundary")
    p.add_argument("--fold", type=int)
    p.add_argument("--head-lr", type=float, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accumulation-steps", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--evaluation-variants", nargs="+", choices=("raw", "ema"),
                   default=("raw", "ema"))
    p.add_argument("--device", default="cuda")
    return p


def validate_config(cfg):
    if (cfg.epochs < 1 or cfg.batch_size < 1 or cfg.accumulation_steps < 1
            or cfg.workers < 0 or cfg.seed < 0):
        raise ValueError("invalid training sizes, workers, or seed")
    if not math.isfinite(cfg.head_lr) or cfg.head_lr <= 0:
        raise ValueError("head-lr must be finite and positive")
    if not math.isfinite(cfg.ema_decay) or not 0 <= cfg.ema_decay < 1:
        raise ValueError("ema-decay must be finite and in [0, 1)")
    if (cfg.phase == "cv") != (cfg.fold is not None):
        raise ValueError("a fold is required only for CV")
    if getattr(cfg, "resume", False) and cfg.phase != "full":
        raise ValueError("resume is available only for a full refit")
    if cfg.phase == "full" and cfg.workers != 0:
        raise ValueError("full refits require workers=0 for epoch-boundary resume")


def select_records(records, partition, phase, fold):
    if phase == "full" and fold is None:
        subjects = {row["subject_id"] for row in records}
        if not records or subjects != set(partition["subject_partition"]):
            raise ValueError("full refit must cover all locked training patients")
        return list(records), []
    if phase == "cv":
        train, held = partition_records(records, partition, fold)
    elif phase == "audit" and fold is None:
        train, held = audit_records(records, partition)
    else:
        raise ValueError("invalid phase/fold combination")
    train_subjects = {row["subject_id"] for row in train}
    held_subjects = {row["subject_id"] for row in held}
    if not train or not held or train_subjects & held_subjects:
        raise ValueError("training and evaluation patients must be nonempty and disjoint")
    return train, held


def restore_full_refit(path, config, model, ema, optimizer, scheduler, scaler, generator):
    state = safe_load(path)
    if state.get("kind") != "rad_dino_full_refit_epoch_v1" or state.get("config") != config:
        raise ValueError("full refit resume configuration or provenance differs")
    completed = state.get("completed_epochs")
    history = state.get("history")
    if (isinstance(completed, bool) or not isinstance(completed, int)
            or not 1 <= completed <= config["epochs"]
            or not isinstance(history, list) or len(history) != completed):
        raise ValueError("invalid completed epochs in full refit checkpoint")
    model.load_state_dict(state["model"], strict=True)
    ema.load_state_dict(state["ema"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    restore_rng(state["rng"], generator)
    result = (completed, history, state["optimizer_updates"],
              state["skipped_updates"], state["elapsed_seconds"])
    del state
    gc.collect()
    return result


def save_full_refit(path, config, model, ema, optimizer, scheduler, scaler,
                    generator, history, updates, skipped_updates, elapsed):
    # Serialize live state synchronously without cloning the full EMA on CUDA.
    atomic_save({
        "kind": "rad_dino_full_refit_epoch_v1", "config": config,
        "completed_epochs": len(history), "history": history,
        "model": model.state_dict(),
        "ema": {"decay": ema.decay, "updates": ema.updates,
                "module": ema.module.state_dict()},
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(), "rng": capture_rng(generator),
        "optimizer_updates": updates, "skipped_updates": skipped_updates,
        "elapsed_seconds": elapsed,
    }, path)


def verify_image_inventory(records, inventory):
    for row in records:
        expected = inventory.get(row["dicom_id"])
        path = Path(row["path"])
        stat = path.stat()
        if expected != {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}:
            raise ValueError("input image path, size or mtime differs from the locked inventory")


@torch.inference_mode()
def predict_records(model, records, processor, device, batch_size, workers):
    loader = DataLoader(
        ImageRecords(records, processor, training=False), batch_size=batch_size,
        shuffle=False, num_workers=workers, collate_fn=collate,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
    )
    model.eval()
    predictions = []
    for batch_index, (images, _) in enumerate(loader, 1):
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            logits = model(pixel_values=images.to(device, non_blocking=True))
        predictions.append(logits.float().sigmoid().cpu().numpy())
        if batch_index == 1 or batch_index % 250 == 0:
            print(f"evaluation batch={batch_index}/{len(loader)}", flush=True)
    return aggregate_studies(records, np.concatenate(predictions))


def save_prediction(output, variant, model, held, processor, device, cfg, provenance):
    studies, probabilities, targets = predict_records(
        model, held, processor, device, cfg.batch_size, cfg.workers,
    )
    metrics = evaluate(targets, probabilities)
    if metrics["macro_auc_defined_classes"] != 10:
        raise ValueError("all ten AUCs must be defined")
    folder = output / variant
    folder.mkdir()
    np.save(folder / "study_probabilities.npy", probabilities)
    np.save(folder / "study_targets.npy", targets)
    atomic_json(folder / "study_ids.json", [
        [row["subject_id"], row["study_id"]] for row in studies
    ])
    torch.save({
        "model": model.state_dict(), "labels": LOCAL_LABELS,
        "epoch": cfg.epochs, "metrics": metrics,
        "processor": str(cfg.model_dir.resolve()), "variant": variant,
        "provenance": provenance,
    }, folder / "last.pt")
    report = {"variant": variant, "metrics": metrics,
              "images": len(held), "studies": len(studies),
              "patients": len({row["subject_id"] for row in held}),
              "checkpoint_sha256": sha256(folder / "last.pt")}
    atomic_json(folder / "report.json", report)
    return report


def main(argv=None):
    cfg = parser().parse_args(argv)
    validate_config(cfg)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    partition = json.loads(cfg.partition.read_text(encoding="utf-8"))
    metadata = cfg.data_root / "train.csv"
    if sha256(metadata) != partition["train_metadata_sha256"]:
        raise ValueError("train.csv differs from the locked partition")
    initial = cfg.model_dir / "pytorch_model.bin"
    if sha256(initial) != partition["public_checkpoint_sha256"]:
        raise ValueError("public initialization differs from the locked partition")
    train, held = select_records(
        read_records(metadata, cfg.data_root / "train"), partition, cfg.phase, cfg.fold,
    )
    inventory_path = Path(partition["image_inventory_path"])
    if sha256(inventory_path) != partition["image_inventory_sha256"]:
        raise ValueError("locked image inventory changed")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    verify_image_inventory([*train, *held], inventory)
    if cfg.resume:
        output = safe_output(cfg.output, cfg.data_root).resolve(strict=True)
        if not (output / "resume.pt").is_file():
            raise ValueError("full refit epoch-boundary resume.pt is missing")
    else:
        output = fresh_run_directory(cfg.output, cfg.data_root)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA is unavailable")
    processor, model = initialize_model(cfg.model_dir.resolve(), device)
    ema = ModelEMA(model, decay=cfg.ema_decay)
    train_labels = np.asarray([row["labels"] for row in train], dtype=np.float64)
    positives = train_labels.sum(axis=0)
    if ((positives == 0) | (positives == len(train_labels))).any():
        raise ValueError("training split requires both outcomes for all ten labels")
    pos_weight = torch.tensor(
        (len(train_labels) - positives) / positives, dtype=torch.float32, device=device,
    )
    dataset = ImageRecords(train, processor, training=True)
    generator = torch.Generator().manual_seed(cfg.seed) if cfg.phase == "full" else None
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers,
        collate_fn=collate, pin_memory=device.type == "cuda",
        persistent_workers=cfg.workers > 0,
        generator=generator,
    )
    backbone = [p for layer in model.dinov2.encoder.layer[-2:]
                for p in layer.parameters() if p.requires_grad]
    backbone += [p for p in model.dinov2.layernorm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": 1e-5},
        {"params": model.classifier.parameters(), "lr": cfg.head_lr},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    provenance = {
        "phase": cfg.phase, "fold": cfg.fold, "seed": cfg.seed,
        "head_lr": cfg.head_lr, "backbone_lr": 1e-5, "weight_decay": 1e-4,
        "epochs": cfg.epochs, "checkpoint_policy": "fixed_final_epoch",
        "batch_size": cfg.batch_size, "accumulation_steps": cfg.accumulation_steps,
        "ema_decay": cfg.ema_decay, "train_images": len(train),
        "train_patients": len({row["subject_id"] for row in train}),
        "evaluation_images": len(held),
        "evaluation_patients": len({row["subject_id"] for row in held}),
        "train_metadata_sha256": sha256(metadata),
        "partition_sha256": sha256(cfg.partition),
        "public_checkpoint_sha256": sha256(initial),
        "official_valid_read": False, "test_read": False,
    }
    if cfg.phase == "full":
        provenance.update(purpose="deployment_refit", independent_validation_metrics=None,
                          eligible_for_oof=False, includes_former_audit_patients=True)
    resume_config = {
        **provenance, "workers": cfg.workers, "device": str(device),
        "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "python_version": sys.version, "cuda_version": str(torch.version.cuda),
    }
    report = {"status": "training", **provenance, "history": []}
    first_epoch = 0
    updates = skipped_updates = 0
    previous_elapsed = 0.0
    if cfg.resume:
        (first_epoch, report["history"], updates, skipped_updates,
         previous_elapsed) = restore_full_refit(
            output / "resume.pt", resume_config, model, ema, optimizer,
            scheduler, scaler, generator,
        )
        report["resumed_from_completed_epoch"] = first_epoch
    atomic_json(output / "report.json", report)
    print(json.dumps(provenance), flush=True)
    started = time.monotonic()
    for epoch in range(first_epoch, cfg.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        seen = 0
        for index, (images, labels) in enumerate(loader):
            count = len(images)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            window = accumulation_window_images(
                index, cfg.batch_size, len(dataset), cfg.accumulation_steps,
            )
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                loss = loss_fn(model(pixel_values=images), labels)
            scaler.scale(loss * count / window).backward()
            total_loss += float(loss.detach()) * count
            seen += count
            if (index + 1) % cfg.accumulation_steps == 0 or index + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    ema.update(model)
                    updates += 1
                else:
                    skipped_updates += 1
                optimizer.zero_grad(set_to_none=True)
            if index == 0 or (index + 1) % 100 == 0:
                elapsed = previous_elapsed + time.monotonic() - started
                item = {"epoch": epoch + 1, "epochs": cfg.epochs,
                        "batch": index + 1, "batches": len(loader),
                        "loss": total_loss / seen, "elapsed_seconds": elapsed,
                        "optimizer_updates": updates, "skipped_updates": skipped_updates}
                atomic_json(output / "progress.json", item)
                print(json.dumps(item), flush=True)
        scheduler.step()
        report["history"].append({"epoch": epoch + 1, "loss": total_loss / seen})
        if cfg.phase == "full":
            save_full_refit(
                output / "resume.pt", resume_config, model, ema, optimizer,
                scheduler, scaler, generator, report["history"], updates,
                skipped_updates, previous_elapsed + time.monotonic() - started,
            )
        atomic_json(output / "report.json", report)
    if updates == 0:
        raise RuntimeError("no successful optimizer updates")
    report["status"] = ("saving_fixed_final_weights" if cfg.phase == "full"
                        else "evaluating_fixed_final_weights")
    report["optimizer_updates"] = updates
    report["skipped_updates"] = skipped_updates
    atomic_json(output / "report.json", report)
    report["variants"] = {}
    for name in cfg.evaluation_variants:
        evaluated_model = model if name == "raw" else ema.module
        if cfg.phase == "full":
            folder = output / name
            folder.mkdir(exist_ok=True)
            checkpoint = folder / "last.pt"
            atomic_save({
                "model": evaluated_model.state_dict(), "labels": LOCAL_LABELS,
                "epoch": cfg.epochs, "metrics": None,
                "processor": str(cfg.model_dir.resolve()), "variant": name,
                "provenance": provenance,
            }, checkpoint)
            report["variants"][name] = {
                "variant": name, "metrics": None,
                "checkpoint_sha256": sha256(checkpoint),
            }
        else:
            report["variants"][name] = save_prediction(
                output, name, evaluated_model, held, processor, device, cfg, provenance,
            )
        atomic_json(output / "report.json", report)
    report["status"] = "complete"
    report["elapsed_seconds"] = previous_elapsed + time.monotonic() - started
    atomic_json(output / "report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
