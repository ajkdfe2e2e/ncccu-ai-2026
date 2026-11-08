"""Patient-disjoint fixed-budget ResNet18 training for CV and audit folds."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cxr.data import read_records
from cxr.engine import ImageDataset, Predictor, choose_device, create_model, load_imagenet_backbone, seed_everything
from cxr.image_cache import CachedImageDataset
from cxr.metrics import aggregate_studies, evaluate
from cxr.patient_splits import audit_records, partition_records
from cxr.safety import fresh_run_directory, safe_output
from cxr.training_state import atomic_save, capture_rng, restore_rng, safe_load, cpu_snapshot
from scripts.run_aggressive_experiment import atomic_json

EPOCHS = 5
IMAGE_SIZE = 224
BATCH_SIZE = 8
SEED = 2026
WEIGHT_DECAY = 1e-4
INITIAL_LR = 3e-4
FINETUNE_LR = 1e-4
MIN_LR = 1e-6


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        return image, target, index


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_records(records, partition, phase, fold):
    if phase == "cv":
        if fold is None:
            raise ValueError("CV requires a fold")
        train, held = partition_records(records, partition, fold)
    elif phase == "audit":
        if fold is not None:
            raise ValueError("audit does not accept a fold")
        train, held = audit_records(records, partition)
    else:
        raise ValueError("phase must be cv or audit")
    train_subjects = {row["subject_id"] for row in train}
    held_subjects = {row["subject_id"] for row in held}
    if not train or not held or train_subjects & held_subjects:
        raise ValueError("training and held patients must be nonempty and disjoint")
    return train, held


def smoke_subset(records, limit):
    if not limit:
        return list(records)
    return list(records[:limit])


def study_weights(records):
    counts = Counter(row["study_id"] for row in records)
    image_weights = torch.tensor([1.0 / counts[row["study_id"]] for row in records], dtype=torch.float32)
    unique = {row["study_id"]: row for row in records}
    targets = np.asarray([row["labels"] for row in unique.values()], dtype=np.float64)
    positives = targets.sum(0)
    pos_weight = np.clip((len(targets) - positives) / np.maximum(positives, 1), 1, 10)
    return image_weights, torch.tensor(pos_weight, dtype=torch.float32)


def phase_spec(epoch):
    if not 0 <= epoch < EPOCHS:
        raise ValueError("epoch is outside the fixed five-epoch schedule")
    return ("initial", INITIAL_LR, None) if epoch == 0 else ("finetune", FINETUNE_LR, 4)


def optimizer_step(model, optimizer, scaler):
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return scaler.get_scale() >= scale_before


def verify_image_inventory(records, inventory):
    for row in records:
        expected = inventory.get(row["dicom_id"])
        path = Path(row["path"])
        stat = path.stat()
        if expected != {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}:
            raise ValueError("input image path, size or mtime differs from the locked inventory")


def _optimizer(model, epoch):
    phase, lr, cosine_epochs = phase_spec(epoch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_epochs, eta_min=MIN_LR,
    ) if cosine_epochs else None)
    return phase, optimizer, scheduler


def _save_resume(path, config, epoch, model, optimizer, scheduler, scaler, generator,
                 history, updates, skipped):
    atomic_save({
        "kind": "resnet_grouped_epoch_v1", "config": config, "completed_epochs": epoch + 1,
        "phase": phase_spec(epoch)[0],
        "model": cpu_snapshot(model.state_dict()), "optimizer": cpu_snapshot(optimizer.state_dict()),
        "scheduler": cpu_snapshot(scheduler.state_dict()) if scheduler else None,
        "scaler": cpu_snapshot(scaler.state_dict()), "rng": capture_rng(generator),
        "history": history, "optimizer_updates": updates, "skipped_updates": skipped,
    }, path)


def _restore_resume(path, config, model, optimizer, scheduler, scaler, generator):
    state = safe_load(path)
    if state.get("kind") != "resnet_grouped_epoch_v1" or state.get("config") != config:
        raise ValueError("resume configuration or provenance differs")
    completed = state.get("completed_epochs")
    if not isinstance(completed, int) or isinstance(completed, bool) or not 1 <= completed <= EPOCHS:
        raise ValueError("resume checkpoint has an invalid completed epoch count")
    last_phase, _, _ = phase_spec(completed - 1)
    _, optimizer, scheduler = _optimizer(model, completed - 1)
    if last_phase != state.get("phase") or (scheduler is None) != (state.get("scheduler") is None):
        raise ValueError("resume phase state is inconsistent")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    if scheduler:
        scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    restore_rng(state["rng"], generator)
    return completed, optimizer, scheduler, state["history"], state["optimizer_updates"], state["skipped_updates"]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--partition", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--phase", choices=("cv", "audit"), required=True)
    p.add_argument("--fold", type=int)
    p.add_argument("--pretrained", type=Path, required=True)
    p.add_argument("--cache", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke-images", type=int, default=0)
    return p


def main(argv=None):
    cfg = parser().parse_args(argv)
    if cfg.smoke_images < 0:
        raise ValueError("smoke-images must be nonnegative")
    if (cfg.phase == "cv") != (cfg.fold is not None):
        raise ValueError("a fold is required only for CV")
    data_root = cfg.data_root.resolve(strict=True)
    partition_path = cfg.partition.resolve(strict=True)
    pretrained = cfg.pretrained.resolve(strict=True)
    cache = safe_output(cfg.cache, data_root) if cfg.cache else None
    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    metadata = data_root / "train.csv"
    if sha256(metadata) != partition.get("train_metadata_sha256"):
        raise ValueError("train.csv differs from the locked patient partition")
    records = read_records(metadata, data_root / "train")
    train, held = select_records(records, partition, cfg.phase, cfg.fold)
    inventory_path = Path(partition["image_inventory_path"]).resolve(strict=True)
    if sha256(inventory_path) != partition.get("image_inventory_sha256"):
        raise ValueError("locked image inventory changed")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    verify_image_inventory([*train, *held], inventory)
    train, held = smoke_subset(train, cfg.smoke_images), smoke_subset(held, cfg.smoke_images)
    if not train or not held:
        raise ValueError("training and held subsets must be nonempty")
    smoke = bool(cfg.smoke_images)
    device = choose_device(cfg.device)
    cfg.output = safe_output(cfg.output, data_root, inputs=(pretrained, partition_path))
    if cfg.resume:
        output = cfg.output.resolve(strict=True)
        resume_path = output / "resume.pt"
        if not resume_path.is_file():
            raise ValueError("resume.pt is missing")
    else:
        output = fresh_run_directory(cfg.output, data_root)
        resume_path = output / "resume.pt"

    seed_everything(SEED)
    torch.set_num_threads(4)
    model = create_model()
    load_imagenet_backbone(model, pretrained)
    model.to(device)
    generator = torch.Generator().manual_seed(SEED)
    weights, pos_weight = study_weights(train)
    dataset = CachedImageDataset(train, IMAGE_SIZE, training=True, cache_root=cache) if cache else ImageDataset(train, IMAGE_SIZE, training=True)
    indexed = IndexedDataset(dataset)
    loader = DataLoader(indexed, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                        generator=generator, pin_memory=device.type == "cuda", drop_last=False)
    config = {
        "phase": cfg.phase, "fold": cfg.fold, "seed": SEED, "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE, "epochs": EPOCHS, "weight_decay": WEIGHT_DECAY,
        "schedule": [[INITIAL_LR, 1, "none"], [FINETUNE_LR, 4, "cosine"],
                     [MIN_LR, "cosine_floor"]],
        "train_metadata_sha256": sha256(metadata), "partition_sha256": sha256(partition_path),
        "pretrained_sha256": sha256(pretrained), "smoke_images": cfg.smoke_images,
        "train_subjects": sorted({r["subject_id"] for r in train}),
        "held_subjects": sorted({r["subject_id"] for r in held}), "smoke": smoke,
        "device": str(device), "torch_version": str(torch.__version__),
    }
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    atomic_json(output / "run.json", {"status": "training", **config})
    start_epoch, history, updates, skipped = 0, [], 0, 0
    optimizer = scheduler = None
    if cfg.resume:
        state = safe_load(resume_path)
        completed = state.get("completed_epochs")
        if not isinstance(completed, int) or isinstance(completed, bool) or not 1 <= completed <= EPOCHS:
            raise ValueError("resume checkpoint has an invalid completed epoch count")
        _, optimizer, scheduler = _optimizer(model, completed - 1)
        start_epoch, optimizer, scheduler, history, updates, skipped = _restore_resume(
            resume_path, config, model, optimizer, scheduler, scaler, generator,
        )

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device), reduction="none")
    for epoch in range(start_epoch, EPOCHS):
        phase, _, _ = phase_spec(epoch)
        if epoch == 0 or epoch == 1:
            phase, optimizer, scheduler = _optimizer(model, epoch)
        lr = optimizer.param_groups[0]["lr"]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        epoch_started = time.monotonic()
        updates_before, skipped_before = updates, skipped
        for step, (images, labels, indices) in enumerate(loader, 1):
            if len(images) == 1:
                model.eval()
            else:
                model.train()
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                per_image = loss_fn(model(images), labels).mean(1)
                batch_weights = weights[indices].to(device)
                loss = (per_image * batch_weights).mean() / weights.mean().to(device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch={epoch + 1}, batch={step}")
            scaler.scale(loss).backward()
            if optimizer_step(model, optimizer, scaler):
                updates += 1
            else:
                skipped += 1
            total_loss += float(loss.detach()) * len(images)
            if step % 100 == 0:
                elapsed = time.monotonic() - epoch_started
                report = {"epoch": epoch + 1, "phase": phase, "batch": step,
                          "batches": len(loader), "loss": total_loss / min(step * BATCH_SIZE, len(train)),
                          "lr": lr, "elapsed_seconds": elapsed,
                          "eta_seconds": elapsed / step * (len(loader) - step),
                          "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")}
                atomic_json(output / "progress.json", report)
        if scheduler:
            scheduler.step()
        history.append({"epoch": epoch + 1, "phase": phase,
                        "train_loss": total_loss / len(train), "lr": lr})
        _save_resume(resume_path, config, epoch, model, optimizer, scheduler, scaler,
                     generator, history, updates, skipped)
        atomic_json(output / "progress.json", {
            "epoch": epoch + 1, "phase": phase, "batch": len(loader), "batches": len(loader),
            "epoch_optimizer_updates": updates - updates_before,
            "epoch_skipped_updates": skipped - skipped_before, "status": "epoch_complete",
            "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        })

    if updates == 0:
        raise RuntimeError("no successful optimizer updates")
    model.eval()
    final_path = output / "final.pt"
    atomic_save({"architecture": "resnet18", "class_ids": list(range(1, 11)),
                 "image_size": IMAGE_SIZE, "thresholds": [0.5] * 10,
                 "inference_precision": "fp32", "model": cpu_snapshot(model.state_dict()),
                 "epoch": EPOCHS, "weights_source": "model", "smoke": smoke}, final_path)
    predictor = Predictor(final_path, device=device)
    raw_image_probs = predictor.predict_records(held)
    studies, probabilities, targets = aggregate_studies(held, raw_image_probs)
    metrics = evaluate(targets, probabilities)
    (output / "study_ids.json").write_text(json.dumps([
        [row["subject_id"], row["study_id"]] for row in studies
    ], indent=2) + "\n", encoding="utf-8")
    np.save(output / "study_targets.npy", targets)
    np.save(output / "study_probabilities.npy", probabilities)
    report = {"status": "complete", **config, "train_images": len(train), "held_images": len(held),
              "train_studies": len({r["study_id"] for r in train}),
              "held_studies": len(studies), "optimizer_updates": updates,
              "skipped_updates": skipped, "history": history, "metrics": metrics,
              "final_checkpoint_sha256": sha256(final_path), "official_valid_read": False,
              "test_read": False, "eligible_for_oof": not smoke}
    atomic_json(output / "report.json", report)
    atomic_json(output / "run.json", {**config, "status": "complete",
                                      "final_checkpoint_sha256": report["final_checkpoint_sha256"]})


if __name__ == "__main__":
    main()
