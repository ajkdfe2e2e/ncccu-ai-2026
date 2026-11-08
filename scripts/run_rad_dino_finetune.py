"""Fine-tune the public Rad-DINO CheXpert head on the local ten-label task.

The backbone starts from the public checkpoint. The ten-label classifier is
initialized by copying the matching CheXpert rows, then the first ten DINO
blocks remain frozen while blocks 10-11 and the head adapt to the local labels.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cxr.data import read_records
from cxr.metrics import aggregate_studies, evaluate
from cxr.safety import fresh_run_directory

MODEL_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices",
]
LOCAL_LABELS = [
    "Enlarged Cardiomediastinum", "Pneumothorax", "Consolidation",
    "Pneumonia", "Edema", "Cardiomegaly", "Atelectasis",
    "Lung Opacity", "Pleural Effusion", "No Finding",
]
MODEL_INDICES = tuple(MODEL_LABELS.index(name) for name in LOCAL_LABELS)


class ImageRecords(Dataset):
    def __init__(self, records, processor, training=False):
        self.records = list(records)
        self.processor = processor
        self.training = training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(row["path"]) as image:
            image = image.convert("RGB")
            pixels = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        if self.training and random.random() < 0.5:
            # Apply the mild affine after fixed-size processing, avoiding a
            # costly full-resolution PIL rotation on multi-megapixel images.
            angle = random.uniform(-5.0, 5.0) * np.pi / 180.0
            theta = pixels.new_tensor([[np.cos(angle), -np.sin(angle), 0.0],
                                       [np.sin(angle), np.cos(angle), 0.0]])
            grid = torch.nn.functional.affine_grid(theta.unsqueeze(0), (1, *pixels.shape), align_corners=False)
            pixels = torch.nn.functional.grid_sample(pixels.unsqueeze(0), grid, mode="bilinear",
                                                    padding_mode="zeros", align_corners=False)[0]
        return pixels, torch.tensor(row["labels"], dtype=torch.float32)


def collate(batch):
    images, labels = zip(*batch)
    return torch.stack(images), torch.stack(labels)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path(r"G:\cxr_first(1)\cxr_first"))
    p.add_argument("--model-dir", type=Path, default=ROOT / "weights/medical/rad-dino-chexpert-upstream")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/rad-dino-finetune-20261105")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accumulation-steps", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--train-limit", type=int, default=0)
    p.add_argument("--valid-limit", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def initialize_model(model_dir: Path, device: torch.device):
    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True, use_safetensors=False)
    old = model.classifier
    new = nn.Linear(old.in_features, 10)
    with torch.no_grad():
        new.weight.copy_(old.weight.detach()[list(MODEL_INDICES)])
        new.bias.copy_(old.bias.detach()[list(MODEL_INDICES)])
    model.classifier = new
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for layer in model.dinov2.encoder.layer[-2:]:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
    for parameter in model.dinov2.layernorm.parameters():
        parameter.requires_grad_(True)
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)
    model.to(device)
    return processor, model


@torch.inference_mode()
def predict(model, records, processor, device, batch_size):
    loader = DataLoader(ImageRecords(records, processor, training=False), batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate, pin_memory=device.type == "cuda")
    output = []
    model.eval()
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(pixel_values=images)
        output.append(logits.float().sigmoid().cpu().numpy())
    return np.concatenate(output, axis=0)


def main():
    cfg = args()
    if cfg.epochs < 1 or cfg.batch_size < 1 or cfg.accumulation_steps < 1:
        raise ValueError("epochs, batch-size, and accumulation-steps must be positive")
    cfg.output = fresh_run_directory(cfg.output, cfg.data_root)
    random.seed(2026); np.random.seed(2026); torch.manual_seed(2026)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(2026)
    device = torch.device(cfg.device)
    train = read_records(cfg.data_root / "train.csv", cfg.data_root / "train")
    valid = read_records(cfg.data_root / "valid.csv", cfg.data_root / "valid")
    if cfg.train_limit:
        train = train[:cfg.train_limit]
    if cfg.valid_limit:
        valid = valid[:cfg.valid_limit]
    processor, model = initialize_model(cfg.model_dir.resolve(), device)
    train_labels = np.asarray([row["labels"] for row in train], dtype=np.float64)
    positives = train_labels.sum(axis=0)
    pos_weight = torch.tensor((len(train_labels) - positives) / np.maximum(positives, 1.0), dtype=torch.float32, device=device)
    dataset = ImageRecords(train, processor, training=True)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers,
                        collate_fn=collate, pin_memory=device.type == "cuda",
                        persistent_workers=cfg.workers > 0, prefetch_factor=2 if cfg.workers > 0 else None)
    backbone_params = [p for p in model.dinov2.encoder.layer[-2:].parameters() if p.requires_grad]
    backbone_params += [p for p in model.dinov2.layernorm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 1e-5},
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    cfg.output.mkdir(parents=True, exist_ok=True)
    checkpoint = cfg.model_dir.resolve() / "pytorch_model.bin"
    report = {"status": "running", "model": "yaLLMan/rad-dino-chexpert",
              "model_revision": "db02e1b7234dd83c6d7c4485963ef5b22df9e5db",
              "checkpoint_sha256": sha256(checkpoint), "model_indices": MODEL_INDICES,
              "local_labels": LOCAL_LABELS, "processor": {"size": 518, "mean": [0.5307]*3, "std": [0.2583]*3},
              "epochs": cfg.epochs, "batch_size": cfg.batch_size, "accumulation_steps": cfg.accumulation_steps,
              "train_images": len(train), "valid_images": len(valid), "history": []}
    best_auc = -1.0
    best_path = cfg.output / "best.pt"
    for epoch in range(cfg.epochs):
        model.train(); optimizer.zero_grad(set_to_none=True); running = 0.0
        for step, (images, labels) in enumerate(loader, 1):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = loss_fn(model(pixel_values=images), labels) / cfg.accumulation_steps
            scaler.scale(loss).backward(); running += float(loss.detach()) * cfg.accumulation_steps
            if step % cfg.accumulation_steps == 0 or step == len(loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            if step == 1 or step % 500 == 0:
                print(f"epoch={epoch + 1}/{cfg.epochs} batch={step}/{len(loader)} loss={running / step:.5f}",
                      flush=True)
        scheduler.step()
        image_probs = predict(model, valid, processor, device, cfg.batch_size)
        studies, probs, targets = aggregate_studies(valid, image_probs)
        metrics = evaluate(targets, probs)
        item = {"epoch": epoch + 1, "loss": running / len(loader), "learning_rates": [g["lr"] for g in optimizer.param_groups], **metrics}
        report["history"].append(item); print(json.dumps(item), flush=True)
        if metrics["macro_auc"] > best_auc:
            best_auc = metrics["macro_auc"]
            torch.save({"model": model.state_dict(), "processor": str(cfg.model_dir.resolve()), "epoch": epoch + 1, "metrics": metrics, "labels": LOCAL_LABELS}, best_path)
    report["status"] = "complete"; report["best_macro_auc"] = best_auc
    (cfg.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(cfg.output), "best_macro_auc": best_auc}, indent=2), flush=True)


if __name__ == "__main__":
    main()
