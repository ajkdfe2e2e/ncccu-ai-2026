"""Transfer chest-xray-bench RAD-DINO features to the local ten-label task.

The public five-label head is used as initialization for the matching rows.
The RAD-DINO backbone is frozen while a new ten-label head is trained on the
local train split; validation Macro-AUC selects the exported head. Features are
cached outside Git so repeated head experiments do not rerun the ViT.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cxr.data import read_records
from cxr.metrics import aggregate_studies, evaluate

LOCAL_LABELS = [
    "Enlarged Cardiomediastinum", "Pneumothorax", "Consolidation",
    "Pneumonia", "Edema", "Cardiomegaly", "Atelectasis",
    "Lung Opacity", "Pleural Effusion", "No Finding",
]
BENCH_LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
BENCH_REVISION = "1735afb88371d98f87667c5f9966ed47bf4ef678"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("chest_xray_bench_modeling_transfer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FeatureDataset(Dataset):
    def __init__(self, records, module, cfg):
        self.records = list(records)
        self.module = module
        self.cfg = cfg

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(row["path"]) as image:
            gray = np.asarray(image.convert("L"))
        pixels = self.module.preprocess(gray, self.cfg)[0]
        return pixels, torch.tensor(row["labels"], dtype=torch.float32)


def extract_features(model, records, module, cfg, device, batch_size, workers, cache_path=None):
    if cache_path and cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        return payload["features"].float(), payload.get("labels")
    loader = DataLoader(FeatureDataset(records, module, cfg), batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=device.type == "cuda",
                        persistent_workers=workers > 0)
    features, labels = [], []
    model.eval()
    with torch.inference_mode():
        for step, (pixels, target) in enumerate(loader, 1):
            pixels = pixels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                sequence = model.backbone(pixel_values=pixels, interpolate_pos_encoding=True).last_hidden_state
                pooled = torch.cat([sequence[:, 0], sequence[:, 1:].mean(dim=1)], dim=1)
            features.append(pooled.float().cpu())
            labels.append(target)
            if step == 1 or step % 100 == 0:
                print(f"features {min(step * batch_size, len(records))}/{len(records)}", flush=True)
    result = torch.cat(features), torch.cat(labels)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": result[0], "labels": result[1]}, cache_path)
    return result


def aggregate_feature_predictions(records, probabilities):
    studies, study_probabilities, targets = aggregate_studies(records, probabilities)
    return studies, study_probabilities, targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"G:\cxr_first(1)\cxr_first"))
    parser.add_argument("--model-root", type=Path, default=ROOT / "weights/medical/chest-xray-bench")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/chest-xray-bench-transfer-20261105")
    parser.add_argument("--cache-root", type=Path, default=ROOT / ".cache/chest-xray-bench-features")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--feature-batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--valid-limit", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.feature_batch_size < 1:
        raise ValueError("epochs and batch sizes must be positive")
    random.seed(2026); np.random.seed(2026); torch.manual_seed(2026)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    module = load_module(args.model_root / "modeling.py")
    model_dir = args.model_root / "rad_dino_vitB_768"
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    source_model, loaded_cfg = module.load_model(str(model_dir), device=str(device))
    if loaded_cfg != cfg:
        raise ValueError("local model config changed while loading")
    if list(cfg["tasks"]) != BENCH_LABELS:
        raise ValueError("unexpected public task order")
    train = read_records(args.data_root / "train.csv", args.data_root / "train")
    valid = read_records(args.data_root / "valid.csv", args.data_root / "valid")
    if args.train_limit:
        train = train[:args.train_limit]
    if args.valid_limit:
        valid = valid[:args.valid_limit]

    train_features, train_labels = extract_features(source_model, train, module, cfg, device, args.feature_batch_size, args.workers, args.cache_root / "train.pt")
    valid_features, valid_labels = extract_features(source_model, valid, module, cfg, device, args.feature_batch_size, args.workers, args.cache_root / "valid.pt")
    hidden = source_model.head.in_features
    head = nn.Linear(hidden, len(LOCAL_LABELS))
    with torch.no_grad():
        head.weight.normal_(mean=0.0, std=0.02)
        head.bias.zero_()
        for local_index, label in enumerate(LOCAL_LABELS):
            if label in BENCH_LABELS:
                source_index = BENCH_LABELS.index(label)
                head.weight[local_index].copy_(source_model.head.weight[source_index])
                head.bias[local_index].copy_(source_model.head.bias[source_index])
    head.to(device)
    train_features, train_labels = train_features.to(device), train_labels.to(device)
    valid_features = valid_features.to(device)
    positives = train_labels.sum(dim=0)
    pos_weight = ((len(train_labels) - positives) / torch.clamp(positives, min=1.0)).clamp(max=10.0)
    loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=2e-5)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    args.output.mkdir(parents=True, exist_ok=True)
    best_auc = -1.0
    best_epoch = 0
    history = []
    best_path = args.output / "best_head.pt"
    for epoch in range(args.epochs):
        head.train(); running = 0.0
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(head(features), labels)
            loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); optimizer.step()
            running += float(loss.detach())
        head.eval()
        with torch.inference_mode():
            train_valid_probs = head(valid_features).sigmoid().float().cpu().numpy()
        _, study_probs, targets = aggregate_feature_predictions(valid, train_valid_probs)
        metrics = evaluate(targets, study_probs)
        item = {"epoch": epoch + 1, "loss": running / max(1, len(loader)), "lr": optimizer.param_groups[0]["lr"], **metrics}
        history.append(item); print(json.dumps(item), flush=True)
        if metrics["macro_auc"] > best_auc:
            best_auc = metrics["macro_auc"]; best_epoch = epoch + 1
            torch.save({"head": head.state_dict(), "epoch": best_epoch, "metrics": metrics, "labels": LOCAL_LABELS, "source_model": "mamounyosef/chest-xray-bench/rad_dino_vitB_768"}, best_path)
        scheduler.step()
    head.load_state_dict(torch.load(best_path, map_location=device, weights_only=True)["head"]); head.eval()
    with torch.inference_mode():
        valid_image_probs = head(valid_features).sigmoid().float().cpu().numpy()
    studies, study_probs, targets = aggregate_feature_predictions(valid, valid_image_probs)
    final_metrics = evaluate(targets, study_probs)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "complete",
        "model": "mamounyosef/chest-xray-bench/rad_dino_vitB_768 -> ten-label head",
        "model_revision": BENCH_REVISION,
        "weights_sha256": sha256(model_dir / "model.safetensors"),
        "config_sha256": sha256(model_dir / "config.json"),
        "modeling_py_sha256": sha256(args.model_root / "modeling.py"),
        "local_labels": LOCAL_LABELS,
        "source_labels": BENCH_LABELS,
        "copied_source_rows": {label: LOCAL_LABELS.index(label) for label in BENCH_LABELS},
        "preprocessing": cfg["image"],
        "feature_cache": {"train": str((args.cache_root / "train.pt").resolve()), "valid": str((args.cache_root / "valid.pt").resolve())},
        "train_images": len(train), "valid_images": len(valid), "valid_studies": len(studies),
        "epochs": args.epochs, "batch_size": args.batch_size, "best_epoch": best_epoch,
        "history": history, "metrics": final_metrics,
        "data_overlap_status": "unknown for the local dataset",
        "license": "CC BY-NC 4.0 per upstream model card",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.save(args.output / "study_probabilities.npy", study_probs)
    np.save(args.output / "study_targets.npy", targets)
    (args.output / "study_ids.json").write_text(json.dumps([[row["subject_id"], row["study_id"]] for row in studies], indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "best_epoch": best_epoch, **final_metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
