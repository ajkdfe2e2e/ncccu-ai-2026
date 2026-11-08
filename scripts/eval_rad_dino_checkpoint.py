"""Independently reload and evaluate a fine-tuned ten-class Rad-DINO checkpoint."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
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

LOCAL_LABELS = [
    "Enlarged Cardiomediastinum", "Pneumothorax", "Consolidation",
    "Pneumonia", "Edema", "Cardiomegaly", "Atelectasis",
    "Lung Opacity", "Pleural Effusion", "No Finding",
]


class ImageRecords(Dataset):
    def __init__(self, records, processor):
        self.records = records
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(row["path"]) as image:
            pixels = self.processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"][0]
        return pixels


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"G:\cxr_first(1)\cxr_first"))
    parser.add_argument("--model-dir", type=Path, default=ROOT / "weights/medical/rad-dino-chexpert-upstream")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/rad-dino-finetune-20261105/best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/rad-dino-finetuned-valid-independent")
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--horizontal-flip", action="store_true", help="average original and horizontally flipped probabilities")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    args.output = fresh_run_directory(args.output, args.data_root)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload.get("labels") != LOCAL_LABELS:
        raise ValueError("checkpoint labels do not match this project's ten-class order")
    processor = AutoImageProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(
        args.model_dir, trust_remote_code=True, local_files_only=True, use_safetensors=False
    )
    model.classifier = nn.Linear(model.classifier.in_features, len(LOCAL_LABELS))
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()

    records = read_records(args.data_root / f"{args.split}.csv", args.data_root / args.split)
    loader = DataLoader(
        ImageRecords(records, processor), batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    predictions = []
    original_predictions = []
    flipped_predictions = []
    with torch.inference_mode():
        for index, images in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(pixel_values=images)
            if not isinstance(logits, torch.Tensor):
                logits = logits.logits if hasattr(logits, "logits") else logits[0]
            if logits.ndim != 2 or logits.shape[1] != len(LOCAL_LABELS):
                raise ValueError(f"unexpected logits shape: {tuple(logits.shape)}")
            original = logits.float().sigmoid()
            original_predictions.append(original.cpu().numpy())
            if args.horizontal_flip:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    flipped_logits = model(pixel_values=images.flip(-1))
                if not isinstance(flipped_logits, torch.Tensor):
                    flipped_logits = flipped_logits.logits if hasattr(flipped_logits, "logits") else flipped_logits[0]
                flipped = flipped_logits.float().sigmoid()
                flipped_predictions.append(flipped.cpu().numpy())
                original = (original + flipped) / 2.0
            predictions.append(original.cpu().numpy())
            if index == 1 or index % 100 == 0:
                print(f"predicted {min(index * args.batch_size, len(records))}/{len(records)} images", flush=True)

    image_probabilities = np.concatenate(predictions, axis=0)
    studies, probabilities, targets = aggregate_studies(records, image_probabilities)
    metrics = evaluate(targets, probabilities) if targets is not None else None
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = sha256(args.checkpoint)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "model": "yaLLMan/rad-dino-chexpert fine-tuned on the local train split",
        "source_revision": "db02e1b7234dd83c6d7c4485963ef5b22df9e5db",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "selected_epoch": payload["epoch"],
        "stored_validation_metrics": payload["metrics"],
        "local_labels": LOCAL_LABELS,
        "image_processor": {"size": 518, "mean": [0.5307] * 3, "std": [0.2583] * 3},
        "data_root": str(args.data_root.resolve()),
        "split": args.split,
        f"{args.split}_images": len(records),
        f"{args.split}_studies": len(studies),
        "device": str(device),
        "batch_size": args.batch_size,
        "horizontal_flip_tta": args.horizontal_flip,
        "data_overlap_status": "unknown for the local dataset",
    }
    if metrics is not None:
        report["independently_reloaded_metrics"] = metrics
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.save(args.output / "study_probabilities.npy", probabilities)
    if targets is not None:
        np.save(args.output / "study_targets.npy", targets)
    if args.horizontal_flip:
        original_images = np.concatenate(original_predictions, axis=0)
        flipped_images = np.concatenate(flipped_predictions, axis=0)
        _, original_scores, original_targets = aggregate_studies(records, original_images)
        _, flipped_scores, _ = aggregate_studies(records, flipped_images)
        if original_targets is not None:
            report["original_only_metrics"] = evaluate(original_targets, original_scores)
            report["flipped_only_metrics"] = evaluate(original_targets, flipped_scores)
        np.save(args.output / "original_study_probabilities.npy", original_scores)
        np.save(args.output / "flipped_study_probabilities.npy", flipped_scores)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "study_ids.json").write_text(
        json.dumps([[row["subject_id"], row["study_id"]] for row in studies], indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {"output": str(args.output)}
    if metrics is not None:
        summary.update(metrics)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
