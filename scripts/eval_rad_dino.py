"""Evaluate the public Rad-DINO CheXpert checkpoint on this project's valid split.

The upstream checkpoint has a 14-class CheXpert head. This script maps the ten
local labels by name, preserves the upstream 518x518 processor, aggregates
image probabilities by study, and writes an auditable report. It does not train
or modify the local baseline checkpoint.
"""
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(r"G:\cxr_first(1)\cxr_first"))
    parser.add_argument("--model-dir", type=Path, default=ROOT / "weights/medical/rad-dino-chexpert-upstream")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model(model_dir: Path, device: torch.device):
    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=True, use_safetensors=False
    )
    model.to(device).eval()
    return processor, model


def predict(records, processor, model, device: torch.device, batch_size: int) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, len(records), batch_size):
        rows = records[start:start + batch_size]
        images = []
        for row in rows:
            with Image.open(row["path"]) as image:
                images.append(image.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        pixels = inputs["pixel_values"].to(device, non_blocking=True)
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(pixel_values=pixels)
            else:
                logits = model(pixel_values=pixels)
        if not isinstance(logits, torch.Tensor):
            logits = logits.logits if hasattr(logits, "logits") else logits[0]
        if logits.ndim != 2 or logits.shape[1] != 14:
            raise ValueError(f"unexpected Rad-DINO output shape: {tuple(logits.shape)}")
        probabilities = logits.float().sigmoid()[:, MODEL_INDICES]
        outputs.append(probabilities.cpu().numpy())
        if start == 0 or (start + batch_size) % (batch_size * 25) == 0:
            print(f"predicted {min(start + batch_size, len(records))}/{len(records)} images", flush=True)
    return np.concatenate(outputs, axis=0)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    output = args.output or (ROOT / "outputs" / f"rad-dino-valid-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    output = fresh_run_directory(output, args.data_root)
    model_dir = args.model_dir.resolve()
    data_root = args.data_root.resolve()
    records = read_records(data_root / "valid.csv", data_root / "valid")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    processor, model = load_model(model_dir, device)
    image_probabilities = predict(records, processor, model, device, args.batch_size)
    studies, probabilities, targets = aggregate_studies(records, image_probabilities)
    metrics = evaluate(targets, probabilities)
    checkpoint = model_dir / "pytorch_model.bin"
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "model": "yaLLMan/rad-dino-chexpert",
        "model_revision": "db02e1b7234dd83c6d7c4485963ef5b22df9e5db",
        "model_checkpoint_sha256": sha256(checkpoint),
        "model_labels": MODEL_LABELS,
        "local_labels": LOCAL_LABELS,
        "model_indices": MODEL_INDICES,
        "processor": {"size": 518, "mean": [0.5307] * 3, "std": [0.2583] * 3},
        "data_root": str(data_root),
        "valid_images": len(records),
        "valid_studies": len(studies),
        "device": str(device),
        "batch_size": args.batch_size,
        "metrics": metrics,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.save(output / "study_probabilities.npy", probabilities)
    np.save(output / "study_targets.npy", targets)
    print(json.dumps({"output": str(output), **metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
