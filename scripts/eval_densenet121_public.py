"""Evaluate the pinned public CheXpert DenseNet121 checkpoint on a local split."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cxr.data import read_records
from cxr.metrics import aggregate_studies, evaluate
from cxr.safety import fresh_run_directory

MODEL_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]
LOCAL_LABELS = [
    "Enlarged Cardiomediastinum", "Pneumothorax", "Consolidation", "Pneumonia",
    "Edema", "Cardiomegaly", "Atelectasis", "Lung Opacity", "Pleural Effusion", "No Finding",
]
MODEL_INDICES = tuple(MODEL_LABELS.index(name) for name in LOCAL_LABELS)


class ImageRecords(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(row["path"]) as image:
            return self.transform(image.convert("RGB"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"G:\cxr_first(1)\cxr_first"))
    parser.add_argument("--model-dir", type=Path, default=ROOT / "weights/medical/itsomk-chexpert-densenet121")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/itsomk-densenet121-valid")
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    args.output = fresh_run_directory(args.output, args.data_root)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    cfg = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    if cfg["labels"] != MODEL_LABELS or cfg["num_labels"] != 14:
        raise ValueError("public DenseNet config labels do not match the pinned CheXpert order")

    class DenseNet121CheXpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.densenet = models.densenet121(weights=None)
            self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, 14)

        def forward(self, x):
            return self.densenet(x)

    model = DenseNet121CheXpert()
    state = load_file(str(args.model_dir / "pytorch_model.safetensors"))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    data_root = args.data_root.resolve()
    records = read_records(data_root / f"{args.split}.csv", data_root / args.split)
    loader = DataLoader(ImageRecords(records, transform), batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=device.type == "cuda")
    output = []
    with torch.inference_mode():
        for index, images in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images)
            output.append(logits.float().sigmoid()[:, MODEL_INDICES].cpu().numpy())
            if index == 1 or index % 25 == 0:
                print(f"predicted {min(index * args.batch_size, len(records))}/{len(records)} images", flush=True)
    image_probabilities = np.concatenate(output, axis=0)
    studies, probabilities, targets = aggregate_studies(records, image_probabilities)
    metrics = evaluate(targets, probabilities) if targets is not None else None
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "model": "itsomk/chexpert-densenet121",
        "model_revision": "81be45e4d43011e8f50e17d002190c76cb069edd",
        "weights_sha256": sha256(args.model_dir / "pytorch_model.safetensors"),
        "model_labels": MODEL_LABELS,
        "local_labels": LOCAL_LABELS,
        "model_indices": MODEL_INDICES,
        "preprocessing": {"size": [224, 224], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "data_root": str(data_root),
        "split": args.split,
        f"{args.split}_images": len(records), f"{args.split}_studies": len(studies),
        "device": str(device), "batch_size": args.batch_size,
        "data_overlap_status": "unknown for the local dataset",
    }
    if metrics is not None:
        report["metrics"] = metrics
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.save(args.output / "study_probabilities.npy", probabilities)
    if targets is not None:
        np.save(args.output / "study_targets.npy", targets)
    (args.output / "study_ids.json").write_text(json.dumps([[row["subject_id"], row["study_id"]] for row in studies], indent=2) + "\n", encoding="utf-8")
    summary = {"output": str(args.output)}
    if metrics is not None:
        summary.update(metrics)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
