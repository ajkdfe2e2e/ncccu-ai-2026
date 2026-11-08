"""Evaluate the pinned RussellALA five-finding CheXpert ConvNeXt checkpoint."""
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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoModelForImageClassification

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cxr.data import read_records
from cxr.metrics import aggregate_studies, evaluate
from cxr.safety import fresh_run_directory

LOCAL_LABELS = [
    "Enlarged Cardiomediastinum", "Pneumothorax", "Consolidation", "Pneumonia",
    "Edema", "Cardiomegaly", "Atelectasis", "Lung Opacity", "Pleural Effusion", "No Finding",
]


class ImageRecords(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        with Image.open(self.records[index]["path"]) as image:
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
    parser.add_argument("--model-dir", type=Path, default=ROOT / "weights/medical/russellala-convnext-chexpert-seed1")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/russellala-convnext-valid")
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
    label_map = cfg.get("id2label")
    if not isinstance(label_map, dict) or not label_map:
        raise ValueError("RussellALA config is missing id2label")
    model_labels = [label_map[str(index)] for index in range(len(label_map))]
    model = AutoModelForImageClassification.from_pretrained(
        args.model_dir, local_files_only=True, use_safetensors=True
    ).to(device).eval()
    if len(model_labels) != 5 or model_labels != [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"
    ]:
        raise ValueError("RussellALA output labels do not match its published five-label order")
    transform = transforms.Compose([
        transforms.Resize((384, 384)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    records = read_records(args.data_root / f"{args.split}.csv", args.data_root / args.split)
    loader = DataLoader(ImageRecords(records, transform), batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=device.type == "cuda")
    output = []
    with torch.inference_mode():
        for index, images in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(pixel_values=images).logits
            if logits.ndim != 2 or logits.shape[1] != 5:
                raise ValueError(f"unexpected RussellALA output shape: {tuple(logits.shape)}")
            output.append(logits.float().sigmoid().cpu().numpy())
            if index == 1 or index % 25 == 0:
                print(f"predicted {min(index * args.batch_size, len(records))}/{len(records)} images", flush=True)
    image_probabilities = np.concatenate(output, axis=0)
    studies, _, targets = aggregate_studies(
        records, np.full((len(records), len(LOCAL_LABELS)), 0.5, dtype=np.float32)
    )
    indices = [LOCAL_LABELS.index(label) for label in model_labels]
    study_probabilities = np.empty((len(studies), 5), dtype=np.float32)
    positions = {(str(row["subject_id"]), str(row["study_id"])): index
                 for index, row in enumerate(studies)}
    sums = np.zeros_like(study_probabilities, dtype=np.float64)
    counts = np.zeros(len(studies), dtype=np.int64)
    for row, probabilities in zip(records, image_probabilities):
        index = positions[(str(row["subject_id"]), str(row["study_id"]))]
        sums[index] += probabilities
        counts[index] += 1
    study_probabilities[:] = (sums / counts[:, None]).astype(np.float32)
    metrics = None
    if targets is not None:
        full_probabilities = np.full_like(targets, 0.5, dtype=np.float64)
        full_probabilities[:, indices] = study_probabilities
        mapped_metrics = evaluate(targets, full_probabilities)
        per_class = [mapped_metrics["per_class"][index] for index in indices]
        macro_auc = float(np.mean([row["auc"] for row in per_class if row["auc"] is not None]))
        metrics = {"macro_auc_mapped_classes": macro_auc, "defined_classes": len(per_class),
                   "per_class": per_class}
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "model": "RussellALA/convnext-chexpert-seed1",
        "model_revision": "0444dbe62e26e0206b485c7771b06530b19e66f7",
        "weights_sha256": sha256(args.model_dir / "model.safetensors"),
        "local_labels": LOCAL_LABELS, "mapped_labels": model_labels, "local_label_indices": indices,
        "preprocessing": {"size": [384, 384], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "data_root": str(args.data_root.resolve()), "split": args.split,
        f"{args.split}_images": len(records), f"{args.split}_studies": len(studies),
        "device": str(device), "batch_size": args.batch_size,
        "data_overlap_status": "unknown for the local dataset",
    }
    if metrics is not None:
        report["metrics"] = metrics
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.save(args.output / "study_probabilities.npy", study_probabilities)
    if targets is not None:
        np.save(args.output / "study_targets.npy", targets)
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
