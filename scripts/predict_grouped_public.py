"""Run frozen public models on an explicit patient-grouped training partition."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cxr.data import read_records
from cxr.safety import fresh_run_directory
from PIL import Image

LOCAL_LABELS = [
    "Enlarged Cardiomediastinum", "Pneumothorax", "Consolidation", "Pneumonia",
    "Edema", "Cardiomegaly", "Atelectasis", "Lung Opacity", "Pleural Effusion", "No Finding",
]
DENSENET_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]
FIVE_LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
BRANCHES = (
    "densenet121", "russellala_convnext", "rad_dino_vitB_768",
    "medmae_vitb_nih_B_768_s2", "medmae_vitb_nih_B_768_s2_seed1337",
)
BRANCH_PATHS = {
    "densenet121": "itsomk-chexpert-densenet121",
    "russellala_convnext": "russellala-convnext-chexpert-seed1",
    "rad_dino_vitB_768": "chest-xray-bench/rad_dino_vitB_768",
    "medmae_vitb_nih_B_768_s2": "chest-xray-bench/medmae_vitb_nih_B_768_s2",
    "medmae_vitb_nih_B_768_s2_seed1337": "chest-xray-bench/medmae_vitb_nih_B_768_s2_seed1337",
}


class ImageDataset:
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.rows[index]["path"]) as image:
            return self.transform(image.convert("RGB"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_partition_records(records, partition, phase):
    """Select all non-audit development rows or the isolated audit rows."""
    if phase not in {"development", "audit"}:
        raise ValueError("phase must be development or audit")
    if not isinstance(partition, dict) or not isinstance(partition.get("subject_partition"), dict):
        raise ValueError("partition must contain a subject_partition mapping")
    assignment = partition["subject_partition"]
    subjects = {str(row.get("subject_id")) for row in records}
    if set(assignment) != subjects:
        raise ValueError("subject_partition must cover train records exactly")
    values = list(assignment.values())
    if any(value == "audit" for value in values) is False or not any(value != "audit" for value in values):
        raise ValueError("partition must contain non-empty development and audit subjects")
    if any(isinstance(value, bool) or not (value == "audit" or isinstance(value, int) and value >= 0)
           for value in values):
        raise ValueError("partition contains invalid subject assignments")
    selected = [row for row in records if (assignment[str(row["subject_id"])] == "audit") == (phase == "audit")]
    if not selected:
        raise ValueError(f"partition has no {phase} records")
    return selected


def aggregate_mapped_predictions(records, image_probabilities, local_indices):
    """Average mapped image scores per study and retain full ten-label targets."""
    records = list(records)
    probabilities = np.asarray(image_probabilities, dtype=np.float64)
    indices = list(local_indices)
    if not indices or len(set(indices)) != len(indices) or any(
        isinstance(index, bool) or not isinstance(index, (int, np.integer)) or index < 0 or index >= 10
        for index in indices
    ):
        raise ValueError("local_indices must be unique class indices from 0 through 9")
    if probabilities.shape != (len(records), len(indices)):
        raise ValueError("image probabilities do not align with records and mapped classes")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("image probabilities must be finite values in [0, 1]")
    studies, sums, targets, counts, seen_dicom = [], [], [], [], set()
    positions = {}
    study_owner = {}
    for record, probability in zip(records, probabilities):
        if not all(key in record for key in ("dicom_id", "subject_id", "study_id", "labels")):
            raise ValueError("record is missing required identifiers or labels")
        dicom, subject, study = (str(record[key]) for key in ("dicom_id", "subject_id", "study_id"))
        if not dicom or not subject or not study or dicom in seen_dicom:
            raise ValueError("missing identifier or duplicate image probability record")
        seen_dicom.add(dicom)
        labels = np.asarray(record["labels"], dtype=np.float64)
        if labels.shape != (10,) or not np.isfinite(labels).all() or ((labels != 0) & (labels != 1)).any():
            raise ValueError("records must have ten binary training labels")
        key = (subject, study)
        if study in study_owner and study_owner[study] != subject:
            raise ValueError(f"study {study!r} has conflicting patients")
        study_owner[study] = subject
        if key in positions:
            slot = positions[key]
            if not np.array_equal(targets[slot], labels):
                raise ValueError(f"study {study!r} has inconsistent labels")
            sums[slot] += probability
            counts[slot] += 1
        else:
            positions[key] = len(studies)
            studies.append(key)
            sums.append(probability.copy())
            targets.append(labels.copy())
            counts.append(1)
    if not studies:
        raise ValueError("no study predictions supplied")
    return studies, (np.asarray(sums) / np.asarray(counts)[:, None]).astype(np.float32), np.asarray(targets, dtype=np.float32)


def _model_labels(branch, model_dir):
    if branch == "densenet121":
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        if cfg.get("labels") != DENSENET_LABELS or cfg.get("num_labels") != 14:
            raise ValueError("public DenseNet config labels do not match the pinned 14-label order")
        return LOCAL_LABELS, list(range(10)), cfg
    if branch == "russellala_convnext":
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        label_map = cfg.get("id2label")
        if not isinstance(label_map, dict) or not label_map:
            raise ValueError("RussellALA config is missing id2label")
        labels = [label_map[str(index)] for index in range(len(label_map))]
        if labels != FIVE_LABELS:
            raise ValueError("RussellALA labels do not match its published five-label order")
        return labels, [LOCAL_LABELS.index(name) for name in labels], cfg
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    labels = list(cfg.get("tasks", []))
    if cfg.get("head", {}).get("n_logits") != len(labels) or not labels:
        raise ValueError("invalid benchmark task configuration")
    if any(label not in LOCAL_LABELS for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("benchmark tasks contain unsupported or duplicate labels")
    return labels, [LOCAL_LABELS.index(name) for name in labels], cfg


def _predict(branch, records, model_dir, device, batch_size, workers):
    import torch
    from torch.utils.data import DataLoader
    from torchvision import models, transforms

    model_labels, local_indices, cfg = _model_labels(branch, model_dir)
    if branch == "densenet121":
        from safetensors.torch import load_file
        from torch import nn
        class DenseNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.densenet = models.densenet121(weights=None)
                self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, 14)
            def forward(self, x): return self.densenet(x)
        model = DenseNet()
        model.load_state_dict(load_file(str(model_dir / "pytorch_model.safetensors")), strict=True)
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        input_key = None
    elif branch == "russellala_convnext":
        from transformers import AutoModelForImageClassification
        model = AutoModelForImageClassification.from_pretrained(model_dir, local_files_only=True, use_safetensors=True)
        transform = transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        input_key = "pixel_values"
    else:
        module_path = model_dir.parent / "modeling.py"
        spec = importlib.util.spec_from_file_location("grouped_public_bench_modeling", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        model, loaded_cfg = module.load_model(str(model_dir), device="cpu")
        if loaded_cfg != cfg:
            raise ValueError("upstream loader returned a config different from the pinned file")
        model.to(device).eval()
        probabilities = []
        with torch.inference_mode():
            for index, row in enumerate(records, 1):
                with Image.open(row["path"]) as image:
                    gray = np.asarray(image.convert("L"))
                pixels = module.preprocess(gray, cfg).to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = model(pixels)
                if logits.ndim != 2 or logits.shape[1] != len(model_labels):
                    raise ValueError("unexpected benchmark model output shape")
                probabilities.append(logits.float().sigmoid()[0].cpu().numpy())
                if index == 1 or index % 100 == 0:
                    print(f"{branch}: predicted {index}/{len(records)} images", flush=True)
        return np.asarray(probabilities, dtype=np.float32), model_labels, local_indices, cfg

    model.to(device).eval()
    loader = DataLoader(ImageDataset(records, transform), batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=device.type == "cuda")
    chunks = []
    with torch.inference_mode():
        for batch_index, images in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(pixel_values=images).logits if input_key else model(images)
            expected_outputs = 14 if branch == "densenet121" else 5
            if logits.ndim != 2 or logits.shape[1] != expected_outputs:
                raise ValueError("unexpected public model output shape")
            probs = logits.float().sigmoid()
            if branch == "densenet121": probs = probs[:, [DENSENET_LABELS.index(name) for name in LOCAL_LABELS]]
            chunks.append(probs.cpu().numpy())
            interval = 25 if branch == "densenet121" else 25
            if batch_index == 1 or batch_index % interval == 0:
                done = min(batch_index * batch_size, len(records))
                print(f"{branch}: predicted {done}/{len(records)} images", flush=True)
    return np.concatenate(chunks), model_labels, local_indices, cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--phase", choices=("development", "audit"), required=True)
    parser.add_argument("--branch", choices=BRANCHES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--medical-root", type=Path, default=ROOT / "weights/medical")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.workers < 0: parser.error("batch-size must be positive and workers nonnegative")
    import torch
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA was requested but is unavailable")
    data_root = args.data_root.resolve(strict=True)
    train_csv = data_root / "train.csv"
    records = read_records(train_csv, data_root / "train")
    partition = json.loads(args.partition.read_text(encoding="utf-8"))
    actual_train_sha = sha256(train_csv)
    if partition.get("train_metadata_sha256") != actual_train_sha:
        raise ValueError("partition training metadata hash does not match train.csv")
    selected = select_partition_records(records, partition, args.phase)
    inventory_path = partition.get("image_inventory_path")
    if inventory_path:
        inventory_path = Path(inventory_path)
        if sha256(inventory_path) != partition.get("image_inventory_sha256"):
            raise ValueError("partition image inventory hash does not match its file")
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        for row in selected:
            item = inventory.get(row["dicom_id"])
            image_path = Path(row["path"])
            if not item or str(image_path.resolve()) != str(Path(item["path"]).resolve()):
                raise ValueError(f"selected image is absent from locked inventory: {row['dicom_id']}")
            stat = image_path.stat()
            if stat.st_size != item.get("bytes") or stat.st_mtime_ns != item.get("mtime_ns"):
                raise ValueError(f"selected image differs from locked inventory: {row['dicom_id']}")
    out = fresh_run_directory(args.output, data_root)
    model_dir = args.medical_root / BRANCH_PATHS[args.branch]
    image_probs, model_labels, local_indices, cfg = _predict(args.branch, selected, model_dir, device, args.batch_size, args.workers)
    study_keys, study_probs, targets = aggregate_mapped_predictions(selected, image_probs, local_indices)
    report = {
        "status": "complete", "branch": args.branch, "phase": args.phase,
        "class_indices": local_indices, "local_label_indices": local_indices,
        "class_names": [LOCAL_LABELS[i] for i in local_indices],
        "model_labels": model_labels, "config": cfg,
        "model_sha256": {p.name: sha256(p) for p in sorted(model_dir.iterdir()) if p.is_file()},
        "config_sha256": sha256(model_dir / "config.json"),
        "code_sha256": sha256(Path(__file__)), "partition_sha256": sha256(args.partition),
        "train_csv_sha256": sha256(train_csv), "public_pretraining_overlap": "unknown",
        "local_fit": False, "official_valid_read": False, "test_read": False,
        "selected_images": len(selected), "selected_studies": len(study_keys),
        "device": str(device), "batch_size": args.batch_size,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    if args.branch in BRANCHES[2:]:
        report["modeling_py_sha256"] = sha256(model_dir.parent / "modeling.py")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "study_ids.json").write_text(json.dumps([list(key) for key in study_keys], indent=2) + "\n", encoding="utf-8")
    np.save(out / "study_probabilities.npy", study_probs, allow_pickle=False)
    np.save(out / "study_targets.npy", targets, allow_pickle=False)
    print(json.dumps({"output": str(out), "status": "complete", "phase": args.phase, "studies": len(study_keys)}))


if __name__ == "__main__":
    main()
