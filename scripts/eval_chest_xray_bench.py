"""Evaluate pinned public CheXpert benchmark models on a local split.

The upstream models predict five CheXpert findings. Scores are aggregated per
study and reported only over those five mapped labels; they are never compared
to ten-class Macro-AUC as if missing outputs were predictions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

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
BENCH_MODELS = [
    "rad_dino_vitB_768",
    "medmae_vitb_nih_B_768_s2",
    "convnext_base_22k_1600x1312",
    "medmae_vitb_nih_B_768_s2_seed1337",
    "medmae_vitb_nih_B_448_s1_seed7",
]
BENCH_REVISION = "1735afb88371d98f87667c5f9966ed47bf4ef678"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_upstream_module(path: Path):
    spec = importlib.util.spec_from_file_location("chest_xray_bench_modeling", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load upstream model module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def evaluate_model(model_name: str, records, module, model_root: Path, output: Path,
                   device: torch.device, max_images: int, split: str) -> dict:
    model_dir = model_root / model_name
    config_path = model_dir / "config.json"
    weights_path = model_dir / "model.safetensors"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    tasks = list(cfg["tasks"])
    if cfg["head"]["n_logits"] != len(tasks) or not tasks:
        raise ValueError(f"invalid five-finding classifier config: {config_path}")
    local_indices = [LOCAL_LABELS.index(name) for name in tasks]
    subset = records[:max_images] if max_images else records

    model, loaded_cfg = module.load_model(str(model_dir), device="cpu")
    if loaded_cfg != cfg:
        raise ValueError("upstream loader returned a config different from the pinned local file")
    model.to(device).eval()
    image_probs = []
    with torch.inference_mode():
        for index, row in enumerate(subset, 1):
            with Image.open(row["path"]) as image:
                gray = np.asarray(image.convert("L"))
            pixels = module.preprocess(gray, cfg).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(pixels)
            if logits.ndim != 2 or logits.shape[1] != len(tasks):
                raise ValueError(f"unexpected {model_name} output shape: {tuple(logits.shape)}")
            image_probs.append(logits.float().sigmoid()[0].cpu().numpy())
            if index == 1 or index % 100 == 0:
                print(f"{model_name}: predicted {index}/{len(subset)} images", flush=True)

    image_probs = np.asarray(image_probs, dtype=np.float32)
    # Use mapped scores for available classes and a neutral placeholder elsewhere;
    # the aggregate targets and full metric structure come from the official local records.
    full_image_probs = np.full((len(subset), len(LOCAL_LABELS)), 0.5, dtype=np.float32)
    full_image_probs[:, local_indices] = image_probs
    studies, _, targets = aggregate_studies(subset, full_image_probs)
    study_probs = np.full((len(studies), len(tasks)), np.nan, dtype=np.float32)
    position = {row["study_id"]: index for index, row in enumerate(studies)}
    sums = np.zeros((len(studies), len(tasks)), dtype=np.float64)
    counts = np.zeros(len(studies), dtype=np.int64)
    for row, prob in zip(subset, image_probs):
        slot = position[row["study_id"]]
        sums[slot] += prob
        counts[slot] += 1
    study_probs[:] = (sums / counts[:, None]).astype(np.float32)
    metrics = None
    if targets is not None:
        metric_input = np.full_like(targets, 0.5, dtype=np.float32)
        metric_input[:, local_indices] = study_probs
        all_metrics = evaluate(targets, metric_input)
        per_class = [all_metrics["per_class"][index] for index in local_indices]
        defined = [row["auc"] for row in per_class if row["auc"] is not None]
        metrics = {
            "macro_auc_mapped_classes": float(np.mean(defined)) if defined else None,
            "defined_classes": len(defined),
            "per_class": per_class,
        }

    model_output = output / model_name
    model_output.mkdir(parents=True, exist_ok=True)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "model": f"mamounyosef/chest-xray-bench/{model_name}",
        "model_revision": BENCH_REVISION,
        "modeling_py_sha256": sha256(model_root / "modeling.py"),
        "weights_sha256": sha256(weights_path),
        "config_sha256": sha256(config_path),
        "upstream_config": cfg,
        "local_label_indices": local_indices,
        "mapped_labels": tasks,
        "split": split,
        f"{split}_images": len(subset),
        f"{split}_studies": len(studies),
        "device": str(device),
        "data_overlap_status": "unknown for the local dataset",
        "license": "CC BY-NC 4.0 per upstream model card",
    }
    if metrics is not None:
        report["metrics"] = metrics
    (model_output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.save(model_output / "study_probabilities.npy", study_probs)
    if targets is not None:
        np.save(model_output / "study_targets.npy", targets)
    (model_output / "study_ids.json").write_text(
        json.dumps([[row["subject_id"], row["study_id"]] for row in studies], indent=2) + "\n",
        encoding="utf-8",
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"G:\cxr_first(1)\cxr_first"))
    parser.add_argument("--model-root", type=Path, default=ROOT / "weights/medical/chest-xray-bench")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/chest-xray-bench-valid")
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--models", nargs="+", choices=BENCH_MODELS, default=BENCH_MODELS[:3])
    parser.add_argument("--max-images", type=int, default=0, help="positive value limits to a valid-only smoke run")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.max_images < 0:
        parser.error("--max-images must be nonnegative")
    if args.max_images and args.split == "test":
        parser.error("--max-images is allowed only with --split valid")
    args.output = fresh_run_directory(args.output, args.data_root)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    module = load_upstream_module(args.model_root / "modeling.py")
    records = read_records(args.data_root / f"{args.split}.csv", args.data_root / args.split)
    if args.max_images:
        records = records[:args.max_images]
    results = [
        evaluate_model(name, records, module, args.model_root, args.output, device, args.max_images, args.split)
        for name in args.models
    ]
    summary = {
        "model_revision": BENCH_REVISION,
        "split": args.split,
        "data_root": str(args.data_root.resolve()),
        f"{args.split}_images": len(records),
        f"{args.split}_studies": len({row["study_id"] for row in records}),
        "results": [
            {"model": item["model"], **({
                "macro_auc_mapped_classes": item["metrics"]["macro_auc_mapped_classes"],
                "per_class": item["metrics"]["per_class"],
            } if "metrics" in item else {})}
            for item in results
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
