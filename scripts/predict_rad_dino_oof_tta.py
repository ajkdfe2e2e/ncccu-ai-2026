"""Predict held patient groups with raw Rad-DINO original/flip TTA checkpoints."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cxr.data import read_records
from cxr.metrics import aggregate_studies
from cxr.patient_splits import audit_records, partition_records
from scripts.run_aggressive_experiment import atomic_json, file_sha256
from scripts.run_rad_dino_finetune import ImageRecords, LOCAL_LABELS, collate, initialize_model, sha256
from scripts.train_rad_dino_grouped import verify_image_inventory

PUBLIC_SHA = "a81fea07272e6cff836b838ba4c12dd26d128b5fb16b3eb2e8c089a4a5ca14f8"
CONFIG_LR = {"A": 1e-3, "B": 1e-4}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--source-stage", type=Path, required=True,
                        help="completed grouped stage containing report.json and raw/last.pt")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("cv", "audit"), required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-images", type=int, default=0,
                        help="predict only the first N held images and mark output as smoke-only")
    return parser.parse_args(argv)


def validate_args(args):
    if (args.phase == "cv") != (args.fold is not None):
        raise ValueError("--fold is required for cv and forbidden for audit")
    if isinstance(args.batch_size, bool) or args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if isinstance(args.smoke_images, bool) or args.smoke_images < 0:
        raise ValueError("smoke-images must be a nonnegative integer")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")


def select_records(records, partition, phase, fold):
    if phase == "cv":
        train, held = partition_records(records, partition, fold)
    else:
        train, held = audit_records(records, partition)
    train_subjects = {str(row["subject_id"]) for row in train}
    held_subjects = {str(row["subject_id"]) for row in held}
    if not train or not held or train_subjects & held_subjects:
        raise ValueError("source training and held patients must be nonempty and disjoint")
    return train, held, train_subjects, held_subjects


def select_smoke_records(records, minimum_images, batch_size=1):
    if not minimum_images:
        return records
    if batch_size < 1:
        raise ValueError("batch-size must be positive")
    study_counts, study_order = {}, []
    for row in records:
        key = (str(row["subject_id"]), str(row["study_id"]))
        if key not in study_counts:
            study_counts[key] = 0
            study_order.append(key)
        study_counts[key] += 1
    selected_keys, images = set(), 0
    for key in study_order:
        selected_keys.add(key)
        images += study_counts[key]
        if images >= minimum_images and images % batch_size == 0:
            return [row for row in records
                    if (str(row["subject_id"]), str(row["study_id"])) in selected_keys]
    return records


def stage_config(source_stage, phase, fold):
    name = Path(source_stage).name
    if phase == "cv":
        match = re.fullmatch(r"cv_([AB])_fold([0-2])", name)
        if match is None or int(match.group(2)) != fold:
            raise ValueError("source-stage name does not match requested CV fold")
        return match.group(1)
    if name != "audit_A_raw":
        raise ValueError("audit source-stage must be the selected audit_A_raw stage")
    return "A"


def validate_source_provenance(
    source_stage, report, payload, *, phase, fold, train_records, held_records,
    train_metadata_sha256, partition_sha256, public_checkpoint_sha256, checkpoint_sha256,
):
    config = stage_config(source_stage, phase, fold)
    expected_fold = fold if phase == "cv" else None
    if report.get("status") != "complete" or report.get("phase") != phase or report.get("fold") != expected_fold:
        raise ValueError("source-stage report does not match requested phase/fold")
    if report.get("head_lr") != CONFIG_LR[config] or report.get("epochs") != 3:
        raise ValueError("source-stage is not the expected three-epoch configuration")
    if report.get("checkpoint_policy") != "fixed_final_epoch":
        raise ValueError("source-stage did not use the fixed final epoch")
    if report.get("train_metadata_sha256") != train_metadata_sha256:
        raise ValueError("source-stage train.csv fingerprint differs")
    if report.get("partition_sha256") != partition_sha256:
        raise ValueError("source-stage partition fingerprint differs")
    if report.get("public_checkpoint_sha256") != public_checkpoint_sha256:
        raise ValueError("source-stage public initialization differs")
    variant = report.get("variants", {}).get("raw")
    if not isinstance(variant, dict) or variant.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("raw source checkpoint fingerprint differs from stage report")
    provenance = payload.get("provenance") if isinstance(payload, dict) else None
    if payload.get("labels") != LOCAL_LABELS or payload.get("epoch") != 3 or payload.get("variant") != "raw":
        raise ValueError("checkpoint labels, epoch, or variant do not match raw ten-class evaluation")
    expected = {
        "phase": phase, "fold": expected_fold,
        "head_lr": CONFIG_LR[config], "epochs": 3,
        "checkpoint_policy": "fixed_final_epoch",
        "train_metadata_sha256": train_metadata_sha256,
        "partition_sha256": partition_sha256,
        "public_checkpoint_sha256": public_checkpoint_sha256,
        "train_images": len(train_records), "evaluation_images": len(held_records),
        "train_patients": len({str(row["subject_id"]) for row in train_records}),
        "evaluation_patients": len({str(row["subject_id"]) for row in held_records}),
    }
    if not isinstance(provenance, dict) or any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError("checkpoint training provenance differs from locked split/configuration")
    if report.get("train_images") != expected["train_images"] or report.get("train_patients") != expected["train_patients"]:
        raise ValueError("source-stage training coverage differs from locked split")
    if report.get("evaluation_images") != expected["evaluation_images"] or report.get("evaluation_patients") != expected["evaluation_patients"]:
        raise ValueError("source-stage held coverage differs from locked split")


@torch.inference_mode()
def predict_image_views(model, records, processor, device, batch_size, progress_callback=None):
    loader = DataLoader(
        ImageRecords(records, processor, training=False), batch_size=batch_size,
        shuffle=False, num_workers=0, collate_fn=collate, pin_memory=device.type == "cuda",
    )
    original_chunks, tta_chunks = [], []
    model.eval()
    for batch_index, (images, _) in enumerate(loader, 1):
        images = images.to(device, non_blocking=device.type == "cuda")
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            original_logits = model(pixel_values=images)
        if not isinstance(original_logits, torch.Tensor):
            original_logits = (original_logits.logits if hasattr(original_logits, "logits")
                               else original_logits[0])
        if original_logits.ndim != 2 or original_logits.shape[1] != len(LOCAL_LABELS):
            raise ValueError(f"unexpected original logits shape: {tuple(original_logits.shape)}")
        original = original_logits.float().sigmoid()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            flipped_logits = model(pixel_values=images.flip(-1))
        if not isinstance(flipped_logits, torch.Tensor):
            flipped_logits = flipped_logits.logits if hasattr(flipped_logits, "logits") else flipped_logits[0]
        if flipped_logits.ndim != 2 or flipped_logits.shape[1] != len(LOCAL_LABELS):
            raise ValueError(f"unexpected flipped logits shape: {tuple(flipped_logits.shape)}")
        flipped = flipped_logits.float().sigmoid()
        tta = (original + flipped) / 2.0
        original_chunks.append(original.cpu().numpy())
        tta_chunks.append(tta.cpu().numpy())
        done = min(batch_index * batch_size, len(records))
        if progress_callback is not None and (batch_index % 100 == 0 or done == len(records)):
            progress_callback(done, len(records))
        if batch_index == 1 or batch_index % 100 == 0:
            print(f"predicted {done}/{len(records)} held images", flush=True)
    if not original_chunks:
        raise ValueError("selected held image set is empty")
    return np.concatenate(original_chunks), np.concatenate(tta_chunks)


def align_and_compare_original(folder, studies, original_scores):
    reference_ids = [tuple(map(str, row)) for row in json.loads(
        (folder / "study_ids.json").read_text(encoding="utf-8"))]
    reference = np.load(folder / "study_probabilities.npy", allow_pickle=False)
    if len(reference_ids) != len(set(reference_ids)) or reference.shape != (len(reference_ids), 10):
        raise ValueError("saved raw predictions have invalid IDs or shape")
    if not np.isfinite(reference).all() or ((reference < 0) | (reference > 1)).any():
        raise ValueError("saved raw probabilities are invalid")
    expected_ids = [(str(row["subject_id"]), str(row["study_id"])) for row in studies]
    positions = {key: index for index, key in enumerate(reference_ids)}
    if any(key not in positions for key in expected_ids):
        raise ValueError("saved raw predictions do not cover selected held studies")
    aligned = reference[[positions[key] for key in expected_ids]]
    if aligned.shape != original_scores.shape:
        raise ValueError("original reprojection shape differs from saved raw predictions")
    max_abs = float(np.max(np.abs(aligned - original_scores)))
    if max_abs > 1e-5:
        raise ValueError(f"original predictions differ from saved raw checkpoint output: max_abs={max_abs}")
    return max_abs


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    data_root = args.data_root.resolve(strict=True)
    partition_path = args.partition.resolve(strict=True)
    source_stage = args.source_stage.resolve(strict=True)
    model_dir = args.model_dir.resolve(strict=True)
    output = args.output.resolve()
    if output == data_root or output.is_relative_to(data_root):
        raise ValueError("output must be outside the source dataset")
    if output == source_stage or output.is_relative_to(source_stage) or source_stage.is_relative_to(output):
        raise ValueError("output and source-stage must be separate directories")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty")
    metadata = data_root / "train.csv"
    metadata_sha = sha256(metadata)
    if metadata_sha != json.loads(partition_path.read_text(encoding="utf-8"))["train_metadata_sha256"]:
        raise ValueError("train.csv does not match the locked partition")
    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    partition_sha = sha256(partition_path)
    public_checkpoint = model_dir / "pytorch_model.bin"
    public_sha = sha256(public_checkpoint)
    if public_sha != PUBLIC_SHA or partition.get("public_checkpoint_sha256") != public_sha:
        raise ValueError("public initialization does not match the locked checkpoint")
    records = read_records(metadata, data_root / "train")
    train, held, train_subjects, held_subjects = select_records(records, partition, args.phase, args.fold)
    inventory_path = Path(partition["image_inventory_path"])
    if sha256(inventory_path) != partition.get("image_inventory_sha256"):
        raise ValueError("locked image inventory fingerprint differs")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    verify_image_inventory(records, inventory)

    source_report = json.loads((source_stage / "report.json").read_text(encoding="utf-8"))
    source_checkpoint = source_stage / "raw" / "last.pt"
    if not source_checkpoint.is_file():
        raise ValueError("source-stage raw/last.pt is missing")
    checkpoint_sha = sha256(source_checkpoint)
    payload = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
    validate_source_provenance(
        source_stage, source_report, payload, phase=args.phase, fold=args.fold,
        train_records=train, held_records=held, train_metadata_sha256=metadata_sha,
        partition_sha256=partition_sha, public_checkpoint_sha256=public_sha,
        checkpoint_sha256=checkpoint_sha,
    )
    selected = select_smoke_records(held, args.smoke_images, args.batch_size)
    if not selected:
        raise ValueError("smoke image selection is empty")

    device = torch.device(args.device)
    processor, model = initialize_model(model_dir, device)
    model.load_state_dict(payload["model"], strict=True)
    del payload
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "progress.json", {"status": "running", "completed_images": 0,
                                            "total_images": len(selected), "smoke": bool(args.smoke_images)})
    def write_progress(completed, total):
        atomic_json(output / "progress.json", {
            "status": "running", "completed_images": completed,
            "total_images": total, "smoke": bool(args.smoke_images),
        })

    original_images, tta_images = predict_image_views(
        model, selected, processor, device, args.batch_size, progress_callback=write_progress,
    )
    studies, tta_scores, targets = aggregate_studies(selected, tta_images)
    original_studies, original_scores, original_targets = aggregate_studies(selected, original_images)
    if [(r["subject_id"], r["study_id"]) for r in studies] != [
        (r["subject_id"], r["study_id"]) for r in original_studies
    ] or not np.array_equal(targets, original_targets):
        raise ValueError("original and TTA study aggregation changed IDs or labels")
    max_abs = align_and_compare_original(source_stage / "raw", studies, original_scores)
    study_ids = [[row["subject_id"], row["study_id"]] for row in studies]
    np.save(output / "study_probabilities.npy", tta_scores)
    np.save(output / "original_study_probabilities.npy", original_scores)
    np.save(output / "study_targets.npy", targets)
    atomic_json(output / "study_ids.json", study_ids)
    report = {
        "status": "complete", "phase": args.phase, "fold": args.fold,
        "smoke": bool(args.smoke_images), "smoke_images": len(selected) if args.smoke_images else None,
        "train_metadata_sha256": metadata_sha, "partition_sha256": partition_sha,
        "train_subjects": sorted(train_subjects),
        "held_subjects": sorted({str(row["subject_id"]) for row in selected}),
        "source_checkpoint_sha256": checkpoint_sha,
        "source_stage": str(source_stage), "source_config": stage_config(source_stage, args.phase, args.fold),
        "images": len(selected), "studies": len(studies),
        "original_max_abs_difference": max_abs, "original_comparison_tolerance": 1e-5,
        "horizontal_flip_tta": True, "aggregation": "float32 mean(original sigmoid, horizontal-flip sigmoid) per image, then study mean",
        "official_valid_read": False, "test_read": False,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    atomic_json(output / "report.json", report)
    atomic_json(output / "progress.json", {"status": "complete", "completed_images": len(selected),
                                            "total_images": len(selected), "smoke": bool(args.smoke_images)})
    print(json.dumps({"output": str(output), "report": report}, indent=2), flush=True)


if __name__ == "__main__":
    main()
