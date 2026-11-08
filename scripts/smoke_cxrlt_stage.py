"""Discarded engineering updates: split parity and worst-case study memory."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys
import time

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cxr.cxrlt_transfer import load_encoder
from cxr.cxrlt_stage_adapter import split_official_model
from cxr.training_state import cpu_snapshot
from cxr.study_probe import predict_linear_probe
from scripts.evaluate_frozen_study_probe import _load_head
from scripts.run_frozen_study_probe import atomic_json, read_json, sha256, environment, now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/cxrlt-supervised-study-probe-20261108")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    torch.manual_seed(20261108)
    device = torch.device("cuda")
    plan = read_json(args.source / "plan.json")
    records = read_json(args.source / "train_records.json")["records"]
    payload = _load_head(args.source / "final_head.npz")
    adapter, processor = load_encoder(plan["model_dir"], device)
    prefix, tail = split_official_model(adapter.model, payload)
    prefix, tail = prefix.to(device), tail.to(device)
    selected = [0, 1, len(records) // 2, len(records) // 2 + 1, len(records) - 2, len(records) - 1]
    prefixes, full_features, split_features = [], [], []
    with torch.inference_mode():
        for begin in range(0, len(selected), 2):
            images = []
            for index in selected[begin:begin + 2]:
                with Image.open(records[index]["path"]) as image:
                    images.append(image.convert("RGB"))
            try:
                pixels = processor(images=images)["pixel_values"].to(device)
            finally:
                for image in images:
                    image.close()
            with torch.autocast("cuda", dtype=torch.float16):
                full = adapter(pixels)
                cached = prefix(pixels)
                split = tail.features(cached)
            full_features.append(full.float().cpu().numpy())
            split_features.append(split.float().cpu().numpy())
            prefixes.append(cached.cpu().numpy())
    full = np.concatenate(full_features)
    split = np.concatenate(split_features)
    spatial = np.concatenate(prefixes)
    old = np.load(args.source / "train_features/image_features.npy", mmap_mode="r")[selected]
    if not np.array_equal(full, split) or not np.array_equal(full, old):
        raise ValueError(f"initial split/cache forward differs: split={np.max(abs(full-split))}, old={np.max(abs(full-old))}")
    # NumPy -> Tensor outside inference mode yields normal tensors for backward.
    prefix.cpu()
    del adapter, pixels, cached
    gc.collect()
    torch.cuda.empty_cache()
    initial_batches = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for start in range(0, len(spatial), 2):
            initial_batches.append(tail.study_logits(torch.from_numpy(spatial[start:start+2]).to(device),
                                                    torch.arange(2, device=device), 2).sigmoid().cpu().numpy())
    initial = np.concatenate(initial_batches)
    expected = predict_linear_probe(payload, old)
    probability_error = float(np.max(abs(initial - expected)))
    if probability_error > 3e-5:
        raise ValueError(f"local head float32 parity exceeds engineering tolerance: {probability_error}")
    original = cpu_snapshot(tail.state_dict())
    counts = {}
    for record in records:
        key = (record["subject_id"], record["study_id"])
        counts[key] = counts.get(key, 0) + 1
    worst = sorted(counts.values(), reverse=True)
    trials = []
    chosen = None
    for studies in (4, 2, 1):
        tail.load_state_dict(original)
        tail.train()
        optimizer = torch.optim.AdamW([
            {"params": tail.stage.parameters(), "lr": 1e-5, "weight_decay": .01},
            {"params": tail.local_head.parameters(), "lr": 1e-4, "weight_decay": 0},
        ])
        scaler = torch.amp.GradScaler("cuda")
        image_count = sum(worst[:studies])
        x = torch.from_numpy(spatial[np.arange(image_count) % len(spatial)].copy()).to(device)
        image_to_study = torch.repeat_interleave(torch.arange(studies, device=device), torch.tensor(worst[:studies], device=device))
        targets = torch.randint(0, 2, (studies, 10), device=device).float()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = tail.study_logits(x, image_to_study, studies)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
                loss = loss + .5 * .01 * tail.local_head.weight.square().sum() / 10
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in tail.parameters() if p.requires_grad], 1.)
                scaler.step(optimizer)
                scaler.update()
            torch.cuda.synchronize()
            if not np.isfinite(float(loss)):
                raise ValueError("nonfinite engineering smoke loss")
            if any(parameter.grad is not None for parameter in prefix.parameters()):
                raise ValueError("frozen prefix received gradients")
            unchanged_bn = all(torch.equal(value.cpu(), original[name]) for name, value in tail.state_dict().items() if name.startswith("official_bn."))
            if not unchanged_bn:
                raise ValueError("frozen BN changed during engineering update")
            changed_stage = any(not torch.equal(value.cpu(), original[name]) for name, value in tail.state_dict().items() if name.startswith("stage."))
            if not changed_stage:
                raise ValueError("synthetic updates did not change any stage parameter")
            trials.append({"studies": studies, "images": image_count, "success": True,
                           "seconds_per_microbatch": (time.perf_counter() - started) / 2,
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                           "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                           "synthetic_updates_discarded": True})
            chosen = studies
        except torch.cuda.OutOfMemoryError as exc:
            trials.append({"studies": studies, "images": image_count, "success": False, "error": str(exc)})
        finally:
            tail.zero_grad(set_to_none=True)
            del optimizer, scaler, x, image_to_study, targets
            gc.collect()
            torch.cuda.empty_cache()
        if chosen is not None:
            break
    if chosen is None:
        raise RuntimeError("final-stage backward cannot fit even one worst-case study")
    tail.load_state_dict(original)
    result = {"status": "complete", "created_utc": now(), "selected_microbatch_studies": chosen,
              "effective_batch_studies": 16, "prefix_shape": list(spatial.shape[1:]),
              "prefix_dtype": str(spatial.dtype), "image_rows_checked": selected,
              "split_features_exact_equal": True, "old_image_cache_exact_equal": True,
              "initial_head_probability_max_error": probability_error,
              "worst_image_counts": worst[:4], "trials": trials,
              "real_candidate_fits": 0, "real_auc_evaluations": 0,
              "environment": environment(), "source_plan_sha256": sha256(args.source / "plan.json"),
              "source_head_sha256": sha256(args.source / "final_head.npz"),
              "script_sha256": sha256(__file__),
              "adapter_sha256": sha256(ROOT / "cxr/cxrlt_stage_adapter.py")}
    atomic_json(args.output, result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
