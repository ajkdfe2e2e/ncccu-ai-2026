"""Measure train-only decoding and CUDA compute separately before a long run."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from cxr.data import _read_metadata, _attach_paths
from cxr.engine import ImageDataset, load_imagenet_backbone, seed_everything
from cxr.image_cache import CachedImageDataset
from cxr.models import create_model
from cxr.optimization import ModelEMA
from cxr.safety import safe_output
from train import training_logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--pretrained', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[2, 4, 8])
    parser.add_argument('--examples', type=int, default=16)
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--image-size', type=int, default=384)
    parser.add_argument('--gradient-checkpointing', action='store_true')
    parser.add_argument('--image-cache', type=Path)
    args = parser.parse_args()
    if args.examples < max(args.batch_sizes) or min(args.batch_sizes) < 1 or args.steps < 1:
        parser.error('Need positive batches/steps and examples >= largest batch')
    args.output = safe_output(args.output, args.data_root, inputs=[args.pretrained])
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this profile')
    seed_everything(2026)
    torch.set_num_threads(4)
    metadata = _read_metadata(args.data_root / 'train.csv')
    records, _ = _attach_paths(metadata[:args.examples], args.data_root / 'train')
    dataset = ImageDataset(records, args.image_size, training=True)
    started = time.perf_counter()
    torch.manual_seed(2026)
    examples = [dataset[i] for i in range(len(dataset))]
    decode_elapsed = time.perf_counter() - started
    images = torch.stack([item[0] for item in examples]).cuda()
    labels = torch.stack([item[1] for item in examples]).cuda()
    report = dict(device=torch.cuda.get_device_name(), image_size=args.image_size,
                  source='official train only; profiling updates are discarded',
                  decode_images=len(dataset), decode_seconds=decode_elapsed,
                  decode_images_per_s=len(dataset) / decode_elapsed,
                  gradient_checkpointing=args.gradient_checkpointing, batches=[])
    if args.image_cache:
        cache_root = safe_output(args.image_cache, args.data_root)
        cached = CachedImageDataset(records, args.image_size, training=True, cache_root=cache_root)
        cache_times = []
        for _ in range(2):
            torch.manual_seed(2026)
            started = time.perf_counter()
            actual = [cached[i] for i in range(len(cached))]
            cache_times.append(time.perf_counter() - started)
            if not all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(examples, actual)):
                raise AssertionError('Cached preprocessing differs from original tensors')
        report['cache'] = dict(path=str(cache_root), cold_seconds=cache_times[0], warm_seconds=cache_times[1],
                               warm_images_per_s=len(dataset) / cache_times[1], pixel_and_target_equality=True)
    for batch_size in args.batch_sizes:
        model = optimizer = scaler = ema = loss = None
        try:
            model = create_model('convnext_tiny')
            load_imagenet_backbone(model, args.pretrained, 'convnext_tiny')
            model.cuda().train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.01)
            scaler = torch.amp.GradScaler('cuda')
            ema = ModelEMA(model, .999)
            torch.cuda.reset_peak_memory_stats()
            for step in range(args.steps + 2):
                if step == 2:
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    logits = training_logits(model, images[:batch_size], args.gradient_checkpointing)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels[:batch_size])
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    ema.update(model)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            optimizer.zero_grad(set_to_none=True)
            with torch.inference_mode():
                probabilities = ema.module(images[:1]).float().sigmoid()
            if not torch.isfinite(probabilities).all():
                raise FloatingPointError('Profile validation probabilities are not finite')
            row = dict(batch_size=batch_size, status='ok', compute_seconds=elapsed,
                       compute_images_per_s=args.steps * batch_size / elapsed,
                       peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
        except torch.cuda.OutOfMemoryError as error:
            row = dict(batch_size=batch_size, status='out-of-memory', error=str(error))
        finally:
            model = optimizer = scaler = ema = loss = None
            if 'logits' in locals():
                del logits
            gc.collect()
            torch.cuda.empty_cache()
        report['batches'].append(row)
        print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
