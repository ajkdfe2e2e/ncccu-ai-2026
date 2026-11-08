"""Train on official train split, select checkpoints on study-level validation AUC."""
import argparse
import json
import math
import platform
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint_sequential

from cxr.data import read_records
from cxr.engine import ImageDataset, create_model, load_imagenet_backbone, seed_everything, choose_device, sha256
from cxr.models import SUPPORTED_ARCHITECTURES
from cxr.image_cache import CachedImageDataset
from cxr.metrics import aggregate_studies, evaluate
from cxr.optimization import ModelEMA, accumulation_window_images
from cxr.safety import fresh_run_directory, safe_output
from cxr.training_state import (atomic_save, safe_load, validate_init, validate_resume, cpu_snapshot,
                                capture_training_state, restore_training_state)


def create_scheduler(optimizer, epochs, scheduler='none', min_lr=1e-6, warmup_epochs=0):
    """Epoch schedule; warmup never changes the fixed total training horizon."""
    if scheduler == 'none':
        return None
    if not warmup_epochs:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    peak_lr = optimizer.param_groups[0]['lr']
    floor = min_lr / peak_lr

    def multiplier(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = min(1., (epoch - warmup_epochs) / (epochs - warmup_epochs))
        return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def training_logits(model, images, gradient_checkpointing=False):
    if not gradient_checkpointing:
        return model(images)
    # ConvNeXt's four feature stages retain their stochastic-depth RNG on recompute.
    features = checkpoint_sequential(model.features, 4, images, use_reentrant=False,
                                     preserve_rng_state=True)
    return model.classifier(model.avgpool(features))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('cxr_first'))
    parser.add_argument('--output', type=Path, default=Path('outputs/resnet18'))
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument('--pretrained', type=Path, help='Audited torchvision ImageNet weights for --architecture')
    initialization.add_argument('--from-scratch', action='store_true', help='Explicit random initialization, for smoke tests')
    initialization.add_argument('--init-checkpoint', type=Path,
                                help='Inference best.pt: fine-tune with a NEW optimizer/horizon; image size must match')
    initialization.add_argument('--resume', type=Path,
                                help='Epoch-boundary last-training.pt: restore complete state into a fresh output directory')
    parser.add_argument('--epochs', type=int, default=12, help='Total scheduler/training horizon, unchanged on resume')
    parser.add_argument('--stop-after-epochs', type=int,
                        help='Stop after this many epochs in this invocation, without changing the total horizon')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--architecture', choices=SUPPORTED_ARCHITECTURES, default='resnet18')
    parser.add_argument('--accumulation-steps', type=int, default=1,
                        help='Microbatches per optimizer update, normalized by actual images including the last window')
    parser.add_argument('--validation-batch-size', type=int, default=1,
                        help='Default 1 matches Predictor numerical ordering; larger batches may differ slightly')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--image-cache', type=Path,
                        help='Optional lossless RGB/letterbox cache outside raw data; random augmentation remains live')
    parser.add_argument('--workers', type=int, default=0, help='Exact resume requires 0 in both segments')
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--scheduler', choices=('none', 'cosine'), default='none')
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--warmup-epochs', type=int, default=0, help='Linear warmup before cosine, within total --epochs')
    parser.add_argument('--ema-decay', type=float, default=0., help='0 disables EMA; positive decay selects/exports EMA weights')
    parser.add_argument('--ema-device', choices=('model', 'cpu'), default='model', help='CPU EMA saves GPU memory')
    parser.add_argument('--gradient-checkpointing', action='store_true', help='Recompute ConvNeXt stages to reduce activation memory')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--smoke-studies', type=int, default=0, help='Train/valid only, entire studies; not a leaderboard estimate')
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < (2 if args.architecture == 'resnet18' else 1) or args.validation_batch_size < 1
            or args.accumulation_steps < 1
            or not 64 <= args.image_size <= 2048 or args.smoke_studies < 0 or args.workers < 0
            or not np.isfinite(args.lr) or args.lr <= 0 or not np.isfinite(args.min_lr) or args.min_lr < 0
            or (args.scheduler == 'cosine' and args.min_lr > args.lr)
            or not np.isfinite(args.weight_decay) or args.weight_decay < 0
            or not 0 <= args.warmup_epochs < args.epochs
            or (args.warmup_epochs and args.scheduler != 'cosine')
            or not np.isfinite(args.ema_decay) or not 0 <= args.ema_decay < 1
            or (args.gradient_checkpointing and args.architecture != 'convnext_tiny')
            or (args.stop_after_epochs is not None and args.stop_after_epochs < 1)):
        parser.error('Invalid training arguments: positive epochs/lr/batches/accumulation, ResNet batch >=2, '
                     'image-size 64..2048, workers/smoke >=0, finite weight-decay/min-lr >=0 (min-lr <=lr for cosine), '
                     '0 <= warmup < epochs (cosine only), EMA decay in [0,1), ConvNeXt-only checkpointing, stop-after >=1')
    if args.resume and args.workers != 0:
        parser.error('--resume requires --workers 0 for exact continuation')
    if args.image_cache is not None:
        args.image_cache = safe_output(args.image_cache, args.data_root)
    seed_everything(args.seed)
    torch.set_num_threads(4)
    device = choose_device(args.device)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(class_ids=list(range(1, 11)), device=str(device),
                  train_manifest_sha256=sha256(args.data_root / 'train.csv'),
                  valid_manifest_sha256=sha256(args.data_root / 'valid.csv'),
                  torch_version=str(torch.__version__), torchvision_version=version('torchvision'),
                  python_version=platform.python_version(), numpy_version=np.__version__,
                  pillow_version=version('Pillow'), cuda_runtime=torch.version.cuda,
                  cuda_device=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                  source_sha256={name: sha256(Path(__file__).resolve().parent / name)
                                 for name in ('train.py', 'cxr/models.py', 'cxr/engine.py', 'cxr/data.py',
                                              'cxr/image_cache.py', 'cxr/metrics.py', 'cxr/optimization.py',
                                              'cxr/training_state.py')})
    resume_state = safe_load(args.resume) if args.resume else None
    init_payload = safe_load(args.init_checkpoint) if args.init_checkpoint else None
    if resume_state is not None:
        validate_resume(resume_state, config)
    if init_payload is not None:
        validate_init(init_payload, args.image_size, bool(args.smoke_studies), args.architecture)
    # No input checkpoint is ever updated; nonempty output directories are refused.
    args.output = fresh_run_directory(args.output, args.data_root)
    train = read_records(args.data_root / 'train.csv', args.data_root / 'train')
    valid = read_records(args.data_root / 'valid.csv', args.data_root / 'valid')
    if any(r['labels'] is None for r in train + valid):
        raise ValueError('Training and validation require labels')
    overlaps = set(r['subject_id'] for r in train) & set(r['subject_id'] for r in valid)
    if overlaps:
        print(f'WARNING: {len(overlaps)} patients overlap official train/valid; not patient-independent validation', flush=True)
    if args.smoke_studies:
        def subset(records):
            ids = set(list(dict.fromkeys(r['study_id'] for r in records))[:args.smoke_studies])
            return [r for r in records if r['study_id'] in ids]
        train, valid = subset(train), subset(valid)
    if len(train) < 2 or not valid:
        raise ValueError('Insufficient data')
    # Weight each image inversely to images per study so multi-view studies do not dominate.
    counts = Counter(r['study_id'] for r in train)
    image_weights = torch.tensor([1 / counts[r['study_id']] for r in train])
    unique = {r['study_id']: r for r in train}
    targets = np.asarray([r['labels'] for r in unique.values()])
    positives = targets.sum(0)
    pos_weight = np.clip((len(targets) - positives) / np.maximum(positives, 1), 1, 10)
    model = create_model() if args.architecture == 'resnet18' else create_model(args.architecture)
    if args.pretrained:
        load_imagenet_backbone(model, args.pretrained, args.architecture)
    elif init_payload is not None:
        model.load_state_dict(init_payload['model'])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    scheduler = create_scheduler(optimizer, args.epochs, args.scheduler, args.min_lr, args.warmup_epochs)
    ema = (ModelEMA(model, args.ema_decay, device='cpu' if args.ema_device == 'cpu' else device)
           if args.ema_decay else None)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device), reduction='none')
    def dataset(records, training=False):
        return (CachedImageDataset(records, args.image_size, training, cache_root=args.image_cache)
                if args.image_cache is not None else ImageDataset(records, args.image_size, training))

    train_ds = IndexedDataset(dataset(train, True))
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        generator=generator, pin_memory=device.type == 'cuda', drop_last=False)
    valid_loader = DataLoader(dataset(valid), batch_size=args.validation_batch_size,
                              num_workers=args.workers, pin_memory=device.type == 'cuda')
    config.update(output=str(args.output), train_images=len(train), valid_images=len(valid),
                  effective_batch_size=args.batch_size * args.accumulation_steps,
                  validation_weights='ema' if ema else 'model',
                  validation_patient_overlap=len(overlaps), pretrained_sha256=sha256(args.pretrained) if args.pretrained else None,
                  init_checkpoint_sha256=sha256(args.init_checkpoint) if args.init_checkpoint else None,
                  resume_sha256=sha256(args.resume) if args.resume else None, pos_weight=pos_weight.tolist())
    config['initialization_lineage'] = (resume_state['config'].get('initialization_lineage')
                                        if resume_state is not None else
                                        dict(mode='init-checkpoint' if args.init_checkpoint else
                                             'pretrained' if args.pretrained else 'from-scratch',
                                             source=str(args.init_checkpoint or args.pretrained)
                                             if args.init_checkpoint or args.pretrained else None,
                                             sha256=config['init_checkpoint_sha256'] or config['pretrained_sha256']))
    start_epoch, best, best_payload = 0, -float('inf'), None
    if resume_state is not None:
        start_epoch, best, best_payload = restore_training_state(
            resume_state, model, optimizer, scaler, scheduler, generator, config, ema=ema)
        atomic_save(best_payload, args.output / 'best.pt')
        del resume_state
    config['start_epoch'] = start_epoch
    (args.output / 'run.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(json.dumps(config, indent=2), flush=True)
    final_epoch = min(args.epochs, start_epoch + args.stop_after_epochs) if args.stop_after_epochs else args.epochs
    mean_image_weight = image_weights.mean().to(device)
    for epoch in range(start_epoch, final_epoch):
        model.train()
        total_loss = 0.
        seen_images = 0
        started = time.perf_counter()
        epoch_lr = optimizer.param_groups[0]['lr']
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps = skipped_steps = 0
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        for step, (images, labels, indices) in enumerate(loader, 1):
            # BatchNorm cannot train with one image and a 1x1 final feature map at tiny smoke sizes.
            if args.architecture == 'resnet18' and len(images) == 1:
                model.eval()
            else:
                model.train()
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                losses = loss_fn(training_logits(model, images, args.gradient_checkpointing), labels).mean(1)
                weights = image_weights[indices].to(device)
                loss = (losses * weights).mean() / mean_image_weight
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at epoch={epoch + 1}, step={step}')
            window_images = accumulation_window_images(step - 1, args.batch_size, len(train), args.accumulation_steps)
            scaler.scale(loss * len(images) / window_images).backward()
            if step % args.accumulation_steps == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    optimizer_steps += 1
                    if ema is not None:
                        ema.update(model)
                else:
                    skipped_steps += 1
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach()) * len(images)
            seen_images += len(images)
            if step % 100 == 0:
                elapsed = time.perf_counter() - started
                print(f'epoch={epoch+1} step={step}/{len(loader)} optimizer_steps={optimizer_steps} lr={epoch_lr:.8g} '
                      f'elapsed_s={elapsed:.1f} images_per_s={seen_images / elapsed:.2f}', flush=True)
        train_elapsed = time.perf_counter() - started
        model.eval()
        validation_model = ema.module.to(device) if ema is not None else model
        validation_model.eval()
        probabilities = []
        with torch.inference_mode():
            for images, _ in valid_loader:
                # Full precision, batch=1 by default: same forward ordering as Predictor.
                probabilities.append(validation_model(images.to(device)).float().sigmoid().cpu().numpy())
        _, study_probs, study_targets = aggregate_studies(valid, np.concatenate(probabilities))
        metrics = evaluate(study_targets, study_probs)
        score = metrics['macro_auc']
        if score is None:
            if not args.smoke_studies:
                raise ValueError('Validation AUC undefined for all classes')
            score = -1.
        if score > best:
            best = score
            best_payload = dict(architecture=args.architecture, class_ids=list(range(1, 11)), image_size=args.image_size,
                                thresholds=[0.5] * 10, inference_precision='fp32',
                                model=cpu_snapshot(validation_model.state_dict()), epoch=epoch + 1,
                                weights_source='ema' if ema is not None else 'model', ema_decay=args.ema_decay,
                                validation_macro_auc=metrics['macro_auc'], smoke=bool(args.smoke_studies))
            atomic_save(best_payload, args.output / 'best.pt')
        if ema is not None and args.ema_device == 'cpu':
            ema.module.cpu()
        if scheduler is not None:
            scheduler.step()
        state = capture_training_state(model, optimizer, scaler, scheduler, generator, config,
                                       epoch + 1, best, best_payload, ema=ema)
        atomic_save(state, args.output / 'last-training.pt')
        del state
        row = dict(epoch=epoch + 1, train_loss=total_loss / len(train), lr=epoch_lr,
                   optimizer_steps=optimizer_steps, skipped_optimizer_steps=skipped_steps,
                   ema_updates=ema.updates if ema is not None else None,
                   peak_cuda_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None,
                   next_lr=optimizer.param_groups[0]['lr'], elapsed_s=time.perf_counter() - started,
                   train_elapsed_s=train_elapsed, train_images_per_s=len(train) / train_elapsed, **metrics)
        with (args.output / 'history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row, allow_nan=False), flush=True)
    print(f'Best checkpoint: {args.output / "best.pt"}', flush=True)
    print(f'Epoch-boundary state: {args.output / "last-training.pt"}', flush=True)


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        image, target = self.dataset[index]
        return image, target, index


if __name__ == '__main__':
    main()
