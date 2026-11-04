"""Train on official train split, select checkpoints on study-level validation AUC."""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cxr.data import read_records
from cxr.engine import ImageDataset, create_model, load_imagenet_backbone, seed_everything, choose_device, sha256
from cxr.metrics import aggregate_studies, evaluate
from cxr.safety import fresh_run_directory
from cxr.training_state import (atomic_save, safe_load, validate_init, validate_resume, cpu_snapshot,
                                capture_training_state, restore_training_state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('cxr_first'))
    parser.add_argument('--output', type=Path, default=Path('outputs/resnet18'))
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument('--pretrained', type=Path, help='Audited torchvision ImageNet ResNet18 weights')
    initialization.add_argument('--from-scratch', action='store_true', help='Explicit random initialization, for smoke tests')
    initialization.add_argument('--init-checkpoint', type=Path,
                                help='Inference best.pt: fine-tune with a NEW optimizer/horizon; image size must match')
    initialization.add_argument('--resume', type=Path,
                                help='Epoch-boundary last-training.pt: restore complete state into a fresh output directory')
    parser.add_argument('--epochs', type=int, default=12, help='Total scheduler/training horizon, unchanged on resume')
    parser.add_argument('--stop-after-epochs', type=int,
                        help='Stop after this many epochs in this invocation, without changing the total horizon')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--validation-batch-size', type=int, default=1,
                        help='Default 1 matches Predictor numerical ordering; larger batches may differ slightly')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--workers', type=int, default=0, help='Exact resume requires 0 in both segments')
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--scheduler', choices=('none', 'cosine'), default='none')
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--smoke-studies', type=int, default=0, help='Train/valid only, entire studies; not a leaderboard estimate')
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 2 or args.validation_batch_size < 1
            or not 64 <= args.image_size <= 2048 or args.smoke_studies < 0 or args.workers < 0
            or not np.isfinite(args.lr) or args.lr <= 0 or not np.isfinite(args.min_lr) or args.min_lr < 0
            or (args.scheduler == 'cosine' and args.min_lr > args.lr)
            or (args.stop_after_epochs is not None and args.stop_after_epochs < 1)):
        parser.error('Invalid training arguments: positive epochs/lr, batch-size >=2, validation batch >=1, '
                     'image-size 64..2048, workers/smoke >=0, finite min-lr >=0 (<=lr for cosine), stop-after >=1')
    if args.resume and args.workers != 0:
        parser.error('--resume requires --workers 0 for exact continuation')
    seed_everything(args.seed)
    torch.set_num_threads(4)
    device = choose_device(args.device)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(architecture='resnet18', class_ids=list(range(1, 11)), device=str(device),
                  train_manifest_sha256=sha256(args.data_root / 'train.csv'),
                  valid_manifest_sha256=sha256(args.data_root / 'valid.csv'),
                  torch_version=str(torch.__version__))
    resume_state = safe_load(args.resume) if args.resume else None
    init_payload = safe_load(args.init_checkpoint) if args.init_checkpoint else None
    if resume_state is not None:
        validate_resume(resume_state, config)
    if init_payload is not None:
        validate_init(init_payload, args.image_size, bool(args.smoke_studies))
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
    model = create_model()
    if args.pretrained:
        load_imagenet_backbone(model, args.pretrained)
    elif init_payload is not None:
        model.load_state_dict(init_payload['model'])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
                 if args.scheduler == 'cosine' else None)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device), reduction='none')
    train_ds = IndexedDataset(ImageDataset(train, args.image_size, True))
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        generator=generator, pin_memory=device.type == 'cuda', drop_last=False)
    valid_loader = DataLoader(ImageDataset(valid, args.image_size), batch_size=args.validation_batch_size,
                              num_workers=args.workers, pin_memory=device.type == 'cuda')
    config.update(output=str(args.output), train_images=len(train), valid_images=len(valid),
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
            resume_state, model, optimizer, scaler, scheduler, generator, config)
        atomic_save(best_payload, args.output / 'best.pt')
        del resume_state
    config['start_epoch'] = start_epoch
    (args.output / 'run.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(json.dumps(config, indent=2), flush=True)
    final_epoch = min(args.epochs, start_epoch + args.stop_after_epochs) if args.stop_after_epochs else args.epochs
    for epoch in range(start_epoch, final_epoch):
        model.train()
        total_loss = 0.
        seen_images = 0
        started = time.perf_counter()
        epoch_lr = optimizer.param_groups[0]['lr']
        for step, (images, labels, indices) in enumerate(loader, 1):
            # BatchNorm cannot train with one image and a 1x1 final feature map at tiny smoke sizes.
            if len(images) == 1:
                model.eval()
            else:
                model.train()
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                losses = loss_fn(model(images), labels).mean(1)
                weights = image_weights[indices].to(device)
                loss = (losses * weights).mean() / image_weights.mean().to(device)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(images)
            seen_images += len(images)
            if step % 100 == 0:
                elapsed = time.perf_counter() - started
                print(f'epoch={epoch+1} step={step}/{len(loader)} lr={epoch_lr:.8g} '
                      f'elapsed_s={elapsed:.1f} images_per_s={seen_images / elapsed:.2f}', flush=True)
        train_elapsed = time.perf_counter() - started
        model.eval()
        probabilities = []
        with torch.inference_mode():
            for images, _ in valid_loader:
                # Full precision, batch=1 by default: same forward ordering as Predictor.
                probabilities.append(model(images.to(device)).float().sigmoid().cpu().numpy())
        _, study_probs, study_targets = aggregate_studies(valid, np.concatenate(probabilities))
        metrics = evaluate(study_targets, study_probs)
        score = metrics['macro_auc']
        if score is None:
            if not args.smoke_studies:
                raise ValueError('Validation AUC undefined for all classes')
            score = -1.
        if score > best:
            best = score
            best_payload = dict(architecture='resnet18', class_ids=list(range(1, 11)), image_size=args.image_size,
                                thresholds=[0.5] * 10, inference_precision='fp32',
                                model=cpu_snapshot(model.state_dict()), epoch=epoch + 1,
                                validation_macro_auc=metrics['macro_auc'], smoke=bool(args.smoke_studies))
            atomic_save(best_payload, args.output / 'best.pt')
        if scheduler is not None:
            scheduler.step()
        state = capture_training_state(model, optimizer, scaler, scheduler, generator, config,
                                       epoch + 1, best, best_payload)
        atomic_save(state, args.output / 'last-training.pt')
        del state
        row = dict(epoch=epoch + 1, train_loss=total_loss / len(train), lr=epoch_lr,
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
