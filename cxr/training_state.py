"""Safe epoch-boundary state: primitive containers/tensors, never pickle globals.

Exact continuation requires workers=0 and the same software/hardware configuration.
NumPy's legacy MT19937 keys are serialized as int64 tensors, not NumPy objects.
"""
import copy
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

FORMAT_VERSION = 1
RESUME_KEYS = (
    'architecture', 'class_ids', 'image_size', 'batch_size', 'validation_batch_size',
    'workers', 'seed', 'epochs', 'lr', 'scheduler', 'min_lr', 'smoke_studies',
    'train_manifest_sha256', 'valid_manifest_sha256', 'device', 'torch_version',
)


def cpu_snapshot(value):
    """Detach AND clone so subsequent CPU training cannot mutate saved best weights."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)


def atomic_save(payload, path):
    """Replace only a completed checkpoint, using a sibling temporary file."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def safe_load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def validate_init(payload, image_size, smoke):
    if not isinstance(payload, dict) or payload.get('architecture') != 'resnet18':
        raise ValueError('Init checkpoint architecture must be resnet18')
    if payload.get('class_ids') != list(range(1, 11)):
        raise ValueError('Init checkpoint class order does not match')
    if payload.get('image_size') != image_size:
        raise ValueError('Init checkpoint image_size must match --image-size; implicit resolution changes are not supported')
    if payload.get('smoke', False) and not smoke:
        raise ValueError('Cannot use a smoke checkpoint for real training')
    if 'model' not in payload:
        raise ValueError('Init checkpoint has no model weights')


def capture_rng(generator):
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return dict(
        python=random.getstate(),
        numpy=dict(algorithm=algorithm, keys=torch.tensor(keys.astype(np.int64)),
                   position=position, has_gauss=has_gauss, cached_gaussian=cached_gaussian),
        torch_cpu=torch.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        loader=generator.get_state(),
    )


def restore_rng(state, generator):
    cuda_states = state['torch_cuda']
    if len(cuda_states) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
        raise ValueError('CUDA device count differs from checkpoint; exact RNG resume is unavailable')
    random.setstate(state['python'])
    numpy = state['numpy']
    np.random.set_state((numpy['algorithm'], numpy['keys'].numpy().astype(np.uint32),
                         numpy['position'], numpy['has_gauss'], numpy['cached_gaussian']))
    torch.set_rng_state(state['torch_cpu'])
    if cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)
    generator.set_state(state['loader'])


def validate_resume(state, config):
    if not isinstance(state, dict) or state.get('format_version') != FORMAT_VERSION:
        raise ValueError('Not an epoch-boundary training checkpoint; use --init-checkpoint for inference-only best.pt')
    saved = state.get('config', {})
    if config.get('workers') != 0 or saved.get('workers') != 0:
        raise ValueError('Exact resume requires --workers 0 in both original and resumed runs')
    differences = [key for key in RESUME_KEYS if key not in config or key not in saved or config[key] != saved[key]]
    if differences:
        raise ValueError('Incompatible resume configuration: ' + ', '.join(differences))
    epoch = state.get('completed_epochs')
    if not isinstance(epoch, int) or isinstance(epoch, bool) or not 1 <= epoch < config['epochs']:
        raise ValueError('Resume checkpoint must have completed epochs remaining in the original horizon')
    required = ('model', 'optimizer', 'scaler', 'scheduler', 'rng', 'best_score', 'best_payload')
    if any(key not in state for key in required) or state['best_payload'] is None:
        raise ValueError('Incomplete training state or missing previous best payload')
    validate_init(state['best_payload'], config['image_size'], bool(config['smoke_studies']))
    if (state['scheduler'] is None) != (config['scheduler'] == 'none'):
        raise ValueError('Scheduler state does not match configuration')


def capture_training_state(model, optimizer, scaler, scheduler, generator, config,
                           completed_epochs, best_score, best_payload):
    return dict(format_version=FORMAT_VERSION, config=copy.deepcopy(config),
                completed_epochs=completed_epochs, model=cpu_snapshot(model.state_dict()),
                optimizer=cpu_snapshot(optimizer.state_dict()), scaler=cpu_snapshot(scaler.state_dict()),
                scheduler=cpu_snapshot(scheduler.state_dict()) if scheduler is not None else None,
                rng=capture_rng(generator), best_score=best_score,
                best_payload=cpu_snapshot(best_payload))


def restore_training_state(state, model, optimizer, scaler, scheduler, generator, config):
    validate_resume(state, config)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scaler.load_state_dict(state['scaler'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])
    # Restore last: construction/loading must not consume randomness after this point.
    restore_rng(state['rng'], generator)
    return state['completed_epochs'], state['best_score'], cpu_snapshot(state['best_payload'])
