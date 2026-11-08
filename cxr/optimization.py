"""Small training helpers for EMA weights and image-weighted accumulation."""

import copy
import math

import torch


class ModelEMA:
    """Maintain a detached exponential moving average of a model's state."""

    def __init__(self, model, decay=0.999, device=None):
        if isinstance(decay, bool) or not isinstance(decay, (int, float)) or not math.isfinite(decay) or not 0 <= decay < 1:
            raise ValueError('decay must be finite and in [0, 1)')
        self.decay = float(decay)
        self.updates = 0
        self.module = copy.deepcopy(model).eval()
        self.module.requires_grad_(False)
        if device is not None:
            self.module.to(device)

    @torch.no_grad()
    def update(self, model):
        source = model.state_dict()
        target = self.module.state_dict()
        if source.keys() != target.keys():
            raise ValueError('Model state keys are incompatible with EMA module')
        for key, averaged in target.items():
            current = source[key].detach().to(device=averaged.device)
            if current.shape != averaged.shape or current.dtype != averaged.dtype:
                raise ValueError(f'Model state is incompatible at {key}')
            if torch.is_floating_point(averaged) or torch.is_complex(averaged):
                averaged.mul_(self.decay).add_(current, alpha=1 - self.decay)
            else:
                averaged.copy_(current)
        self.updates += 1

    def state_dict(self):
        return {
            'decay': self.decay,
            'updates': self.updates,
            'module': {key: value.detach().clone() for key, value in self.module.state_dict().items()},
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict) or set(state) != {'decay', 'updates', 'module'}:
            raise ValueError('Invalid EMA state')
        decay = state['decay']
        updates = state['updates']
        if isinstance(decay, bool) or not isinstance(decay, (int, float)) or not math.isfinite(decay) or float(decay) != self.decay:
            raise ValueError('EMA decay is incompatible')
        if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
            raise ValueError('EMA update count is invalid')
        saved = state['module']
        current = self.module.state_dict()
        if not isinstance(saved, dict) or saved.keys() != current.keys():
            raise ValueError('EMA module state keys are incompatible')
        for key, value in saved.items():
            if not isinstance(value, torch.Tensor) or value.shape != current[key].shape or value.dtype != current[key].dtype:
                raise ValueError(f'EMA module state is incompatible at {key}')
        self.module.load_state_dict(saved, strict=True)
        self.module.eval()
        self.module.requires_grad_(False)
        self.updates = updates


def accumulation_window_images(batch_index, batch_size, dataset_size, accumulation_steps):
    """Return the actual image count in the accumulation window for a batch."""
    values = (batch_index, batch_size, dataset_size, accumulation_steps)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError('batch index, sizes, and accumulation steps must be integers')
    if batch_index < 0 or batch_size <= 0 or dataset_size <= 0 or accumulation_steps <= 0:
        raise ValueError('batch index must be nonnegative and sizes/steps must be positive')
    batch_count = (dataset_size + batch_size - 1) // batch_size
    if batch_index >= batch_count:
        raise ValueError('batch index is outside the dataset')
    first_batch = (batch_index // accumulation_steps) * accumulation_steps
    first_image = first_batch * batch_size
    end_image = min(dataset_size, (first_batch + accumulation_steps) * batch_size)
    return end_image - first_image
