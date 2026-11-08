"""Study-level one-epoch fine-tuning for a cached ConvNeXt final stage."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import time
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from cxr.training_state import atomic_save, capture_rng, cpu_snapshot, restore_rng, safe_load


FORMAT_VERSION = 1


@dataclass(frozen=True)
class StageTrainingConfig:
    device: str = "cpu"
    amp: bool = False
    microbatch_studies: int = 4
    accumulation_studies: int = 16
    stage_lr: float = 1e-5
    stage_weight_decay: float = 0.01
    head_lr: float = 1e-4
    head_weight_decay: float = 0.0
    l2: float = 0.01
    clip_grad_norm: float = 1.0
    seed: int = 20261108
    checkpoint_every_windows: int = 20

    def validate(self):
        for name in ("microbatch_studies", "accumulation_studies", "checkpoint_every_windows"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.accumulation_studies < self.microbatch_studies:
            raise ValueError("accumulation_studies must be at least microbatch_studies")
        try:
            torch.device(self.device)
        except (TypeError, RuntimeError) as error:
            raise ValueError("device must name a valid torch device") from error
        if not isinstance(self.amp, bool):
            raise ValueError("amp must be a boolean")
        if self.amp and torch.device(self.device).type != "cuda":
            raise ValueError("AMP requires a CUDA device")
        for name in ("stage_lr", "stage_weight_decay", "head_lr", "head_weight_decay", "l2", "clip_grad_norm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.stage_lr <= 0 or self.head_lr <= 0 or self.clip_grad_norm <= 0:
            raise ValueError("learning rates and clip_grad_norm must be positive")


@dataclass(frozen=True)
class StudyBatch:
    """Cached image features and labels for a microbatch of studies.

    `image_to_study` maps each image to its local study row in `targets`; it is
    an index vector, not CSR offsets. `study_indices` preserves global row ids.
    """

    prefix_features: torch.Tensor
    image_to_study: torch.Tensor
    targets: torch.Tensor
    study_indices: torch.Tensor

    def validate(self, requested_indices: Sequence[int]):
        indices = torch.as_tensor(requested_indices, dtype=torch.long).reshape(-1)
        if self.prefix_features.ndim != 4:
            raise ValueError("prefix_features must have shape [images, channels, height, width]")
        if not torch.is_floating_point(self.prefix_features) or not torch.isfinite(self.prefix_features).all():
            raise ValueError("prefix_features must be finite floating-point data")
        if self.targets.ndim != 2 or self.targets.shape[1] != 10:
            raise ValueError("targets must have shape [studies, 10]")
        if self.targets.shape[0] != len(indices) or self.study_indices.shape != indices.shape:
            raise ValueError("batch study count does not match requested indices")
        if not torch.equal(self.study_indices.to(dtype=torch.long, device="cpu"), indices):
            raise ValueError("batch study indices differ from the requested order")
        mapping = self.image_to_study
        if mapping.ndim != 1 or mapping.numel() != self.prefix_features.shape[0]:
            raise ValueError("image_to_study must map every image to one study")
        if mapping.dtype not in (torch.int32, torch.int64):
            raise ValueError("image_to_study must contain integer indices")
        mapping = mapping.to(device="cpu", dtype=torch.long)
        if mapping.numel() == 0 or mapping.min() < 0 or mapping.max() >= len(indices):
            raise ValueError("image_to_study contains an invalid local study index")
        counts = torch.bincount(mapping, minlength=len(indices))
        if torch.any(counts == 0):
            raise ValueError("each study must contain at least one image")
        if not torch.isfinite(self.targets).all() or torch.any((self.targets < 0) | (self.targets > 1)):
            raise ValueError("targets must be finite values in [0, 1]")


FetchStudyBatch = Callable[[Sequence[int]], StudyBatch]
ForwardStudies = Callable[[nn.Module, StudyBatch], torch.Tensor]


def forward_studies(model: nn.Module, batch: StudyBatch) -> torch.Tensor:
    """Default model interface: ``study_logits(features, image_to_study, n)``."""
    method = getattr(model, "study_logits", None)
    if method is None:
        raise TypeError("model must expose study_logits(prefix_features, image_to_study, study_count)")
    return method(batch.prefix_features, batch.image_to_study, batch.targets.shape[0])


def configure_trainable_tail(model: nn.Module) -> None:
    """Freeze the public tail except its final stage and local linear head."""
    if not all(hasattr(model, name) for name in ("stage", "local_head")):
        raise TypeError("model must expose stage and local_head modules")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.stage, model.local_head):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    if not any(parameter.requires_grad for parameter in model.stage.parameters()):
        raise ValueError("stage has no trainable parameters")
    if not any(parameter.requires_grad for parameter in model.local_head.parameters()):
        raise ValueError("local_head has no trainable parameters")


def build_optimizer(model: nn.Module, config: StageTrainingConfig | None = None):
    config = config or StageTrainingConfig()
    config.validate()
    configure_trainable_tail(model)
    return torch.optim.AdamW([
        {"params": list(model.stage.parameters()), "lr": config.stage_lr,
         "weight_decay": config.stage_weight_decay},
        {"params": list(model.local_head.parameters()), "lr": config.head_lr,
         "weight_decay": config.head_weight_decay},
    ])


def _sha256_indices(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def _weight_parameter(model: nn.Module) -> torch.Tensor:
    weight = getattr(model.local_head, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or weight.shape[0] != 10:
        raise ValueError("local_head must expose a [10, features] weight tensor")
    return weight


def _regularization(model: nn.Module, l2: float) -> torch.Tensor:
    return 0.5 * l2 * _weight_parameter(model).square().sum() / 10.0


def _trainable_state(model: nn.Module) -> dict:
    buffers = {}
    for name in ("probe_mean", "probe_std", "_probe_loaded"):
        value = getattr(model, name, None)
        if isinstance(value, torch.Tensor):
            buffers[name] = cpu_snapshot(value)
        elif name in ("probe_mean", "probe_std", "_probe_loaded"):
            raise ValueError(f"model must expose tensor buffer {name}")
    if not bool(buffers["_probe_loaded"].item()):
        raise ValueError("local probe initialization has not been loaded")
    return {
        "stage": cpu_snapshot(model.stage.state_dict()),
        "local_head": cpu_snapshot(model.local_head.state_dict()),
        "buffers": buffers,
    }


def _load_trainable_state(model: nn.Module, saved: Mapping):
    if set(saved) != {"stage", "local_head", "buffers"}:
        raise ValueError("checkpoint trainable state is incomplete")
    model.stage.load_state_dict(saved["stage"], strict=True)
    model.local_head.load_state_dict(saved["local_head"], strict=True)
    if set(saved["buffers"]) != {"probe_mean", "probe_std", "_probe_loaded"}:
        raise ValueError("checkpoint probe buffers are incomplete")
    for name, value in saved["buffers"].items():
        target = getattr(model, name, None)
        if not isinstance(target, torch.Tensor) or target.shape != value.shape or target.dtype != value.dtype:
            raise ValueError(f"checkpoint buffer {name} is incompatible")
        target.copy_(value.to(device=target.device))
    if not bool(model._probe_loaded.item()):
        raise ValueError("checkpoint local probe initialization is incomplete")


def _save_checkpoint(path: Path, *, model, optimizer, scaler, generator, binding, config,
                     order: np.ndarray, next_study: int, progress: Mapping):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save({
        "format_version": FORMAT_VERSION,
        "binding": binding,
        "config": asdict(config),
        "study_order": torch.as_tensor(order, dtype=torch.int64),
        "next_study": next_study,
        "progress": dict(progress),
        "model": _trainable_state(model),
        "optimizer": cpu_snapshot(optimizer.state_dict()),
        "scaler": cpu_snapshot(scaler.state_dict()),
        "rng": capture_rng(generator),
    }, path)


def _restore_checkpoint(path: Path, *, model, optimizer, scaler, generator, binding,
                        config, order: np.ndarray):
    state = safe_load(path)
    if not isinstance(state, dict) or state.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported stage-training checkpoint format")
    if state.get("binding") != binding:
        raise ValueError("checkpoint initialization or index binding differs")
    if state.get("config") != asdict(config):
        raise ValueError("checkpoint training configuration differs")
    saved_order = state.get("study_order")
    if not isinstance(saved_order, torch.Tensor) or not torch.equal(
            saved_order.to(dtype=torch.int64, device="cpu"), torch.as_tensor(order, dtype=torch.int64)):
        raise ValueError("checkpoint fixed epoch study order differs")
    cursor = state.get("next_study")
    if isinstance(cursor, bool) or not isinstance(cursor, int) or not 0 <= cursor <= len(order):
        raise ValueError("checkpoint next-study cursor is invalid")
    _load_trainable_state(model, state.get("model", {}))
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    restore_rng(state["rng"], generator)
    progress = state.get("progress")
    if not isinstance(progress, dict) or progress.get("next_study") != cursor:
        raise ValueError("checkpoint progress cursor is inconsistent")
    return cursor, progress


def restore_trained_tail(path: str | Path, model: nn.Module, *, binding: Mapping[str, str],
                         config: StageTrainingConfig | None = None, require_complete: bool = True):
    """Load only the trained stage, local head and probe buffers for prediction."""
    state = safe_load(path)
    if not isinstance(state, dict) or state.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported stage-training checkpoint format")
    saved_binding = state.get("binding", {})
    if any(saved_binding.get(key) != value for key, value in binding.items()):
        raise ValueError("checkpoint initialization or data binding differs")
    if config is not None and state.get("config") != asdict(config):
        raise ValueError("checkpoint training configuration differs")
    order = state.get("study_order")
    cursor = state.get("next_study")
    if not isinstance(order, torch.Tensor) or isinstance(cursor, bool) or not isinstance(cursor, int):
        raise ValueError("checkpoint order or cursor is invalid")
    if not 0 <= cursor <= order.numel() or (require_complete and cursor != order.numel()):
        raise ValueError("checkpoint does not contain a completed epoch")
    _load_trainable_state(model, state.get("model", {}))
    model.eval()
    return {"next_study": cursor, "total_studies": int(order.numel()),
            "binding": saved_binding, "config": state["config"]}


def train_one_epoch(
    model: nn.Module,
    study_indices: Sequence[int],
    fetch_batch: FetchStudyBatch,
    *,
    config: StageTrainingConfig | None = None,
    forward: ForwardStudies = forward_studies,
    optimizer=None,
    scaler=None,
    device: str | torch.device | None = None,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    binding: Mapping[str, str] | None = None,
    progress_callback: Callable[[Mapping], None] | None = None,
):
    """Train one fixed epoch with study-weighted accumulation and atomic resume.

    `fetch_batch(global_study_ids)` reads the frozen-prefix 4D NumPy memmap.
    `forward(model, batch)` must return logits with shape `[studies, 10]`.
    The caller supplies data and initialization fingerprints in `binding`.
    """
    config = config or StageTrainingConfig()
    config.validate()
    if len(study_indices) == 0:
        raise ValueError("study_indices must be nonempty")
    indices = np.asarray(study_indices, dtype=np.int64)
    if indices.ndim != 1 or len(np.unique(indices)) != len(indices) or np.any(indices < 0):
        raise ValueError("study_indices must be unique nonnegative global indices")
    if binding is None or not isinstance(binding, Mapping) or not binding:
        raise ValueError("binding must include immutable initialization and data fingerprints")
    if any(not isinstance(k, str) or not isinstance(v, str) or not v for k, v in binding.items()):
        raise ValueError("binding entries must be nonempty string keys and values")
    binding = dict(binding) | {"study_indices_sha256": _sha256_indices(indices)}
    device = torch.device(config.device if device is None else device)
    if device != torch.device(config.device):
        raise ValueError("device argument must match the frozen training configuration")
    configure_trainable_tail(model)
    optimizer = optimizer or build_optimizer(model, config)
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=config.amp)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    permutation = torch.randperm(len(indices), generator=generator).numpy()
    order = indices[permutation]

    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if resume and checkpoint is None:
        raise ValueError("resume requires checkpoint_path")
    if resume and not checkpoint.is_file():
        raise FileNotFoundError("resume checkpoint does not exist")
    cursor, restored_progress = _restore_checkpoint(
        checkpoint, model=model, optimizer=optimizer, scaler=scaler,
        generator=generator, binding=binding, config=config, order=order,
    ) if resume else (0, {"total_bce": 0.0, "total_studies": 0, "steps": 0,
                         "skipped_steps": 0, "windows": 0, "elapsed_seconds": 0.0,
                         "next_study": 0})
    if cursor % config.accumulation_studies and cursor != len(order):
        raise ValueError("checkpoint cursor is not at an accumulation-window boundary")

    model.eval()
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    total_bce = float(restored_progress["total_bce"])
    total_studies = int(restored_progress["total_studies"])
    steps = int(restored_progress["steps"])
    skipped_steps = int(restored_progress["skipped_steps"])
    windows = int(restored_progress["windows"])
    elapsed_before = float(restored_progress["elapsed_seconds"])
    peak_memory = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    while cursor < len(order):
        window_end = min(cursor + config.accumulation_studies, len(order))
        window_size = window_end - cursor
        optimizer.zero_grad(set_to_none=True)
        window_bce = 0.0
        micro_start = cursor
        while micro_start < window_end:
            micro_end = min(micro_start + config.microbatch_studies, window_end)
            requested = order[micro_start:micro_end].tolist()
            batch = fetch_batch(requested)
            if not isinstance(batch, StudyBatch):
                raise TypeError("fetch_batch must return StudyBatch")
            batch.validate(requested)
            batch = StudyBatch(
                prefix_features=batch.prefix_features.to(device, non_blocking=True),
                image_to_study=batch.image_to_study.to(device=device, dtype=torch.long, non_blocking=True),
                targets=batch.targets.to(device=device, dtype=torch.float32, non_blocking=True),
                study_indices=batch.study_indices,
            )
            model.eval()
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=config.amp and device.type == "cuda"):
                logits = forward(model, batch)
                if logits.shape != batch.targets.shape:
                    raise ValueError("forward must return one ten-class logit row per study")
                per_study = nn.functional.binary_cross_entropy_with_logits(
                    logits.float(), batch.targets, reduction="none").mean(dim=1)
                if not torch.isfinite(per_study).all():
                    raise FloatingPointError("non-finite study BCE; refusing to advance the cursor")
                bce = per_study.mean()
                scaled_loss = bce * ((micro_end - micro_start) / window_size)
                if micro_end == window_end:
                    scaled_loss = scaled_loss + _regularization(model, config.l2)
            scaler.scale(scaled_loss).backward()
            window_bce += float(per_study.detach().sum().cpu())
            micro_start = micro_end

        scaler.unscale_(optimizer)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, config.clip_grad_norm)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale_before:
            skipped_steps += 1
        else:
            steps += 1
        optimizer.zero_grad(set_to_none=True)
        cursor = window_end
        total_bce += window_bce
        total_studies += window_size
        windows += 1
        checkpoint_due = windows % config.checkpoint_every_windows == 0 or cursor == len(order)
        elapsed = elapsed_before + (time.monotonic() - started)
        progress = {
            "next_study": cursor,
            "total_studies": total_studies,
            "total_bce": total_bce,
            "steps": steps,
            "skipped_steps": skipped_steps,
            "windows": windows,
            "loss": total_bce / max(total_studies, 1),
            "elapsed_seconds": elapsed,
            "studies_per_second": total_studies / max(elapsed, 1e-9),
        }
        if checkpoint is not None and checkpoint_due:
            _save_checkpoint(checkpoint, model=model, optimizer=optimizer, scaler=scaler,
                             generator=generator, binding=binding, config=config,
                             order=order, next_study=cursor, progress=progress)
        if checkpoint_due and progress_callback is not None:
            progress_callback(progress)

    if device.type == "cuda":
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    elapsed = elapsed_before + (time.monotonic() - started)
    return {
        "completed": cursor == len(order),
        "studies": total_studies,
        "steps": steps,
        "skipped_steps": skipped_steps,
        "windows": windows,
        "loss": total_bce / max(total_studies, 1),
        "elapsed_seconds": elapsed,
        "studies_per_second": total_studies / max(elapsed, 1e-9),
        "peak_memory_bytes": peak_memory,
        "next_study": cursor,
    }
