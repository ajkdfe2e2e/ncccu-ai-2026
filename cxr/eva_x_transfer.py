"""Local adapter for the pinned EVA-X-S CheXpert checkpoint."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from functools import partial
import importlib.util
from pathlib import Path
import pickle
import sys
import warnings

import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode


MODEL_REPOSITORY = "MapleF/eva_x"
MODEL_REVISION = "35ddcd6dab6ca99bbdb6cb45c8d1b093aefbd0ee"
UPSTREAM_REPOSITORY = "HUSTVL/EVA-X"
UPSTREAM_COMMIT = "eedd642c85339b25682ceb574b644cb9afe88e62"
WEIGHTS_NAME = "eva_x_small_patch16_merged520k_mim_chexpert_ft.pth"
FEATURE_SIZE = 389
INPUT_SIZE = 224
INPUT_RESIZE = 256
SOURCE_FEATURES = 384
SOURCE_CLASSES = 5
ROPE_PATCH_COUNT = 14 * 14
ROPE_HALF_WIDTH = 64
FEATURE_POOLING = "eva_x_mean_pooled_patch_features_384_plus_raw_logits_5"
_MEAN = (0.49185243, 0.49185243, 0.49185243)
_STD = (0.28509309, 0.28509309, 0.28509309)


class EvaXProcessor:
    """Apply the pinned official RGB evaluation transform."""

    input_size = INPUT_SIZE

    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(INPUT_RESIZE, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])

    def __call__(self, *, images, return_tensors="pt"):
        if return_tensors != "pt":
            raise ValueError("only return_tensors='pt' is supported")
        if not images:
            raise ValueError("images must be a nonempty sequence")
        values = []
        for image in images:
            if not hasattr(image, "convert"):
                raise ValueError("images must be PIL images")
            rgb = image.convert("RGB")
            values.append(self.transform(rgb))
        return {"pixel_values": torch.stack(values)}


class EvaXFeatureAdapter(nn.Module):
    """Expose the official mean-pooled representation and unmodified logits."""

    feature_size = FEATURE_SIZE
    input_size = INPUT_SIZE
    feature_pooling = FEATURE_POOLING

    def __init__(self, model):
        super().__init__()
        if (not callable(getattr(model, "forward_features", None))
                or not callable(getattr(model, "forward_head", None))
                or not isinstance(getattr(model, "head", None), nn.Module)):
            raise ValueError("EVA-X model must expose forward_features, forward_head, and head")
        self.model = model
        self.model.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, pixel_values):
        if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 4:
            raise ValueError("pixel_values must be a BCHW tensor")
        if pixel_values.shape[0] == 0 or pixel_values.shape[1] != 3:
            raise ValueError("pixel_values must contain a nonempty RGB batch")
        tokens = self.model.forward_features(pixel_values)
        if (not isinstance(tokens, torch.Tensor) or tokens.ndim != 3
                or tokens.shape[0] != pixel_values.shape[0]
                or tokens.shape[2] != SOURCE_FEATURES):
            raise ValueError(
                f"EVA-X forward_features must return BxNx{SOURCE_FEATURES} tokens"
            )
        features = self.model.forward_head(tokens, pre_logits=True)
        if features.shape != (pixel_values.shape[0], SOURCE_FEATURES):
            raise ValueError(
                f"EVA-X forward_head(pre_logits=True) must return Bx{SOURCE_FEATURES} features"
            )
        logits = self.model.head(features)
        if logits.shape != (pixel_values.shape[0], SOURCE_CLASSES):
            raise ValueError(f"EVA-X head must return Bx{SOURCE_CLASSES} raw logits")
        return torch.cat((features, logits), dim=1)


def adapt_model(model):
    """Wrap an already constructed official EVA-X model."""
    return EvaXFeatureAdapter(model)


def _load_official_module(model_dir: Path):
    source = model_dir / "eva_x.py"
    if not source.is_file():
        raise FileNotFoundError(f"pinned EVA-X source not found: {source}")
    module_name = f"_eva_x_official_{UPSTREAM_COMMIT.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned EVA-X model definition: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    if not callable(getattr(module, "EVA_X", None)) or not callable(
            getattr(module, "checkpoint_filter_fn", None)):
        raise ImportError("pinned EVA-X source must expose EVA_X and checkpoint_filter_fn")
    return module


def _load_checkpoint(path: Path):
    """Load a tensor-only state dict, permitting only legacy argparse metadata."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        safe_globals = getattr(torch.serialization, "safe_globals", None)
        if safe_globals is None:
            raise ValueError(
                "checkpoint is not weights_only-compatible and this PyTorch lacks safe_globals"
            )
        try:
            with safe_globals([argparse.Namespace]):
                payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(
                "checkpoint is not weights_only-compatible with the explicit argparse.Namespace allowlist"
            ) from exc
        warnings.warn(
            "Loaded legacy EVA-X checkpoint metadata with the explicit argparse.Namespace safe-global allowlist; "
            "weights_only remained enabled.",
            RuntimeWarning,
            stacklevel=2,
        )
    if not isinstance(payload, Mapping):
        raise ValueError("EVA-X checkpoint must contain a state_dict mapping")
    for key in ("model", "state_dict"):
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            payload = nested
            break
    if not payload or not all(isinstance(key, str) and isinstance(value, torch.Tensor)
                              for key, value in payload.items()):
        raise ValueError("EVA-X checkpoint does not contain a tensor-only state_dict")
    if all(key.startswith("module.") for key in payload):
        payload = {key[len("module."):]: value for key, value in payload.items()}
    return payload


def _restore_checkpoint_rope(state, model):
    """Restore the pinned 224px checkpoint RoPE values into timm's static buffer."""
    expected_keys = {f"rope.freqs_{name}" for name in ("sin", "cos")}
    expected_keys.update(
        f"blocks.{index}.attn.rope.freqs_{name}"
        for index in range(12)
        for name in ("sin", "cos")
    )
    rope_keys = {key for key in state if "rope" in key}
    if rope_keys != expected_keys:
        missing = sorted(expected_keys - rope_keys)
        unexpected = sorted(rope_keys - expected_keys)
        raise ValueError(f"unexpected EVA-X RoPE checkpoint keys; missing={missing}, unexpected={unexpected}")

    root = {}
    for name in ("sin", "cos"):
        key = f"rope.freqs_{name}"
        value = state[key]
        if (value.shape != (ROPE_PATCH_COUNT, ROPE_HALF_WIDTH)
                or value.dtype != torch.float32 or not torch.isfinite(value).all()):
            raise ValueError(f"EVA-X checkpoint {key} must be finite float32 196x64")
        root[name] = value

    for index in range(12):
        for name in ("sin", "cos"):
            key = f"blocks.{index}.attn.rope.freqs_{name}"
            value = state[key]
            if (value.shape != (ROPE_PATCH_COUNT, ROPE_HALF_WIDTH)
                    or value.dtype != torch.float32 or not torch.isfinite(value).all()
                    or not torch.equal(value, root[name])):
                raise ValueError(f"EVA-X checkpoint {key} does not match the fixed root RoPE buffer")

    rope = getattr(model, "rope", None)
    target = getattr(rope, "pos_embed", None)
    if not isinstance(target, torch.Tensor) or target.shape != (ROPE_PATCH_COUNT, 2 * ROPE_HALF_WIDTH):
        raise ValueError("EVA-X model must expose a static 196x128 rope.pos_embed buffer")
    with torch.no_grad():
        target.copy_(torch.cat((root["sin"], root["cos"]), dim=1).to(device=target.device, dtype=target.dtype))


def load_encoder(model_dir, device):
    """Load the pinned official EVA-X-S CheXpert checkpoint and processor."""
    model_dir = Path(model_dir)
    device = torch.device(device)
    module = _load_official_module(model_dir)
    model = module.EVA_X(
        img_size=INPUT_SIZE,
        patch_size=16,
        embed_dim=SOURCE_FEATURES,
        depth=12,
        num_heads=6,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(14, 14),
        num_classes=SOURCE_CLASSES,
        global_pool="avg",
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    weights = model_dir / WEIGHTS_NAME
    if not weights.is_file():
        raise FileNotFoundError(f"pinned EVA-X checkpoint not found: {weights}")
    checkpoint = _load_checkpoint(weights)
    _restore_checkpoint_rope(checkpoint, model)
    filtered_weights = module.checkpoint_filter_fn(checkpoint, model)
    model.load_state_dict(filtered_weights, strict=True)
    adapter = adapt_model(model).to(device).eval()
    return adapter, EvaXProcessor()
