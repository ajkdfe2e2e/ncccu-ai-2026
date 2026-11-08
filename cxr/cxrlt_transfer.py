"""Local adapter for the supervised CXR-LT 2026 ConvNeXtV2 checkpoint."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn


REVISION = "24bf8bdfd8d0262899167b030be16ab89f78ae20"
WEIGHTS_NAME = "convnextv2_base_mimic-cxr_padchest_csra_dbcas.safetensors"
FEATURE_SIZE = 1054
INPUT_SIZE = 512
FEATURE_POOLING = "supervised_bn_gap_1024_plus_csra_raw_logits_30"
_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


class CXRLTProcessor:
    """Preprocess PIL images like the model card's OpenCV inference example."""

    input_size = INPUT_SIZE

    def __call__(self, *, images, return_tensors="pt"):
        if return_tensors != "pt":
            raise ValueError("only return_tensors='pt' is supported")
        if not images:
            raise ValueError("images must be a nonempty sequence")
        batch = []
        for image in images:
            if not hasattr(image, "mode"):
                raise ValueError("images must be PIL images")
            if image.mode == "L":
                gray = np.asarray(image, dtype=np.uint8)
            else:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE))
            rgb_gray = np.repeat(resized[..., None], 3, axis=2).astype(np.float32) / 255.0
            normalized = (rgb_gray - _MEAN) / _STD
            batch.append(np.transpose(normalized, (2, 0, 1)))
        return {"pixel_values": torch.from_numpy(np.stack(batch, axis=0))}


class ConvNeXtV2FeatureAdapter(nn.Module):
    """Expose frozen BN spatial GAP features and the official head's raw logits."""

    feature_size = FEATURE_SIZE
    input_size = INPUT_SIZE
    feature_pooling = FEATURE_POOLING

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Keep all BatchNorm and other encoder modules in inference mode.
        super().train(False)
        return self

    def forward(self, pixel_values):
        spatial = self.model.bn(self.model.backbone(pixel_values))
        if spatial.ndim != 4 or spatial.shape[1] != 1024:
            raise ValueError(f"expected BN feature map with 1024 channels, got {tuple(spatial.shape)}")
        pooled = spatial.mean(dim=(2, 3))
        logits = self.model.head(spatial)
        if logits.shape != (spatial.shape[0], 30):
            raise ValueError(f"expected official head logits with shape (B, 30), got {tuple(logits.shape)}")
        return torch.cat((pooled, logits), dim=1)


def _load_official_module(model_dir: Path):
    source = model_dir / "model.py"
    if not source.is_file():
        raise FileNotFoundError(f"official CXR-LT model.py not found: {source}")
    module_name = f"_cxrlt2026_official_{REVISION}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official CXR-LT model code: {source}")
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def load_encoder(model_dir, device):
    """Load the pinned official architecture and local safetensors checkpoint.

    Returns ``(adapter, processor)``. No class-name interpretation is applied:
    the last 30 feature columns are exactly the official head logits.
    """
    model_dir = Path(model_dir)
    device = torch.device(device)
    module = _load_official_module(model_dir)
    model = module.ConvNeXtV2CXR()
    weights = model_dir / WEIGHTS_NAME
    if not weights.is_file():
        raise FileNotFoundError(f"CXR-LT safetensors checkpoint not found: {weights}")
    from safetensors.torch import load_file

    model.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
    adapter = ConvNeXtV2FeatureAdapter(model).to(device).eval()
    processor = CXRLTProcessor()
    return adapter, processor
