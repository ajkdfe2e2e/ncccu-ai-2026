"""Train the final ConvNeXt stage while caching its frozen spatial prefix."""

from collections.abc import Mapping

import numpy as np
import torch
from torch import nn


PREFIX_CHANNELS = 512
SPATIAL_CHANNELS = 1024
OFFICIAL_CLASSES = 30
LOCAL_FEATURES = SPATIAL_CHANNELS + OFFICIAL_CLASSES
LOCAL_CLASSES = 10
INPUT_SIZE = 512
PREFIX_SPATIAL_SIZE = 32


def _numeric_array(payload, key, shape):
    try:
        value = np.asarray(payload[key], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"probe payload is missing numeric {key}") from exc
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"probe {key} must be finite with shape {shape}")
    return value


class ConvNeXtPrefix(nn.Module):
    """Frozen ConvNeXt stem and stages zero through two."""

    def __init__(self, stem, stages):
        super().__init__()
        if len(stages) != 3:
            raise ValueError("prefix requires exactly the first three ConvNeXt stages")
        self.stem = stem
        self.stages = nn.Sequential(*stages)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, pixel_values):
        if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 4:
            raise ValueError("pixel_values must be a BCHW tensor")
        if pixel_values.shape[1] != 3 or pixel_values.shape[0] == 0:
            raise ValueError("pixel_values must contain a nonempty three-channel batch")
        features = self.stages(self.stem(pixel_values))
        if features.ndim != 4 or features.shape[1] != PREFIX_CHANNELS:
            raise ValueError(f"prefix must return Bx{PREFIX_CHANNELS}xHxW features")
        if tuple(pixel_values.shape[-2:]) == (INPUT_SIZE, INPUT_SIZE):
            if tuple(features.shape[-2:]) != (PREFIX_SPATIAL_SIZE, PREFIX_SPATIAL_SIZE):
                raise ValueError("512x512 inputs must produce 512x32x32 prefix features")
        return features.detach()


class ConvNeXtTail(nn.Module):
    """Trainable final stage, frozen official heads, and local ten-class probe."""

    def __init__(self, backbone, official_bn, official_csra):
        super().__init__()
        if len(backbone.stages) != 4:
            raise ValueError("ConvNeXt backbone must expose four stages")
        self.stage = backbone.stages[3]
        self.norm_pre = backbone.norm_pre
        self.backbone_head = backbone.head
        self.official_bn = official_bn
        self.official_csra = official_csra
        self.local_head = nn.Linear(LOCAL_FEATURES, LOCAL_CLASSES, dtype=torch.float32)
        self.register_buffer("probe_mean", torch.zeros(LOCAL_FEATURES, dtype=torch.float32))
        self.register_buffer("probe_std", torch.ones(LOCAL_FEATURES, dtype=torch.float32))
        self.register_buffer("_probe_loaded", torch.tensor(False))
        self.probe_l2 = 0.0

        self.requires_grad_(False)
        for parameter in self.stage.parameters():
            parameter.requires_grad_(True)
        for parameter in self.local_head.parameters():
            parameter.requires_grad_(True)
        self.train(False)

    def _freeze_official_eval(self):
        self.stage.eval()
        self.norm_pre.eval()
        self.backbone_head.eval()
        self.official_bn.eval()
        self.official_csra.eval()

    def train(self, mode=True):
        super().train(mode)
        self._freeze_official_eval()
        self.local_head.train(mode)
        return self

    def load_probe(self, payload):
        """Initialize the local classifier from a standardized NumPy head."""
        if not isinstance(payload, Mapping):
            raise ValueError("probe payload must be a mapping")
        if bool(payload.get("class_balance", False)):
            raise ValueError("local stage probe requires an ordinary BCE head")
        mean = _numeric_array(payload, "mean", (LOCAL_FEATURES,))
        std = _numeric_array(payload, "std", (LOCAL_FEATURES,))
        weight = _numeric_array(payload, "weight", (LOCAL_CLASSES, LOCAL_FEATURES))
        bias = _numeric_array(payload, "bias", (LOCAL_CLASSES,))
        if np.any(std <= 0):
            raise ValueError("probe standard deviations must be positive")
        try:
            l2 = float(payload.get("l2", 0.01))
        except (TypeError, ValueError) as exc:
            raise ValueError("probe l2 must be finite and nonnegative") from exc
        if not np.isfinite(l2) or l2 < 0:
            raise ValueError("probe l2 must be finite and nonnegative")

        with torch.no_grad():
            self.probe_mean.copy_(torch.as_tensor(mean, dtype=torch.float32,
                                                  device=self.probe_mean.device))
            self.probe_std.copy_(torch.as_tensor(std, dtype=torch.float32,
                                                 device=self.probe_std.device))
            self.local_head.weight.copy_(torch.as_tensor(weight, dtype=torch.float32,
                                                         device=self.local_head.weight.device))
            self.local_head.bias.copy_(torch.as_tensor(bias, dtype=torch.float32,
                                                       device=self.local_head.bias.device))
            self._probe_loaded.fill_(True)
        self.probe_l2 = l2
        self.local_head.train(self.training)
        return self

    def features(self, prefix_tensor):
        """Return GAP plus the official 30 raw logits for each image."""
        if not isinstance(prefix_tensor, torch.Tensor) or prefix_tensor.ndim != 4:
            raise ValueError("prefix_tensor must be a BCHW tensor")
        if (prefix_tensor.shape[0] == 0 or prefix_tensor.shape[1] != PREFIX_CHANNELS
                or not prefix_tensor.is_floating_point()):
            raise ValueError(f"prefix_tensor must be a nonempty floating Bx{PREFIX_CHANNELS}xHxW tensor")
        spatial = self.stage(prefix_tensor)
        spatial = self.norm_pre(spatial)
        spatial = self.backbone_head(spatial)
        spatial = self.official_bn(spatial)
        if spatial.ndim != 4 or spatial.shape[1] != SPATIAL_CHANNELS:
            raise ValueError(f"official feature map must have {SPATIAL_CHANNELS} channels")
        pooled = spatial.mean(dim=(2, 3))
        logits = self.official_csra(spatial)
        if logits.shape != (spatial.shape[0], OFFICIAL_CLASSES):
            raise ValueError("official CSRA head must return a Bx30 logits tensor")
        if pooled.dtype != logits.dtype:
            pooled = pooled.to(dtype=logits.dtype)
        return torch.cat((pooled, logits), dim=1)

    def _local_logits(self, study_features):
        if not bool(self._probe_loaded.item()):
            raise RuntimeError("load_probe must be called before local study prediction")
        device_type = study_features.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            values = study_features.to(dtype=torch.float32)
            normalized = (values - self.probe_mean) / self.probe_std
            return self.local_head(normalized)

    def study_logits(self, prefix_tensor, study_indices, study_count):
        """Pool image features by study before applying the local ten-class head."""
        if isinstance(study_count, bool) or not isinstance(study_count, (int, np.integer)) or study_count <= 0:
            raise ValueError("study_count must be a positive integer")
        image_features = self.features(prefix_tensor)
        try:
            indices = torch.as_tensor(study_indices, device=image_features.device)
        except (TypeError, ValueError) as exc:
            raise ValueError("study_indices must be an integer vector") from exc
        if indices.ndim != 1 or len(indices) != len(image_features):
            raise ValueError("study_indices length must match the image batch")
        if indices.dtype == torch.bool or indices.dtype.is_floating_point or indices.dtype.is_complex:
            raise ValueError("study_indices must contain integers")
        indices = indices.to(dtype=torch.long)
        if torch.any(indices < 0) or torch.any(indices >= study_count):
            raise ValueError("study_indices are outside the study range")
        counts = torch.bincount(indices, minlength=int(study_count))
        if counts.shape[0] != study_count or torch.any(counts == 0):
            raise ValueError("each study must contain at least one image")
        image_features = image_features.to(dtype=torch.float32)
        sums = image_features.new_zeros((int(study_count), LOCAL_FEATURES))
        sums.index_add_(0, indices, image_features)
        means = sums / counts.to(dtype=torch.float32).unsqueeze(1)
        return self._local_logits(means)


def split_official_model(model, probe_payload=None):
    """Split the pinned timm ConvNeXtV2CXR into frozen prefix and train tail."""
    try:
        backbone = model.backbone
        stages = list(backbone.stages)
        stem = backbone.stem
        norm_pre = backbone.norm_pre
        backbone_head = backbone.head
        official_bn = model.bn
        official_csra = model.head
    except AttributeError as exc:
        raise ValueError("official model is missing required ConvNeXt/CSRA modules") from exc
    if len(stages) != 4:
        raise ValueError("official ConvNeXt backbone must have four stages")
    model.eval()
    prefix = ConvNeXtPrefix(stem, stages[:3])
    tail = ConvNeXtTail(backbone, official_bn, official_csra)
    # Keep the exact modules discovered above when the source uses an indexed stage container.
    tail.norm_pre = norm_pre
    tail.backbone_head = backbone_head
    if probe_payload is not None:
        tail.load_probe(probe_payload)
    return prefix, tail
