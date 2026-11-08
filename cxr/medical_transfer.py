"""TorchXRayVision CheX transfer model and medical-image input pipeline."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
import hashlib
import importlib.metadata
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F

WEIGHTS_NAME = "densenet121-res224-chex"
TARGET_NAMES = [
    "Enlarged Cardiomediastinum", "Pneumothorax", "Consolidation",
    "Pneumonia", "Edema", "Cardiomegaly", "Atelectasis",
    "Lung Opacity", "Pleural Effusion", "No Finding",
]
SOURCE_INDICES = (17, 3, 1, 8, 4, 10, 0, 16, 7, None)
_SOURCE_NAMES = (
    "Enlarged Cardiomediastinum", "Pneumothorax", "Consolidation",
    "Pneumonia", "Edema", "Cardiomegaly", "Atelectasis", "Lung Opacity",
    "Effusion",
)


def _xrv():
    try:
        import torchxrayvision as xrv
    except ImportError as exc:
        raise ImportError(
            "TorchXRayVision is required for the medical transfer model; install torchxrayvision."
        ) from exc
    return xrv


class _RawLogitDenseNet(nn.Module):
    """Keep XRV's DenseNet trunk while always exposing unmodified logits."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.features = backbone.features
        self.classifier = backbone.classifier
        self.pathologies = list(TARGET_NAMES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = F.relu(features, inplace=False)
        features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        return self.classifier(features)


def _checkpoint_state(path: Path) -> tuple[Mapping[str, torch.Tensor], Mapping]:
    """Read a tensor-only checkpoint; legacy serialized modules require trusted input."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            "Checkpoint must be a weights_only-compatible state_dict. "
            "Convert legacy serialized XRV module checkpoints to a state_dict first."
        ) from exc
    if isinstance(payload, nn.Module):
        payload = payload.state_dict()
    metadata = payload if isinstance(payload, Mapping) else {}
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in payload and isinstance(payload[key], Mapping):
                payload = payload[key]
                break
    if not isinstance(payload, Mapping) or not payload or not all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in payload.items()
    ):
        raise ValueError("Checkpoint does not contain a tensor state_dict")
    return payload, metadata


def build_model(pretrained_path: Path | None = None) -> nn.Module:
    """Build a 10-logit XRV DenseNet; optionally copy the nine CheX heads.

    The supplied checkpoint must be a tensor-only state_dict (or a mapping
    containing one), with the original 18-output XRV classifier.
    """
    xrv = _xrv()
    source_labels = xrv.models.model_urls[WEIGHTS_NAME]["labels"]
    if len(source_labels) != 18 or any(source_labels[i] != n for n, i in zip(_SOURCE_NAMES, (17, 3, 1, 8, 4, 10, 0, 16, 7))):
        raise RuntimeError("Installed TorchXRayVision CheX pathology metadata does not match expected indices")

    backbone = xrv.models.DenseNet(weights=None, num_classes=10, in_channels=1, op_threshs=None,
                                   apply_sigmoid=False)
    model = _RawLogitDenseNet(backbone)
    model.source_pathologies = list(source_labels)
    if pretrained_path is None:
        return model

    state, metadata = _checkpoint_state(Path(pretrained_path))
    checkpoint_pathologies = metadata.get("pathologies")
    if checkpoint_pathologies is None or list(checkpoint_pathologies) != list(source_labels):
        raise ValueError("Checkpoint pathology metadata does not match TorchXRayVision CheX labels")
    if metadata.get("weights_name") not in (None, WEIGHTS_NAME):
        raise ValueError("Checkpoint weights_name is not densenet121-res224-chex")
    # Accept raw XRV state dict keys or state_dicts prefixed by a wrapper name.
    state = {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}
    if "classifier.weight" not in state or state["classifier.weight"].shape[0] != 18:
        raise ValueError("Expected the 18-output densenet121-res224-chex checkpoint classifier")
    model_state = model.state_dict()
    for key, value in state.items():
        if key.startswith("classifier."):
            continue
        if key not in model_state or model_state[key].shape != value.shape:
            raise ValueError(f"Incompatible XRV checkpoint tensor: {key}")
        model_state[key] = value
    for target_index, source_index in enumerate(SOURCE_INDICES):
        if source_index is not None:
            model_state["classifier.weight"][target_index].copy_(state["classifier.weight"][source_index])
            model_state["classifier.bias"][target_index].copy_(state["classifier.bias"][source_index])
    model.load_state_dict(model_state)
    return model


def _load_medical_image(path: str | Path) -> np.ndarray:
    """Return one grayscale plane normalized by TorchXRayVision's loader."""
    xrv = _xrv()
    image = np.asarray(xrv.utils.load_image(str(path)), dtype=np.float32)
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 2:
        raise ValueError(f"Expected a grayscale medical image, got shape {image.shape}")
    return np.asarray(image, dtype=np.float32)


def _cache_key(path: Path) -> str:
    stat = path.stat()
    try:
        xrv_version = importlib.metadata.version("torchxrayvision")
    except importlib.metadata.PackageNotFoundError:
        xrv_version = "unknown"
    try:
        pillow_version = importlib.metadata.version("Pillow")
    except importlib.metadata.PackageNotFoundError:
        pillow_version = "unknown"
    identity = f"medical-v1|{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{xrv_version}|{pillow_version}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _preprocess(path: str | Path, *, training: bool = False, cache_root=None) -> torch.Tensor:
    xrv = _xrv()
    cache_path = None
    if cache_root is not None:
        cache_path = Path(cache_root) / (_cache_key(Path(path)) + ".npy")
    image = None
    if cache_path is not None and cache_path.is_file():
        try:
            cached = np.load(cache_path, allow_pickle=False)
            if (cached.shape == (1, 224, 224) and cached.dtype == np.float32
                    and np.isfinite(cached).all() and cached.min() >= -1024 and cached.max() <= 1024):
                image = cached
        except (OSError, ValueError):
            image = None
    if image is None:
        image = _load_medical_image(path)
        if image.ndim != 2:
            raise ValueError(f"Expected a grayscale medical image, got shape {image.shape}")
        image = np.clip(image, -1024.0, 1024.0)
        image = xrv.datasets.XRayCenterCrop()(image[None, ...])
        image = np.asarray(xrv.datasets.XRayResizer(224)(image), dtype=np.float32).reshape(1, 224, 224)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = cache_path.with_suffix(f".tmp-{os.getpid()}")
            try:
                with temp_path.open("wb") as stream:
                    np.save(stream, image, allow_pickle=False)
                os.replace(temp_path, cache_path)
            finally:
                if temp_path.exists():
                    temp_path.unlink()
    tensor = torch.from_numpy(np.ascontiguousarray(image)).float().reshape(1, 1, 224, 224)
    if training:
        angle = float(torch.empty(()).uniform_(-7, 7)) * np.pi / 180
        scale = float(torch.empty(()).uniform_(0.95, 1.05))
        theta = tensor.new_tensor([[scale * np.cos(angle), -scale * np.sin(angle), 0.0],
                                   [scale * np.sin(angle), scale * np.cos(angle), 0.0]])
        grid = F.affine_grid(theta[None], tensor.shape, align_corners=False)
        tensor = F.grid_sample(tensor + 1024.0, grid, mode="bilinear", padding_mode="zeros", align_corners=False) - 1024.0
    return tensor[0]


class MedicalDataset(Dataset):
    def __init__(self, records, training: bool = False, cache_root=None):
        self.records = records
        self.training = training
        self.cache_root = cache_root

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image = _preprocess(record["path"], training=self.training, cache_root=self.cache_root)
        labels = record.get("labels")
        target = torch.tensor(labels if labels is not None else [0.0] * 10, dtype=torch.float32)
        return image, target


@torch.inference_mode()
def predict_records(model, records, device, batch_size=1) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    records = list(records)
    if not records:
        return np.empty((0, 10), dtype=np.float32)
    device = torch.device(device)
    prior_training = model.training
    model.eval()
    outputs = []
    try:
        for start in range(0, len(records), batch_size):
            batch = torch.stack([_preprocess(row["path"]) for row in records[start:start + batch_size]])
            logits = model(batch.to(device, dtype=torch.float32))
            if logits.ndim != 2 or logits.shape[1] != 10:
                raise ValueError("medical model must return an N x 10 logits tensor")
            outputs.append(logits.float().sigmoid().cpu().numpy())
    finally:
        model.train(prior_training)
    return np.concatenate(outputs, axis=0)
