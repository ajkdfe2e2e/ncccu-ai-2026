"""Image-only adapter for the locally pinned Microsoft BioViL-T checkpoint."""

from __future__ import annotations

import importlib
import hashlib
import json
from pathlib import Path
import sys
import threading

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor


MODEL_REPOSITORY = "microsoft/BiomedVLP-BioViL-T"
MODEL_REVISION = "301f526e823b805d3fe712d0cf06a66f042789c5"
SOURCE_REPOSITORY = "microsoft/hi-ml"
SOURCE_REVISION = "b67c1d27c6b17d8e8ff01f8c507f3cabdb307388"
WEIGHTS_NAME = "biovil_t_image_model_proj_size_128.pt"
WEIGHTS_SHA256 = "b2399d73dc2a68b9f3a1950e864ae0ecd24093fb07aa459d7e65807ebdc0fb77"
WEIGHTS_BYTES = 109745561
INPUT_RESIZE = 512
INPUT_SIZE = 448
RAW_FEATURES = 512
STATIC_FEATURES = 256
PROJECTED_FEATURES = 128
FEATURE_SIZE = STATIC_FEATURES + PROJECTED_FEATURES
FEATURE_POOLING = "biovil_t_static_pooled_256_plus_l2_normalized_projected_128"
SOURCE_MANIFEST = Path(__file__).resolve().parents[1] / "docs/biovil_t_source_manifest_20261108.json"
_IMPORT_LOCK = threading.RLock()


class BioViLTProcessor:
    """Apply BioViL-T input remapping, grayscale conversion, resize and crop."""

    input_size = INPUT_SIZE

    def __init__(self):
        self.transform = Compose([Resize(INPUT_RESIZE), CenterCrop(INPUT_SIZE), ToTensor()])

    @staticmethod
    def _remap_to_uint8(array):
        array = np.asarray(array).astype(float)
        low, high = array.min(), array.max()
        if high == low:
            # The upstream formula divides by zero here; its uint8 cast yields zeros.
            return np.zeros(array.shape, dtype=np.uint8)
        array = array - low
        array = array / array.max()
        array = array * 255
        return array.astype(np.uint8)

    def _preprocess(self, image):
        if not callable(getattr(image, "convert", None)):
            raise ValueError("images must be PIL images")
        if image.width <= 0 or image.height <= 0:
            raise ValueError("images must have nonzero width and height")
        pixels = np.asarray(image)
        if pixels.ndim not in (2, 3) or (pixels.ndim == 3 and pixels.shape[2] not in (3, 4)):
            raise ValueError("images must be grayscale, RGB, or RGBA PIL images")
        # Match load_image: remap the source array before PIL converts it to luminance.
        remapped = self._remap_to_uint8(pixels)
        gray = Image.fromarray(remapped).convert("L")
        return self.transform(gray).expand(3, -1, -1)

    def __call__(self, *, images, return_tensors="pt"):
        if return_tensors != "pt":
            raise ValueError("only return_tensors='pt' is supported")
        if not isinstance(images, (list, tuple)) or not images:
            raise ValueError("images must be a nonempty sequence")
        return {"pixel_values": torch.stack([self._preprocess(image) for image in images])}


class BioViLTFeatureAdapter(nn.Module):
    """Expose frozen pooled and normalized projected features for a single image."""

    feature_size = FEATURE_SIZE
    input_size = INPUT_SIZE
    feature_pooling = FEATURE_POOLING

    def __init__(self, model):
        super().__init__()
        if not callable(getattr(model, "forward", None)):
            raise ValueError("BioViL-T ImageModel must expose forward")
        self.model = model
        self.model.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def encode_components(self, pixel_values):
        _validate_pixels(pixel_values)
        # Omitting previous_image deliberately uses the upstream learned missing-image embedding.
        outputs = self.model(pixel_values)
        raw = outputs.img_embedding
        projected = outputs.projected_global_embedding
        batch = pixel_values.shape[0]
        if not isinstance(raw, torch.Tensor) or raw.shape != (batch, RAW_FEATURES):
            raise ValueError(f"BioViL-T img_embedding must have shape Bx{RAW_FEATURES}")
        if not isinstance(projected, torch.Tensor) or projected.shape != (batch, PROJECTED_FEATURES):
            raise ValueError(f"BioViL-T projected_global_embedding must have shape Bx{PROJECTED_FEATURES}")
        if not torch.isfinite(raw).all() or not torch.isfinite(projected).all():
            raise ValueError("BioViL-T features contain non-finite values")
        norms = torch.linalg.vector_norm(projected, dim=1, keepdim=True)
        if not torch.isfinite(norms).all() or torch.any(norms <= 0):
            raise ValueError("BioViL-T projected features have zero norm")
        return raw, projected / norms

    @torch.no_grad()
    def forward(self, pixel_values):
        raw, projected = self.encode_components(pixel_values)
        return torch.cat((raw[:, :STATIC_FEATURES], projected), dim=1)


def _validate_pixels(pixel_values):
    if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 4:
        raise ValueError("pixel_values must be a BCHW tensor")
    if (pixel_values.shape[0] == 0 or pixel_values.shape[1] != 3
            or tuple(pixel_values.shape[-2:]) != (INPUT_SIZE, INPUT_SIZE)):
        raise ValueError(f"pixel_values must have shape Bx3x{INPUT_SIZE}x{INPUT_SIZE}")
    if not torch.is_floating_point(pixel_values) or not torch.isfinite(pixel_values).all():
        raise ValueError("pixel_values must be finite floating-point values")


def adapt_model(model):
    return BioViLTFeatureAdapter(model)


def _source_root(model_dir):
    root = Path(model_dir) / "source/hi-ml-multimodal/src"
    if not (root / "health_multimodal/image/model/model.py").is_file():
        raise FileNotFoundError(f"pinned BioViL-T source tree is incomplete: {root}")
    return root.resolve()


def _load_official_model_class(model_dir):
    """Import only the pinned image-model dependency tree; do not execute package initializers."""
    root = _source_root(model_dir)
    hm = root / "health_multimodal"
    package_paths = {
        "health_multimodal": hm,
        "health_multimodal.common": hm / "common",
        "health_multimodal.image": hm / "image",
        "health_multimodal.image.model": hm / "image/model",
    }
    with _IMPORT_LOCK:
        previous_names = {name for name in sys.modules
                          if name == "health_multimodal" or name.startswith("health_multimodal.")}
        added = set()
        try:
            for name in previous_names:
                module = sys.modules[name]
                module_path = getattr(module, "__file__", None)
                module_paths = getattr(module, "__path__", ())
                paths = ([Path(module_path).resolve()] if module_path else []) + [Path(p).resolve() for p in module_paths]
                if paths and not all(path == root or root in path.parents for path in paths):
                    raise ImportError(f"refusing preloaded {name} outside pinned source: {paths}")
            for name, path in package_paths.items():
                existing = sys.modules.get(name)
                existing_path = getattr(existing, "__path__", ()) if existing else ()
                if existing is not None and not any(Path(p).resolve() == path for p in existing_path):
                    raise ImportError(f"refusing preloaded {name} outside pinned source: {existing_path}")
                if existing is None:
                    module = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
                    package = importlib.util.module_from_spec(module)
                    package.__path__ = [str(path)]
                    sys.modules[name] = package
                    added.add(name)
            model_module = importlib.import_module("health_multimodal.image.model.model")
            cls = getattr(model_module, "ImageModel", None)
            if cls is None or Path(model_module.__file__).resolve() != (hm / "image/model/model.py").resolve():
                raise ImportError("BioViL-T ImageModel did not resolve to the pinned local source")
            return cls
        finally:
            for name in list(sys.modules):
                if (name == "health_multimodal" or name.startswith("health_multimodal.")) \
                        and name not in previous_names and name not in package_paths:
                    sys.modules.pop(name, None)
            for name in reversed(list(package_paths)):
                if name in added:
                    sys.modules.pop(name, None)


def load_encoder(model_dir, device):
    """Load the pinned image-only checkpoint with safe deserialization and strict keys."""
    model_dir = Path(model_dir)
    device = torch.device(device)
    provenance_path = model_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if (provenance.get("model") != MODEL_REPOSITORY
            or provenance.get("revision") != MODEL_REVISION
            or provenance.get("code_revision") != SOURCE_REVISION
            or manifest.get("model") != MODEL_REPOSITORY
            or manifest.get("revision") != MODEL_REVISION
            or manifest.get("code_revision") != SOURCE_REVISION
            or provenance.get("files") != manifest.get("files")):
        raise ValueError("BioViL-T provenance differs from the pinned protocol")
    files = provenance.get("files")
    if not isinstance(files, dict) or len(files) != 22:
        raise ValueError("BioViL-T provenance must register the fixed 22 source and model files")
    for relative, lock in files.items():
        path = (model_dir / relative).resolve()
        try:
            path.relative_to(model_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"BioViL-T provenance path escapes model directory: {relative}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"pinned BioViL-T asset is missing: {relative}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if path.stat().st_size != lock.get("bytes") or digest.hexdigest() != lock.get("sha256"):
            raise ValueError(f"BioViL-T asset differs from provenance: {relative}")
        if relative == WEIGHTS_NAME and (lock.get("sha256") != WEIGHTS_SHA256
                                         or lock.get("bytes") != WEIGHTS_BYTES):
            raise ValueError("BioViL-T checkpoint lock differs from the fixed protocol")
    weights_path = model_dir / WEIGHTS_NAME
    if not weights_path.is_file():
        raise FileNotFoundError(f"pinned BioViL-T checkpoint is missing: {weights_path}")
    model_class = _load_official_model_class(model_dir)
    model = model_class(img_encoder_type="resnet50_multi_image", joint_feature_size=PROJECTED_FEATURES,
                        pretrained_model_path=None)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ValueError("BioViL-T checkpoint must be a tensor-only state dictionary")
    model.load_state_dict(state, strict=True)
    adapter = adapt_model(model).to(device).eval()
    return adapter, BioViLTProcessor()
