"""Image-only adapter for the pinned CheXficient checkpoint."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import threading
from collections.abc import Mapping

import numpy as np
import torch
from torch import nn
from PIL import Image


MODEL_REPOSITORY = "StanfordAIMI/CheXficient"
MODEL_REVISION = "34bb5ea905c4cb504d2b73c9a6687a5d5082f52a"
WEIGHTS_NAME = "model.safetensors"
INPUT_SIZE = 378
PATCH_SIZE = 14
SOURCE_FEATURES = 768
PROJECTED_FEATURES = 512
FEATURE_SIZE = PROJECTED_FEATURES
FEATURE_POOLING = "dinov2_vitb14_normalized_cls_projected_512_l2_normalized"
_MEAN = (0.48145466, 0.4578275, 0.40821073)
_STD = (0.26862954, 0.26130258, 0.27577711)
_DINO_IMPORT_LOCK = threading.RLock()


class CheXficientProcessor:
    """Apply the pinned HF RGB image preprocessing configuration."""

    input_size = INPUT_SIZE

    def __init__(self):
        self.mean = torch.tensor(_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(_STD, dtype=torch.float32).view(3, 1, 1)

    def _preprocess(self, image):
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("images must have nonzero width and height")
        if width < height:
            resized = (INPUT_SIZE, int(height * INPUT_SIZE / width))
        else:
            resized = (int(width * INPUT_SIZE / height), INPUT_SIZE)
        image = image.resize(resized, resample=Image.Resampling.BICUBIC)
        left = (image.width - INPUT_SIZE) // 2
        top = (image.height - INPUT_SIZE) // 2
        image = image.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))
        pixels = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True))
        pixels = pixels.permute(2, 0, 1).to(torch.float32).div_(255.0)
        return (pixels - self.mean) / self.std

    def __call__(self, *, images, return_tensors="pt"):
        if return_tensors != "pt":
            raise ValueError("only return_tensors='pt' is supported")
        if not isinstance(images, (list, tuple)) or not images:
            raise ValueError("images must be a nonempty sequence")
        values = []
        for image in images:
            if not callable(getattr(image, "convert", None)):
                raise ValueError("images must be PIL images")
            values.append(self._preprocess(image.convert("RGB")))
        return {"pixel_values": torch.stack(values)}


class CheXficientImageAdapter(nn.Module):
    """Expose frozen CLS and normalized projected image embeddings."""

    feature_size = FEATURE_SIZE
    input_size = INPUT_SIZE
    feature_pooling = FEATURE_POOLING

    def __init__(self, image_model: nn.Module, image_projection: nn.Module):
        super().__init__()
        if not isinstance(image_model, nn.Module) or not isinstance(image_projection, nn.Module):
            raise ValueError("image_model and image_projection must be torch modules")
        self.image_model = image_model
        self.image_projection = image_projection
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def encode_components(self, pixel_values):
        _validate_pixels(pixel_values)
        cls_features = self.image_model(pixel_values)
        batch = pixel_values.shape[0]
        if (not isinstance(cls_features, torch.Tensor)
                or cls_features.shape != (batch, SOURCE_FEATURES)):
            shape = getattr(cls_features, "shape", None)
            raise ValueError(f"DINOv2 must return Bx{SOURCE_FEATURES} CLS features, got {shape}")
        if not torch.isfinite(cls_features).all():
            raise ValueError("DINOv2 CLS features contain non-finite values")

        projected = self.image_projection(cls_features)
        if (not isinstance(projected, torch.Tensor)
                or projected.shape != (batch, PROJECTED_FEATURES)):
            shape = getattr(projected, "shape", None)
            raise ValueError(f"image projection must return Bx{PROJECTED_FEATURES}, got {shape}")
        if not torch.isfinite(projected).all():
            raise ValueError("projected image features contain non-finite values")
        norms = torch.linalg.vector_norm(projected, dim=1, keepdim=True)
        if not torch.isfinite(norms).all() or torch.any(norms <= 0):
            raise ValueError("projected image features have zero norm")
        return cls_features, projected / norms

    @torch.no_grad()
    def forward(self, pixel_values):
        return self.encode_components(pixel_values)[1]


def _validate_pixels(pixel_values):
    if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 4:
        raise ValueError("pixel_values must be a BCHW tensor")
    if (pixel_values.shape[0] == 0 or pixel_values.shape[1] != 3
            or tuple(pixel_values.shape[-2:]) != (INPUT_SIZE, INPUT_SIZE)):
        raise ValueError(f"pixel_values must have shape Bx3x{INPUT_SIZE}x{INPUT_SIZE}")
    if not torch.is_floating_point(pixel_values) or not torch.isfinite(pixel_values).all():
        raise ValueError("pixel_values must be a finite floating-point tensor")


def adapt_model(image_model, image_projection):
    """Wrap already-created image-only modules without loading any weights."""
    return CheXficientImageAdapter(image_model, image_projection)


def _path_from_module(module):
    path = getattr(module, "__file__", None)
    return Path(path).resolve() if path else None


def _under(path, root):
    if path is None:
        return False
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _load_vit_base(model_dir):
    """Import vit_base only from this model directory's vendored source."""
    model_dir = Path(model_dir).resolve()
    dino_root = (model_dir / "dinov2").resolve()
    source = dino_root / "models" / "vision_transformer.py"
    required = (dino_root / "__init__.py", dino_root / "models" / "__init__.py",
                dino_root / "layers" / "__init__.py", source)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"pinned vendored DINOv2 source is incomplete: {missing[0]}")

    with _DINO_IMPORT_LOCK:
        previous_names = {name for name in sys.modules
                          if name == "dinov2" or name.startswith("dinov2.")}
        for name in previous_names:
            module_path = _path_from_module(sys.modules[name])
            if not _under(module_path, dino_root):
                raise ImportError(
                    f"refusing to use preloaded {name!r} outside pinned source: {module_path}"
                )

        source_root = str(model_dir)
        inserted_path = source_root not in sys.path
        if inserted_path:
            sys.path.insert(0, source_root)
        try:
            module = importlib.import_module("dinov2.models.vision_transformer")
            if _path_from_module(module) != source.resolve():
                raise ImportError(
                    f"loaded DINOv2 source differs from pinned local file: {_path_from_module(module)}"
                )
            for name in set(sys.modules) - previous_names:
                if name == "dinov2" or name.startswith("dinov2."):
                    if not _under(_path_from_module(sys.modules[name]), dino_root):
                        raise ImportError(f"DINOv2 dependency {name!r} escaped pinned source tree")
            builder = getattr(module, "vit_base", None)
            if not callable(builder):
                raise ImportError("pinned DINOv2 source must expose vit_base")
            return builder
        finally:
            if inserted_path:
                try:
                    sys.path.remove(source_root)
                except ValueError:
                    pass
            if not previous_names:
                for name in list(sys.modules):
                    if name == "dinov2" or name.startswith("dinov2."):
                        sys.modules.pop(name, None)


def _read_config(model_dir):
    path = Path(model_dir) / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"pinned CheXficient config not found: {path}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid CheXficient config: {path}") from exc
    expected = {
        "model_type": "chexficient_clip",
        "vision_model_name": "dinov2_vitb14",
        "image_size": INPUT_SIZE,
        "projection_dim": PROJECTED_FEATURES,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"pinned CheXficient config has unexpected {key}: {config.get(key)!r}")
    return config


class _LinearProjectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(SOURCE_FEATURES, PROJECTED_FEATURES)

    def forward(self, features):
        return self.projection(features)


def _make_projection():
    # Preserve the pinned LinearProjectionHead state key prefix.
    return _LinearProjectionHead()


def _load_image_state(model_dir, image_model, image_projection):
    weights_path = Path(model_dir) / WEIGHTS_NAME
    if not weights_path.is_file():
        raise FileNotFoundError(f"pinned CheXficient checkpoint not found: {weights_path}")

    expected_vit = {f"image_encoder.model.{key}" for key in image_model.state_dict()}
    expected_projection = {f"image_projection.{key}" for key in image_projection.state_dict()}
    expected_image = expected_vit | expected_projection
    text_prefixes = ("text_encoder.", "text_projection.")
    allowed_standalone = {"logit_scale"}

    from safetensors import safe_open

    with safe_open(str(weights_path), framework="pt", device="cpu") as checkpoint:
        all_keys = set(checkpoint.keys())
        unknown = sorted(key for key in all_keys
                         if key not in expected_image
                         and key not in allowed_standalone
                         and not key.startswith("image_encoder.model.")
                         and not key.startswith(text_prefixes))
        if unknown:
            raise ValueError(f"unexpected top-level checkpoint key: {unknown[0]}")
        image_keys = {key for key in all_keys
                      if key.startswith("image_encoder.model.")
                      or key.startswith("image_projection.")}
        if image_keys != expected_image:
            missing = sorted(expected_image - image_keys)
            unexpected = sorted(image_keys - expected_image)
            raise ValueError(
                f"image checkpoint keys do not match pinned modules; missing={missing[:3]}, "
                f"unexpected={unexpected[:3]}"
            )
        vit_state = {
            key.removeprefix("image_encoder.model."): checkpoint.get_tensor(key)
            for key in sorted(expected_vit)
        }
        projection_state = {
            key.removeprefix("image_projection."): checkpoint.get_tensor(key)
            for key in sorted(expected_projection)
        }

    image_model.load_state_dict(vit_state, strict=True)
    image_projection.load_state_dict(projection_state, strict=True)


def load_encoder(model_dir, device):
    """Load only the pinned image encoder/projection and its RGB processor.

    This function never constructs the Hugging Face full model, tokenizer, or
    either text module, and does not perform network initialization.
    """
    model_dir = Path(model_dir)
    device = torch.device(device)
    _read_config(model_dir)
    vit_base = _load_vit_base(model_dir)
    image_model = vit_base(
        patch_size=PATCH_SIZE,
        img_size=INPUT_SIZE,
        init_values=1.0,
        block_chunks=0,
        num_register_tokens=0,
    )
    if getattr(image_model, "embed_dim", None) != SOURCE_FEATURES:
        raise ValueError(f"pinned DINOv2 ViT-B/14 must expose {SOURCE_FEATURES} features")
    if getattr(image_model, "num_register_tokens", None) != 0:
        raise ValueError("pinned CheXficient checkpoint requires a DINOv2 model without register tokens")
    image_projection = _make_projection()
    _load_image_state(model_dir, image_model, image_projection)
    adapter = CheXficientImageAdapter(image_model, image_projection).to(device).eval()
    return adapter, CheXficientProcessor()
