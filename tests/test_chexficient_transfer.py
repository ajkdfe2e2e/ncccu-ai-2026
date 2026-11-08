"""Synthetic contract tests for the pinned CheXficient image-only adapter."""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from cxr.chexficient_transfer import (
    FEATURE_SIZE,
    INPUT_SIZE,
    PROJECTED_FEATURES,
    SOURCE_FEATURES,
    CheXficientImageAdapter,
    CheXficientProcessor,
    _make_projection,
    _load_vit_base,
    adapt_model,
    load_encoder,
)


FAKE_VIT_SOURCE = '''\
import torch
from torch import nn

class FakeViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 768
        self.num_register_tokens = 0
        self.scale = nn.Parameter(torch.tensor(0.75))
        self.offset = nn.Parameter(torch.linspace(-0.2, 0.2, 768))

    def forward(self, pixels):
        mean = pixels.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        return mean * self.scale + self.offset.unsqueeze(0)

def vit_base(patch_size=16, img_size=224, init_values=1.0,
             block_chunks=1, num_register_tokens=0):
    expected = (14, 378, 1.0, 0, 0)
    actual = (patch_size, img_size, init_values, block_chunks, num_register_tokens)
    if actual != expected:
        raise ValueError(f"wrong pinned DINOv2 construction: {actual}")
    return FakeViT()
'''


def _pinned_v4513_reference(image):
    image = image.convert("RGB")
    width, height = image.size
    if width < height:
        resized_size = (INPUT_SIZE, int(height * INPUT_SIZE / width))
    else:
        resized_size = (int(width * INPUT_SIZE / height), INPUT_SIZE)
    image = image.resize(resized_size, resample=Image.Resampling.BICUBIC)
    left = (image.width - INPUT_SIZE) // 2
    top = (image.height - INPUT_SIZE) // 2
    image = image.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))
    pixels = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True))
    pixels = pixels.permute(2, 0, 1).to(torch.float32).div_(255.0)
    mean = torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(3, 1, 1)
    std = torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(3, 1, 1)
    return (pixels - mean) / std


def _write_fake_model_dir(directory):
    model_dir = Path(directory)
    (model_dir / "dinov2" / "models").mkdir(parents=True)
    (model_dir / "dinov2" / "layers").mkdir()
    (model_dir / "dinov2" / "__init__.py").write_text("", encoding="utf-8")
    (model_dir / "dinov2" / "models" / "__init__.py").write_text("", encoding="utf-8")
    (model_dir / "dinov2" / "layers" / "__init__.py").write_text("", encoding="utf-8")
    (model_dir / "dinov2" / "models" / "vision_transformer.py").write_text(
        FAKE_VIT_SOURCE, encoding="utf-8"
    )
    config = {
        "model_type": "chexficient_clip",
        "vision_model_name": "dinov2_vitb14",
        "image_size": INPUT_SIZE,
        "projection_dim": PROJECTED_FEATURES,
        "text_model_name": "unused-for-image-tests",
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return model_dir


def _fake_checkpoint(model_dir, *, extra=None, projection_shape=None):
    from safetensors.torch import save_file

    vit_base = _load_vit_base(model_dir)
    image_model = vit_base(
        patch_size=14, img_size=378, init_values=1.0,
        block_chunks=0, num_register_tokens=0,
    )
    projection = _make_projection()
    tensors = {
        f"image_encoder.model.{key}": value.detach().cpu().contiguous()
        for key, value in image_model.state_dict().items()
    }
    tensors.update({
        f"image_projection.{key}": value.detach().cpu().contiguous()
        for key, value in projection.state_dict().items()
    })
    tensors["text_encoder.model.embeddings.word_embeddings.weight"] = torch.ones(3, 4)
    tensors["text_projection.projection.weight"] = torch.ones(4, 3)
    tensors["logit_scale"] = torch.tensor(0.01)
    if projection_shape is not None:
        tensors["image_projection.projection.weight"] = torch.ones(projection_shape)
    if extra:
        tensors.update(extra)
    path = model_dir / "model.safetensors"
    save_file(tensors, str(path))
    return image_model, projection, path


class CheXficientTransferTests(unittest.TestCase):
    def test_processor_matches_fixed_rgb_transform(self):
        gray = Image.fromarray(np.arange(63, dtype=np.uint8).reshape(7, 9), mode="L")
        rgb_data = np.zeros((5, 7, 3), dtype=np.uint8)
        rgb_data[..., 0] = np.arange(7, dtype=np.uint8)[None, :] * 29
        rgb_data[..., 1] = np.arange(5, dtype=np.uint8)[:, None] * 43
        rgb_data[..., 2] = 191
        rgb = Image.fromarray(rgb_data, mode="RGB")
        actual = CheXficientProcessor()(images=[gray, rgb])["pixel_values"]
        resized_rgb_height = int(7 * INPUT_SIZE / 5)
        self.assertEqual((resized_rgb_height - INPUT_SIZE) % 2, 1)
        expected = torch.stack([_pinned_v4513_reference(image) for image in (gray, rgb)])
        self.assertEqual(tuple(actual.shape), (2, 3, INPUT_SIZE, INPUT_SIZE))
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_adapter_freezes_modules_and_returns_normalized_projection(self):
        class FakeImageModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(0.5))

            def forward(self, pixels):
                mean = pixels.mean(dim=(1, 2, 3)).unsqueeze(1)
                return mean.expand(-1, SOURCE_FEATURES) * self.scale

        image_model = FakeImageModel()
        projection = nn.Linear(SOURCE_FEATURES, PROJECTED_FEATURES)
        adapter = adapt_model(image_model, projection)
        pixels = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE, requires_grad=True)
        cls, embedding = adapter.encode_components(pixels)
        self.assertIsInstance(adapter, CheXficientImageAdapter)
        self.assertEqual(tuple(cls.shape), (2, SOURCE_FEATURES))
        self.assertEqual(tuple(embedding.shape), (2, FEATURE_SIZE))
        torch.testing.assert_close(torch.linalg.vector_norm(embedding, dim=1), torch.ones(2))
        self.assertFalse(adapter(pixels).requires_grad)
        self.assertTrue(all(not parameter.requires_grad for parameter in adapter.parameters()))
        adapter.train()
        self.assertFalse(adapter.training)
        self.assertFalse(image_model.training)
        self.assertFalse(projection.training)

    def test_adapter_rejects_bad_input_and_invalid_embedding(self):
        class ConstantImageModel(nn.Module):
            def forward(self, pixels):
                return torch.ones(len(pixels), SOURCE_FEATURES, device=pixels.device)

        adapter = adapt_model(ConstantImageModel(), nn.Linear(SOURCE_FEATURES, PROJECTED_FEATURES))
        with self.assertRaisesRegex(ValueError, "shape"):
            adapter(torch.zeros(1, 1, INPUT_SIZE, INPUT_SIZE))
        with self.assertRaisesRegex(ValueError, "shape"):
            adapter(torch.zeros(1, 3, INPUT_SIZE - 1, INPUT_SIZE))
        bad_pixels = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
        bad_pixels[0, 0, 0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            adapter(bad_pixels)

        zero_projection = nn.Linear(SOURCE_FEATURES, PROJECTED_FEATURES)
        nn.init.zeros_(zero_projection.weight)
        nn.init.zeros_(zero_projection.bias)
        zero_adapter = adapt_model(ConstantImageModel(), zero_projection)
        with self.assertRaisesRegex(ValueError, "zero norm"):
            zero_adapter(torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE))

    def test_loader_strictly_loads_only_image_branch_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = _write_fake_model_dir(directory)
            expected_model, expected_projection, weights_path = _fake_checkpoint(model_dir)
            from safetensors import safe_open as original_safe_open

            reads = []

            class TrackingSafeOpen:
                def __init__(self, *args, **kwargs):
                    self.inner = original_safe_open(*args, **kwargs)

                def __enter__(self):
                    self.inner.__enter__()
                    return self

                def __exit__(self, *args):
                    return self.inner.__exit__(*args)

                def keys(self):
                    return self.inner.keys()

                def get_tensor(self, key):
                    reads.append(key)
                    return self.inner.get_tensor(key)

            with patch("safetensors.safe_open", TrackingSafeOpen), patch.object(
                torch.hub, "load_state_dict_from_url",
                side_effect=AssertionError("network initialization is forbidden"),
            ):
                adapter, processor = load_encoder(model_dir, "cpu")

            self.assertTrue(weights_path.is_file())
            self.assertIsInstance(processor, CheXficientProcessor)
            self.assertEqual(set(reads), {
                "image_encoder.model.scale",
                "image_encoder.model.offset",
                "image_projection.projection.weight",
                "image_projection.projection.bias",
            })
            self.assertFalse(any(key.startswith("text_") for key in reads))
            self.assertFalse(any(key.startswith("text_") for key in adapter.state_dict()))
            for key, value in expected_model.state_dict().items():
                torch.testing.assert_close(adapter.image_model.state_dict()[key], value)
            for key, value in expected_projection.state_dict().items():
                torch.testing.assert_close(adapter.image_projection.state_dict()[key], value)
            self.assertEqual(adapter.image_model.num_register_tokens, 0)
            self.assertFalse(adapter.training)
            with torch.inference_mode():
                output = adapter(torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE))
            self.assertEqual(tuple(output.shape), (1, PROJECTED_FEATURES))

    def test_loader_rejects_unexpected_image_key_and_wrong_tensor_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = _write_fake_model_dir(directory)
            _fake_checkpoint(model_dir, extra={"image_encoder.model.rogue": torch.ones(1)})
            with self.assertRaisesRegex(ValueError, "image checkpoint keys"):
                load_encoder(model_dir, "cpu")

            _fake_checkpoint(model_dir, projection_shape=(511, SOURCE_FEATURES))
            with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                load_encoder(model_dir, "cpu")

    def test_loader_rejects_preloaded_dinov2_from_another_source(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = _write_fake_model_dir(directory)
            conflict = types.ModuleType("dinov2")
            conflict.__file__ = str(Path(directory) / "other" / "dinov2" / "__init__.py")
            with patch.dict(sys.modules, {"dinov2": conflict}):
                with self.assertRaisesRegex(ImportError, "outside pinned source"):
                    _load_vit_base(model_dir)


if __name__ == "__main__":
    unittest.main()
