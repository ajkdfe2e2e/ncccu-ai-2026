"""Synthetic contract tests for the pinned BioViL-T image adapter."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

from cxr import biovil_t_transfer as transfer


class FakeImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.25))
        self.projector = nn.Linear(transfer.RAW_FEATURES, transfer.PROJECTED_FEATURES)

    def forward(self, pixels):
        raw = pixels.mean(dim=(1, 2, 3)).unsqueeze(1).expand(-1, transfer.RAW_FEATURES) * self.scale
        projected = self.projector(raw)
        return type("Output", (), {"img_embedding": raw, "projected_global_embedding": projected})()


class BioViLTransferTests(unittest.TestCase):
    def test_processor_matches_official_order_for_rgb_gray_and_constant_images(self):
        rgb_array = np.zeros((7, 5, 3), dtype=np.uint8)
        rgb_array[..., 0] = np.arange(5, dtype=np.uint8)[None, :] * 30
        rgb_array[..., 1] = np.arange(7, dtype=np.uint8)[:, None] * 17
        gray_array = np.arange(35, dtype=np.uint8).reshape(5, 7)
        inputs = [Image.fromarray(rgb_array), Image.fromarray(gray_array), Image.fromarray(np.full((3, 4), 9, dtype=np.uint8))]

        expected = []
        transform = Compose([Resize(512), CenterCrop(448), ToTensor()])
        for image in inputs:
            source = np.asarray(image).astype(float)
            span = source.max() - source.min()
            remapped = (np.zeros(source.shape, dtype=np.uint8) if span == 0 else
                        ((source - source.min()) * (255.0 / span)).astype(np.uint8))
            expected.append(transform(Image.fromarray(remapped).convert("L")).expand(3, -1, -1))
        actual = transfer.BioViLTProcessor()(images=inputs)["pixel_values"]
        self.assertEqual(tuple(actual.shape), (3, 3, 448, 448))
        torch.testing.assert_close(actual, torch.stack(expected), rtol=0, atol=0)
        self.assertTrue(torch.equal(actual[2], torch.zeros_like(actual[2])))

    def test_adapter_is_frozen_and_returns_raw_plus_normalized_projection(self):
        adapter = transfer.adapt_model(FakeImageModel())
        pixels = torch.rand(2, 3, 448, 448, requires_grad=True)
        features = adapter(pixels)
        self.assertEqual(tuple(features.shape), (2, transfer.FEATURE_SIZE))
        self.assertFalse(features.requires_grad)
        _, projected = adapter.encode_components(pixels)
        torch.testing.assert_close(torch.linalg.vector_norm(projected, dim=1), torch.ones(2))
        self.assertTrue(all(not parameter.requires_grad for parameter in adapter.parameters()))
        adapter.train()
        self.assertFalse(adapter.training)

    def test_loader_uses_weights_only_and_strict_state_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source/hi-ml-multimodal/src/health_multimodal/image/model/model.py"
            source.parent.mkdir(parents=True)
            source.write_text(
                "import torch\nfrom torch import nn\nfrom types import SimpleNamespace\n"
                "class ImageModel(nn.Module):\n"
                " def __init__(self, img_encoder_type, joint_feature_size, pretrained_model_path=None):\n"
                "  super().__init__(); assert img_encoder_type == 'resnet50_multi_image' and joint_feature_size == 128 and pretrained_model_path is None\n"
                "  self.scale=nn.Parameter(torch.tensor(1.25)); self.projector=nn.Linear(512,128)\n"
                " def forward(self, x):\n"
                "  raw=x.mean((1,2,3)).unsqueeze(1).expand(-1,512)*self.scale\n"
                "  return SimpleNamespace(img_embedding=raw, projected_global_embedding=self.projector(raw))\n",
                encoding="utf-8")
            weights = root / transfer.WEIGHTS_NAME
            reference = FakeImageModel()
            torch.save(reference.state_dict(), weights)
            relative_files = [
                transfer.WEIGHTS_NAME, "README.md", "source/LICENSE", "source/hi-ml-multimodal/setup.py",
                "source/hi-ml-multimodal/src/health_multimodal/__init__.py",
                "source/hi-ml-multimodal/src/health_multimodal/common/__init__.py",
                "source/hi-ml-multimodal/src/health_multimodal/common/device.py",
                "source/hi-ml-multimodal/src/health_multimodal/common/visualization.py",
                "source/hi-ml-multimodal/src/health_multimodal/image/__init__.py",
                "source/hi-ml-multimodal/src/health_multimodal/image/data/__init__.py",
                "source/hi-ml-multimodal/src/health_multimodal/image/data/io.py",
                "source/hi-ml-multimodal/src/health_multimodal/image/data/transforms.py",
                "source/hi-ml-multimodal/src/health_multimodal/image/inference_engine.py",
                "source/hi-ml-multimodal/src/health_multimodal/image/model/__init__.py",
                "source/hi-ml-multimodal/src/health_multimodal/image/model/encoder.py",
                "source/hi-ml-multimodal/src/health_multimodal/image/model/model.py",
                "source/hi-ml-multimodal/src/health_multimodal/image/model/modules.py",
                "source/hi-ml-multimodal/src/health_multimodal/image/model/pretrained.py",
                "source/hi-ml-multimodal/src/health_multimodal/image/model/resnet.py",
                "source/hi-ml-multimodal/src/health_multimodal/image/model/transformer.py",
                "source/hi-ml-multimodal/src/health_multimodal/image/model/types.py",
                "source/hi-ml-multimodal/src/health_multimodal/image/utils.py",
            ]
            locks = {}
            for relative in relative_files:
                asset = root / relative
                if relative != transfer.WEIGHTS_NAME and not relative.endswith("/model.py"):
                    asset.parent.mkdir(parents=True, exist_ok=True)
                    asset.write_bytes(b"synthetic pinned asset")
                locks[relative] = {"sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
                                   "bytes": asset.stat().st_size}
            manifest = {"model": transfer.MODEL_REPOSITORY, "revision": transfer.MODEL_REVISION,
                        "code_revision": transfer.SOURCE_REVISION, "files": locks}
            provenance_path = root / "provenance.json"
            provenance_path.write_text(json.dumps({
                "model": transfer.MODEL_REPOSITORY, "revision": transfer.MODEL_REVISION,
                "code_revision": transfer.SOURCE_REVISION, "files": locks}), encoding="utf-8")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            digest = locks[transfer.WEIGHTS_NAME]["sha256"]
            with patch.object(transfer, "WEIGHTS_BYTES", weights.stat().st_size), \
                    patch.object(transfer, "WEIGHTS_SHA256", digest), \
                    patch.object(transfer, "SOURCE_MANIFEST", manifest_path), \
                    patch("torch.load", wraps=torch.load) as load:
                adapter, processor = transfer.load_encoder(root, "cpu")
            self.assertIsInstance(adapter, transfer.BioViLTFeatureAdapter)
            self.assertIsInstance(processor, transfer.BioViLTProcessor)
            self.assertTrue(all(call.kwargs.get("weights_only") is True for call in load.call_args_list))
            bad_state = reference.state_dict()
            bad_state["unexpected.weight"] = torch.ones(1)
            torch.save(bad_state, weights)
            bad_digest = hashlib.sha256(weights.read_bytes()).hexdigest()
            locks[transfer.WEIGHTS_NAME] = {"sha256": bad_digest, "bytes": weights.stat().st_size}
            (root / "provenance.json").write_text(json.dumps({
                "model": transfer.MODEL_REPOSITORY, "revision": transfer.MODEL_REVISION,
                "code_revision": transfer.SOURCE_REVISION, "files": locks}), encoding="utf-8")
            manifest_path.write_text(json.dumps({**manifest, "files": locks}), encoding="utf-8")
            with patch.object(transfer, "WEIGHTS_BYTES", weights.stat().st_size), \
                    patch.object(transfer, "WEIGHTS_SHA256", bad_digest), \
                    patch.object(transfer, "SOURCE_MANIFEST", manifest_path), \
                    self.assertRaises(RuntimeError):
                transfer.load_encoder(root, "cpu")

    def test_adapter_rejects_nonfinite_and_wrong_input(self):
        adapter = transfer.adapt_model(FakeImageModel())
        with self.assertRaisesRegex(ValueError, "shape"):
            adapter(torch.zeros(1, 3, 224, 224))
        with self.assertRaisesRegex(ValueError, "PIL"):
            transfer.BioViLTProcessor()(images=[object()])


if __name__ == "__main__":
    unittest.main()
