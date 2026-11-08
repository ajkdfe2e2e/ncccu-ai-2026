"""Offline tests for the CXR-LT ConvNeXtV2 feature adapter."""

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
import torch

from cxr.cxrlt_transfer import (
    CXRLTProcessor,
    FEATURE_POOLING,
    ConvNeXtV2FeatureAdapter,
    load_encoder,
)


FAKE_MODEL_SOURCE = '''\
import torch
from torch import nn

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.calls = 0
    def forward(self, pixels):
        self.calls += 1
        value = pixels.mean(dim=(1, 2, 3)).view(-1, 1, 1, 1) * self.scale
        return value.expand(-1, 1024, 2, 2)

class Head(nn.Module):
    def forward(self, spatial):
        return spatial.mean(dim=(2, 3))[:, :30] + 7.0

class ConvNeXtV2CXR(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.bn = nn.BatchNorm2d(1024)
        self.head = Head()
        self.strict_used = None
    def load_state_dict(self, state_dict, strict=True):
        self.strict_used = strict
        return super().load_state_dict(state_dict, strict=strict)
'''


def reference(image):
    if image.mode == "L":
        gray = np.asarray(image, dtype=np.uint8)
    else:
        gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (512, 512))
    rgb = np.repeat(resized[..., None], 3, axis=2).astype(np.float32) / 255.0
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    return np.transpose((rgb - mean) / std, (2, 0, 1))


class CXRLTTransferTests(unittest.TestCase):
    def test_processor_matches_opencv_reference_for_l_and_rgb(self):
        gray = Image.fromarray(np.arange(35, dtype=np.uint8).reshape(5, 7), mode="L")
        rgb_data = np.zeros((6, 4, 3), dtype=np.uint8)
        rgb_data[..., 0] = np.arange(4, dtype=np.uint8)[None, :] * 50
        rgb_data[..., 1] = np.arange(6, dtype=np.uint8)[:, None] * 30
        rgb_data[..., 2] = 180
        rgb = Image.fromarray(rgb_data, mode="RGB")
        output = CXRLTProcessor()(images=[gray, rgb], return_tensors="pt")["pixel_values"]
        self.assertEqual(tuple(output.shape), (2, 3, 512, 512))
        self.assertEqual(output.dtype, torch.float32)
        np.testing.assert_allclose(output[0].numpy(), reference(gray), rtol=0, atol=0)
        np.testing.assert_allclose(output[1].numpy(), reference(rgb), rtol=0, atol=0)

    def test_adapter_returns_gap_and_raw_logits_from_one_backbone_call(self):
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
            def forward(self, pixels):
                self.calls += 1
                v = pixels.mean(dim=(1, 2, 3)).view(-1, 1, 1, 1)
                return v.expand(-1, 1024, 3, 2)

        class Head(torch.nn.Module):
            def forward(self, spatial):
                pooled = spatial.mean(dim=(2, 3))
                return pooled[:, :30] * 2.5 + 13.0

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = Backbone()
                self.bn = torch.nn.BatchNorm2d(1024)
                self.head = Head()

        base = Model().eval()
        adapter = ConvNeXtV2FeatureAdapter(base)
        result = adapter(torch.ones(2, 3, 8, 8))
        self.assertEqual(tuple(result.shape), (2, 1054))
        self.assertEqual(base.backbone.calls, 1)
        torch.testing.assert_close(result[:, :1024], torch.ones(2, 1024))
        torch.testing.assert_close(result[:, 1024:], torch.full((2, 30), 15.5))
        self.assertEqual(adapter.feature_size, 1054)
        self.assertEqual(adapter.input_size, 512)
        self.assertEqual(adapter.feature_pooling, FEATURE_POOLING)
        self.assertFalse(adapter.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in adapter.parameters()))
        adapter.train()
        self.assertFalse(adapter.training)

    def test_local_loader_uses_strict_safetensors_and_freezes_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "model.py").write_text(FAKE_MODEL_SOURCE, encoding="utf-8")
            weights = model_dir / "convnextv2_base_mimic-cxr_padchest_csra_dbcas.safetensors"
            weights.write_bytes(b"mock safetensors path")

            # The fake model's expected state keys let the real strict loader path run
            # without constructing or downloading the 88M parameter ConvNeXt.
            namespace = {}
            exec(FAKE_MODEL_SOURCE, namespace)
            state = namespace["ConvNeXtV2CXR"]().state_dict()
            with patch("safetensors.torch.load_file", return_value=state) as load_file:
                adapter, processor = load_encoder(model_dir, "cpu")
            load_file.assert_called_once_with(str(weights), device="cpu")
            self.assertIsInstance(adapter, ConvNeXtV2FeatureAdapter)
            self.assertIsInstance(processor, CXRLTProcessor)
            self.assertIs(adapter.model.strict_used, True)
            self.assertFalse(adapter.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in adapter.parameters()))
            self.assertEqual(adapter.feature_size, 1054)
            self.assertEqual(adapter.input_size, 512)


if __name__ == "__main__":
    unittest.main()
