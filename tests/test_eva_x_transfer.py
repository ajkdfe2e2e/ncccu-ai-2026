"""Synthetic contract tests for the pinned EVA-X-S transfer adapter."""

import argparse
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from cxr.eva_x_transfer import (
    FEATURE_SIZE,
    INPUT_SIZE,
    SOURCE_CLASSES,
    SOURCE_FEATURES,
    WEIGHTS_NAME,
    EvaXFeatureAdapter,
    EvaXProcessor,
    _load_checkpoint,
    _restore_checkpoint_rope,
    adapt_model,
    load_encoder,
)


FAKE_MODEL_SOURCE = '''\
import torch
from torch import nn

class FakeEVA(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        expected = {
            "img_size": 224, "patch_size": 16, "embed_dim": 384,
            "depth": 12, "num_heads": 6, "mlp_ratio": 4 * 2 / 3,
            "swiglu_mlp": True, "use_rot_pos_emb": True,
            "ref_feat_shape": (14, 14), "num_classes": 5,
            "global_pool": "avg",
        }
        for key, value in expected.items():
            if kwargs.get(key) != value:
                raise ValueError(f"wrong EVA-X config: {key}")
        self.anchor = nn.Parameter(torch.tensor(1.25))
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(3, 4)
        self.mlp.fc2 = nn.Linear(2, 3)
        self.fc_norm = nn.LayerNorm(384)
        self.head = nn.Linear(384, 5)
        self.rope = nn.Module()
        self.rope.register_buffer("pos_embed", torch.zeros(196, 128), persistent=False)
    def forward_features(self, pixels):
        pooled = pixels.mean(dim=(1, 2, 3)).unsqueeze(1).expand(-1, 384) * self.anchor
        return pooled.unsqueeze(1).expand(-1, 197, -1)
    def forward_head(self, tokens, pre_logits=False):
        pooled = tokens[:, 1:].mean(dim=1)
        pooled = self.fc_norm(pooled)
        return pooled if pre_logits else self.head(pooled)
    def forward(self, pixels):
        raise AssertionError("adapter must use forward_features and head explicitly")

def EVA_X(**kwargs):
    return FakeEVA(**kwargs)

def checkpoint_filter_fn(state, model):
    filtered = {}
    for key, value in state.items():
        if "rope" in key:
            continue
        key = key.replace("mlp.w12", "mlp.fc1").replace("mlp.w3", "mlp.fc2")
        filtered[key] = value
    return filtered
'''


def _checkpoint_state(model):
    state = {key: value for key, value in model.state_dict().items() if key != "rope.pos_embed"}
    sin = torch.arange(196 * 64, dtype=torch.float32).reshape(196, 64) / 10000
    cos = torch.ones(196, 64, dtype=torch.float32)
    state["rope.freqs_sin"] = sin
    state["rope.freqs_cos"] = cos
    for index in range(12):
        state[f"blocks.{index}.attn.rope.freqs_sin"] = sin.clone()
        state[f"blocks.{index}.attn.rope.freqs_cos"] = cos.clone()
    return {
        key.replace("mlp.fc1", "mlp.w12").replace("mlp.fc2", "mlp.w3"): value
        for key, value in state.items()
    }


class EvaXTransferTests(unittest.TestCase):
    def test_processor_matches_pinned_torchvision_transform_exactly(self):
        gray = Image.fromarray(np.arange(35, dtype=np.uint8).reshape(5, 7), mode="L")
        rgb_data = np.zeros((6, 4, 3), dtype=np.uint8)
        rgb_data[..., 0] = np.arange(4, dtype=np.uint8)[None, :] * 51
        rgb_data[..., 1] = np.arange(6, dtype=np.uint8)[:, None] * 31
        rgb_data[..., 2] = 173
        rgb = Image.fromarray(rgb_data, mode="RGB")
        reference = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.49185243,) * 3, (0.28509309,) * 3),
        ])

        output = EvaXProcessor()(images=[gray, rgb])['pixel_values']
        expected = torch.stack([reference(image.convert("RGB")) for image in (gray, rgb)])
        self.assertEqual(tuple(output.shape), (2, 3, INPUT_SIZE, INPUT_SIZE))
        self.assertEqual(output.dtype, torch.float32)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        self.assertEqual(EvaXProcessor.input_size, INPUT_SIZE)

    def test_adapter_uses_explicit_feature_path_and_freezes_model(self):
        class FakeEVA(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(2.0))
                self.fc_norm = nn.LayerNorm(SOURCE_FEATURES)
                self.head = nn.Linear(SOURCE_FEATURES, SOURCE_CLASSES)

            def forward_features(self, pixels):
                value = pixels.mean(dim=(1, 2, 3)).unsqueeze(1)
                return (value.expand(-1, SOURCE_FEATURES) * self.scale).unsqueeze(1).expand(-1, 3, -1)

            def forward_head(self, tokens, pre_logits=False):
                features = self.fc_norm(tokens[:, 1:].mean(dim=1))
                return features if pre_logits else self.head(features)

            def forward(self, pixels):
                raise AssertionError("must not call inherited forward")

        model = FakeEVA()
        adapter = adapt_model(model)
        pixels = torch.randn(3, 3, 8, 8, requires_grad=True)
        features = model.forward_head(model.forward_features(pixels), pre_logits=True)
        expected = torch.cat((features, model.head(features)), dim=1).detach()
        actual = adapter(pixels)
        self.assertIsInstance(adapter, EvaXFeatureAdapter)
        self.assertEqual(tuple(actual.shape), (3, FEATURE_SIZE))
        torch.testing.assert_close(actual, expected)
        self.assertFalse(actual.requires_grad)
        self.assertTrue(all(not parameter.requires_grad for parameter in adapter.parameters()))
        adapter.train()
        self.assertFalse(adapter.training)
        self.assertFalse(model.training)

    def test_local_loader_filters_official_keys_then_strictly_loads_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "eva_x.py").write_text(FAKE_MODEL_SOURCE, encoding="utf-8")
            namespace = {}
            exec(FAKE_MODEL_SOURCE, namespace)
            torch.manual_seed(29)
            from functools import partial
            reference_model = namespace["EVA_X"](
                img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6,
                mlp_ratio=4 * 2 / 3, swiglu_mlp=True, use_rot_pos_emb=True,
                ref_feat_shape=(14, 14), num_classes=5, global_pool="avg",
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
            state = _checkpoint_state(reference_model)
            checkpoint = model_dir / WEIGHTS_NAME
            torch.save({"model": state, "optimizer": {}, "epoch": 4}, checkpoint)

            with patch("torch.load", wraps=torch.load) as load:
                adapter, processor = load_encoder(model_dir, "cpu")

            self.assertIsInstance(adapter, EvaXFeatureAdapter)
            self.assertIsInstance(processor, EvaXProcessor)
            self.assertEqual(adapter.feature_size, FEATURE_SIZE)
            self.assertTrue(all(call.kwargs.get("weights_only") is True for call in load.call_args_list))
            expected_rope = torch.cat((state["rope.freqs_sin"], state["rope.freqs_cos"]), dim=1)
            torch.testing.assert_close(adapter.model.rope.pos_embed, expected_rope, rtol=0, atol=0)
            with torch.no_grad():
                result = adapter(torch.zeros(2, 3, 16, 16))
            self.assertEqual(tuple(result.shape), (2, FEATURE_SIZE))

    def test_loader_strictly_rejects_unexpected_learned_checkpoint_tensor(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "eva_x.py").write_text(FAKE_MODEL_SOURCE, encoding="utf-8")
            namespace = {}
            exec(FAKE_MODEL_SOURCE, namespace)
            model = namespace["EVA_X"](
                img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6,
                mlp_ratio=4 * 2 / 3, swiglu_mlp=True, use_rot_pos_emb=True,
                ref_feat_shape=(14, 14), num_classes=5, global_pool="avg",
            )
            state = _checkpoint_state(model)
            state["unexpected.learned_weight"] = torch.ones(1)
            torch.save({"model": state}, model_dir / WEIGHTS_NAME)
            with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
                load_encoder(model_dir, "cpu")

    def test_rope_restore_rejects_inconsistent_repeated_checkpoint_buffer(self):
        class RopeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.rope = nn.Module()
                self.rope.register_buffer("pos_embed", torch.zeros(196, 128), persistent=False)

        state = _checkpoint_state(RopeModel())
        state["blocks.7.attn.rope.freqs_sin"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "does not match"):
            _restore_checkpoint_rope(state, RopeModel())

    def test_legacy_namespace_checkpoint_uses_narrow_safe_global(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pth"
            torch.save({"model": {"weight": torch.ones(2)},
                        "args": argparse.Namespace(input_size=224)}, path)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                state = _load_checkpoint(path)
            self.assertEqual(set(state), {"weight"})
            self.assertTrue(torch.equal(state["weight"], torch.ones(2)))
            self.assertTrue(any("argparse.Namespace" in str(item.message) for item in caught))

    def test_adapter_rejects_wrong_feature_or_logit_width(self):
        class WrongFeatureModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Linear(383, 5)

            def forward_features(self, pixels):
                return torch.zeros(len(pixels), 2, 383)

            def forward_head(self, tokens, pre_logits=False):
                return torch.zeros(len(tokens), 383)

        with self.assertRaisesRegex(ValueError, "tokens"):
            adapt_model(WrongFeatureModel())(torch.zeros(1, 3, 8, 8))


if __name__ == "__main__":
    unittest.main()
