"""Synthetic tests for the CXR-LT ConvNeXt stage split."""

import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cxr.cxrlt_stage_adapter import (
    LOCAL_CLASSES,
    LOCAL_FEATURES,
    ConvNeXtPrefix,
    split_official_model,
)


class TinyStem(nn.Module):
    def forward(self, x):
        x = F.avg_pool2d(x.mean(dim=1, keepdim=True), kernel_size=4, stride=4)
        return x.expand(-1, 512, -1, -1)


class TinyDownsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=2, stride=2) * self.scale


class TinyFinalStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.linspace(0.5, 1.5, 1024))

    def forward(self, x):
        x = F.avg_pool2d(x.mean(dim=1, keepdim=True), kernel_size=2, stride=2)
        return x * self.scale.view(1, 1024, 1, 1)


class TinyCSRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(1024, 30)

    def forward(self, x):
        return self.classifier(x.mean(dim=(2, 3)))


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = TinyStem()
        self.stages = nn.Sequential(
            nn.Identity(), TinyDownsample(), TinyDownsample(), TinyFinalStage()
        )
        self.norm_pre = nn.BatchNorm2d(1024)
        self.head = nn.Identity()

    def forward(self, x):
        return self.head(self.norm_pre(self.stages(self.stem(x))))


class TinyOfficialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TinyBackbone()
        self.bn = nn.BatchNorm2d(1024)
        self.head = TinyCSRA()

    def forward(self, x):
        return self.head(self.bn(self.backbone(x)))


def probe_payload():
    rng = np.random.default_rng(17)
    return {
        "mean": np.zeros(LOCAL_FEATURES),
        "std": np.ones(LOCAL_FEATURES),
        "weight": rng.normal(size=(LOCAL_CLASSES, LOCAL_FEATURES)).astype(np.float32),
        "bias": rng.normal(size=LOCAL_CLASSES).astype(np.float32),
        "l2": 0.01,
    }


class CXRLTStageAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.model = TinyOfficialModel().eval()
        self.prefix, self.tail = split_official_model(self.model, probe_payload())

    def test_split_features_match_official_forward_and_prefix_is_no_grad(self):
        pixels = torch.randn(2, 3, 32, 32, requires_grad=True)
        with torch.no_grad():
            spatial = self.model.bn(self.model.backbone(pixels))
            expected = torch.cat(
                (spatial.mean(dim=(2, 3)), self.model.head(spatial)), dim=1
            )
        prefix_features = self.prefix(pixels)
        self.assertEqual(tuple(prefix_features.shape), (2, 512, 2, 2))
        self.assertFalse(prefix_features.requires_grad)
        torch.testing.assert_close(self.tail.features(prefix_features), expected)
        self.assertIsInstance(self.prefix, ConvNeXtPrefix)

    def test_train_keeps_official_modules_eval_and_updates_stage_and_local_head(self):
        self.tail.train()
        self.assertFalse(self.tail.stage.training)
        self.assertFalse(self.tail.norm_pre.training)
        self.assertFalse(self.tail.backbone_head.training)
        self.assertFalse(self.tail.official_bn.training)
        self.assertFalse(self.tail.official_csra.training)
        self.assertTrue(self.tail.local_head.training)

        running_mean = self.tail.official_bn.running_mean.clone()
        prefix_features = self.prefix(torch.randn(2, 3, 32, 32))
        logits = self.tail._local_logits(self.tail.features(prefix_features).mean(0, keepdim=True))
        logits.sum().backward()

        self.assertIsNotNone(self.tail.stage.scale.grad)
        self.assertIsNotNone(self.tail.local_head.weight.grad)
        self.assertTrue(all(p.grad is None for p in self.tail.official_bn.parameters()))
        self.assertTrue(all(p.grad is None for p in self.tail.official_csra.parameters()))
        torch.testing.assert_close(self.tail.official_bn.running_mean, running_mean)

    def test_study_pool_promotes_low_precision_features_before_mean(self):
        image_features = torch.zeros(3, LOCAL_FEATURES, dtype=torch.float16)
        image_features[:, 0] = torch.tensor([0.1, 0.2, 0.4], dtype=torch.float16)
        image_features[:, 1] = torch.tensor([3.0, 6.0, 9.0], dtype=torch.float16)
        captured = {}

        def capture(values):
            captured["values"] = values
            return values

        with patch.object(self.tail, "features", return_value=image_features):
            with patch.object(self.tail, "_local_logits", side_effect=capture):
                pooled = self.tail.study_logits(torch.empty(0), [0, 0, 1], 2)

        expected = torch.stack(
            (image_features[:2].float().mean(0), image_features[2].float()), dim=0
        )
        self.assertEqual(pooled.dtype, torch.float32)
        torch.testing.assert_close(pooled, expected)
        torch.testing.assert_close(captured["values"], expected)

    def test_state_dict_reload_preserves_features_and_local_logits(self):
        reloaded_prefix, reloaded_tail = split_official_model(
            TinyOfficialModel().eval(), probe_payload()
        )
        reloaded_prefix.load_state_dict(self.prefix.state_dict())
        reloaded_tail.load_state_dict(self.tail.state_dict())
        pixels = torch.randn(2, 3, 32, 32)
        expected = self.tail._local_logits(self.tail.features(self.prefix(pixels)))
        actual = reloaded_tail._local_logits(
            reloaded_tail.features(reloaded_prefix(pixels))
        )
        torch.testing.assert_close(actual, expected)

    def test_invalid_probe_payloads_and_study_indices_are_rejected(self):
        invalid = probe_payload()
        invalid["std"] = np.zeros(LOCAL_FEATURES)
        with self.assertRaisesRegex(ValueError, "standard deviations"):
            self.tail.load_probe(invalid)

        invalid = probe_payload()
        invalid["class_balance"] = True
        with self.assertRaisesRegex(ValueError, "ordinary BCE"):
            self.tail.load_probe(invalid)

        prefix_features = self.prefix(torch.randn(3, 3, 32, 32))
        with self.assertRaisesRegex(ValueError, "contain integers"):
            self.tail.study_logits(prefix_features, [0.0, 0.0, 1.0], 2)
        with self.assertRaisesRegex(ValueError, "outside the study range"):
            self.tail.study_logits(prefix_features, [0, 0, 2], 2)
        with self.assertRaisesRegex(ValueError, "each study"):
            self.tail.study_logits(prefix_features[:2], [0, 0], 2)


if __name__ == "__main__":
    unittest.main()
