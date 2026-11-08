"""Architecture and audited backbone-loading tests."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from cxr import models as model_api


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_architectures_return_ten_logits(self):
        for architecture in model_api.SUPPORTED_ARCHITECTURES:
            with self.subTest(architecture=architecture):
                model = model_api.create_model(architecture).eval()
                with torch.inference_mode():
                    self.assertEqual(tuple(model(torch.zeros(1, 3, 64, 64)).shape), (1, 10))

    def test_resnet18_legacy_default_and_backbone_loading(self):
        model = model_api.create_model()
        source = model_api.create_model()
        source.fc = torch.nn.Linear(source.fc.in_features, 1000)
        state = source.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.pth'
            torch.save(state, path)
            with patch.object(model_api, '_file_sha256', return_value=model_api.IMAGENET_WEIGHT_SHA256['resnet18']):
                model_api.load_imagenet_backbone(model, path)
        self.assertEqual(model.fc.out_features, 10)
        self.assertTrue(torch.equal(model.conv1.weight, source.conv1.weight))

    def test_convnext_loading_keeps_classifier_normalization(self):
        model = model_api.create_model('convnext_tiny')
        source = model_api.models.convnext_tiny(weights=None)
        state = source.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.pth'
            torch.save(state, path)
            with patch.object(model_api, '_file_sha256', return_value=model_api.IMAGENET_WEIGHT_SHA256['convnext_tiny']):
                model_api.load_imagenet_backbone(model, path, 'convnext_tiny')
        self.assertEqual(model.classifier[2].out_features, 10)
        self.assertTrue(torch.equal(model.classifier[0].weight, source.classifier[0].weight))

    def test_predictor_constructs_checkpoint_architecture(self):
        payload = dict(architecture='convnext_tiny', class_ids=list(range(1, 11)),
                       model=model_api.create_model('convnext_tiny').state_dict(), image_size=64,
                       thresholds=[.5] * 10, inference_precision='fp32')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            torch.save(payload, path)
            from cxr.engine import Predictor
            predictor = Predictor(path, device='cpu')
        self.assertEqual(predictor.model.classifier[2].out_features, 10)
        self.assertFalse(predictor.model.training)

    def test_convnext_missing_classifier_norm_is_rejected(self):
        model = model_api.create_model('convnext_tiny')
        source = model_api.models.convnext_tiny(weights=None)
        state = source.state_dict()
        del state['classifier.0.weight']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.pth'
            torch.save(state, path)
            with patch.object(model_api, '_file_sha256', return_value=model_api.IMAGENET_WEIGHT_SHA256['convnext_tiny']):
                with self.assertRaises(ValueError):
                    model_api.load_imagenet_backbone(model, path, 'convnext_tiny')

    def test_wrong_source_is_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'untrusted.pth'
            path.write_bytes(b'not the allowlisted torchvision checkpoint')
            with self.assertRaises(ValueError):
                model_api.load_imagenet_backbone(model_api.create_model(), path)


if __name__ == '__main__':
    unittest.main()
