import sys
import types
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from cxr import medical_transfer as mt


class TinyBackbone(nn.Module):
    def __init__(self, weights, num_classes, in_channels, op_threshs, apply_sigmoid):
        super().__init__()
        assert weights is None
        self.features = nn.Sequential(nn.Conv2d(in_channels, 3, 1), nn.BatchNorm2d(3))
        self.classifier = nn.Linear(3, num_classes)


def fake_xrv():
    labels = ["Atelectasis", "Consolidation", "", "Pneumothorax", "Edema", "", "",
              "Effusion", "Pneumonia", "", "Cardiomegaly", "", "", "", "Lung Lesion",
              "Fracture", "Lung Opacity", "Enlarged Cardiomediastinum"]
    xrv = types.SimpleNamespace(
        models=types.SimpleNamespace(DenseNet=TinyBackbone,
                                     model_urls={mt.WEIGHTS_NAME: {"labels": labels}}),
        utils=types.SimpleNamespace(load_image=lambda path: np.zeros((1, 20, 28), dtype=np.float32)),
        datasets=types.SimpleNamespace(XRayCenterCrop=lambda: lambda image: image,
                                       XRayResizer=lambda size: lambda image: np.resize(image, (1, size, size))),
    )
    return xrv


class MedicalTransferTests(unittest.TestCase):
    def setUp(self):
        module_patch = patch.dict(sys.modules, {"torchxrayvision": fake_xrv()})
        module_patch.start()
        self.addCleanup(module_patch.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_transfer_copies_nine_rows_and_leaves_no_finding_independent(self):
        source = TinyBackbone(None, 18, 1, None, False).state_dict()
        source["classifier.weight"] = torch.arange(54, dtype=torch.float32).reshape(18, 3)
        source["classifier.bias"] = torch.arange(18, dtype=torch.float32)
        checkpoint = self.root / "xrv.pt"
        torch.save({"model": source, "pathologies": [
            "Atelectasis", "Consolidation", "", "Pneumothorax", "Edema", "", "", "Effusion",
            "Pneumonia", "", "Cardiomegaly", "", "", "", "Lung Lesion", "Fracture",
            "Lung Opacity", "Enlarged Cardiomediastinum",
        ], "weights_name": mt.WEIGHTS_NAME}, checkpoint)

        torch.manual_seed(42)
        random_model = mt.build_model()
        torch.manual_seed(42)
        model = mt.build_model(checkpoint)
        for target, source_index in enumerate(mt.SOURCE_INDICES[:-1]):
            self.assertTrue(torch.equal(model.classifier.weight[target], source["classifier.weight"][source_index]))
            self.assertTrue(torch.equal(model.classifier.bias[target], source["classifier.bias"][source_index]))
        self.assertTrue(torch.equal(model.classifier.weight[-1], random_model.classifier.weight[-1]))
        self.assertTrue(torch.equal(model.classifier.bias[-1], random_model.classifier.bias[-1]))
        self.assertEqual(model.pathologies, mt.TARGET_NAMES)
        with torch.no_grad():
            model.classifier.weight.zero_()
            model.classifier.bias.fill_(2.5)
        self.assertTrue(torch.equal(model(torch.zeros(2, 1, 224, 224)), torch.full((2, 10), 2.5)))

    def test_medical_preprocessing_crops_resizes_and_preserves_grayscale_range(self):
        image = np.linspace(-1300, 1300, 20 * 32, dtype=np.float32).reshape(20, 32)
        with patch.object(mt, "_load_medical_image", return_value=image):
            tensor = mt._preprocess("synthetic")
        self.assertEqual(tensor.shape, (1, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(tensor.min() >= -1024 and tensor.max() <= 1024)

    def test_predict_records_uses_eval_for_inference_and_restores_training(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm2d(1)
                self.seen_training = None

            def forward(self, x):
                self.seen_training = self.training or self.bn.training
                return torch.zeros((len(x), 10))

        model = Model().train()
        records = [{"path": "0"}, {"path": "1"}]
        preprocess = lambda path, training=False: torch.full((1, 224, 224), float(path))
        with patch.object(mt, "_preprocess", side_effect=preprocess):
            probs = mt.predict_records(model, records, "cpu", batch_size=2)
        self.assertEqual(probs.shape, (2, 10))
        self.assertTrue(np.all(probs == 0.5))
        self.assertIs(model.seen_training, False)
        self.assertTrue(model.training and model.bn.training)
