"""Synthetic inference tests; skip if the isolated torch environment isn't ready."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

HAS_TORCH = importlib.util.find_spec('torch') is not None and importlib.util.find_spec('torchvision') is not None


@unittest.skipUnless(HAS_TORCH, 'PyTorch/torchvision not installed in this Python')
class EngineTests(unittest.TestCase):
    def setUp(self):
        import torch
        from cxr.engine import create_model
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payload = dict(architecture='resnet18', class_ids=list(range(1, 11)),
                            model=create_model().state_dict(), image_size=64,
                            thresholds=[.5] * 10, inference_precision='fp32')

    def checkpoint(self):
        import torch
        path = self.root / 'model.pt'
        torch.save(self.payload, path)
        return path

    def test_predictor_repeatability_and_shape(self):
        import numpy as np
        from PIL import Image
        from cxr.engine import Predictor
        image = self.root / 'image.png'
        Image.new('L', (95, 71), 123).save(image)
        predictor = Predictor(self.checkpoint(), 'cpu')
        first = predictor.predict_image(image)
        second = predictor.predict_image(image)
        self.assertEqual(first.shape, (10,))
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.all((first >= 0) & (first <= 1)))
        self.assertFalse(predictor.model.training)

    def test_invalid_threshold_rejected(self):
        from cxr.engine import Predictor
        self.payload['thresholds'] = [float('nan')] * 10
        with self.assertRaises(ValueError):
            Predictor(self.checkpoint(), 'cpu')

    def test_invalid_size_rejected(self):
        from cxr.engine import Predictor
        self.payload['image_size'] = 0
        with self.assertRaises(ValueError):
            Predictor(self.checkpoint(), 'cpu')

    def test_invalid_class_order_rejected(self):
        from cxr.engine import Predictor
        self.payload['class_ids'] = list(range(10))
        with self.assertRaises(ValueError):
            Predictor(self.checkpoint(), 'cpu')


if __name__ == '__main__':
    unittest.main()
