import unittest
from pathlib import Path
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from scripts.predict_grouped_public import (
    DENSENET_LABELS,
    FIVE_LABELS,
    LOCAL_LABELS,
    _predict,
    aggregate_mapped_predictions,
    select_partition_records,
)


def record(dicom, patient, study, label):
    labels = [0.0] * 10
    labels[label] = 1.0
    return {"dicom_id": dicom, "subject_id": patient, "study_id": study,
            "labels": labels, "path": f"/{dicom}.jpg"}


class GroupedPublicTests(unittest.TestCase):
    def setUp(self):
        self.records = [record("d1", "p1", "s1", 0), record("d2", "p1", "s1", 0),
                        record("d3", "p2", "s2", 1), record("d4", "p3", "s3", 2)]
        self.partition = {"subject_partition": {"p1": 0, "p2": 1, "p3": "audit"}}

    def test_development_and_audit_are_patient_disjoint(self):
        dev = select_partition_records(self.records, self.partition, "development")
        audit = select_partition_records(self.records, self.partition, "audit")
        self.assertEqual({row["subject_id"] for row in dev}, {"p1", "p2"})
        self.assertEqual({row["subject_id"] for row in audit}, {"p3"})
        self.assertFalse({row["subject_id"] for row in dev} & {row["subject_id"] for row in audit})

    def test_partition_missing_or_extra_patients_fails_closed(self):
        for assignments in ({"p1": 0, "p3": "audit"},
                            {"p1": 0, "p2": 1, "p3": "audit", "p4": 0}):
            with self.subTest(assignments=assignments), self.assertRaises(ValueError):
                select_partition_records(self.records, {"subject_partition": assignments}, "development")

    def test_development_never_includes_audit(self):
        dev = select_partition_records(self.records, self.partition, "development")
        self.assertNotIn("p3", {row["subject_id"] for row in dev})

    def test_mapped_classes_and_variable_image_study_mean(self):
        probs = np.array([[.2, .8], [.6, .4], [.9, .1], [.3, .7]], dtype=np.float32)
        studies, result, targets = aggregate_mapped_predictions(self.records, probs, [1, 8])
        self.assertEqual(studies, [("p1", "s1"), ("p2", "s2"), ("p3", "s3")])
        np.testing.assert_allclose(result, [[.4, .6], [.9, .1], [.3, .7]])
        self.assertEqual(result.shape, (3, 2))
        self.assertEqual(targets.shape, (3, 10))
        self.assertEqual(targets[1, 1], 1)

    def test_invalid_class_maps_and_probability_alignment_fail(self):
        with self.assertRaises(ValueError):
            aggregate_mapped_predictions(self.records, np.ones((4, 2)), [1, 1])
        with self.assertRaises(ValueError):
            aggregate_mapped_predictions(self.records, np.ones((3, 2)), [1, 8])
        with self.assertRaises(ValueError):
            aggregate_mapped_predictions(self.records, np.ones((4, 2)), [1, 10])

    def test_nonfinite_out_of_range_missing_or_duplicate_records_fail(self):
        for probs in (np.full((4, 1), np.nan), np.full((4, 1), 1.1)):
            with self.subTest(probs=probs[0]), self.assertRaises(ValueError):
                aggregate_mapped_predictions(self.records, probs, [0])
        duplicate = self.records + [dict(self.records[0])]
        with self.assertRaisesRegex(ValueError, "duplicate image"):
            aggregate_mapped_predictions(duplicate, np.ones((5, 1)) * .5, [0])
        missing = [dict(row) for row in self.records]
        del missing[0]["study_id"]
        with self.assertRaisesRegex(ValueError, "missing required"):
            aggregate_mapped_predictions(missing, np.ones((4, 1)) * .5, [0])

    def test_predict_maps_dense14_logits_into_local10_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "image.png"
            Image.new("RGB", (12, 12), color=(80, 90, 100)).save(image_path)
            rows = [{"path": str(image_path)}]

            class FakeBackbone(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.classifier = nn.Linear(2, 14)

                def forward(self, images):
                    return torch.arange(14, dtype=images.dtype, device=images.device).repeat(images.shape[0], 1)

            with patch("scripts.predict_grouped_public._model_labels",
                       return_value=(LOCAL_LABELS, list(range(10)), {})), \
                 patch("torchvision.models.densenet121", return_value=FakeBackbone()), \
                 patch("safetensors.torch.load_file", return_value={}), \
                 patch.object(nn.Module, "load_state_dict", return_value=None):
                probabilities, labels, indices, _ = _predict(
                    "densenet121", rows, Path(temporary), torch.device("cpu"), 1, 0)

            model_order = np.arange(14)
            expected_indices = [DENSENET_LABELS.index(name) for name in LOCAL_LABELS]
            np.testing.assert_allclose(probabilities[0], 1 / (1 + np.exp(-model_order[expected_indices])))
            self.assertEqual(probabilities.shape, (1, 10))
            self.assertEqual(labels, LOCAL_LABELS)
            self.assertEqual(indices, list(range(10)))

    def test_predict_preserves_convnext5_model_order_with_local_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "image.png"
            Image.new("RGB", (12, 12), color=(80, 90, 100)).save(image_path)
            rows = [{"path": str(image_path)}]

            class FakeConvNext(nn.Module):
                def forward(self, pixel_values):
                    logits = torch.arange(5, dtype=pixel_values.dtype).repeat(pixel_values.shape[0], 1)
                    return SimpleNamespace(logits=logits)

            labels = [LOCAL_LABELS.index(name) for name in FIVE_LABELS]
            fake_transformers = SimpleNamespace(
                AutoModelForImageClassification=SimpleNamespace(
                    from_pretrained=lambda *args, **kwargs: FakeConvNext()))
            with patch("scripts.predict_grouped_public._model_labels",
                       return_value=(FIVE_LABELS, labels, {})), \
                 patch.dict("sys.modules", {"transformers": fake_transformers}):
                probabilities, model_labels, indices, _ = _predict(
                    "russellala_convnext", rows, Path(temporary), torch.device("cpu"), 1, 0)

            np.testing.assert_allclose(probabilities[0], 1 / (1 + np.exp(-np.arange(5))))
            self.assertEqual(probabilities.shape, (1, 5))
            self.assertEqual(model_labels, FIVE_LABELS)
            self.assertEqual(indices, labels)

    def test_predict_runs_benchmark_branch_and_reports_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "branch"
            model_dir.mkdir()
            image_path = root / "image.png"
            Image.new("RGB", (12, 12), color=(80, 90, 100)).save(image_path)
            rows = [{"path": str(image_path)}]
            labels = [LOCAL_LABELS.index(name) for name in FIVE_LABELS]
            cfg = {"tasks": FIVE_LABELS, "head": {"n_logits": 5}}

            class FakeModel:
                def to(self, _device): return self
                def eval(self): return self
                def __call__(self, pixels): return torch.arange(5, dtype=torch.float32).reshape(1, 5)

            fake_module = SimpleNamespace(
                load_model=lambda *_args, **_kwargs: (FakeModel(), cfg),
                preprocess=lambda _gray, _cfg: torch.zeros((1, 1, 2, 2)),
            )
            fake_loader = SimpleNamespace(exec_module=lambda _module: None)
            fake_spec = SimpleNamespace(loader=fake_loader, name="fake_grouped_public_modeling")
            with patch("scripts.predict_grouped_public._model_labels",
                       return_value=(FIVE_LABELS, labels, cfg)), \
                 patch("scripts.predict_grouped_public.importlib.util.spec_from_file_location",
                       return_value=fake_spec), \
                 patch("scripts.predict_grouped_public.importlib.util.module_from_spec",
                       return_value=fake_module), \
                 redirect_stdout(StringIO()) as stdout:
                probabilities, model_labels, indices, returned_cfg = _predict(
                    "rad_dino_vitB_768", rows, model_dir, torch.device("cpu"), 4, 0)

            np.testing.assert_allclose(probabilities[0], 1 / (1 + np.exp(-np.arange(5))))
            self.assertEqual(probabilities.shape, (1, 5))
            self.assertEqual(model_labels, FIVE_LABELS)
            self.assertEqual(indices, labels)
            self.assertEqual(returned_cfg, cfg)
            self.assertIn("rad_dino_vitB_768: predicted 1/1 images", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
