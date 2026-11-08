"""CPU-only tests for frozen study runner cache and legacy OOF guards."""

import importlib.util
import gc
import itertools
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_frozen_study_probe.py"


def load_runner():
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoImageProcessor = object
    fake_transformers.AutoModel = object
    spec = importlib.util.spec_from_file_location("test_frozen_runner_module", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"transformers": fake_transformers}):
        spec.loader.exec_module(module)
    return module


runner = load_runner()


class FakeProcessor:
    def __init__(self, fail_on_call=None):
        self.fail_on_call = fail_on_call
        self.calls = 0

    def __call__(self, *, images, return_tensors):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("injected interruption")
        means = [float(np.asarray(image, dtype=np.float32).mean()) / 255.0 for image in images]
        pixels = torch.tensor(means, dtype=torch.float32)[:, None, None, None]
        pixels = pixels.expand(len(means), 3, 518, 518).contiguous()
        return {"pixel_values": pixels}


class FakeEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=2)

    def forward(self, *, pixel_values):
        means = pixel_values.mean(dim=(1, 2, 3))
        tokens = torch.stack((torch.stack((means, means + 1), dim=1),
                              torch.stack((means + 2, means + 3), dim=1),
                              torch.stack((means + 4, means + 5), dim=1)), dim=1)
        return types.SimpleNamespace(last_hidden_state=tokens)


class FakeDirectEncoder(torch.nn.Module):
    feature_size = 3
    input_size = 518
    feature_pooling = "fake_supervised_features"

    def forward(self, *, pixel_values):
        means = pixel_values.mean(dim=(1, 2, 3))
        return torch.stack((means, means + 1, means - 1), dim=1)


class FrozenRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.records = []
        for index, value in enumerate((20, 80, 140, 220)):
            path = self.root / f"image-{index}.png"
            Image.new("RGB", (8, 8), (value, value, value)).save(path)
            self.records.append({"dicom_id": f"d{index}", "path": str(path)})
        self.device = torch.device("cpu")

    def extract(self, folder, records=None, processor=None):
        ticks = itertools.count()
        with patch.object(runner.time, "monotonic", side_effect=lambda: next(ticks) / 10):
            values = runner.extract_features(FakeEncoder(), processor or FakeProcessor(), records or self.records,
                                             folder, batch_size=1, device=self.device)
        copied = np.asarray(values).copy()
        values._mmap.close()
        return copied

    def test_resume_starts_at_committed_cursor_and_matches_continuous(self):
        cache = self.root / "interrupted"
        with self.assertRaisesRegex(RuntimeError, "injected interruption"):
            self.extract(cache, processor=FakeProcessor(fail_on_call=2))
        gc.collect()
        progress = json.loads((cache / "progress.json").read_text())
        self.assertEqual(progress["completed"], 1)

        resumed_processor = FakeProcessor()
        resumed = self.extract(cache, processor=resumed_processor)
        self.assertEqual(resumed_processor.calls, 3)
        continuous = self.extract(self.root / "continuous")
        np.testing.assert_array_equal(resumed, continuous)

    def test_completed_cache_rejects_matrix_hash_change(self):
        cache = self.root / "complete"
        self.extract(cache)
        values = np.load(cache / "image_features.npy", mmap_mode="r+", allow_pickle=False)
        values[0, 0] += 1
        values.flush()
        values._mmap.close()
        with self.assertRaisesRegex(ValueError, "completed feature cache hash mismatch"):
            self.extract(cache)

    def test_cache_rejects_image_identity_or_order_change(self):
        cache = self.root / "ordered"
        self.extract(cache)
        changed_identity = [dict(row) for row in self.records]
        changed_identity[0]["dicom_id"] = "different"
        with self.assertRaisesRegex(ValueError, "feature cache input mismatch"):
            self.extract(cache, records=changed_identity)
        with self.assertRaisesRegex(ValueError, "feature cache input mismatch"):
            self.extract(cache, records=list(reversed(self.records)))

    def test_threaded_preprocessing_preserves_batches_order_and_values(self):
        serial = list(runner.pixel_batches(self.records, FakeProcessor(), 0, 3, 0))
        threaded = list(runner.pixel_batches(self.records, FakeProcessor(), 0, 3, 2))
        self.assertEqual([offset for offset, _ in threaded], [0, 3])
        for (serial_offset, serial_pixels), (threaded_offset, threaded_pixels) in zip(serial, threaded):
            self.assertEqual(serial_offset, threaded_offset)
            self.assertTrue(torch.equal(serial_pixels, threaded_pixels))

    def test_supervised_tensor_features_are_saved_without_token_pooling(self):
        folder = self.root / "direct"
        values = runner.extract_features(FakeDirectEncoder(), FakeProcessor(), self.records,
                                         folder, batch_size=2, device=self.device)
        self.assertEqual(values.shape, (4, 3))
        expected = np.asarray([20, 80, 140, 220]) / 255
        np.testing.assert_allclose(values[:, 0], expected, atol=1e-6)
        np.testing.assert_allclose(values[:, 1] - values[:, 0], 1, atol=1e-6)
        spec = json.loads((folder / "input.json").read_text())
        self.assertEqual(spec["pooling"], FakeDirectEncoder.feature_pooling)
        self.assertEqual(spec["input_size"], 518)
        values._mmap.close()

    def test_encoder_dispatch_passes_registered_model_directory(self):
        expected = (FakeDirectEncoder(), FakeProcessor())
        module = types.ModuleType("cxr.cxrlt_transfer")
        calls = []

        def load(model_dir, device):
            calls.append((model_dir, str(device)))
            return expected

        module.load_encoder = load
        with patch.dict(sys.modules, {"cxr.cxrlt_transfer": module}):
            model, processor, device = runner.load_encoder(
                {"encoder": "cxrlt", "model_dir": "locked-model", "device": "cpu"})
        self.assertIs(model, expected[0])
        self.assertIs(processor, expected[1])
        self.assertEqual(str(device), "cpu")
        self.assertEqual(calls, [("locked-model", "cpu")])

    def test_legacy_oof_rejects_wrong_study_order_and_missing_head_weights(self):
        directory = self.root / "old-oof"
        directory.mkdir()
        studies = [{"subject_id": "p1", "study_id": "s1"},
                   {"subject_id": "p2", "study_id": "s2"}]
        ids_path = directory / "oof_study_ids.json"
        plan_path = directory / "plan.json"
        np.save(directory / "oof_component_probabilities.npy", np.zeros((2, 8, 10)))
        ids_path.write_text(json.dumps([["p2", "s2"], ["p1", "s1"]]))
        plan_path.write_text(json.dumps({"prior": np.ones((8, 10)).tolist()}))
        plan = {"old_oof": str(directory)}
        with self.assertRaisesRegex(ValueError, "study order differs"):
            runner.old_oof_predictions(plan, studies)

        ids_path.write_text(json.dumps([["p1", "s1"], ["p2", "s2"]]))
        plan_path.write_text(json.dumps({"prior": np.ones((7, 10)).tolist()}))
        with self.assertRaisesRegex(ValueError, "unexpected legacy OOF input shape"):
            runner.old_oof_predictions(plan, studies)


if __name__ == "__main__":
    unittest.main()
