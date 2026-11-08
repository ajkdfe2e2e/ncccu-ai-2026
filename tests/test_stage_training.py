"""Synthetic CPU checks for study-weighted final-stage training and resume."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn

from cxr.stage_training import (
    StageTrainingConfig,
    StudyBatch,
    build_optimizer,
    restore_trained_tail,
    train_one_epoch,
)


class SyntheticTail(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage = nn.Linear(2, 3)
        self.frozen_bn = nn.BatchNorm1d(3)
        self.local_head = nn.Linear(3, 10)
        self.register_buffer("probe_mean", torch.zeros(3))
        self.register_buffer("probe_std", torch.ones(3))
        self.register_buffer("_probe_loaded", torch.tensor(True))

    def study_logits(self, features, image_to_study, study_count):
        image_features = self.frozen_bn(self.stage(features.flatten(1)))
        pooled = torch.zeros(study_count, image_features.shape[1], device=image_features.device)
        pooled.index_add_(0, image_to_study, image_features)
        counts = torch.bincount(image_to_study, minlength=study_count).to(image_features.dtype).unsqueeze(1)
        return self.local_head(pooled / counts)


def synthetic_data(study_count=8):
    generator = torch.Generator().manual_seed(91)
    images = torch.randn(study_count, 1, 1, 2, generator=generator)
    targets = (torch.rand(study_count, 10, generator=generator) > 0.7).float()

    def fetch(indices):
        rows = torch.as_tensor(indices, dtype=torch.long)
        return StudyBatch(
            prefix_features=images[rows].clone(),
            image_to_study=torch.arange(len(rows), dtype=torch.long),
            targets=targets[rows].clone(),
            study_indices=rows.clone(),
        )

    return images, targets, fetch


def scaler_cpu():
    return torch.amp.GradScaler("cuda", enabled=False)


class StageTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)

    @mock.patch("cxr.training_state.torch.cuda.is_available", return_value=False)
    def test_microbatch_accumulation_matches_one_full_study_batch(self, _cuda):
        images, targets, fetch = synthetic_data(8)
        initial = SyntheticTail().state_dict()
        left, right = SyntheticTail(), SyntheticTail()
        left.load_state_dict(initial)
        right.load_state_dict(initial)
        left_config = StageTrainingConfig(microbatch_studies=2, accumulation_studies=8, l2=0.01)
        right_config = StageTrainingConfig(microbatch_studies=8, accumulation_studies=8, l2=0.01)
        left_optimizer = build_optimizer(left, left_config)
        right_optimizer = build_optimizer(right, right_config)
        binding = {"initialization_sha256": "init", "prefix_cache_sha256": "cache"}

        np.random.seed(7)
        left_report = train_one_epoch(left, list(range(8)), fetch, config=left_config,
                                      optimizer=left_optimizer, scaler=scaler_cpu(), binding=binding)
        np.random.seed(7)
        right_report = train_one_epoch(right, list(range(8)), fetch, config=right_config,
                                       optimizer=right_optimizer, scaler=scaler_cpu(), binding=binding)

        self.assertEqual(left_report["steps"], 1)
        self.assertEqual(right_report["steps"], 1)
        for key, expected in left.state_dict().items():
            self.assertTrue(torch.equal(expected, right.state_dict()[key]), key)

    @mock.patch("cxr.training_state.torch.cuda.is_available", return_value=False)
    def test_tail_window_is_processed_and_summary_counts_studies(self, _cuda):
        _, _, fetch = synthetic_data(5)
        model = SyntheticTail()
        config = StageTrainingConfig(microbatch_studies=2, accumulation_studies=4,
                                     checkpoint_every_windows=1)
        progress = []
        report = train_one_epoch(
            model, np.arange(5), fetch, config=config, optimizer=build_optimizer(model, config),
            scaler=scaler_cpu(), binding={"initialization_sha256": "init"},
            progress_callback=progress.append,
        )
        self.assertTrue(report["completed"])
        self.assertEqual(report["studies"], 5)
        self.assertEqual(report["steps"], 2)
        self.assertEqual(report["next_study"], 5)
        self.assertEqual([item["next_study"] for item in progress], [4, 5])

    def test_study_logits_pool_all_views_before_local_head(self):
        model = SyntheticTail().eval()
        features = torch.tensor([[[[1.0, 2.0]]], [[[3.0, 4.0]]], [[[5.0, 6.0]]]])
        image_to_study = torch.tensor([0, 0, 1])
        actual = model.study_logits(features, image_to_study, 2)
        image_features = model.frozen_bn(model.stage(features.flatten(1)))
        expected = model.local_head(torch.stack((image_features[:2].mean(0), image_features[2])))
        self.assertTrue(torch.equal(actual, expected))

    @mock.patch("cxr.training_state.torch.cuda.is_available", return_value=False)
    def test_checkpoint_resume_matches_uninterrupted_epoch(self, _cuda):
        _, _, fetch = synthetic_data(8)
        initial = SyntheticTail().state_dict()
        config = StageTrainingConfig(microbatch_studies=2, accumulation_studies=4,
                                     checkpoint_every_windows=1)
        binding = {"initialization_sha256": "init", "prefix_cache_sha256": "cache"}
        baseline = SyntheticTail()
        baseline.load_state_dict(initial)
        baseline_optimizer = build_optimizer(baseline, config)
        torch.manual_seed(101)
        train_one_epoch(baseline, list(range(8)), fetch, config=config,
                        optimizer=baseline_optimizer, scaler=scaler_cpu(), binding=binding)

        interrupted = SyntheticTail()
        interrupted.load_state_dict(initial)
        interrupted_optimizer = build_optimizer(interrupted, config)
        original_fetch = fetch
        calls = 0

        def failing_fetch(indices):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("simulated interruption after first committed window")
            return original_fetch(indices)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "resume.pt"
            torch.manual_seed(101)
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                train_one_epoch(interrupted, list(range(8)), failing_fetch, config=config,
                                optimizer=interrupted_optimizer, scaler=scaler_cpu(),
                                checkpoint_path=checkpoint, binding=binding)
            resumed = SyntheticTail()
            resumed_optimizer = build_optimizer(resumed, config)
            report = train_one_epoch(resumed, list(range(8)), fetch, config=config,
                                     optimizer=resumed_optimizer, scaler=scaler_cpu(),
                                     checkpoint_path=checkpoint, resume=True, binding=binding)
            self.assertEqual(report["studies"], 8)
            inference_tail = SyntheticTail()
            loaded = restore_trained_tail(checkpoint, inference_tail, binding=binding)
            self.assertEqual(loaded["next_study"], 8)
            self.assertFalse(inference_tail.training)
            for key, expected in resumed.state_dict().items():
                self.assertTrue(torch.equal(expected, inference_tail.state_dict()[key]), key)
            for key, expected in baseline.state_dict().items():
                self.assertTrue(torch.equal(expected, resumed.state_dict()[key]), key)
            for expected, actual in zip(baseline_optimizer.state_dict()["state"].values(),
                                        resumed_optimizer.state_dict()["state"].values()):
                for key in expected:
                    if isinstance(expected[key], torch.Tensor):
                        self.assertTrue(torch.equal(expected[key], actual[key]), key)
                    else:
                        self.assertEqual(expected[key], actual[key])

    @mock.patch("cxr.training_state.torch.cuda.is_available", return_value=False)
    def test_resume_rejects_changed_binding_configuration_and_indices(self, _cuda):
        _, _, fetch = synthetic_data(4)
        config = StageTrainingConfig(microbatch_studies=2, accumulation_studies=2,
                                     checkpoint_every_windows=1)
        binding = {"initialization_sha256": "init"}
        model = SyntheticTail()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "resume.pt"
            train_one_epoch(model, list(range(4)), fetch, config=config, optimizer=build_optimizer(model, config),
                            scaler=scaler_cpu(), checkpoint_path=checkpoint, binding=binding)
            for changed_binding, changed_config, changed_indices in (
                ({"initialization_sha256": "other"}, config, list(range(4))),
                (binding, StageTrainingConfig(microbatch_studies=2, accumulation_studies=4,
                                              checkpoint_every_windows=1), list(range(4))),
                (binding, config, list(range(1, 5))),
            ):
                candidate = SyntheticTail()
                candidate_optimizer = build_optimizer(candidate, changed_config)
                with self.subTest(binding=changed_binding, config=changed_config, indices=changed_indices):
                    with self.assertRaisesRegex(ValueError, "binding|configuration|order"):
                        train_one_epoch(candidate, changed_indices, fetch, config=changed_config,
                                        optimizer=candidate_optimizer, scaler=scaler_cpu(),
                                        checkpoint_path=checkpoint, resume=True, binding=changed_binding)

    def test_only_stage_and_local_head_are_trainable_and_model_stays_eval(self):
        model = SyntheticTail()
        optimizer = build_optimizer(model)
        self.assertFalse(model.frozen_bn.training)
        self.assertFalse(any(parameter.requires_grad for parameter in model.frozen_bn.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.stage.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.local_head.parameters()))
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [1e-5, 1e-4])

    def test_batch_rejects_invalid_image_to_study_mapping(self):
        batch = StudyBatch(
            prefix_features=torch.ones(2, 2, 1, 1),
            image_to_study=torch.tensor([0, 2]),
            targets=torch.zeros(2, 10),
            study_indices=torch.tensor([4, 5]),
        )
        with self.assertRaisesRegex(ValueError, "invalid local study index"):
            batch.validate([4, 5])


if __name__ == "__main__":
    unittest.main()
