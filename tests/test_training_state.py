"""Synthetic CPU-only continuation tests; no dataset or pretrained weights needed."""
import copy
import io
import random
import unittest
from unittest import mock

import numpy as np
import torch

from cxr.training_state import (RESUME_KEYS, OPTIMIZATION_DEFAULTS, capture_rng, restore_rng, cpu_snapshot,
                                capture_training_state, restore_training_state,
                                validate_init, validate_resume, safe_load, atomic_save)
from cxr.optimization import ModelEMA


def configuration():
    return dict(architecture='resnet18', class_ids=list(range(1, 11)), image_size=64,
                batch_size=2, validation_batch_size=1, workers=0, seed=2026, epochs=4,
                lr=.01, scheduler='cosine', min_lr=1e-6, smoke_studies=2,
                train_manifest_sha256='train-hash', valid_manifest_sha256='valid-hash',
                device='cpu', torch_version=str(torch.__version__))


def payload(model):
    return dict(architecture='resnet18', class_ids=list(range(1, 11)), image_size=64,
                model=cpu_snapshot(model.state_dict()), smoke=True, epoch=1,
                thresholds=[.5] * 10, validation_macro_auc=.6, inference_precision='fp32')


def components():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.2), torch.nn.Linear(4, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    # CPU scaler is genuinely enabled: tests cover nonempty scaler state without GPU training.
    scaler = torch.amp.GradScaler('cpu', init_scale=32., growth_interval=2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4, eta_min=1e-6)
    generator = torch.Generator().manual_seed(123)
    return model, optimizer, scaler, scheduler, generator


def epoch(parts):
    model, optimizer, scaler, scheduler, generator = parts
    loader = torch.utils.data.DataLoader(torch.arange(24, dtype=torch.float32).reshape(8, 3) / 24,
                                        batch_size=2, shuffle=True, generator=generator, num_workers=0)
    model.train()
    order = []
    for images in loader:
        order.extend(images[:, 0].tolist())
        optimizer.zero_grad(set_to_none=True)
        target = random.random() + float(np.random.random())
        loss = (model(images) - target).square().mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    scheduler.step()
    return order


def roundtrip(state):
    stream = io.BytesIO()
    torch.save(state, stream)
    stream.seek(0)
    return safe_load(stream)


class TrainingStateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        self.parts = components()
        epoch(self.parts)
        self.best = payload(self.parts[0])
        self.config = configuration()
        self.state = capture_training_state(*self.parts, self.config, 1, .6, self.best)

    def test_rng_roundtrip_weights_only(self):
        generator = self.parts[-1]
        # Exercise cached Gaussian state as well as MT19937 uniform state.
        np.random.normal()
        state = roundtrip(capture_rng(generator))
        expected = (random.random(), np.random.normal(), np.random.random(), torch.rand(4),
                    torch.randperm(20, generator=generator))
        restore_rng(state, generator)
        actual = (random.random(), np.random.normal(), np.random.random(), torch.rand(4),
                  torch.randperm(20, generator=generator))
        self.assertEqual(expected[:3], actual[:3])
        self.assertTrue(torch.equal(expected[3], actual[3]))
        self.assertTrue(torch.equal(expected[4], actual[4]))

    def test_split_equals_uninterrupted_optimizer_scheduler_scaler_and_loader(self):
        saved = roundtrip(self.state)
        expected_order = epoch(self.parts)
        expected_weights = cpu_snapshot(self.parts[0].state_dict())
        expected_optimizer = cpu_snapshot(self.parts[1].state_dict())
        expected_scaler = copy.deepcopy(self.parts[2].state_dict())
        expected_scheduler = copy.deepcopy(self.parts[3].state_dict())
        resumed = components()  # deliberately consumes fresh RNG before restore
        start, best, best_payload = restore_training_state(saved, *resumed, self.config)
        self.assertEqual((start, best), (1, .6))
        self.assertEqual(best_payload['epoch'], 1)
        self.assertEqual(epoch(resumed), expected_order)
        for key, expected in expected_weights.items():
            self.assertTrue(torch.equal(expected, resumed[0].state_dict()[key]), key)
        for index, state in expected_optimizer['state'].items():
            for key, expected in state.items():
                self.assertTrue(torch.equal(expected, resumed[1].state_dict()['state'][index][key]))
        self.assertEqual(expected_optimizer['param_groups'], resumed[1].state_dict()['param_groups'])
        self.assertEqual(expected_scaler, resumed[2].state_dict())
        self.assertEqual(expected_scheduler, resumed[3].state_dict())

    def test_every_compatibility_key_is_enforced(self):
        for key in RESUME_KEYS:
            with self.subTest(key=key):
                changed = dict(self.config, **{key: 'incompatible'})
                with self.assertRaisesRegex(ValueError, 'resume|workers'):
                    validate_resume(self.state, changed)

    def test_workers_nonzero_rejected_even_when_matching(self):
        changed = copy.deepcopy(self.state)
        changed['config']['workers'] = 1
        with self.assertRaisesRegex(ValueError, 'workers 0'):
            validate_resume(changed, dict(self.config, workers=1))

    def test_horizon_cannot_be_extended_or_already_complete(self):
        with self.assertRaisesRegex(ValueError, 'epochs'):
            validate_resume(self.state, dict(self.config, epochs=5))
        self.state['completed_epochs'] = 4
        with self.assertRaisesRegex(ValueError, 'remaining'):
            validate_resume(self.state, self.config)

    def test_segment_output_and_stop_limit_are_not_compatibility_keys(self):
        validate_resume(self.state, dict(self.config, output='new-output', stop_after_epochs=1))

    def test_inference_only_resume_rejected(self):
        with self.assertRaisesRegex(ValueError, 'init-checkpoint'):
            validate_resume(self.best, self.config)

    def test_init_metadata_and_smoke_validation(self):
        validate_init(self.best, 64, True)
        for key, value in [('architecture', 'other'), ('class_ids', list(range(10))), ('image_size', 224)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_init(dict(self.best, **{key: value}), 64, True)
        with self.assertRaisesRegex(ValueError, 'smoke'):
            validate_init(self.best, 64, False)
        validate_init(dict(self.best, smoke=False), 64, False)

    def test_best_payload_and_model_snapshot_do_not_alias_live_cpu_weights(self):
        first = cpu_snapshot(self.state['best_payload']['model'])
        epoch(self.parts)
        for key, expected in first.items():
            self.assertTrue(torch.equal(expected, self.state['best_payload']['model'][key]))
        with torch.no_grad():
            next(self.parts[0].parameters()).add_(100)
        self.assertTrue(torch.equal(first['0.weight'], self.best['model']['0.weight']))

    def test_missing_best_or_scheduler_rejected(self):
        for key, value in [('best_payload', None), ('scheduler', None)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_resume(dict(self.state, **{key: value}), self.config)

    def test_no_scheduler_roundtrip(self):
        parts = (*self.parts[:3], None, self.parts[-1])
        config = dict(self.config, scheduler='none')
        state = roundtrip(capture_training_state(*parts, config, 1, .6, self.best))
        restore_training_state(state, *parts, config)
        self.assertIsNone(state['scheduler'])

    def test_legacy_state_without_optimization_fields_keeps_original_defaults(self):
        legacy = copy.deepcopy(self.state)
        legacy.pop('ema', None)
        validate_resume(legacy, dict(self.config, **OPTIMIZATION_DEFAULTS))

    def test_optimization_changes_cannot_be_silently_resumed(self):
        for key, value in dict(accumulation_steps=2, weight_decay=.01, warmup_epochs=1,
                               ema_decay=.999, ema_device='cpu', gradient_checkpointing=True).items():
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate_resume(self.state, dict(self.config, **{key: value}))

    def test_convnext_init_and_resume_validate_selected_architecture(self):
        config = dict(self.config, architecture='convnext_tiny')
        state = copy.deepcopy(self.state)
        state['config'] = config
        state['best_payload']['architecture'] = 'convnext_tiny'
        validate_init(state['best_payload'], 64, True, 'convnext_tiny')
        validate_resume(state, config)
        with self.assertRaisesRegex(ValueError, 'architecture'):
            validate_init(state['best_payload'], 64, True)

    def test_ema_epoch_boundary_roundtrip_matches_uninterrupted_updates(self):
        ema = ModelEMA(self.parts[0], .9)
        ema.update(self.parts[0])
        config = dict(self.config, ema_decay=.9)
        saved = roundtrip(capture_training_state(*self.parts, config, 1, .6, self.best, ema=ema))
        epoch(self.parts)
        ema.update(self.parts[0])
        resumed = components()
        resumed_ema = ModelEMA(resumed[0], .9)
        restore_training_state(saved, *resumed, config, ema=resumed_ema)
        epoch(resumed)
        resumed_ema.update(resumed[0])
        self.assertEqual(ema.updates, resumed_ema.updates)
        for key, value in ema.module.state_dict().items():
            self.assertTrue(torch.equal(value, resumed_ema.module.state_dict()[key]), key)
        with self.assertRaisesRegex(ValueError, 'EMA'):
            restore_training_state(saved, *components(), config)
        with self.assertRaisesRegex(ValueError, 'EMA'):
            validate_resume(dict(saved, ema=None), config)

    def test_atomic_save_commits_only_after_success(self):
        # In-memory file descriptor mock avoids host temporary-directory ACL assumptions.
        stream = io.BytesIO()
        stream.close = lambda: None
        with mock.patch('cxr.training_state.tempfile.mkstemp', return_value=(123, 'temporary')), \
             mock.patch('cxr.training_state.os.fdopen', return_value=stream), \
             mock.patch('cxr.training_state.os.fsync'), \
             mock.patch.object(stream, 'fileno', return_value=123), \
             mock.patch('cxr.training_state.os.replace') as replace, \
             mock.patch('cxr.training_state.os.path.exists', return_value=False):
            atomic_save({'tensor': torch.ones(1)}, 'last-training.pt')
            replace.assert_called_once_with('temporary', __import__('pathlib').Path('last-training.pt'))
        with mock.patch('cxr.training_state.tempfile.mkstemp', return_value=(123, 'temporary')), \
             mock.patch('cxr.training_state.os.fdopen', return_value=stream), \
             mock.patch('cxr.training_state.torch.save', side_effect=RuntimeError('write failed')), \
             mock.patch('cxr.training_state.os.replace') as replace, \
             mock.patch('cxr.training_state.os.path.exists', return_value=True), \
             mock.patch('cxr.training_state.os.unlink') as unlink:
            with self.assertRaisesRegex(RuntimeError, 'write failed'):
                atomic_save({}, 'last-training.pt')
            replace.assert_not_called()
            unlink.assert_called_once_with('temporary')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_rng_restore_without_gpu_training(self):
        state = capture_rng(self.parts[-1])
        expected = [torch.rand(3, device=f'cuda:{index}').cpu() for index in range(torch.cuda.device_count())]
        restore_rng(state, self.parts[-1])
        actual = [torch.rand(3, device=f'cuda:{index}').cpu() for index in range(torch.cuda.device_count())]
        self.assertTrue(all(torch.equal(left, right) for left, right in zip(expected, actual)))


if __name__ == '__main__':
    unittest.main()
