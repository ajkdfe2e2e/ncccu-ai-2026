"""Synthetic tests for optimization helpers; no image data is accessed."""

import io
import unittest

import torch

from cxr.optimization import ModelEMA, accumulation_window_images


class OptimizationTests(unittest.TestCase):
    def test_ema_module_is_independent_eval_and_frozen(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2)).train()
        ema = ModelEMA(model)
        self.assertIsNot(ema.module, model)
        self.assertFalse(ema.module.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in ema.module.parameters()))
        with torch.no_grad():
            next(model.parameters()).add_(1)
        self.assertFalse(torch.equal(next(model.parameters()), next(ema.module.parameters())))

    def test_ema_float_math_and_integer_buffer_copy(self):
        model = torch.nn.BatchNorm1d(2)
        ema = ModelEMA(model, decay=.75)
        with torch.no_grad():
            model.weight.fill_(5)
            model.num_batches_tracked.fill_(7)
        ema.update(model)
        self.assertTrue(torch.equal(ema.module.weight, torch.full((2,), 2.0)))
        self.assertEqual(ema.module.num_batches_tracked.item(), 7)
        self.assertEqual(ema.updates, 1)

    def test_weights_only_roundtrip_and_decay_compatibility(self):
        ema = ModelEMA(torch.nn.Linear(3, 2), decay=.9)
        ema.update(torch.nn.Linear(3, 2))
        stream = io.BytesIO()
        torch.save(ema.state_dict(), stream)
        stream.seek(0)
        restored_state = torch.load(stream, map_location='cpu', weights_only=True)
        restored = ModelEMA(torch.nn.Linear(3, 2), decay=.9)
        restored.load_state_dict(restored_state)
        self.assertEqual(restored.updates, 1)
        for key, value in ema.module.state_dict().items():
            self.assertTrue(torch.equal(value, restored.module.state_dict()[key]))
        with self.assertRaisesRegex(ValueError, 'decay'):
            ModelEMA(torch.nn.Linear(3, 2), decay=.8).load_state_dict(restored_state)

    def test_invalid_ema_state_rejected(self):
        ema = ModelEMA(torch.nn.Linear(2, 1))
        state = ema.state_dict()
        state['updates'] = -1
        with self.assertRaisesRegex(ValueError, 'update'):
            ema.load_state_dict(state)
        state = ema.state_dict()
        state['module'] = {}
        with self.assertRaisesRegex(ValueError, 'keys'):
            ema.load_state_dict(state)

    def test_accumulation_denominator_includes_partial_tail_window(self):
        self.assertEqual(accumulation_window_images(0, 4, 13, 2), 8)
        self.assertEqual(accumulation_window_images(2, 4, 13, 2), 5)
        self.assertEqual(accumulation_window_images(3, 4, 13, 2), 5)
        for args in [(-1, 4, 13, 2), (4, 4, 13, 2), (0, 0, 13, 2), (0, 4, 0, 2), (0, 4, 13, 0)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                accumulation_window_images(*args)

    def test_nondivisible_microbatches_match_large_batch_gradient(self):
        torch.manual_seed(4)
        examples = torch.randn(11, 3)
        targets = torch.randn(11, 1)
        micro_model = torch.nn.Linear(3, 1)
        large_model = torch.nn.Linear(3, 1)
        large_model.load_state_dict(micro_model.state_dict())
        batch_size, steps, dataset_size = 4, 2, len(examples)
        for batch_index, start in enumerate(range(0, dataset_size, batch_size)):
            end = min(start + batch_size, dataset_size)
            denominator = accumulation_window_images(batch_index, batch_size, dataset_size, steps)
            loss_sum = (micro_model(examples[start:end]) - targets[start:end]).square().sum()
            (loss_sum / denominator).backward()
            if batch_index % steps == steps - 1 or end == dataset_size:
                window_start = (batch_index // steps) * steps * batch_size
                window_x = examples[window_start:end]
                window_y = targets[window_start:end]
                (large_model(window_x) - window_y).square().mean().backward()
                for micro_param, large_param in zip(micro_model.parameters(), large_model.parameters()):
                    self.assertTrue(torch.allclose(micro_param.grad, large_param.grad, atol=1e-6))
                micro_model.zero_grad(set_to_none=True)
                large_model.zero_grad(set_to_none=True)


if __name__ == '__main__':
    unittest.main()
