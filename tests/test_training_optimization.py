"""Check schedule recovery and activation recomputation on synthetic inputs."""
import unittest

import torch

from cxr.models import create_model
from train import create_scheduler, training_logits


class TrainingOptimizationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_warmup_cosine_recovery_uses_original_horizon(self):
        def parts():
            optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=.01)
            return optimizer, create_scheduler(optimizer, 6, 'cosine', .001, 2)

        optimizer, scheduler = parts()
        rates = []
        for _ in range(3):
            rates.append(optimizer.param_groups[0]['lr'])
            optimizer.step()
            scheduler.step()
        self.assertEqual(rates, [.005, .01, .01])
        resumed_optimizer, resumed_scheduler = parts()
        resumed_optimizer.load_state_dict(optimizer.state_dict())
        resumed_scheduler.load_state_dict(scheduler.state_dict())
        for _ in range(3):
            self.assertEqual(optimizer.param_groups[0]['lr'], resumed_optimizer.param_groups[0]['lr'])
            optimizer.step()
            scheduler.step()
            resumed_optimizer.step()
            resumed_scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], .001)

    def test_checkpointing_preserves_convnext_logits_and_stochastic_gradients(self):
        torch.manual_seed(7)
        model = create_model('convnext_tiny').train()
        images = torch.randn(1, 3, 64, 64)
        torch.manual_seed(123)
        expected_logits = training_logits(model, images)
        expected_logits.square().mean().backward()
        expected_gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        torch.manual_seed(123)
        actual_logits = training_logits(model, images, gradient_checkpointing=True)
        actual_logits.square().mean().backward()
        self.assertTrue(torch.equal(expected_logits, actual_logits))
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.allclose(parameter.grad, expected_gradients[name], atol=1e-6, rtol=1e-5), name)


if __name__ == '__main__':
    unittest.main()
