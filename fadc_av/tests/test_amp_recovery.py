import unittest
import torch
from fadc_av.run import step


class AMPRecoveryTests(unittest.TestCase):
    def exercise(self, device):
        model = torch.nn.Conv3d(2, 2, 1).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler(device.type, init_scale=128)
        batch = {'image': torch.randn(2, 2, 4, 4, 4, device=device),
                 'label': torch.zeros(2, 1, 4, 4, 4, device=device)}
        before = model.weight.detach().clone()
        hook = model.weight.register_hook(lambda g: torch.full_like(g, float('inf')))
        try:
            step(model, batch, optimizer, scaler, device)
        finally:
            hook.remove()
        torch.testing.assert_close(before, model.weight, rtol=0, atol=0)
        self.assertFalse(optimizer.state)
        self.assertEqual(scaler.get_scale(), 64)
        self.assertEqual(optimizer.amp_skips, 1)
        step(model, batch, optimizer, scaler, device)
        self.assertFalse(torch.equal(before, model.weight))
        self.assertEqual(optimizer.consecutive_amp_skips, 0)

    def test_scaler_skips_overflow_then_recovers_cpu(self):
        self.exercise(torch.device('cpu'))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_scaler_skips_overflow_then_recovers_cuda(self):
        self.exercise(torch.device('cuda'))

    def test_unscaled_nonfinite_gradients_fail(self):
        model = torch.nn.Conv3d(2, 2, 1)
        model.weight.register_hook(lambda g: torch.full_like(g, float('nan')))
        optimizer = torch.optim.AdamW(model.parameters())
        with self.assertRaisesRegex(RuntimeError, 'without AMP'):
            step(model, {'image': torch.randn(2, 2, 4, 4, 4), 'label': torch.zeros(2, 1, 4, 4, 4)},
                 optimizer, torch.amp.GradScaler('cpu', enabled=False), torch.device('cpu'))
