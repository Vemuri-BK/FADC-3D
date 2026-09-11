import io
import math
import unittest

import torch

from fadc_av import FrequencyConfig, SelectionConfig, FrequencySelection3D


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)

    def test_identity_init_and_parameter_stability(self):
        model = FrequencySelection3D(4, SelectionConfig(spatial_groups=2))
        original = tuple((n, id(p)) for n, p in model.named_parameters())
        for shape in [(16,16,16), (16,16,8), (8,8,4), (7,9,5), (1,1,1)]:
            with self.subTest(shape=shape):
                x = torch.randn(2,4,*shape)
                result = model.forward_with_gates(x)
                torch.testing.assert_close(result.output, x)
                self.assertIsNone(result.low_gain)
                for gain in result.high_gains:
                    if gain is not None:
                        torch.testing.assert_close(gain, torch.ones_like(gain))
                self.assertEqual(original, tuple((n,id(p)) for n,p in model.named_parameters()))

    def test_nonidentity_gates_preserve_constant(self):
        model = FrequencySelection3D(2).double()
        with torch.no_grad():
            for gate in model.high_gates:
                gate.weight.normal_()
                gate.bias.fill_(1.1)
        for shape in [(16,16,8), (8,8,4), (7,9,5)]:
            x = torch.ones(1,2,*shape, dtype=torch.float64)
            torch.testing.assert_close(model(x), x, atol=1e-12, rtol=1e-12)

    def test_group_gains_against_known_high_frequency_signal(self):
        config = SelectionConfig(FrequencyConfig((2,)), spatial_groups=2)
        model = FrequencySelection3D(4, config).double()
        with torch.no_grad():
            model.high_gates[0].bias.copy_(torch.tensor([math.log(3), -math.log(3)]))
        wave = torch.cos(2*torch.pi*5*torch.arange(16,dtype=torch.float64)/16)
        wave = wave.view(1,1,16,1,1).expand(1,4,16,8,8)
        expected = 1 + wave * torch.tensor([1.5,1.5,0.5,0.5]).view(1,4,1,1,1)
        torch.testing.assert_close(model(1+wave), expected, atol=1e-7, rtol=1e-7)

    def test_gates_can_vary_spatially(self):
        model = FrequencySelection3D(1, SelectionConfig(FrequencyConfig((2,)), kernel_size=1))
        with torch.no_grad():
            model.high_gates[0].weight.fill_(1)
        x = torch.randn(1,1,8,8,8)
        result = model.forward_with_gates(x)
        torch.testing.assert_close(result.high_gains[0], 2*torch.sigmoid(x))
        self.assertGreater(result.high_gains[0].std().item(), 0)

    def test_optional_low_frequency_gate(self):
        model = FrequencySelection3D(2, SelectionConfig(low_frequency_attention=True))
        x = torch.ones(1,2,8,8,4)
        torch.testing.assert_close(model(x), x)
        with torch.no_grad():
            model.low_gate.bias.fill_(math.log(3))
        torch.testing.assert_close(model(x), 1.5*x)

    def test_inactive_head_is_not_called_or_differentiated(self):
        model = FrequencySelection3D(2)
        calls = []
        hook = model.high_gates[2].register_forward_hook(lambda *args: calls.append(True))
        try:
            result = model.forward_with_gates(torch.randn(1,2,8,8,4))
            result.output.square().mean().backward()
        finally:
            hook.remove()
        self.assertEqual(calls, [])
        self.assertIsNone(result.high_gains[2])
        self.assertIsNone(model.high_gates[2].weight.grad)
        for gate in model.high_gates[:2]:
            self.assertGreater(gate.weight.grad.abs().sum().item(), 0)

    def test_amp_forward_backward(self):
        model = FrequencySelection3D(2)
        x = torch.randn(2,2,16,16,16, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = model(x)
            loss = output.square().mean()
        self.assertEqual(output.dtype, torch.float32)
        loss.backward()
        for tensor in [output, x.grad, *(gate.weight.grad for gate in model.high_gates)]:
            self.assertTrue(torch.isfinite(tensor).all().item())
        for gate in model.high_gates:
            self.assertGreater(gate.weight.grad.abs().sum().item(), 0)

    def test_nonidentity_serialization_roundtrip(self):
        config = SelectionConfig(low_frequency_attention=True)
        model = FrequencySelection3D(2, config)
        with torch.no_grad():
            for param in model.parameters():
                param.uniform_(-0.2, 0.2)
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)
        restored = FrequencySelection3D(2, config)
        restored.load_state_dict(torch.load(buf, weights_only=True), strict=True)
        x = torch.randn(1,2,16,16,16)
        torch.testing.assert_close(restored(x), model(x), atol=0, rtol=0)

    def test_validation(self):
        for kwargs in [dict(kernel_size=2), dict(spatial_groups=0), dict(kernel_size=True)]:
            with self.assertRaises(ValueError):
                SelectionConfig(**kwargs)
        with self.assertRaises(ValueError):
            FrequencySelection3D(3, SelectionConfig(spatial_groups=2))
        with self.assertRaises(ValueError):
            FrequencySelection3D(0)
        with self.assertRaises(ValueError):
            FrequencySelection3D(2)(torch.ones(1,3,4,4,4))


if __name__ == "__main__":
    unittest.main()
