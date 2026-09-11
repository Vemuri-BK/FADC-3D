import io
import unittest

import torch
from torch.nn import functional as F

from fadc_av import AdaptiveDilatedConv3D, DilationConfig, SharedKernelConv3D


class DilationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)

    def test_one_kernel_and_uniform_initialization(self):
        model = AdaptiveDilatedConv3D(2,3, DilationConfig(selection=None))
        self.assertEqual(list(dict(model.convolution.named_parameters())), ["weight"])
        x = torch.randn(2,2,7,9,5)
        result = model.forward_with_attention(x)
        reference = sum(F.conv3d(x, model.convolution.weight, padding=d, dilation=d)
                        for d in (1,2,3))/3
        torch.testing.assert_close(result.output, reference)
        torch.testing.assert_close(result.probabilities, torch.full((2,3,7,9,5), 1/3))
        torch.testing.assert_close(result.expected_dilation, torch.full((2,7,9,5), 2.0))

    def test_nonuniform_mixture_against_expanded_kernel_and_gradients(self):
        model = AdaptiveDilatedConv3D(2,3, DilationConfig(selection=None, bias=True)).double()
        with torch.no_grad():
            model.selector.head.weight.normal_(std=0.1)
        x = torch.randn(2,2,7,9,5, dtype=torch.float64, requires_grad=True)
        actual = model(x)
        probabilities = model.selector(x)
        reference = 0
        for i,d in enumerate((1,2,3)):
            # Independent reference: zero-insert kernel, then ordinary convolution.
            expanded = model.convolution.weight.new_zeros(3,2,2*d+1,2*d+1,2*d+1)
            expanded[:,:,::d,::d,::d] = model.convolution.weight
            branch = F.conv3d(x, expanded, model.convolution.bias, padding=d)
            reference = reference + probabilities[:,i:i+1] * branch
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
        params = (x, model.convolution.weight, model.convolution.bias, model.selector.head.weight)
        upstream = torch.randn_like(actual)
        ga = torch.autograd.grad(actual, params, upstream, retain_graph=True)
        gr = torch.autograd.grad(reference, params, upstream)
        for a,b in zip(ga,gr):
            torch.testing.assert_close(a,b, atol=1e-10, rtol=1e-10)

    def test_shapes_and_noncontiguous_input(self):
        model = AdaptiveDilatedConv3D(2,3)
        for shape in [(16,16,8), (8,8,4), (7,9,5), (1,1,1)]:
            x = torch.randn(1,2,*shape).transpose(-1,-2)
            result = model.forward_with_attention(x)
            self.assertEqual(result.output.shape, (1,3,*x.shape[-3:]))
            torch.testing.assert_close(result.probabilities.sum(1), torch.ones_like(result.expected_dilation))
            self.assertTrue(((result.expected_dilation>=1)&(result.expected_dilation<=3)).all().item())

    def test_all_enabled_heads_receive_gradients(self):
        model = AdaptiveDilatedConv3D(2,3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(2,2,16,16,16, requires_grad=True)
        for step in range(2):
            optimizer.zero_grad()
            model(x).square().mean().backward()
            heads = [model.convolution.weight, model.selector.head.weight,
                     *(g.weight for g in model.frequency_selection.high_gates)]
            if step:
                heads.append(model.selector.trunk[0].weight)
            for p in heads:
                self.assertTrue(torch.isfinite(p.grad).all().item())
                self.assertGreater(p.grad.abs().sum().item(), 0)
            optimizer.step()

    def test_runtime_temperature_survives_checkpoint(self):
        model = AdaptiveDilatedConv3D(2,3)
        with torch.no_grad():
            model.selector.head.bias.copy_(torch.tensor([0.,1.,-1.]))
        model.set_temperature(1.5)
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)
        restored = AdaptiveDilatedConv3D(2,3)
        restored.load_state_dict(torch.load(buf, weights_only=True), strict=True)
        self.assertEqual(restored.selector.temperature.item(), 1.5)
        x = torch.randn(1,2,8,8,4)
        torch.testing.assert_close(restored(x), model(x), atol=0, rtol=0)

    def test_semantic_configuration_mismatch_rejected(self):
        model = AdaptiveDilatedConv3D(2,3, DilationConfig(dilations=(1,2)))
        different = AdaptiveDilatedConv3D(2,3, DilationConfig(dilations=(1,3)))
        with self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
            different.load_state_dict(model.state_dict(), strict=True)

    def test_single_branch_equals_ordinary_convolution(self):
        model = AdaptiveDilatedConv3D(2,3, DilationConfig(dilations=(1,), selection=None))
        x = torch.randn(1,2,8,8,4)
        torch.testing.assert_close(model(x), F.conv3d(x,model.convolution.weight,padding=1))

    def test_cpu_amp_backward(self):
        model = AdaptiveDilatedConv3D(2,3)
        x = torch.randn(2,2,16,16,16,requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = model.forward_with_attention(x)
            loss = result.output.square().mean()
        loss.backward()
        self.assertEqual(result.output.dtype, torch.float32)
        for p in model.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all().item())
        self.assertTrue(torch.isfinite(x.grad).all().item())

    @unittest.skipUnless(torch.cuda.is_available(), "Kaggle CUDA verification required")
    def test_cuda_amp_optimizer_step(self):
        model = AdaptiveDilatedConv3D(2,3).cuda()
        opt = torch.optim.AdamW(model.parameters(),lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        x = torch.randn(2,2,16,16,16,device="cuda")
        with torch.autocast("cuda",dtype=torch.float16):
            output = model(x)
            loss = output.square().mean()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        self.assertTrue(torch.isfinite(output).all().item())
        for p in model.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all().item())
        scaler.step(opt)
        scaler.update()

    def test_invalid_configuration_and_temperature(self):
        for rates in [(), (0,), (1,1), (2,1), (1.5,), (True,)]:
            with self.assertRaises(ValueError):
                DilationConfig(dilations=rates)
        model = AdaptiveDilatedConv3D(2,3)
        for value in (0,-1,float("nan"),float("inf"),True):
            with self.assertRaises(ValueError):
                model.set_temperature(value)
        with self.assertRaises(ValueError):
            model(torch.randn(1,3,4,4,4))
        with self.assertRaises(TypeError):
            model(torch.ones(1,2,4,4,4,dtype=torch.int64))
        with self.assertRaises(ValueError):
            SharedKernelConv3D(0,3)


if __name__ == "__main__":
    unittest.main()
