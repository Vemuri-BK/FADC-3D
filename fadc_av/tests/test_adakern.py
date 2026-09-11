import io
import unittest

import torch
from torch.nn import functional as F

from fadc_av import (AdaKernConfig, AdaKern3D, KernelGains, SharedKernelConv3D,
                     AdaptiveDilatedConv3D, DilationConfig)


class AdaKernTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)

    def test_identity_kernel_and_static_block_equivalence(self):
        adapter = AdaKern3D(2,3).double()
        x = torch.randn(2,2,8,8,4,dtype=torch.float64)
        weight = torch.randn(3,2,3,3,3,dtype=torch.float64)
        torch.testing.assert_close(adapter(x,weight), weight[None].expand(2,-1,-1,-1,-1,-1))
        adaptive = AdaptiveDilatedConv3D(2,3,DilationConfig(adaptive_kernel=AdaKernConfig())).double()
        static = AdaptiveDilatedConv3D(2,3).double()
        # Match shared components; architecture identities intentionally differ.
        for name in ("convolution", "frequency_selection", "selector"):
            getattr(static,name).load_state_dict(getattr(adaptive,name).state_dict())
        torch.testing.assert_close(adaptive(x),static(x),atol=1e-12,rtol=1e-12)

    def test_nonidentity_kernel_scalar_reference(self):
        adapter = AdaKern3D(2,3).double()
        weight = torch.randn(3,2,3,3,3,dtype=torch.float64)
        gains = KernelGains(*(torch.rand(2,c,dtype=torch.float64)*2 for c in (2,3,2,3)))
        actual = adapter.apply_gains(weight,gains)
        reference = torch.empty_like(actual)
        for b in range(2):
            for o in range(3):
                for i in range(2):
                    kernel = weight[o,i]
                    reference[b,o,i] = (kernel.mean()*gains.low_input[b,i]*gains.low_output[b,o]
                        + (kernel-kernel.mean())*gains.high_input[b,i]*gains.high_output[b,o])
        torch.testing.assert_close(actual,reference,atol=1e-12,rtol=1e-12)
        self.assertFalse(torch.equal(actual[0],actual[1]))

    def test_packed_convolution_outputs_and_gradients(self):
        conv = SharedKernelConv3D(2,3,bias=True).double()
        x = torch.randn(2,2,7,9,5,dtype=torch.float64).transpose(-1,-2).requires_grad_()
        weights = torch.randn(2,3,2,3,3,3,dtype=torch.float64,requires_grad=True)
        for dilation in (1,2,3):
            actual = conv(x,dilation,sample_weight=weights)
            reference = torch.cat([F.conv3d(x[b:b+1],weights[b],conv.bias,
                                          padding=dilation,dilation=dilation) for b in range(2)])
            torch.testing.assert_close(actual,reference,atol=1e-12,rtol=1e-12)
            upstream = torch.randn_like(actual)
            params = (x,weights,conv.bias)
            ga = torch.autograd.grad(actual,params,upstream,retain_graph=True)
            gr = torch.autograd.grad(reference,params,upstream,retain_graph=True)
            for a,b in zip(ga,gr):
                torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)

    def test_same_adaptive_kernel_reused_across_branches(self):
        model = AdaptiveDilatedConv3D(2,3,DilationConfig(adaptive_kernel=AdaKernConfig()))
        with torch.no_grad():
            for head in model.adakern.heads:
                head.weight.normal_(std=0.1)
        seen, adaptations = [], []
        def capture(module,args,kwargs):
            seen.append(kwargs["sample_weight"])
        handle = model.convolution.register_forward_pre_hook(capture,with_kwargs=True)
        handle2 = model.adakern.register_forward_hook(lambda *args: adaptations.append(True))
        try:
            model(torch.randn(2,2,8,8,4))
        finally:
            handle.remove()
            handle2.remove()
        self.assertEqual(len(adaptations),1)
        self.assertEqual(len(seen),3)
        self.assertTrue(all(w is seen[0] for w in seen))
        self.assertEqual(tuple(seen[0].shape),(2,3,2,3,3,3))

    def test_gradients_reach_heads_then_descriptor(self):
        model = AdaptiveDilatedConv3D(2,3,DilationConfig(adaptive_kernel=AdaKernConfig()))
        opt = torch.optim.SGD(model.parameters(),lr=0.01)
        for step in range(2):
            opt.zero_grad()
            model(torch.randn(2,2,16,16,16)).square().mean().backward()
            params = [model.convolution.weight, *(h.weight for h in model.adakern.heads)]
            if step:
                params.append(model.adakern.trunk[0].weight)
            for param in params:
                self.assertTrue(torch.isfinite(param.grad).all().item())
                self.assertGreater(param.grad.abs().sum().item(),0)
            opt.step()

    def test_nonidentity_checkpoint_and_temperature(self):
        config = DilationConfig(adaptive_kernel=AdaKernConfig())
        model = AdaptiveDilatedConv3D(2,3,config)
        with torch.no_grad():
            for head in model.adakern.heads:
                head.weight.normal_(std=0.1)
            model.selector.head.bias.copy_(torch.tensor([0.,1.,-1.]))
        model.set_temperature(1.7)
        buf = io.BytesIO()
        torch.save(model.state_dict(),buf)
        buf.seek(0)
        restored = AdaptiveDilatedConv3D(2,3,config)
        restored.load_state_dict(torch.load(buf,weights_only=True),strict=True)
        x = torch.randn(2,2,8,8,4)
        torch.testing.assert_close(model(x),restored(x),atol=0,rtol=0)
        with self.assertRaises(RuntimeError):
            AdaptiveDilatedConv3D(2,3).load_state_dict(model.state_dict())

    def test_cpu_amp_backward(self):
        model = AdaptiveDilatedConv3D(2,3,DilationConfig(adaptive_kernel=AdaKernConfig()))
        with torch.autocast("cpu",dtype=torch.bfloat16):
            output = model(torch.randn(2,2,16,16,16))
            loss = output.square().mean()
        loss.backward()
        self.assertEqual(output.dtype,torch.float32)
        self.assertTrue(torch.isfinite(output).all().item())
        for p in model.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all().item())

    @unittest.skipUnless(torch.cuda.is_available(), "Kaggle CUDA verification required")
    def test_cuda_amp_optimizer_step(self):
        model = AdaptiveDilatedConv3D(2,3,DilationConfig(adaptive_kernel=AdaKernConfig())).cuda()
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        with torch.autocast("cuda",dtype=torch.float16):
            output = model(torch.randn(2,2,16,16,16,device="cuda"))
            loss = output.square().mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        self.assertTrue(torch.isfinite(output).all().item())
        for p in model.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all().item())
        scaler.step(optimizer)
        scaler.update()

    def test_invalid_configuration_and_shapes(self):
        with self.assertRaises(ValueError):
            AdaKernConfig(hidden_channels=0)
        with self.assertRaises(TypeError):
            DilationConfig(adaptive_kernel=True)
        with self.assertRaises(ValueError):
            AdaKern3D(0,3)
        adapter = AdaKern3D(2,3)
        with self.assertRaises(ValueError):
            adapter(torch.randn(1,3,4,4,4),torch.randn(3,2,3,3,3))
        with self.assertRaises(ValueError):
            adapter(torch.randn(1,2,4,4,4),torch.randn(3,2,1,1,1))
        with self.assertRaises(ValueError):
            SharedKernelConv3D(2,3)(torch.randn(2,2,4,4,4),1,
                                     sample_weight=torch.randn(1,3,2,3,3,3))


if __name__ == "__main__":
    unittest.main()
