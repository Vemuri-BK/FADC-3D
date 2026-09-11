import unittest
import torch
from fadc_av.model import FADCAVUNet3D, build_model
from fadc_av.dilation import AdaptiveDilatedConv3D
from models.unet_3d import UNet3D


class ModelTests(unittest.TestCase):
    def test_all_encoders_placement_backward_and_reload(self):
        torch.set_num_threads(2)
        cfg = {"variant": "fadc_all_encoders", "base_filters": 2}
        model = build_model(cfg)
        names = [n for n, m in model.named_modules() if isinstance(m, AdaptiveDilatedConv3D)]
        self.assertEqual(names, [f"enc{i}.conv.block.{j}" for i in range(1, 5) for j in (0, 3)])
        x = torch.randn(2, 2, 32, 32, 16)
        y = model(x)
        self.assertEqual(y.shape, x.shape)
        y.square().mean().backward()
        for name, module in model.named_modules():
            if isinstance(module, AdaptiveDilatedConv3D):
                grads = [p.grad for p in module.parameters() if p.grad is not None]
                self.assertTrue(grads, name)
                self.assertTrue(all(torch.isfinite(g).all() for g in grads), name)
        model.eval()
        restored = build_model(cfg).eval()
        restored.load_state_dict(model.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(model(x), restored(x))

    def test_only_enc3_second_convolution_replaced(self):
        model = FADCAVUNet3D(base_filters=2)
        base = UNet3D(in_channels=2, base_filters=2)
        adaptive = [(name, m) for name, m in model.named_modules() if isinstance(m, AdaptiveDilatedConv3D)]
        self.assertEqual([n for n, _ in adaptive], ["enc3.conv.block.3"])
        base_modules = dict(base.named_modules())
        for name, module in model.named_modules():
            if not name.startswith("enc3.conv.block.3"):
                self.assertIs(type(module), type(base_modules[name])) if name else None
        self.assertIsInstance(build_model({"variant": "baseline", "base_filters": 2}), UNet3D)

    def test_volume_backward_and_reload(self):
        torch.set_num_threads(2)
        model = FADCAVUNet3D(base_filters=2)
        x = torch.randn(2, 2, 32, 32, 16)
        y = model(x)
        self.assertEqual(y.shape, x.shape)
        y.square().mean().backward()
        grads = [p.grad for p in model.enc3.conv.block[3].parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        model.eval()
        restored = FADCAVUNet3D(base_filters=2).eval()
        restored.load_state_dict(model.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(model(x), restored(x))


if __name__ == "__main__":
    unittest.main()
