import unittest

import torch

from fadc_av import FrequencyConfig, FrequencyDecomposition3D


class FrequencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_invalid_config(self):
        for cutoffs in [(), (1,), (2, 2), (4, 2), (2.5,), (True,)]:
            with self.subTest(cutoffs=cutoffs), self.assertRaises(ValueError):
                FrequencyConfig(cutoffs)

    def test_reconstruction_all_stages_and_odd_shapes(self):
        module = FrequencyDecomposition3D()
        for shape in [(128,128,64), (64,64,32), (32,32,16),
                      (16,16,8), (8,8,4), (7,9,5), (1,1,1)]:
            with self.subTest(shape=shape):
                x = torch.randn(1, 1, *shape)
                bands = module(x)
                torch.testing.assert_close(sum(bands.high) + bands.low, x)

    def test_dc_survives_nonidentity_high_band_gains(self):
        for shape in [(16,16,8), (8,8,4), (7,9,5)]:
            with self.subTest(shape=shape):
                x = torch.ones(1, 2, *shape)
                bands = FrequencyDecomposition3D()(x)
                result = sum(g * b for g, b in zip((0.2, 1.5, 1.9), bands.high)) + bands.low
                torch.testing.assert_close(result, x)
                torch.testing.assert_close(bands.low, x)

    def test_masks_keep_dc_are_nested_and_symmetric(self):
        for shape in [(128,128,64), (16,16,8), (8,8,4), (7,9,5), (1,1,1)]:
            previous = torch.ones(1, 1, *shape, dtype=torch.bool)
            for mask in FrequencyDecomposition3D().masks(shape, "cpu"):
                self.assertTrue(mask[0,0,0,0,0].item())
                self.assertFalse((mask & ~previous).any().item())
                reflected = mask
                for axis, n in zip((-3,-2,-1), shape):
                    reflected = reflected.index_select(axis, (-torch.arange(n)) % n)
                self.assertTrue(torch.equal(mask, reflected))
                previous = mask

    def test_known_sinusoids_and_cutoff(self):
        module = FrequencyDecomposition3D(FrequencyConfig((2,4)))
        # bin 1 -> low, bin 3 -> middle, bins 4 (cutoff) and 5 -> high.
        for freq, destination in [(1,2), (3,1), (4,0), (5,0)]:
            x = torch.cos(2 * torch.pi * freq * torch.arange(16, dtype=torch.float64) / 16)
            x = x.view(1,1,16,1,1).expand(1,1,16,8,8)
            bands = module(x)
            for i, band in enumerate((*bands.high, bands.low)):
                torch.testing.assert_close(band, x if i == destination else torch.zeros_like(x), atol=1e-12, rtol=1e-12)

    def test_duplicate_band_is_explicitly_inactive(self):
        bands = FrequencyDecomposition3D()(torch.randn(1,2,8,8,4))
        self.assertEqual(bands.active, (True, True, False))
        self.assertEqual(torch.count_nonzero(bands.high[2]).item(), 0)

    def test_masked_inverse_is_real(self):
        x = torch.randn(1,1,16,16,8, dtype=torch.float64)
        spectrum = torch.fft.fftn(x, dim=(-3,-2,-1))
        for mask in FrequencyDecomposition3D().masks((16,16,8), "cpu", torch.float64):
            inverse = torch.fft.ifftn(spectrum * mask, dim=(-3,-2,-1))
            self.assertLess(inverse.imag.abs().max().item(), 1e-12)

    def test_gradients_for_nonidentity_band_weighting(self):
        module = FrequencyDecomposition3D(FrequencyConfig((2,4)))
        x = torch.randn(1,1,3,4,5, dtype=torch.float64, requires_grad=True)
        def weighted(inp):
            bands = module(inp)
            return 0.3 * bands.high[0] + 1.7 * bands.high[1] + bands.low
        self.assertTrue(torch.autograd.gradcheck(weighted, (x,)))

    def test_precision_policy(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            x = torch.randn(1,1,8,8,4).to(dtype).requires_grad_()
            bands = FrequencyDecomposition3D()(x)
            self.assertEqual(bands.low.dtype, torch.float64 if dtype == torch.float64 else torch.float32)
            reconstructed = sum(bands.high) + bands.low
            reconstructed.square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all().item())

    def test_invalid_inputs(self):
        module = FrequencyDecomposition3D()
        for x in (torch.ones(2,3), torch.empty(1,1,0,3,3)):
            with self.assertRaises(ValueError):
                module(x)
        with self.assertRaises(TypeError):
            module(torch.ones(1,1,4,4,4, dtype=torch.int64))


if __name__ == "__main__":
    unittest.main()
