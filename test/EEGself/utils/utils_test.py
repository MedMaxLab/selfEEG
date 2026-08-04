import os
import random
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from EEGself._testtools import get_device, make_grid

from selfeeg import models, utils


def _sine_batch(scale=1.0):
    return torch.zeros(16, 32, 1024) + torch.sin(torch.linspace(0, 8 * torch.pi, 1024)) * scale


class TestUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = get_device(
            probe=lambda dev: utils.scale_range_soft_clip(_sine_batch(500).to(dev), "scale", "uV")
        )

    def _inputs(self, x):
        """Return the input variants (cpu tensor, numpy, gpu tensor) to test."""
        variants = [x, x.numpy()]
        if self.device.type != "cpu":
            variants.append(x.clone().to(self.device))
        return variants

    def _assert_scaled(self, x, x_scaled, asintote):
        if isinstance(x_scaled, torch.Tensor):
            self.assertEqual(torch.isnan(x_scaled).sum().item(), 0)
        else:
            self.assertEqual(np.isnan(x_scaled).sum(), 0)
        self.assertFalse(x.max() <= asintote and x.min() >= -asintote)
        self.assertTrue(x_scaled.max() <= asintote and x_scaled.min() >= -asintote)

    def test_scale_range_soft_clip(self):
        random.seed(1234)
        x = _sine_batch(500)
        grid = make_grid(
            {
                "x": self._inputs(x),
                "Range": [200, 300],
                "asintote": [1.0, 2.3, 3.5],
                "scale": ["uV"],
                "exact": [True, False],
            },
            max_comb=24,
        )
        for args in grid:
            with self.subTest(asintote=args["asintote"], exact=args["exact"]):
                x_scaled = utils.scale_range_soft_clip(**args)
                self._assert_scaled(x, x_scaled, args["asintote"])

    def test_RangeScaler(self):
        random.seed(1234)
        x = _sine_batch(500)
        grid = make_grid(
            {
                "x": self._inputs(x),
                "Range": [200, 300],
                "asintote": [1.0, 2.3, 3.5],
                "scale": ["uV"],
                "exact": [True, False],
            },
            max_comb=24,
        )
        for args in grid:
            with self.subTest(asintote=args["asintote"], exact=args["exact"]):
                scaler = utils.RangeScaler(
                    args["Range"], args["asintote"], args["scale"], args["exact"]
                )
                x_scaled = scaler(args["x"])
                self._assert_scaled(x, x_scaled, args["asintote"])

    def test_torch_zscore(self):
        x = torch.ones(2, 16, 16)
        x[0] += torch.randn(16, 16)
        x = x.to(self.device)

        xz = utils.torch_zscore(x, -2, 1)
        self.assertEqual(torch.isnan(xz).sum().item(), 16 * 16)
        self.assertEqual(xz.device.type, self.device.type)

        x[1] += torch.randn(16, 16).to(self.device)
        xz = utils.torch_zscore(x, -2, 1)
        self.assertEqual(torch.isnan(xz).sum().item(), 0)
        self.assertEqual((xz.mean(-2).abs() > 1e-6).sum().item(), 0)
        self.assertEqual(((xz.std(-2, correction=0).abs() - 1) > 1e-6).sum().item(), 0)

    def test_get_subarray_closest_sum(self):
        random.seed(1235)
        arr = list(range(1, 100))
        for target in (3251, 2497):
            with self.subTest(target=target):
                _, best_sub_arr = utils.get_subarray_closest_sum(
                    arr, target, tolerance=1e-4, perseverance=10000
                )
                self.assertEqual(sum(best_sub_arr), target)

    def test_check_models(self):
        model1 = models.EEGNet(4, 8, 512)
        model2 = models.EEGNet(4, 8, 512)
        self.assertFalse(utils.check_models(model1, model2))
        model2.load_state_dict(model1.state_dict())
        self.assertTrue(utils.check_models(model1, model2))

    def test_count_parameters(self):
        mdl = models.ShallowNet(4, 8, 1024)
        for n, param in enumerate(mdl.parameters()):  # freeze biases
            param.requires_grad = n not in [1, 3, 5, 7]
        _, trainable = utils.count_parameters(mdl, True, True, True)
        self.assertEqual(trainable, 23760)


if __name__ == "__main__":
    unittest.main()
