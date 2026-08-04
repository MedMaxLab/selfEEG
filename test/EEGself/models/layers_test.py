import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from EEGself._testtools import get_device, make_grid

from selfeeg import models

N, CHAN, SAMPLES = 2, 8, 2048


class TestModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = get_device(
            probe=lambda dev: models.ConstrainedConv1d(8, 4, 16).to(dev)(
                torch.randn(2, 8, 2048, device=dev)
            )
        )
        cls.x = torch.randn(N, CHAN, SAMPLES)
        cls.xl = torch.randn(N, 1, 16, SAMPLES)
        cls.xd = torch.randn(N, 128)

    def setUp(self):
        torch.manual_seed(1234)

    def _devices(self):
        return ["cpu"] + ([self.device.type] if self.device.type != "cpu" else [])

    def _assert_constraints(self, weight, out, max_norm, min_norm, norm_axes):
        self.assertEqual(torch.isnan(out).sum().item(), 0)
        if max_norm is not None:
            norms = torch.sqrt(torch.sum(torch.square(weight), axis=norm_axes))
            self.assertEqual(torch.sum(norms > (max_norm + 1e-3)).item(), 0)
            if min_norm is not None:
                self.assertEqual(torch.sum(norms < (min_norm - 1e-3)).item(), 0)

    def test_ConstrainedConv1d(self):
        grid = make_grid(
            {
                "in_channels": [8],
                "out_channels": [4, 16],
                "kernel_size": [16],
                "stride": [1, 2, 3],
                "dilation": [1, 2],
                "bias": [True, False],
                "max_norm": [None, 1, 2],
                "min_norm": [None, 1],
                "padding": ["valid", "causal"],
            },
            max_comb=40,
        )
        for dev in self._devices():
            x = self.x.to(dev)
            for args in grid:
                with self.subTest(device=dev, **{k: args[k] for k in ("stride", "max_norm", "padding")}):
                    model = models.ConstrainedConv1d(**args).to(dev)
                    model.weight = torch.nn.Parameter(model.weight * 10)
                    out = model(x)
                    self._assert_constraints(model.weight, out, args["max_norm"], args["min_norm"], [1, 2])

    def test_ConstrainedConv2d(self):
        grid = make_grid(
            {
                "in_channels": [1],
                "out_channels": [5, 16],
                "kernel_size": [(1, 64), (5, 1), (5, 64)],
                "stride": [1, 2, 3],
                "dilation": [1, 2],
                "bias": [True, False],
                "max_norm": [None, 1, 2, 3],
                "min_norm": [None, 1],
                "padding": ["valid"],
            },
            max_comb=40,
        )
        for dev in self._devices():
            xl = self.xl.to(dev)
            for args in grid:
                with self.subTest(device=dev, kernel=args["kernel_size"], max_norm=args["max_norm"]):
                    model = models.ConstrainedConv2d(**args).to(dev)
                    model.weight = torch.nn.Parameter(model.weight * 10)
                    out = model(xl)
                    self._assert_constraints(
                        model.weight, out, args["max_norm"], args["min_norm"], [1, 2, 3]
                    )

    def test_ConstrainedDense(self):
        grid = make_grid(
            {
                "in_features": [128],
                "out_features": [32],
                "bias": [True, False],
                "max_norm": [None, 1, 3],
                "min_norm": [None, 1],
            }
        )
        for dev in self._devices():
            xd = self.xd.to(dev)
            for args in grid:
                with self.subTest(device=dev, max_norm=args["max_norm"], min_norm=args["min_norm"]):
                    model = models.ConstrainedDense(**args).to(dev)
                    model.weight = torch.nn.Parameter(model.weight * 10)
                    out = model(xd)
                    self._assert_constraints(model.weight, out, args["max_norm"], args["min_norm"], 1)
                    self.assertEqual(out.shape[1], 32)

    def test_DepthwiseConv2d(self):
        grid = make_grid(
            {
                "in_channels": [1],
                "depth_multiplier": [2, 3, 4],
                "kernel_size": [(1, 64), (5, 1), (5, 64)],
                "stride": [1, 2, 3],
                "dilation": [1, 2],
                "bias": [True, False],
                "max_norm": [None, 1, 3],
                "min_norm": [None, 1],
                "padding": ["valid"],
            },
            max_comb=40,
        )
        for dev in self._devices():
            xl = self.xl.to(dev)
            for args in grid:
                with self.subTest(device=dev, depth=args["depth_multiplier"], max_norm=args["max_norm"]):
                    model = models.DepthwiseConv2d(**args).to(dev)
                    model.weight = torch.nn.Parameter(model.weight * 10)
                    out = model(xl)
                    self._assert_constraints(model.weight, out, args["max_norm"], args["min_norm"], [1, 2, 3])
                    self.assertEqual(out.shape[1], args["depth_multiplier"])

    def test_SeparableConv2d(self):
        grid = make_grid(
            {
                "in_channels": [1],
                "out_channels": [5, 16],
                "depth_multiplier": [1, 3],
                "kernel_size": [(1, 64), (5, 1), (5, 64)],
                "stride": [1, 2, 3],
                "dilation": [1, 2],
                "bias": [True, False],
                "depth_max_norm": [None, 1, 2],
                "depth_min_norm": [None, 1],
                "point_max_norm": [None, 1, 2],
                "point_min_norm": [None, 1],
                "padding": ["valid"],
            },
            max_comb=40,
        )
        for dev in self._devices():
            xl = self.xl.to(dev)
            for args in grid:
                with self.subTest(device=dev, out_channels=args["out_channels"], kernel=args["kernel_size"]):
                    model = models.SeparableConv2d(**args).to(dev)
                    out = model(xl)
                    self.assertEqual(torch.isnan(out).sum().item(), 0)
                    self.assertEqual(out.shape[1], args["out_channels"])


if __name__ == "__main__":
    unittest.main()
