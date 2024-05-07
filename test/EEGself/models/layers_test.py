import itertools
import os
import sys
import unittest

import numpy as np
import torch

from selfeeg import models


class TestModels(unittest.TestCase):

    def makeGrid(self, pars_dict):
        keys = pars_dict.keys()
        combinations = itertools.product(*pars_dict.values())
        ds = [dict(zip(keys, cc)) for cc in combinations]
        return ds

    @classmethod
    def setUpClass(cls):
        cls.device = (
            torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        )
        if cls.device.type == "cpu":
            cls.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if cls.device.type == "mps":
            try:
                xx = torch.randn(2,2).to(device=cls.device)
            except Exception:
                cls.device = torch.device("cpu")
        
        print("\n----------------------------")
        print("TESTING MODELS.LAYERS MODULE")
        if cls.device.type != "cpu":
            print("Found gpu device: testing module with both cpu and gpu")
        else:
            print("Didn't found cuda device: testing module with only cpu")
        print("----------------------------")
        cls.N = 2
        cls.Chan = 8
        cls.Samples = 2048
        cls.x = torch.randn(cls.N, cls.Chan, cls.Samples)
        cls.xl = torch.randn(cls.N, 1, 16, cls.Samples)
        cls.xd = torch.randn(cls.N, 128)
        if cls.device.type != "cpu":
            cls.x2 = torch.randn(cls.N, cls.Chan, cls.Samples).to(device=cls.device)
            cls.xl2 = torch.randn(cls.N, 1, 16, cls.Samples).to(device=cls.device)
            cls.xd2 = torch.randn(cls.N, 128).to(device=cls.device)

    def setUp(self):
        torch.manual_seed(1234)

    def test_ConstrainedConv1d(self):
        print("Testing conv1d with norm constraint...", end="", flush=True)
        Conv_args = {
            "in_channels": [8],
            "out_channels": [4, 16],
            "kernel_size": [16],
            "stride": [1, 2, 3],
            "dilation": [1, 2],
            "bias": [True, False],
            "max_norm": [None, 1, 2],
            "min_norm": [None, 1],
            "padding": ["valid", "causal"],
        }
        Conv_args = self.makeGrid(Conv_args)
        for i in Conv_args:
            model = models.ConstrainedConv1d(**i)
            model.weight = torch.nn.Parameter(model.weight * 10)
            out = model(self.x)
            if i["max_norm"] is not None:
                norms = torch.sqrt(torch.sum(torch.square(model.weight), axis=[1, 2]))
                self.assertTrue(torch.sum(norms > (i["max_norm"] + 1e-3)).item() == 0)
                if i["min_norm"] is not None:
                    self.assertTrue(torch.sum(norms < (i["min_norm"] - 1e-3)).item() == 0)
            self.assertEqual(torch.isnan(out).sum(), 0)

        if self.device.type != "cpu":
            for i in Conv_args:
                model = models.ConstrainedConv1d(**i).to(device=self.device)
                model.weight = torch.nn.Parameter(model.weight * 10)
                out = model(self.x2)
                if i["max_norm"] is not None:
                    norms = torch.sqrt(torch.sum(torch.square(model.weight), axis=[1, 2]))
                    self.assertTrue(torch.sum(norms > (i["max_norm"] + 1e-3)).item() == 0)
                    if i["min_norm"] is not None:
                        self.assertTrue(torch.sum(norms < (i["min_norm"] - 1e-3)).item() == 0)
                self.assertEqual(torch.isnan(out).sum(), 0)
        print(
            "   Constrained conv1d OK: tested", len(Conv_args), " combinations of input arguments"
        )

    def test_ConstrainedConv2d(self):
        print("Testing conv2d with norm constraint...", end="", flush=True)
        Conv_args = {
            "in_channels": [1],
            "out_channels": [5, 16],
            "kernel_size": [(1, 64), (5, 1), (5, 64)],
            "stride": [1, 2, 3],
            "dilation": [1, 2],
            "bias": [True, False],
            "max_norm": [None, 1, 2, 3],
            "min_norm": [None, 1],
            "padding": ["valid"],
        }
        Conv_args = self.makeGrid(Conv_args)
        for i in Conv_args:
            model = models.ConstrainedConv2d(**i)
            model.weight = torch.nn.Parameter(model.weight * 10)
            out = model(self.xl)
            if i["max_norm"] is not None:
                norms = torch.sqrt(torch.sum(torch.square(model.weight), axis=[1, 2, 3]))
                self.assertTrue(torch.sum(norms > (i["max_norm"] + 1e-3)).item() == 0)
                if i["min_norm"] is not None:
                    self.assertTrue(torch.sum(norms < (i["min_norm"] - 1e-3)).item() == 0)
            self.assertEqual(torch.isnan(out).sum(), 0)

        if self.device.type != "cpu":
            for i in Conv_args:
                model = models.ConstrainedConv2d(**i).to(device=self.device)
                model.weight = torch.nn.Parameter(model.weight * 10)
                out = model(self.xl2)
                if i["max_norm"] is not None:
                    norms = torch.sqrt(torch.sum(torch.square(model.weight), axis=[1, 2, 3]))
                    self.assertTrue(torch.sum(norms > (i["max_norm"] + 1e-3)).item() == 0)
                    if i["min_norm"] is not None:
                        self.assertTrue(torch.sum(norms < (i["min_norm"] - 1e-3)).item() == 0)
                self.assertEqual(torch.isnan(out).sum(), 0)
        print(
            "   Constrained conv2d OK: tested", len(Conv_args), " combinations of input arguments"
        )

    def test_ConstrainedDense(self):
        print("Testing Dense layer with max norm constraint...", end="", flush=True)
        Dense_args = {
            "in_features": [128],
            "out_features": [32],
            "bias": [True, False],
            "max_norm": [None, 1, 3],
            "min_norm": [None, 1],
        }
        Dense_args = self.makeGrid(Dense_args)
        for i in Dense_args:
            model = models.ConstrainedDense(**i)
            model.weight = torch.nn.Parameter(model.weight * 10)
            out = model(self.xd)
            if i["max_norm"] is not None:
                norms = torch.sqrt(torch.sum(torch.square(model.weight), axis=1))
                self.assertTrue(torch.sum(norms > (i["max_norm"] + 1e-3)).item() == 0)
                if i["min_norm"] is not None:
                    self.assertTrue(torch.sum(norms < (i["min_norm"] - 1e-3)).item() == 0)
            self.assertEqual(torch.isnan(out).sum(), 0)
            self.assertEqual(out.shape[1], 32)

        if self.device.type != "cpu":
            for i in Dense_args:
                model = models.ConstrainedDense(**i).to(device=self.device)
                model.weight = torch.nn.Parameter(model.weight * 10)
                out = model(self.xd2)
                if i["max_norm"] is not None:
                    norms = torch.sqrt(torch.sum(torch.square(model.weight), axis=1))
                    self.assertTrue(torch.sum(norms > (i["max_norm"] + 1e-3)).item() == 0)
                    if i["min_norm"] is not None:
                        self.assertTrue(torch.sum(norms < (i["min_norm"] - 1e-3)).item() == 0)
                self.assertEqual(torch.isnan(out).sum(), 0)
                self.assertEqual(out.shape[1], 32)
        print("   Dense layer OK: tested", len(Dense_args), " combinations of input arguments")

    def test_DepthwiseConv2d(self):
        print("Testing Depthwise conv2d with norm constraint...", end="", flush=True)
        Depthwise_args = {
            "in_channels": [1],
            "depth_multiplier": [2, 3, 4],
            "kernel_size": [(1, 64), (5, 1), (5, 64)],
            "stride": [1, 2, 3],
            "dilation": [1, 2],
            "bias": [True, False],
            "max_norm": [None, 1, 3],
            "min_norm": [None, 1],
            "padding": ["valid"],
        }
        Depthwise_args = self.makeGrid(Depthwise_args)
        for i in Depthwise_args:
            model = models.DepthwiseConv2d(**i)
            model.weight = torch.nn.Parameter(model.weight * 10)
            out = model(self.xl)
            if i["max_norm"] is not None:
                norms = torch.sqrt(torch.sum(torch.square(model.weight), axis=[1, 2, 3]))
                self.assertTrue(torch.sum(norms > (i["max_norm"] + 1e-3)).item() == 0)
            self.assertEqual(torch.isnan(out).sum(), 0)
            self.assertEqual(out.shape[1], i["depth_multiplier"])

        if self.device.type != "cpu":
            for i in Depthwise_args:
                model = models.DepthwiseConv2d(**i).to(device=self.device)
                model.weight = torch.nn.Parameter(model.weight * 10)
                out = model(self.xl2)
                if i["max_norm"] is not None:
                    norms = torch.sqrt(torch.sum(torch.square(model.weight), axis=[1, 2, 3]))
                    self.assertTrue(torch.sum(norms > (i["max_norm"] + 1e-3)).item() == 0)
                    if i["min_norm"] is not None:
                        self.assertTrue(torch.sum(norms < (i["min_norm"] - 1e-3)).item() == 0)
                self.assertEqual(torch.isnan(out).sum(), 0)
                self.assertEqual(out.shape[1], i["depth_multiplier"])
        print(
            "   Depthwise conv2d OK: tested",
            len(Depthwise_args),
            " combinations of input arguments",
        )

    def test_SeparableConv2d(self):
        print("Testing Separable conv2d with norm constraint...", end="", flush=True)
        Separable_args = {
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
        }
        Separable_args = self.makeGrid(Separable_args)
        for i in Separable_args:
            model = models.SeparableConv2d(**i)
            out = model(self.xl)
            self.assertEqual(torch.isnan(out).sum(), 0)
            self.assertEqual(out.shape[1], i["out_channels"])

        if self.device.type != "cpu":
            for i in Separable_args:
                model = models.SeparableConv2d(**i).to(device=self.device)
                out = model(self.xl2)
                self.assertEqual(torch.isnan(out).sum(), 0)
                self.assertEqual(out.shape[1], i["out_channels"])
        print(
            "   Separable conv2d OK: tested", len(Separable_args), "combinations of input arguments"
        )


if __name__ == "__main__":
    unittest.main()
