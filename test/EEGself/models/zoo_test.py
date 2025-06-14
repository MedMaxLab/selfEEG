import itertools
import os
import sys
import unittest
import warnings
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
        warnings.filterwarnings("ignore", message="Using padding='same'", category=UserWarning)
        if torch.backends.mps.is_available():
            cls.device = torch.device("mps")
        elif torch.cuda.is_available():
            cls.device = torch.device("cuda")
        else:
            cls.device = torch.device("cpu")

        if cls.device.type != "cpu":
            try:
                xx = torch.randn(2, 8, 2048).to(device=cls.device)
                model = models.EEGNet(2, 8, 2048).to(device=cls.device)
                xx = model(xx)
            except Exception:
                cls.device = torch.device("cpu")

        print("\n-------------------------")
        print("TESTING MODELS.ZOO MODULE")
        if cls.device.type != "cpu":
            print("Found gpu device: testing module with both cpu and gpu")
        else:
            print("Didn't found cuda device: testing module with only cpu")
        print("-------------------------")
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

    def test_ATCNet(self):
        print("Testing ATCNet...", end="", flush=True)

        DCN_args = {
            "nb_classes": [2, 4],
            "Chans": [self.Chan],
            "Samples": [self.Samples],
            "Fs": [128],
            "num_windows": [4],
            "mha_heads": [2, 4],
            "tcn_depth": [2, 3],
            "F1": [12, 8],
            "D": [2, 3],
            "return_logits": [False],
            "seed": [42],
        }
        DCN_grid = self.makeGrid(DCN_args)
        for i in DCN_grid:
            model = models.ATCNet(**i)
            out = model(self.x)
            self.assertEqual(torch.isnan(out).sum(), 0)
            self.assertEqual(out.shape[1], i["nb_classes"] if i["nb_classes"] > 2 else 1)
            if not (i["return_logits"]):
                self.assertLessEqual(out.max(), 1)
                self.assertGreaterEqual(out.min(), 0)
        if self.device.type != "cpu":
            for i in DCN_grid:
                model = models.ATCNet(**i).to(device=self.device)
                out = model(self.x2)
                self.assertEqual(torch.isnan(out).sum(), 0)
                self.assertEqual(out.shape[1], i["nb_classes"] if i["nb_classes"] > 2 else 1)
                if not (i["return_logits"]):
                    self.assertLessEqual(out.max(), 1)
                    self.assertGreaterEqual(out.min(), 0)
        print("   ATCNet OK: tested ", len(DCN_grid), " combinations of input arguments")

    def test_DeepConvNet(self):
        print("Testing DeepConvNet...", end="", flush=True)
        DCN_args = {
            "nb_classes": [2, 4],
            "Chans": [self.Chan],
            "Samples": [self.Samples],
            "kernLength": [10, 20],
            "F": [12, 25],
            "Pool": [3, 4],
            "stride": [3, 4],
            "max_norm": [2.0],
            "batch_momentum": [0.9],
            "ELUalpha": [1],
            "dropRate": [0.5],
            "max_dense_norm": [1.0],
            "return_logits": [False],
            "seed": [42],
        }
        DCN_grid = self.makeGrid(DCN_args)
        for i in DCN_grid:
            model = models.DeepConvNet(**i)
            out = model(self.x)
            self.assertEqual(torch.isnan(out).sum(), 0)
            self.assertEqual(out.shape[1], i["nb_classes"] if i["nb_classes"] > 2 else 1)
            if not (i["return_logits"]):
                self.assertLessEqual(out.max(), 1)
                self.assertGreaterEqual(out.min(), 0)
        if self.device.type != "cpu":
            for i in DCN_grid:
                model = models.DeepConvNet(**i).to(device=self.device)
                out = model(self.x2)
                self.assertEqual(torch.isnan(out).sum(), 0)
                self.assertEqual(out.shape[1], i["nb_classes"] if i["nb_classes"] > 2 else 1)
                if not (i["return_logits"]):
                    self.assertLessEqual(out.max(), 1)
                    self.assertGreaterEqual(out.min(), 0)
        print("   DeepConvNet OK: tested ", len(DCN_grid), " combinations of input arguments")

    def test_EEGConformer(self):
        print("Testing EEGConformer...", end="", flush=True)
        EEGcon_args = {
            "nb_classes": [2, 4],
            "Samples": [2048],
            "Chans": [self.Chan],
            "F": [40],
            "K1": [25, 12],
            "Pool": [75, 50],
            "stride_pool": [20],
            "nlayers": [4],
            "d_model": [40, 80],
            "nheads": [8, 10],
            "dim_feedforward": [80],
            "activation_transformer": ["gelu"],
            "mlp_dim": [[128, 32], [64, 32]],
            "return_logits": [False],
            "seed": [42],
        }
        EEGcon_args = self.makeGrid(EEGcon_args)
        for i in EEGcon_args:
            model = models.EEGConformer(**i)
            out = model(self.x)
            self.assertEqual(torch.isnan(out).sum(), 0)
            self.assertEqual(out.shape[1], i["nb_classes"] if i["nb_classes"] > 2 else 1)
            if not (i["return_logits"]):
                self.assertLessEqual(out.max(), 1)
                self.assertGreaterEqual(out.min(), 0)

        if self.device.type != "cpu":
            for i in EEGcon_args:
                model = models.EEGConformer(**i).to(device=self.device)
                out = model(self.x2)
                self.assertEqual(torch.isnan(out).sum(), 0)
                self.assertEqual(out.shape[1], i["nb_classes"] if i["nb_classes"] > 2 else 1)
                if not (i["return_logits"]):
                    self.assertLessEqual(out.max(), 1)
                    self.assertGreaterEqual(out.min(), 0)
        print("   EEGConformer OK: tested", len(EEGcon_args), " combinations of input arguments")

    def test_EEGInception(self):
        print("Testing EEGInception...", end="", flush=True)
        EEGin_args = {
            "nb_classes": [2, 4],
            "Chans": [self.Chan],
            "Samples": [self.Samples],
            "kernel_size": [32, 128],
            "F1": [4, 16],
            "D": [2, 4],
            "pool": [4, 8],
            "batch_momentum": [0.9],
            "dropRate": [0.5],
            "max_depth_norm": [1.0],
            "return_logits": [False],
            "bias": [True, False],
            "seed": [42],
        }
        EEGin_args = self.makeGrid(EEGin_args)
        for i in EEGin_args:
            model = models.EEGInception(**i)
            out = model(self.x)
            self.assertEqual(torch.isnan(out).sum(), 0)
            self.assertEqual(out.shape[1], i["nb_classes"] if i["nb_classes"] > 2 else 1)
            if not (i["return_logits"]):
                self.assertLessEqual(out.max(), 1)
                self.assertGreaterEqual(out.min(), 0)

        if self.device.type != "cpu":
            for i in EEGin_args:
                model = models.EEGInception(**i).to(device=self.device)
                out = model(self.x2)
                self.assertEqual(torch.isnan(out).sum(), 0)
                self.assertEqual(out.shape[1], i["nb_classes"] if i["nb_classes"] > 2 else 1)
                if not (i["return_logits"]):
                    self.assertLessEqual(out.max(), 1)
                    self.assertGreaterEqual(out.min(), 0)
        print("   EEGInception OK: tested", len(EEGin_args), " combinations of input arguments")

    def test_EEGNet(self):
        print("Testing EEGnet...", end="", flush=True)
        EEGnet_args = {
            "nb_classes": [2, 4],
            "Chans": [self.Chan],
            "Samples": [self.Samples],
            "kernLength": [32, 64],
            "F1": [4, 8],
            "D": [2, 4],
            "F2": [8, 16],
            "pool1": [4, 8],
            "pool2": [8, 16],
            "separable_kernel": [16, 32],
            "return_logits": [False],
            "seed": [42],
        }
        EEGnet_args = self.makeGrid(EEGnet_args)
        for i in EEGnet_args:
            model = models.EEGNet(**i)
            out = model(self.x)
            self.assertEqual(torch.isnan(out).sum(), 0)
            self.assertEqual(out.shape[1], i["nb_classes"] if i["nb_classes"] > 2 else 1)
            if not (i["return_logits"]):
                self.assertLessEqual(out.max(), 1)
                self.assertGreaterEqual(out.min(), 0)

        if self.device.type != "cpu":
            for i in EEGnet_args:
                model = models.EEGNet(**i).to(device=self.device)
                out = model(self.x2)
                self.assertEqual(torch.isnan(out).sum(), 0)
                self.assertEqual(out.shape[1], i["nb_classes"] if i["nb_classes"] > 2 else 1)
                if not (i["return_logits"]):
                    self.assertLessEqual(out.max(), 1)
                    self.assertGreaterEqual(out.min(), 0)
        print("   EEGnet OK: tested", len(EEGnet_args), " combinations of input arguments")

    def test_EEGSym(self):
        print("Testing EEGsym...", end="", flush=True)
        EEGsym_args = {
            "nb_classes": [2, 4],
            "Samples": [2048],
            "Chans": [self.Chan],
            "Fs": [64],
            "scales_time": [(500, 250, 125), (250, 183, 95)],
            "lateral_chans": [2, 3],
            "first_left": [True, False],
            "F": [8, 24],
            "pool": [2, 3],
            "bias": [True, False],
            "return_logits": [False],
            "seed": [42],
        }
        EEGsym_args = self.makeGrid(EEGsym_args)
        for i in EEGsym_args:
            model = models.EEGSym(**i)
            out = model(self.x)
            self.assertEqual(torch.isnan(out).sum(), 0)
            self.assertEqual(out.shape[1], i["nb_classes"] if i["nb_classes"] > 2 else 1)
            if not (i["return_logits"]):
                self.assertLessEqual(out.max(), 1)
                self.assertGreaterEqual(out.min(), 0)

        if self.device.type not in ["cpu", "mps"]:
            for i in EEGsym_args:
                model = models.EEGSym(**i).to(device=self.device)
                out = model(self.x2)
                self.assertEqual(torch.isnan(out).sum(), 0)
                self.assertEqual(out.shape[1], i["nb_classes"] if i["nb_classes"] > 2 else 1)
                if not (i["return_logits"]):
                    self.assertLessEqual(out.max(), 1)
                    self.assertGreaterEqual(out.min(), 0)
        print("   EEGsym OK: tested", len(EEGsym_args), " combinations of input arguments")

    def test_FBCNet(self):
        print("Testing FBCNet...", end="", flush=True)

        DCN_args = {
            "nb_classes": [2, 4],
            "Chans": [self.Chan],
            "Samples": [self.Samples],
            "Fs": [128],
            "FilterBands": [4, 8],
            "FilterRange": [4, 5],
            "FilterType": ["Cheby2", "ellip"],
            "TemporalType": ["var", "max", "mean", "std", "logvar"],
            "return_logits": [False],
            "seed": [42],
        }
        DCN_grid = self.makeGrid(DCN_args)
        for i in DCN_grid:
            model = models.FBCNet(**i)
            out = model(self.x)
            self.assertEqual(torch.isnan(out).sum(), 0)
            self.assertEqual(out.shape[1], i["nb_classes"] if i["nb_classes"] > 2 else 1)
            if not (i["return_logits"]):
                self.assertLessEqual(out.max(), 1)
                self.assertGreaterEqual(out.min(), 0)
        if self.device.type != "cpu":
            for i in DCN_grid:
                model = models.FBCNet(**i).to(device=self.device)
                out = model(self.x2)
                self.assertEqual(torch.isnan(out).sum(), 0)
                self.assertEqual(out.shape[1], i["nb_classes"] if i["nb_classes"] > 2 else 1)
                if not (i["return_logits"]):
                    self.assertLessEqual(out.max(), 1)
                    self.assertGreaterEqual(out.min(), 0)
        print("   FBCNet OK: tested ", len(DCN_grid), " combinations of input arguments")

    def test_ResNet(self):
        print("Testing ResNet...", end="", flush=True)
        EEGres_args = {
            "nb_classes": [2, 4],
            "Samples": [2048],
            "Chans": [self.Chan],
            "block": [models.BasicBlock1],
            "Layers": [[1, 1, 1, 1], [1, 2, 4, 3]],
            "inplane": [8, 16],
            "kernLength": [7, 13],
            "addConnection": [True, False],
            "return_logits": [False],
            "seed": [42],
        }
        EEGres_args = self.makeGrid(EEGres_args)
        for i in EEGres_args:
            model = models.ResNet1D(**i)
            out = model(self.x)
            self.assertEqual(torch.isnan(out).sum(), 0)
            self.assertEqual(out.shape[1], i["nb_classes"] if i["nb_classes"] > 2 else 1)
            if not (i["return_logits"]):
                self.assertLessEqual(out.max(), 1)
                self.assertGreaterEqual(out.min(), 0)

        if self.device.type != "cpu":
            for i in EEGres_args:
                model = models.ResNet1D(**i).to(device=self.device)
                out = model(self.x2)
                self.assertEqual(torch.isnan(out).sum(), 0)
                self.assertEqual(out.shape[1], i["nb_classes"] if i["nb_classes"] > 2 else 1)
                if not (i["return_logits"]):
                    self.assertLessEqual(out.max(), 1)
                    self.assertGreaterEqual(out.min(), 0)
        print("   ResNet OK: tested", len(EEGres_args), " combinations of input arguments")

    def test_ShallowNet(self):
        print("Testing ShallowNet...", end="", flush=True)
        EEGsha_args = {
            "nb_classes": [2, 4],
            "Samples": [2048],
            "Chans": [self.Chan],
            "F": [20, 40],
            "K1": [25, 12],
            "Pool": [75, 50],
            "return_logits": [False],
            "seed": [42],
        }
        EEGsha_args = self.makeGrid(EEGsha_args)
        for i in EEGsha_args:
            model = models.ShallowNet(**i)
            out = model(self.x)
            self.assertEqual(torch.isnan(out).sum(), 0)
            self.assertEqual(out.shape[1], i["nb_classes"] if i["nb_classes"] > 2 else 1)
            if not (i["return_logits"]):
                self.assertLessEqual(out.max(), 1)
                self.assertGreaterEqual(out.min(), 0)

        if self.device.type != "cpu":
            for i in EEGsha_args:
                model = models.ShallowNet(**i).to(device=self.device)
                out = model(self.x2)
                self.assertEqual(torch.isnan(out).sum(), 0)
                self.assertEqual(out.shape[1], i["nb_classes"] if i["nb_classes"] > 2 else 1)
                if not (i["return_logits"]):
                    self.assertLessEqual(out.max(), 1)
                    self.assertGreaterEqual(out.min(), 0)
        print("   ShallowNet OK: tested", len(EEGsha_args), " combinations of input arguments")

    def test_StagerNet(self):
        print("Testing StageRNet...", end="", flush=True)
        EEGsta_args = {
            "nb_classes": [2, 4],
            "Samples": [2048],
            "Chans": [self.Chan],
            "F": [8, 16],
            "kernLength": [64, 120],
            "Pool": [16, 8],
            "return_logits": [False],
            "seed": [42],
        }
        EEGsta_args = self.makeGrid(EEGsta_args)
        for i in EEGsta_args:
            model = models.StagerNet(**i)
            out = model(self.x)
            self.assertEqual(torch.isnan(out).sum(), 0)
            self.assertEqual(out.shape[1], i["nb_classes"] if i["nb_classes"] > 2 else 1)
            if not (i["return_logits"]):
                self.assertLessEqual(out.max(), 1)
                self.assertGreaterEqual(out.min(), 0)

        if self.device.type != "cpu":
            for i in EEGsta_args:
                model = models.StagerNet(**i).to(device=self.device)
                out = model(self.x2)
                self.assertEqual(torch.isnan(out).sum(), 0)
                self.assertEqual(out.shape[1], i["nb_classes"] if i["nb_classes"] > 2 else 1)
                if not (i["return_logits"]):
                    self.assertLessEqual(out.max(), 1)
                    self.assertGreaterEqual(out.min(), 0)
        print("   StageRNet OK: tested", len(EEGsta_args), " combinations of input arguments")

    def test_STNet(self):
        print("Testing STNet...", end="", flush=True)
        EEGstn_args = {
            "nb_classes": [2, 4],
            "Samples": [2048],
            "grid_size": [5, 9],
            "F": [256, 64],
            "kernlength": [5, 7],
            "dense_size": [1024, 512],
            "return_logits": [False],
            "seed": [42],
        }
        EEGstn_args = self.makeGrid(EEGstn_args)
        for i in EEGstn_args:
            model = models.STNet(**i)
            xst = torch.randn(self.N, i["Samples"], i["grid_size"], i["grid_size"])
            out = model(xst)
            self.assertEqual(torch.isnan(out).sum(), 0)
            self.assertEqual(out.shape[1], i["nb_classes"] if i["nb_classes"] > 2 else 1)
            if not (i["return_logits"]):
                self.assertLessEqual(out.max(), 1)
                self.assertGreaterEqual(out.min(), 0)

        if self.device.type != "cpu":
            for i in EEGstn_args:
                model = models.STNet(**i).to(device=self.device)
                xst2 = torch.randn(self.N, i["Samples"], i["grid_size"], i["grid_size"]).to(
                    device=self.device
                )
                out = model(xst2)
                self.assertEqual(torch.isnan(out).sum(), 0)
                self.assertEqual(out.shape[1], i["nb_classes"] if i["nb_classes"] > 2 else 1)
                if not (i["return_logits"]):
                    self.assertLessEqual(out.max(), 1)
                    self.assertGreaterEqual(out.min(), 0)
        print("   STNet OK: tested", len(EEGstn_args), " combinations of input arguments")

    def test_TinySleepNet(self):
        print("Testing TinySleepNet...", end="", flush=True)
        EEGsleep_args = {
            "nb_classes": [2, 4],
            "Chans": [self.Chan],
            "Fs": [64],
            "F": [128, 32],
            "kernlength": [8, 30],
            "pool": [16, 5],
            "hidden_lstm": [128, 50],
            "return_logits": [False],
            "seed": [42],
        }
        EEGsleep_args = self.makeGrid(EEGsleep_args)
        for i in EEGsleep_args:
            model = models.TinySleepNet(**i)
            out = model(self.x)
            self.assertEqual(torch.isnan(out).sum(), 0)
            self.assertEqual(out.shape[1], i["nb_classes"] if i["nb_classes"] > 2 else 1)
            if not (i["return_logits"]):
                self.assertLessEqual(out.max(), 1)
                self.assertGreaterEqual(out.min(), 0)

        if self.device.type != "cpu":
            for i in EEGsleep_args:
                model = models.TinySleepNet(**i).to(device=self.device)
                out = model(self.x2)
                self.assertEqual(torch.isnan(out).sum(), 0)
                self.assertEqual(out.shape[1], i["nb_classes"] if i["nb_classes"] > 2 else 1)
                if not (i["return_logits"]):
                    self.assertLessEqual(out.max(), 1)
                    self.assertGreaterEqual(out.min(), 0)
        print("   TinySleepNet OK: tested", len(EEGsleep_args), " combinations of input arguments")

    def test_xEEGNet(self):
        print("Testing xEEGNet...", end="", flush=True)
        EEGxeg_args = {
            "nb_classes": [4],
            "Samples": [2048],
            "Chans": [self.Chan],
            "Fs": [125],
            "F1": [7, 126],
            "K1": [125, 75],
            "F2": [7, 126],
            "Pool": [75, 50],
            "random_temporal_filter": [True, False],
            "freeze_temporal": [0, 1e12],
            "spatial_depthwise": [True, False],
            "log_activation_base": ["dB"],
            "norm_type": ["batchnorm"],
            "global_pooling": [True, False],
            "bias": [[False] * 3],
            "dense_hidden": [-1, 32],
            "return_logits": [False],
            "seed": [42],
        }

        EEGxeg_args = self.makeGrid(EEGxeg_args)
        for i in EEGxeg_args:
            if i["F1"] > i["F2"] and i["spatial_depthwise"]:
                continue
            model = models.xEEGNet(**i)
            out = model(self.x)
            self.assertEqual(torch.isnan(out).sum(), 0)
            self.assertEqual(out.shape[1], i["nb_classes"] if i["nb_classes"] > 2 else 1)
            if not (i["return_logits"]):
                self.assertLessEqual(out.max(), 1)
                self.assertGreaterEqual(out.min(), 0)

        if self.device.type != "cpu":
            for i in EEGxeg_args:
                if i["F1"] > i["F2"] and i["spatial_depthwise"]:
                    continue
                model = models.xEEGNet(**i).to(device=self.device)
                out = model(self.x2)
                self.assertEqual(torch.isnan(out).sum(), 0)
                self.assertEqual(out.shape[1], i["nb_classes"] if i["nb_classes"] > 2 else 1)
                if not (i["return_logits"]):
                    self.assertLessEqual(out.max(), 1)
                    self.assertGreaterEqual(out.min(), 0)
        print("   xEEGNet OK: tested", len(EEGxeg_args), " combinations of input arguments")


if __name__ == "__main__":
    unittest.main()
