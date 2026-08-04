import os
import sys
import unittest
import warnings

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from EEGself._testtools import get_device, make_grid

from selfeeg import models

N, CHAN, SAMPLES = 2, 8, 2048


class TestModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore", message="Using padding='same'", category=UserWarning)
        cls.device = get_device(
            probe=lambda dev: models.EEGNet(2, 8, 2048).to(dev)(torch.randn(2, 8, 2048, device=dev))
        )
        cls.x = torch.randn(N, CHAN, SAMPLES)

    def setUp(self):
        torch.manual_seed(1234)

    def _devices(self, allow_mps=True):
        if self.device.type == "cpu":
            return ["cpu"]
        if self.device.type == "mps" and not allow_mps:
            return ["cpu"]
        return ["cpu", self.device.type]

    def _check_classifier(self, model_cls, grid, input_fn=None, skip=None, allow_mps=True, label_keys=()):
        """Build ``model_cls`` for every combination and check its output.

        The output must be NaN-free, have ``nb_classes`` columns (or 1 when
        binary), and, unless ``return_logits`` is set, lie in the [0, 1] range.
        """
        for dev in self._devices(allow_mps):
            for args in grid:
                if skip is not None and skip(args):
                    continue
                with self.subTest(device=dev, **{k: args[k] for k in label_keys}):
                    model = model_cls(**args).to(dev)
                    x = input_fn(args, dev) if input_fn else self.x.to(dev)
                    out = model(x)
                    self.assertEqual(torch.isnan(out).sum().item(), 0)
                    expected = args["nb_classes"] if args["nb_classes"] > 2 else 1
                    self.assertEqual(out.shape[1], expected)
                    if not args["return_logits"]:
                        self.assertLessEqual(out.max().item(), 1)
                        self.assertGreaterEqual(out.min().item(), 0)

    def test_ATCNet(self):
        grid = make_grid(
            {
                "nb_classes": [2, 4],
                "Chans": [CHAN],
                "Samples": [SAMPLES],
                "Fs": [128],
                "num_windows": [4],
                "mha_heads": [2, 4],
                "tcn_depth": [2, 3],
                "F1": [12, 8],
                "D": [2, 3],
                "return_logits": [False],
                "seed": [42],
            },
            max_comb=12,
        )
        self._check_classifier(models.ATCNet, grid, label_keys=("nb_classes", "mha_heads", "tcn_depth"))

    def test_DeepConvNet(self):
        grid = make_grid(
            {
                "nb_classes": [2, 4],
                "Chans": [CHAN],
                "Samples": [SAMPLES],
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
            },
            max_comb=12,
        )
        self._check_classifier(models.DeepConvNet, grid, label_keys=("nb_classes", "kernLength", "F"))

    def test_EEGConformer(self):
        grid = make_grid(
            {
                "nb_classes": [2, 4],
                "Samples": [2048],
                "Chans": [CHAN],
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
            },
            max_comb=12,
        )
        self._check_classifier(models.EEGConformer, grid, label_keys=("nb_classes", "d_model", "nheads"))

    def test_EEGInception(self):
        grid = make_grid(
            {
                "nb_classes": [2, 4],
                "Chans": [CHAN],
                "Samples": [SAMPLES],
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
            },
            max_comb=12,
        )
        self._check_classifier(models.EEGInception, grid, label_keys=("nb_classes", "kernel_size", "bias"))

    def test_EEGNet(self):
        grid = make_grid(
            {
                "nb_classes": [2, 4],
                "Chans": [CHAN],
                "Samples": [SAMPLES],
                "kernLength": [32, 64],
                "F1": [4, 8],
                "D": [2, 4],
                "F2": [8, 16],
                "pool1": [4, 8],
                "pool2": [8, 16],
                "separable_kernel": [16, 32],
                "return_logits": [False],
                "seed": [42],
            },
            max_comb=12,
        )
        self._check_classifier(models.EEGNet, grid, label_keys=("nb_classes", "kernLength", "F1"))

    def test_EEGSym(self):
        grid = make_grid(
            {
                "nb_classes": [2, 4],
                "Samples": [2048],
                "Chans": [CHAN],
                "Fs": [64],
                "scales_time": [(500, 250, 125), (250, 183, 95)],
                "lateral_chans": [2, 3],
                "first_left": [True, False],
                "F": [8, 24],
                "pool": [2, 3],
                "bias": [True, False],
                "return_logits": [False],
                "seed": [42],
            },
            max_comb=12,
        )
        self._check_classifier(
            models.EEGSym, grid, allow_mps=False, label_keys=("nb_classes", "lateral_chans", "F")
        )

    def test_FBCNet(self):
        grid = make_grid(
            {
                "nb_classes": [2, 4],
                "Chans": [CHAN],
                "Samples": [SAMPLES],
                "Fs": [128],
                "FilterBands": [4, 8],
                "FilterRange": [4, 5],
                "FilterType": ["Cheby2", "ellip"],
                "TemporalType": ["var", "max", "mean", "std", "logvar"],
                "return_logits": [False],
                "seed": [42],
            },
            max_comb=12,
        )
        self._check_classifier(models.FBCNet, grid, label_keys=("nb_classes", "FilterType", "TemporalType"))

    def test_ResNet(self):
        grid = make_grid(
            {
                "nb_classes": [2, 4],
                "Samples": [2048],
                "Chans": [CHAN],
                "block": [models.BasicBlock1],
                "Layers": [[1, 1, 1, 1], [1, 2, 4, 3]],
                "inplane": [8, 16],
                "kernLength": [7, 13],
                "addConnection": [True, False],
                "return_logits": [False],
                "seed": [42],
            },
            max_comb=12,
        )
        self._check_classifier(models.ResNet1D, grid, label_keys=("nb_classes", "inplane", "addConnection"))

    def test_ShallowNet(self):
        grid = make_grid(
            {
                "nb_classes": [2, 4],
                "Samples": [2048],
                "Chans": [CHAN],
                "F": [20, 40],
                "K1": [25, 12],
                "Pool": [75, 50],
                "return_logits": [False],
                "seed": [42],
            },
            max_comb=12,
        )
        self._check_classifier(models.ShallowNet, grid, label_keys=("nb_classes", "F", "K1"))

    def test_StagerNet(self):
        grid = make_grid(
            {
                "nb_classes": [2, 4],
                "Samples": [2048],
                "Chans": [CHAN],
                "F": [8, 16],
                "kernLength": [64, 120],
                "Pool": [16, 8],
                "return_logits": [False],
                "seed": [42],
            },
            max_comb=12,
        )
        self._check_classifier(models.StagerNet, grid, label_keys=("nb_classes", "F", "kernLength"))

    def test_STNet(self):
        grid = make_grid(
            {
                "nb_classes": [2, 4],
                "Samples": [2048],
                "grid_size": [5, 9],
                "F": [256, 64],
                "kernlength": [5, 7],
                "dense_size": [1024, 512],
                "return_logits": [False],
                "seed": [42],
            },
            max_comb=12,
        )

        def stnet_input(args, dev):
            return torch.randn(N, args["Samples"], args["grid_size"], args["grid_size"], device=dev)

        self._check_classifier(
            models.STNet, grid, input_fn=stnet_input, label_keys=("nb_classes", "grid_size", "F")
        )

    def test_TinySleepNet(self):
        grid = make_grid(
            {
                "nb_classes": [2, 4],
                "Chans": [CHAN],
                "Fs": [64],
                "F": [128, 32],
                "kernlength": [8, 30],
                "pool": [16, 5],
                "hidden_lstm": [128, 50],
                "return_logits": [False],
                "seed": [42],
            },
            max_comb=12,
        )
        self._check_classifier(models.TinySleepNet, grid, label_keys=("nb_classes", "F", "hidden_lstm"))

    def test_xEEGNet(self):
        grid = make_grid(
            {
                "nb_classes": [4],
                "Samples": [2048],
                "Chans": [CHAN],
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
            },
            max_comb=16,
        )
        self._check_classifier(
            models.xEEGNet,
            grid,
            skip=lambda a: a["F1"] > a["F2"] and a["spatial_depthwise"],
            label_keys=("F1", "F2", "spatial_depthwise"),
        )


if __name__ == "__main__":
    unittest.main()
