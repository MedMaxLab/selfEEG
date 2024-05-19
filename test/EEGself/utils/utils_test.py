import itertools
import random
import unittest

import numpy as np
import torch

from selfeeg import models, utils


class TestUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n--------------------")
        print("TESTING UTILS MODULE")
        if torch.backends.mps.is_available():
            cls.device = torch.device("mps")
        elif torch.cuda.is_available():
            cls.device = torch.device("cuda")
        else:
            cls.device = torch.device("cpu")

        if cls.device.type == "mps":
            try:
                xx = torch.randn(2, 2).to(device=cls.device)
            except Exception:
                cls.device = torch.device("cpu")

        if cls.device.type != "cpu":
            print("Found gpu device: testing module with both cpu and gpu")
        else:
            print("Didn't found cuda device: testing module with only cpu")
        print("--------------------")

    def makeGrid(self, pars_dict):
        keys = pars_dict.keys()
        combinations = itertools.product(*pars_dict.values())
        ds = [dict(zip(keys, cc)) for cc in combinations]
        return ds

    def test_scale_range_soft_clip(self):
        print("testing scale range with soft clip function...", end="", flush=True)
        random.seed(1234)
        x = torch.zeros(16, 32, 1024) + torch.sin(torch.linspace(0, 8 * torch.pi, 1024)) * 500
        xnp = x.numpy()
        inplist = [x, xnp]
        if self.device.type != "cpu":
            xgpu = torch.clone(x).to(device=self.device)
            inplist = [x, xnp, xgpu]
        input_args = {
            "x": inplist,
            "Range": [200, 300],
            "asintote": [1.0, 2.3, 3.5],
            "scale": ["uV"],
            "exact": [True, False],
        }
        input_args = self.makeGrid(input_args)
        for i in input_args:
            x_scaled = utils.scale_range_soft_clip(**i)
            if isinstance(i["x"], torch.Tensor):
                self.assertTrue(torch.isnan(x_scaled).sum() == 0)
            else:
                self.assertTrue(np.isnan(x_scaled).sum() == 0)
            self.assertFalse(x.max() <= i["asintote"] and x.min() >= -i["asintote"])
            self.assertTrue(x_scaled.max() <= i["asintote"] and x_scaled.min() >= -i["asintote"])
        print("   scale range with soft clip OK")

    def test_RangeScaler(self):
        print("testing Range Scaler...", end="", flush=True)
        random.seed(1234)
        x = torch.zeros(16, 32, 1024)
        x += torch.sin(torch.linspace(0, 8 * torch.pi, 1024)) * 500
        xnp = x.numpy()
        inplist = [x, xnp]
        if self.device.type != "cpu":
            xgpu = torch.clone(x).to(device=self.device)
            inplist = [x, xnp, xgpu]
        input_args = {
            "x": inplist,
            "Range": [200, 300],
            "asintote": [1.0, 2.3, 3.5],
            "scale": ["uV"],
            "exact": [True, False],
        }
        input_args = self.makeGrid(input_args)
        for i in input_args:
            Scaler = utils.RangeScaler(i["Range"], i["asintote"], i["scale"], i["exact"])
            x_scaled = Scaler(i["x"])
            if isinstance(i["x"], torch.Tensor):
                self.assertTrue(torch.isnan(x_scaled).sum() == 0)
            else:
                self.assertTrue(np.isnan(x_scaled).sum() == 0)
            self.assertFalse(x.max() <= i["asintote"] and x.min() >= -i["asintote"])
            self.assertTrue(x_scaled.max() <= i["asintote"] and x_scaled.min() >= -i["asintote"])
        print("   Range Scaler OK")

    def test_torch_zscore(self):
        print("testing Zscore Scaler...", end="", flush=True)
        x = torch.ones(2, 16, 16)
        x[0] += torch.randn(16, 16)
        if self.device.type != "cpu":
            x = x.to(device=self.device)
        xz = utils.torch_zscore(x, -2, 1)
        self.assertTrue(torch.isnan(xz).sum() == 16 * 16)
        self.assertTrue(xz.device.type == self.device.type)

        x[1] += torch.randn(16, 16).to(device=self.device)
        xz = utils.torch_zscore(x, -2, 1)
        self.assertTrue(torch.isnan(xz).sum() == 0)
        self.assertTrue((xz.mean(-2).abs() > 1e-6).sum().item() == 0)
        self.assertTrue(((xz.std(-2, correction=0).abs() - 1) > 1e-6).sum().item() == 0)
        print("   Zscore Scaler OK")

    def test_get_subarray_closest_sum(self):
        print("testing subarray closest sum function...", end="", flush=True)
        random.seed(1235)
        arr = [i for i in range(1, 100)]
        _, best_sub_arr = utils.get_subarray_closest_sum(
            arr, 3251, tolerance=1e-4, perseverance=10000
        )
        self.assertEqual(sum(best_sub_arr), 3251)
        _, best_sub_arr = utils.get_subarray_closest_sum(
            arr, 2497, tolerance=1e-4, perseverance=10000
        )
        self.assertEqual(sum(best_sub_arr), 2497)
        print("   subarray closest sum OK")

    def test_check_models(self):
        print("testing check models function...", end="", flush=True)
        model1 = models.EEGNet(4, 8, 512)
        model2 = models.EEGNet(4, 8, 512)
        self.assertFalse(utils.check_models(model1, model2))  # Should return False
        model2.load_state_dict(model1.state_dict())
        self.assertTrue(utils.check_models(model1, model2))  # Should return True
        print("   check models OK")

    def test_count_parameters(self):
        print("testng count parameters function...\n")
        mdl = models.ShallowNet(4, 8, 1024)
        for n, i in enumerate(mdl.parameters()):  # bias require grad put to False
            i.requires_grad = False if n in [1, 3, 5, 7] else True
        a, b = utils.count_parameters(mdl, True, True, True)
        self.assertEqual(b, 23760)  # should return True
        print("\n   count parameters OK")


if __name__ == "__main__":
    unittest.main()
