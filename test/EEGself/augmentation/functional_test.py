import os
import sys
import unittest

sys.path.append(os.getcwd().split("/test")[0])
import itertools
import math

import numpy as np
import torch
from scipy.signal import periodogram

from selfeeg import augmentation as aug


class TestAugmentationFunctional(unittest.TestCase):

    def makeGrid(self, pars_dict):
        keys = pars_dict.keys()
        combinations = itertools.product(*pars_dict.values())
        ds = [dict(zip(keys, cc)) for cc in combinations]
        return ds

    @classmethod
    def setUpClass(cls):
        if torch.backends.mps.is_available():
            cls.device = torch.device("mps")
        elif torch.cuda.is_available():
            cls.device = torch.device("cuda")
        else:
            cls.device = torch.device("cpu")

        if cls.device.type != "cpu":
            try:
                xx = torch.randn(1024).to(device=cls.device)
                xx = aug.add_band_noise(xx, "theta", 128)
            except Exception:
                cls.device = torch.device("cpu")

        device = cls.device
        print("\n---------------------------")
        print("TESTING AUGMENTATION.FUNCTIONAL MODULE")
        if cls.device.type != "cpu":
            print("Found gpu device: testing module on it")
        else:
            print("Didn't found cuda device: testing module on cpu")
        print("---------------------------")
        dims = (32, 2, 32, 1024)
        pi = torch.pi
        cls.x1 = torch.zeros(*dims[-1:]) + torch.sin(torch.linspace(0, 8 * pi, 1024))
        cls.x2 = torch.zeros(*dims[-2:]) + torch.sin(torch.linspace(0, 8 * pi, 1024))
        cls.x3 = torch.zeros(*dims[-3:]) + torch.sin(torch.linspace(0, 8 * pi, 1024))
        cls.x4 = torch.zeros(*dims) + torch.sin(torch.linspace(0, 8 * pi, 1024))
        cls.x1np = cls.x1.numpy()
        cls.x2np = cls.x2.numpy()
        cls.x3np = cls.x3.numpy()
        cls.x4np = cls.x4.numpy()
        if device.type != "cpu":
            cls.x1gpu = torch.clone(cls.x1).to(device=device)
            cls.x2gpu = torch.clone(cls.x2).to(device=device)
            cls.x3gpu = torch.clone(cls.x3).to(device=device)
            cls.x4gpu = torch.clone(cls.x4).to(device=device)

    def test_identity(self):
        print("Testing identity...", end="", flush=True)
        aug_args = {
            "x": [self.x1, self.x2, self.x3, self.x4, self.x1np, self.x2np, self.x3np, self.x4np]
        }
        aug_args = self.makeGrid(aug_args)
        for i in aug_args:
            xaug = aug.identity(**i)
            if isinstance(xaug, torch.Tensor):
                self.assertTrue(torch.isnan(xaug).sum() == 0)
                self.assertTrue(torch.equal(i["x"], xaug))
            else:
                self.assertTrue(np.isnan(xaug).sum() == 0)
                self.assertTrue(np.array_equal(i["x"], xaug))
        N = len(aug_args)
        if self.device.type != "cpu":
            aug_args = {"x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu]}
            aug_args = self.makeGrid(aug_args)
            for i in aug_args:
                xaug = aug.identity(**i)
                self.assertTrue(torch.isnan(xaug).sum() == 0)
                self.assertTrue(torch.equal(i["x"], xaug))
        print("   identity OK: tested", N + len(aug_args), "combinations of input arguments")

    def test_shift_vertical(self):
        print("Testing shift vertical...", end="", flush=True)
        aug_args = {
            "x": [self.x1, self.x2, self.x3, self.x4, self.x1np, self.x2np, self.x3np, self.x4np],
            "value": [1, 2.0, 4],
        }
        aug_args = self.makeGrid(aug_args)
        for i in aug_args:
            xaug = aug.shift_vertical(**i)
            if isinstance(xaug, torch.Tensor):
                self.assertTrue(torch.isnan(xaug).sum() == 0)
                self.assertFalse(torch.equal(i["x"], xaug))
            else:
                self.assertTrue(np.isnan(xaug).sum() == 0)
                self.assertFalse(np.array_equal(i["x"], xaug))
        N = len(aug_args)
        if self.device.type != "cpu":
            aug_args = {"x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu], "value": [1, 2.0, 4]}
            aug_args = self.makeGrid(aug_args)
            for i in aug_args:
                xaug = aug.shift_vertical(**i)

        x = torch.zeros(16, 32, 1024)
        x += torch.sin(torch.linspace(0, 8 * torch.pi, 1024))
        xaug = aug.shift_vertical(x, 4)
        self.assertTrue(torch.equal(x + 4, xaug))  # should return True
        print("   shift vertical OK: tested", N + len(aug_args), "combinations of input arguments")

    def test_shift_horizontal(self):
        print("Testing shift horizontal...", end="", flush=True)
        aug_args = {
            "x": [self.x1, self.x2, self.x3, self.x4, self.x1np, self.x2np, self.x3np, self.x4np],
            "Fs": [128],
            "shift_time": [0.5, 1, 2.0],
            "forward": [None, True, False],
            "random_shift": [False, True],
            "batch_equal": [True, False],
        }
        aug_args = self.makeGrid(aug_args)
        for i in aug_args:
            # change batch equal to avoid function print
            if not (i["batch_equal"]):
                if not (i["random_shift"] or (i["forward"] is None)):
                    i["batch_equal"] = True
            xaug = aug.shift_horizontal(**i)
            if isinstance(xaug, torch.Tensor):
                self.assertTrue(torch.isnan(xaug).sum() == 0)
                self.assertFalse(torch.equal(i["x"], xaug))
            else:
                self.assertTrue(np.isnan(xaug).sum() == 0)
                self.assertFalse(np.array_equal(i["x"], xaug))
        N = len(aug_args)
        if self.device.type != "cpu":
            aug_args = {
                "x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu],
                "Fs": [128],
                "shift_time": [0.5, 1, 2.0],
                "forward": [None, True, False],
                "random_shift": [False, True],
                "batch_equal": [True, False],
            }
            aug_args = self.makeGrid(aug_args)
            for i in aug_args:
                if not (i["batch_equal"]):
                    if not (i["random_shift"] or (i["forward"] is None)):
                        i["batch_equal"] = True
                xaug = aug.shift_horizontal(**i)

        xaug = aug.shift_horizontal(self.x4, 64, 1, True)
        self.assertTrue(xaug[..., 0:64].sum() == 0)
        self.assertFalse(xaug[..., 65].sum() == 0)
        print(
            "   shift vertical OK: tested", N + len(aug_args) + 1, "combinations of input arguments"
        )

    def test_shift_frequency(self):
        print("Testing shift frequency...", end="", flush=True)
        aug_args = {
            "x": [self.x1, self.x2, self.x3, self.x4, self.x1np, self.x2np, self.x3np, self.x4np],
            "Fs": [128],
            "shift_freq": [1.35, 2, 4.12],
            "forward": [None, True, False],
            "random_shift": [False, True],
            "batch_equal": [True, False],
        }
        aug_args = self.makeGrid(aug_args)
        for i in aug_args:
            if not (i["batch_equal"]):
                if not (i["random_shift"] or (i["forward"] is None)):
                    i["batch_equal"] = True
            xaug = aug.shift_frequency(**i)
            if isinstance(xaug, torch.Tensor):
                self.assertTrue(torch.isnan(xaug).sum() == 0)
                self.assertFalse(torch.equal(i["x"], xaug))
            else:
                self.assertTrue(np.isnan(xaug).sum() == 0)
                self.assertFalse(np.array_equal(i["x"], xaug))
        N = len(aug_args)
        if (self.device.type != "cpu") and (self.device.type != "mps"):
            aug_args = {
                "x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu],
                "Fs": [128],
                "shift_freq": [1.35, 2, 4.12],
                "forward": [None, True, False],
                "random_shift": [False, True],
                "batch_equal": [True, False],
            }
            aug_args = self.makeGrid(aug_args)
            for i in aug_args:
                if not (i["batch_equal"]):
                    if not (i["random_shift"] or (i["forward"] is None)):
                        i["batch_equal"] = True
                xaug = aug.shift_frequency(**i)

        Fs = 128
        x = torch.zeros(16, 32, 1024)
        x += torch.sin(torch.linspace(0, 48 * torch.pi, 1024))
        x = x + torch.sin(torch.linspace(0, 8 * torch.pi, 1024))
        xaug = aug.shift_frequency(x, 10, Fs, True)
        f, per1 = periodogram(x[0, 0], fs=Fs)
        per2 = periodogram(xaug[0, 0], fs=Fs)[1]
        self.assertTrue(math.isclose(per1[4], per2[84], rel_tol=1e-5))
        self.assertTrue(math.isclose(per1[24], per2[104], rel_tol=1e-5))
        print("   shift frequency OK: tested", N + len(aug_args), "combinations of input arguments")

    def test_flip_vertical(self):
        print("Testing flip vertical...", end="", flush=True)
        aug_args = {
            "x": [self.x1, self.x2, self.x3, self.x4, self.x1np, self.x2np, self.x3np, self.x4np]
        }
        aug_args = self.makeGrid(aug_args)
        for i in aug_args:
            xaug = aug.flip_vertical(**i)
            if isinstance(xaug, torch.Tensor):
                self.assertTrue(torch.isnan(xaug).sum() == 0)
                self.assertFalse(torch.equal(i["x"], xaug))
            else:
                self.assertTrue(np.isnan(xaug).sum() == 0)
                self.assertFalse(np.array_equal(i["x"], xaug))
        N = len(aug_args)
        if self.device.type != "cpu":
            aug_args = {"x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu]}
            aug_args = self.makeGrid(aug_args)
            for i in aug_args:
                xaug = aug.flip_vertical(**i)

        x = torch.zeros(16, 32, 1024)
        x += torch.sin(torch.linspace(0, 8 * np.pi, 1024))
        xaug = aug.flip_vertical(x)
        self.assertTrue(torch.equal(xaug, x * (-1)))  # should return True
        print("   flip vertical OK: tested", N + len(aug_args), "combinations of input arguments")

    def test_flip_horizontal(self):
        print("Testing flip horizontal...", end="", flush=True)
        aug_args = {
            "x": [self.x1, self.x2, self.x3, self.x4, self.x1np, self.x2np, self.x3np, self.x4np]
        }
        aug_args = self.makeGrid(aug_args)
        for i in aug_args:
            xaug = aug.flip_horizontal(**i)
            if isinstance(xaug, torch.Tensor):
                self.assertTrue(torch.isnan(xaug).sum() == 0)
                self.assertFalse(torch.equal(i["x"], xaug))
            else:
                self.assertTrue(np.isnan(xaug).sum() == 0)
                self.assertFalse(np.array_equal(i["x"], xaug))
        N = len(aug_args)
        if self.device.type != "cpu":
            aug_args = {"x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu]}
            aug_args = self.makeGrid(aug_args)
            for i in aug_args:
                xaug = aug.flip_horizontal(**i)

        x = torch.zeros(16, 32, 1024)
        x += torch.sin(torch.linspace(0, 8 * torch.pi, 1024))
        xaug = aug.flip_horizontal(x)
        self.assertTrue(torch.equal(xaug, torch.flip(x, [len(x.shape) - 1])))
        print("   flip horizontal OK: tested", N + len(aug_args), "combinations of input arguments")

    def test_gaussian_noise(self):
        print("Testing gaussian noise...", end="", flush=True)
        aug_args = {
            "x": [self.x1, self.x2, self.x3, self.x4, self.x1np, self.x2np, self.x3np, self.x4np],
            "mean": [0, 1, 2.5],
            "std": [1.35, 2, 0.72],
        }
        aug_args = self.makeGrid(aug_args)
        for i in aug_args:
            xaug = aug.add_gaussian_noise(**i)
            if isinstance(xaug, torch.Tensor):
                self.assertTrue(torch.isnan(xaug).sum() == 0)
                self.assertFalse(torch.equal(i["x"], xaug))
            else:
                self.assertTrue(np.isnan(xaug).sum() == 0)
                self.assertFalse(np.array_equal(i["x"], xaug))
        N = len(aug_args)
        if self.device.type != "cpu":
            aug_args = {
                "x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu],
                "mean": [0, 1, 2.5],
                "std": [1.35, 2, 0.72],
            }
            aug_args = self.makeGrid(aug_args)
            for i in aug_args:
                xaug = aug.add_gaussian_noise(**i)
        x = torch.zeros(16, 32, 1024)
        x += torch.sin(torch.linspace(0, 8 * torch.pi, 1024))
        xaug, noise = aug.add_gaussian_noise(x, 0.1, get_noise=True)
        self.assertTrue(math.isclose(noise.std(), 0.1, rel_tol=1e-2))
        self.assertTrue(math.isclose(xaug.mean(), 0, rel_tol=1e-4, abs_tol=1e-3))
        print("   gaussian noise OK: tested", N + len(aug_args), "combinations of input arguments")

    def test_add_noise_SNR(self):
        print("Testing noise SNR...", end="", flush=True)
        aug_args = {
            "x": [self.x1, self.x2, self.x3, self.x4, self.x1np, self.x2np, self.x3np, self.x4np],
            "target_snr": [1, 2, 5],
        }
        aug_args = self.makeGrid(aug_args)
        for i in aug_args:
            xaug = aug.add_noise_SNR(**i)
            if isinstance(xaug, torch.Tensor):
                self.assertTrue(torch.isnan(xaug).sum() == 0)
                self.assertFalse(torch.equal(i["x"], xaug))
            else:
                self.assertTrue(np.isnan(xaug).sum() == 0)
                self.assertFalse(np.array_equal(i["x"], xaug))
        N = len(aug_args)
        if self.device.type != "cpu":
            aug_args = {
                "x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu],
                "target_snr": [1, 2, 5],
            }
            aug_args = self.makeGrid(aug_args)
            for i in aug_args:
                xaug = aug.add_noise_SNR(**i)

        x = torch.zeros(16, 32, 1024) + torch.sin(torch.linspace(0, 8 * np.pi, 1024))
        xaug, noise = aug.add_noise_SNR(x, 10, get_noise=True)
        SNR = 10 * np.log10(((x**2).sum().mean()) / ((noise**2).sum().mean()))
        self.assertTrue(math.isclose(SNR, 10, rel_tol=1e-2))
        print("   noise SNR OK: tested", N + len(aug_args), "combinations of input arguments")

    def test_add_band_noise(self):
        print("Testing band noise...", end="", flush=True)
        aug_args = {
            "x": [self.x1, self.x2, self.x3, self.x4, self.x1np, self.x2np, self.x3np, self.x4np],
            "bandwidth": [
                ["theta", "gamma"],
                [(1, 10), (15, 18)],
                [4, 50],
                50,
                ["theta", (10, 20), 50],
            ],
            "samplerate": [128],
            "noise_range": [None, 2, 1.5],
            "std": [None, 1.4, 1.23],
        }
        aug_args = self.makeGrid(aug_args)
        for i in aug_args:
            xaug = aug.add_band_noise(**i)
            if isinstance(xaug, torch.Tensor):
                self.assertTrue(torch.isnan(xaug).sum() == 0)
                self.assertFalse(torch.equal(i["x"], xaug))
            else:
                self.assertTrue(np.isnan(xaug).sum() == 0)
                self.assertFalse(np.array_equal(i["x"], xaug))
        N = len(aug_args)
        if self.device.type != "cpu":
            aug_args = {
                "x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu],
                "bandwidth": [
                    ["theta", "gamma"],
                    [(1, 10), (15, 18)],
                    [4, 50],
                    50,
                    ["theta", (10, 20), 50],
                ],
                "samplerate": [128],
                "noise_range": [None, 2, 1.5],
                "std": [None, 1.4, 1.23],
            }
            aug_args = self.makeGrid(aug_args)
            for i in aug_args:
                xaug = aug.add_band_noise(**i)
        x = torch.zeros(16, 32, 1024)
        x += torch.sin(torch.linspace(0, 8 * torch.pi, 1024))
        xaug, noise = aug.add_band_noise(x, "beta", 128, noise_range=0.2, get_noise=True)
        f, per = periodogram(noise, 128)
        index = np.where(per > 1e-12)[0]
        self.assertTrue(len(np.where(((f[index] < 13) | (f[index] > 30)))[0]) == 0)
        print("   band noise OK: tested", N + len(aug_args), "combinations of input arguments")

    def test_scaling(self):
        print("Testing scaling...", end="", flush=True)
        aug_args = {
            "x": [self.x1, self.x2, self.x3, self.x4, self.x1np, self.x2np, self.x3np, self.x4np],
            "value": [None, 1.5, 2, 0.5],
            "batch_equal": [True, False],
        }
        aug_args = self.makeGrid(aug_args)
        for i in aug_args:
            xaug = aug.scaling(**i)
            if isinstance(xaug, torch.Tensor):
                self.assertTrue(torch.isnan(xaug).sum() == 0)
                self.assertFalse(torch.equal(i["x"], xaug))
            else:
                self.assertTrue(np.isnan(xaug).sum() == 0)
                self.assertFalse(np.array_equal(i["x"], xaug))
        N = len(aug_args)
        if self.device.type != "cpu":
            aug_args = {
                "x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu],
                "value": [None, 1.5, 2, 0.5],
                "batch_equal": [True, False],
            }
            aug_args = self.makeGrid(aug_args)
            for i in aug_args:
                xaug = aug.scaling(**i)
        x = torch.zeros(16, 32, 1024)
        x += torch.sin(torch.linspace(0, 8 * torch.pi, 1024))
        xaug = aug.scaling(x, 1.5)
        self.assertTrue(xaug.max() == x.max() * 1.5)  # should return True
        self.assertTrue(xaug.min() == x.min() * 1.5)  # should return True
        print("   scaling OK: tested", N + len(aug_args), "combinations of input arguments")

    def test_random_slope_scale(self):
        print("Testing random slope scale...", end="", flush=True)
        aug_args = {
            "x": [self.x1, self.x2, self.x3, self.x4, self.x1np, self.x2np, self.x3np, self.x4np],
            "min_scale": [0.7, 0.9],
            "max_scale": [1.2, 1.5],
            "batch_equal": [True, False],
            "keep_memory": [True, False],
        }
        aug_args = self.makeGrid(aug_args)
        for i in aug_args:
            if i["batch_equal"] and len(i["x"].shape) < 2:
                i["batch_equal"] = False
            xaug = aug.random_slope_scale(**i)
            if isinstance(xaug, torch.Tensor):
                self.assertTrue(torch.isnan(xaug).sum() == 0)
                self.assertFalse(torch.equal(i["x"], xaug))
            else:
                self.assertTrue(np.isnan(xaug).sum() == 0)
                self.assertFalse(np.array_equal(i["x"], xaug))
        N = len(aug_args)
        if self.device.type != "cpu":
            aug_args = {
                "x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu],
                "min_scale": [0.7, 0.9],
                "max_scale": [1.2, 1.5],
                "batch_equal": [True, False],
                "keep_memory": [True, False],
            }
            aug_args = self.makeGrid(aug_args)
            for i in aug_args:
                if i["batch_equal"] and len(i["x"].shape) < 2:
                    i["batch_equal"] = False
                xaug = aug.random_slope_scale(**i)

        x = torch.zeros(16, 32, 1024)
        x += torch.sin(torch.linspace(0, 8 * torch.pi, 1024))
        xaug = aug.random_slope_scale(x)
        diff1 = torch.abs(xaug[0, 0, 1:] - xaug[0, 0, :-1])
        diff2 = torch.abs(x[0, 0, 1:] - x[0, 0, :-1])
        self.assertEqual(
            torch.logical_or(diff1 <= (diff2 * 1.2), diff1 >= (diff2 * 0.9)).sum(), 1023
        )
        print(
            "   random slope scale OK: tested", N + len(aug_args), "combinations of input arguments"
        )

    def test_random_FT_phase(self):
        print("Testing random FT phase...", end="", flush=True)
        aug_args = {
            "x": [self.x1, self.x2, self.x3, self.x4, self.x1np, self.x2np, self.x3np, self.x4np],
            "value": [0.2, 0.5, 0.75],
            "batch_equal": [True, False],
        }
        aug_args = self.makeGrid(aug_args)
        for i in aug_args:
            xaug = aug.random_FT_phase(**i)
            if isinstance(xaug, torch.Tensor):
                self.assertTrue(torch.isnan(xaug).sum() == 0)
                self.assertFalse(torch.equal(i["x"], xaug))
            else:
                self.assertTrue(np.isnan(xaug).sum() == 0)
                self.assertFalse(np.array_equal(i["x"], xaug))
        N = len(aug_args)
        if (self.device.type != "cpu") and (self.device.type != "mps"):
            aug_args = {
                "x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu],
                "value": [0.2, 0.5, 0.75],
                "batch_equal": [True, False],
            }
            aug_args = self.makeGrid(aug_args)
            for i in aug_args:
                xaug = aug.random_FT_phase(**i)
        x = torch.zeros(16, 32, 1024)
        x += torch.sin(torch.linspace(0, 8 * torch.pi, 1024))
        xaug = aug.random_FT_phase(x, 0.8)
        phase_shift = torch.arccos(2 * ((x[0, 0, 0:512] * xaug[0, 0, :512]).mean()))
        a = torch.sin(torch.linspace(0, 8 * torch.pi, 1024) + phase_shift)
        if (a[0] - xaug[0, 0, 0]).abs() > 0.1:
            a = torch.sin(torch.linspace(0, 8 * torch.pi, 1024) - phase_shift)
        self.assertTrue((a - xaug[0, 0]).mean() < 1e-3)
        print("   random FT phase OK: tested", N + len(aug_args), "combinations of input arguments")

    def test_moving_average(self):
        print("Testing moving average...", end="", flush=True)
        aug_args = {
            "x": [self.x1, self.x2, self.x3, self.x4, self.x1np, self.x2np, self.x3np, self.x4np],
            "order": [3, 5, 9],
        }
        aug_args = self.makeGrid(aug_args)
        for i in aug_args:
            xaug = aug.moving_avg(**i)
            if isinstance(xaug, torch.Tensor):
                self.assertTrue(torch.isnan(xaug).sum() == 0)
                self.assertFalse(torch.equal(i["x"], xaug))
            else:
                self.assertTrue(np.isnan(xaug).sum() == 0)
                self.assertFalse(np.array_equal(i["x"], xaug))
        N = len(aug_args)
        if self.device.type != "cpu":
            aug_args = {"x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu], "order": [3, 5, 9]}
            aug_args = self.makeGrid(aug_args)
            for i in aug_args:
                xaug = aug.moving_avg(**i)

        x = torch.randn(16, 32, 1024)
        xaug = aug.moving_avg(x, 5)
        self.assertTrue(math.isclose(x[0, 0, 5 : 5 + 5].sum() / 5, xaug[0, 0, 7], rel_tol=1e-5))
        print("   moving average OK: tested", N + len(aug_args), "combinations of input arguments")

    def test_filter_lowpass(self):
        print("Testing lowpass filter...", end="", flush=True)
        aug_args = {
            "x": [self.x1, self.x2, self.x3, self.x4, self.x1np, self.x2np, self.x3np, self.x4np],
            "Fs": [128, 256],
            "Wp": [30],
            "Ws": [50],
            "rp": [-20 * np.log10(0.90)],
            "rs": [-20 * np.log10(0.15)],
            "filter_type": ["butter", "ellip", "cheby1", "cheby2"],
        }
        aug_args = self.makeGrid(aug_args)
        for i in aug_args:
            xaug = aug.filter_lowpass(**i)
            if isinstance(xaug, torch.Tensor):
                self.assertTrue(torch.isnan(xaug).sum() == 0)
                self.assertFalse(torch.equal(i["x"], xaug))
            else:
                self.assertTrue(np.isnan(xaug).sum() == 0)
                self.assertFalse(np.array_equal(i["x"], xaug))
            self.assertTrue((xaug > 1e2).sum() == 0)
            self.assertTrue((xaug < -1e2).sum() == 0)
        N = len(aug_args)
        if self.device.type != "cpu":
            aug_args = {
                "x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu],
                "Fs": [128, 256],
                "Wp": [30],
                "Ws": [50],
                "rp": [-20 * np.log10(0.95)],
                "rs": [-20 * np.log10(0.15)],
                "filter_type": ["butter", "ellip", "cheby1", "cheby2"],
            }
            aug_args = self.makeGrid(aug_args)
            for i in aug_args:
                xaug = aug.filter_lowpass(**i)
                self.assertTrue(torch.isnan(xaug).sum() == 0)
                self.assertFalse(torch.equal(i["x"], xaug))
                self.assertTrue((xaug > 1e2).sum() == 0)
                self.assertTrue((xaug < -1e2).sum() == 0)

        x = torch.zeros(16, 32, 1024)
        x += torch.sin(torch.linspace(0, 8 * torch.pi, 1024))
        x += torch.sin(torch.linspace(0, 48 * 2 * torch.pi, 1024))
        x += torch.sin(torch.linspace(0, 256 * 2 * torch.pi, 1024))
        f, per1 = periodogram(x[0, 0], 128)
        xaug = aug.filter_lowpass(x, 128, 20, 30)
        f, per2 = periodogram(xaug[0, 0], 128)
        self.assertTrue(np.isclose(np.max(per2[f > 30]), 0, rtol=1e-04, atol=1e-04))
        print("   lowpass filter OK: tested", N + len(aug_args), "combinations of input arguments")

    def test_filter_highpass(self):
        print("Testing highpass filter...", end="", flush=True)
        aug_args = {
            "x": [self.x1, self.x2, self.x3, self.x4, self.x1np, self.x2np, self.x3np, self.x4np],
            "Fs": [128, 256],
            "Wp": [40],
            "Ws": [20],
            "rp": [-20 * np.log10(0.9)],
            "rs": [-20 * np.log10(0.15)],
            "filter_type": ["butter", "ellip", "cheby1", "cheby2"],
        }
        aug_args = self.makeGrid(aug_args)
        for i in aug_args:
            xaug = aug.filter_highpass(**i)
            if isinstance(xaug, torch.Tensor):
                self.assertTrue(torch.isnan(xaug).sum() == 0)
                self.assertFalse(torch.equal(i["x"], xaug))
            else:
                self.assertTrue(np.isnan(xaug).sum() == 0)
                self.assertFalse(np.array_equal(i["x"], xaug))
            self.assertTrue((xaug > 1e2).sum() == 0)
            self.assertTrue((xaug < -1e2).sum() == 0)
        N = len(aug_args)
        if self.device.type != "cpu":
            aug_args = {
                "x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu],
                "Fs": [128, 256],
                "Wp": [40],
                "Ws": [20],
                "rp": [-20 * np.log10(0.95)],
                "rs": [-20 * np.log10(0.15)],
                "filter_type": ["butter", "ellip", "cheby1", "cheby2"],
            }
            aug_args = self.makeGrid(aug_args)
            for i in aug_args:
                xaug = aug.filter_highpass(**i)
                self.assertTrue(torch.isnan(xaug).sum() == 0)
                self.assertFalse(torch.equal(i["x"], xaug))
                self.assertTrue((xaug > 1e2).sum() == 0)
                self.assertTrue((xaug < -1e2).sum() == 0)

        x = torch.zeros(16, 32, 1024)
        x += torch.sin(torch.linspace(0, 8 * torch.pi, 1024))
        x += torch.sin(torch.linspace(0, 48 * 2 * torch.pi, 1024))
        x += torch.sin(torch.linspace(0, 256 * 2 * torch.pi, 1024))
        f, per1 = periodogram(x[0, 0], 128)
        xaug = aug.filter_highpass(x, 128, 20, 30)
        f, per2 = periodogram(xaug[0, 0], 128)
        self.assertTrue(np.isclose(np.max(per2[f < 20]), 0, rtol=1e-04, atol=1e-04))
        print("   highpass filter OK: tested", N + len(aug_args), "combinations of input arguments")

    def test_filter_bandpass(self):
        print("Testing bandpass filter...", end="", flush=True)
        aug_args = {
            "x": [self.x1, self.x2, self.x3, self.x4, self.x1np, self.x2np, self.x3np, self.x4np],
            "Fs": [128, 256],
            "eeg_band": ["delta", "alpha", "beta", "gamma", "gamma_low"],
            "filter_type": ["butter", "ellip", "cheby1", "cheby2"],
        }
        aug_args = self.makeGrid(aug_args)
        for i in aug_args:
            xaug = aug.filter_bandpass(**i)
            if isinstance(xaug, torch.Tensor):
                self.assertTrue(torch.isnan(xaug).sum() == 0)
                self.assertFalse(torch.equal(i["x"], xaug))
            else:
                self.assertTrue(np.isnan(xaug).sum() == 0)
                self.assertFalse(np.array_equal(i["x"], xaug))
            self.assertTrue((xaug > 1e2).sum() == 0)
            self.assertTrue((xaug < -1e2).sum() == 0)
        N = len(aug_args)
        if self.device.type != "cpu":
            aug_args = {
                "x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu],
                "Fs": [128, 256],
                "eeg_band": [None, "delta", "alpha", "beta", "gamma", "gamma_low"],
                "filter_type": ["butter", "ellip", "cheby1", "cheby2"],
            }
            aug_args = self.makeGrid(aug_args)
            for i in aug_args:
                xaug = aug.filter_bandpass(**i)
                self.assertTrue(torch.isnan(xaug).sum() == 0)
                self.assertFalse(torch.equal(i["x"], xaug))
                self.assertTrue((xaug > 1e2).sum() == 0)
                self.assertTrue((xaug < -1e2).sum() == 0)

        x = torch.zeros(16, 32, 1024)
        x += torch.sin(torch.linspace(0, 8 * torch.pi, 1024))
        x += torch.sin(torch.linspace(0, 48 * 2 * torch.pi, 1024))
        x += torch.sin(torch.linspace(0, 256 * 2 * torch.pi, 1024))
        f, per1 = periodogram(x[0, 0], 128)
        xaug = aug.filter_bandpass(x, 128, [13, 22], [5, 27])
        f, per2 = periodogram(xaug[0, 0], 128)
        self.assertTrue(np.isclose(np.max(per2[f < 5]), 0, rtol=1e-04, atol=1e-04))
        self.assertTrue(np.isclose(np.max(per2[f > 27]), 0, rtol=1e-04, atol=1e-04))
        print("   bandpass filter OK: tested", N + len(aug_args), "combinations of input arguments")

    def test_filter_bandstop(self):
        print("Testing bandstop filter...", end="", flush=True)
        aug_args = {
            "x": [self.x1, self.x2, self.x3, self.x4, self.x1np, self.x2np, self.x3np, self.x4np],
            "Fs": [128, 256],
            "eeg_band": [None, "delta", "theta", "alpha", "beta", "gamma", "gamma_low"],
            "filter_type": ["butter", "ellip", "cheby1", "cheby2"],
        }
        aug_args = self.makeGrid(aug_args)
        for i in aug_args:
            xaug = aug.filter_bandstop(**i)
            if isinstance(xaug, torch.Tensor):
                self.assertTrue(torch.isnan(xaug).sum() == 0)
                self.assertFalse(torch.equal(i["x"], xaug))
            else:
                self.assertTrue(np.isnan(xaug).sum() == 0)
                self.assertFalse(np.array_equal(i["x"], xaug))
            self.assertTrue((xaug > 1e2).sum() == 0)
            self.assertTrue((xaug < -1e2).sum() == 0)
        N = len(aug_args)
        if self.device.type != "cpu":
            aug_args = {
                "x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu],
                "Fs": [128, 256],
                "eeg_band": [None, "delta", "theta", "alpha", "beta", "gamma", "gamma_low"],
                "filter_type": ["butter", "ellip", "cheby1", "cheby2"],
            }
            aug_args = self.makeGrid(aug_args)
            for i in aug_args:
                xaug = aug.filter_bandstop(**i)
                self.assertTrue(torch.isnan(xaug).sum() == 0)
                self.assertFalse(torch.equal(i["x"], xaug))
                self.assertTrue((xaug > 1e2).sum() == 0)
                self.assertTrue((xaug < -1e2).sum() == 0)
        x = torch.zeros(16, 32, 1024)
        x += torch.sin(torch.linspace(0, 8 * torch.pi, 1024))
        x += torch.sin(torch.linspace(0, 48 * 2 * torch.pi, 1024))
        x += torch.sin(torch.linspace(0, 256 * 2 * torch.pi, 1024))
        f, per1 = periodogram(x[0, 0], 128)
        xaug = aug.filter_bandpass(x, 128, [13, 22], [5, 27])
        f, per2 = periodogram(xaug[0, 0], 128)
        self.assertTrue(
            np.isclose(np.max(per2[np.logical_and(f > 5, f < 27)]), 0, rtol=1e-04, atol=1e-04)
        )
        print("   bandstop filter OK: tested", N + len(aug_args), "combinations of input arguments")

    def test_permute_channels(self):
        print("Testing permute channels...", end="", flush=True)
        channel_map = [
            "FP1",
            "AF3",
            "F1",
            "F3",
            "FC5",
            "FC3",
            "FC1",
            "C1",
            "C5",
            "TP7",
            "CP5",
            "CP3",
            "CP1",
            "P7",
            "PO7",
            "POZ",
            "PZ",
            "FPZ",
            "FP2",
            "AFZ",
            "FZ",
            "F2",
            "F4",
            "F6",
            "FT8",
            "C4",
            "T8",
            "TP8",
            "CP6",
            "CP4",
            "CP2",
            "PO8",
        ]
        aug_args = {
            "x": [self.x2, self.x3, self.x4, self.x2np, self.x3np, self.x4np],
            "chan2shuf": [-1, 5, 10],
            "mode": ["random", "network"],
            "chan_net": ["DMN", "FPN", ["DMN", "FPN"], "all"],
            "batch_equal": [True, False],
            "channel_map": [channel_map],
        }
        aug_args = self.makeGrid(aug_args)
        for i in aug_args:
            i["x"] = i["x"]
            if isinstance(i["x"], torch.Tensor):
                i["x"] += torch.randn(32, 1)
            else:
                i["x"] += np.random.randn(32, 1)
            xaug = aug.permute_channels(**i)
            if isinstance(xaug, torch.Tensor):
                self.assertTrue(torch.isnan(xaug).sum() == 0)
                if torch.equal(i["x"], xaug):
                    print(i["x"][:, 10])
                    print(xaug[:, 10])
                    print(i)
                self.assertFalse(torch.equal(i["x"], xaug))
            else:
                self.assertTrue(np.isnan(xaug).sum() == 0)
                self.assertFalse(np.array_equal(i["x"], xaug))
        N = len(aug_args)
        if self.device.type != "cpu":
            aug_args = {
                "x": [self.x2gpu, self.x3gpu, self.x4gpu],
                "chan2shuf": [-1, 5, 10],
                "mode": ["random", "network"],
                "chan_net": ["DMN", "FPN", ["DMN", "FPN"], "all"],
                "batch_equal": [True, False],
                "channel_map": [channel_map],
            }
            aug_args = self.makeGrid(aug_args)
            for i in aug_args:
                i["x"] = i["x"] + torch.randn(32, 1).to(device=self.device)
                xaug = aug.permute_channels(**i)
        x = torch.zeros(61, 4) + torch.arange(61).reshape(61, 1)
        xaug = aug.permute_channels(x, 10)
        self.assertEqual((x[:, 0] != xaug[:, 0]).sum(), 10)
        eeg1010, chan_net = aug.get_channel_map_and_networks(chan_net=["DMN", "VFN"])
        chan2per = np.union1d(chan_net[0], chan_net[1])
        a = np.intersect1d(eeg1010, chan2per, return_indices=True)[1]
        b = torch.from_numpy(np.delete(np.arange(61), a))
        xaug2 = aug.permute_channels(x, 50, mode="network", chan_net=["DMN", "VFN"])
        self.assertTrue(((x[:, 0] != xaug2[:, 0]).sum()) == 50)
        self.assertTrue(((x[b, 0] == xaug2[b, 0]).sum()) == len(b))
        print(
            "   permute channels OK: tested", N + len(aug_args), "combinations of input arguments"
        )

    def test_permutation_signal(self):
        print("Testing permute signal...", end="", flush=True)
        aug_args = {
            "x": [self.x1, self.x2, self.x3, self.x4, self.x1np, self.x2np, self.x3np, self.x4np],
            "segments": [10, 15, 20],
            "seg_to_per": [-1, 2, 5, 8],
            "batch_equal": [True, False],
        }
        aug_args = self.makeGrid(aug_args)
        for i in aug_args:
            xaug = aug.permutation_signal(**i)
            if isinstance(xaug, torch.Tensor):
                self.assertTrue(torch.isnan(xaug).sum() == 0)
                self.assertFalse(torch.equal(i["x"], xaug))
            else:
                self.assertTrue(np.isnan(xaug).sum() == 0)
                self.assertFalse(np.array_equal(i["x"], xaug))
        N = len(aug_args)
        if self.device.type != "cpu":
            aug_args = {
                "x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu],
                "segments": [10, 15, 20],
                "seg_to_per": [-1, 2, 5, 8],
                "batch_equal": [True, False],
            }
            aug_args = self.makeGrid(aug_args)
            for i in aug_args:
                xaug = aug.permutation_signal(**i)
        torch.manual_seed(1234)
        x = torch.ones(16, 32, 1024) * 2
        x += torch.sin(torch.linspace(0, 8 * torch.pi, 1024))
        xaug = aug.masking(x, 3, 0.5)
        self.assertTrue(
            torch.isclose(
                ((xaug[0, 0] == 0).sum() / len(xaug[0, 0])),
                torch.tensor([0.5]),
                rtol=1e-8,
                atol=1e-8,
            )
        )
        a = xaug[0, 0] == 0
        self.assertTrue((a[:-1].ne(a[1:])).sum() == 6)  # should return True
        print("   permute signal OK: tested", N + len(aug_args), "combinations of input arguments")

    def test_warp_signal(self):
        print("Testing warp signal...", end="", flush=True)
        aug_args = {
            "x": [self.x1, self.x2, self.x3, self.x4, self.x1np, self.x2np, self.x3np, self.x4np],
            "segments": [15],
            "stretch_strength": [2, 1.5],
            "squeeze_strength": [0.4, 0.8],
            "batch_equal": [True, False],
        }
        aug_args = self.makeGrid(aug_args)
        for i in aug_args:
            xaug = aug.warp_signal(**i)
            if isinstance(xaug, torch.Tensor):
                self.assertTrue(torch.isnan(xaug).sum() == 0)
                self.assertFalse(torch.equal(i["x"], xaug))
            else:
                self.assertTrue(np.isnan(xaug).sum() == 0)
                self.assertFalse(np.array_equal(i["x"], xaug))
        N = len(aug_args)
        if self.device.type != "cpu":
            aug_args = {
                "x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu],
                "segments": [15],
                "stretch_strength": [2, 1.5],
                "squeeze_strength": [0.4, 0.8],
                "batch_equal": [True, False],
            }
            aug_args = self.makeGrid(aug_args)
            for i in aug_args:
                xaug = aug.warp_signal(**i)
        print("   warp signal OK: tested", N + len(aug_args), "combinations of input arguments")

    def test_crop_and_resize(self):
        print("Testing crop and resize...", end="", flush=True)
        aug_args = {
            "x": [self.x2, self.x3, self.x4, self.x2np, self.x3np, self.x4np],
            "segments": [15],
            "N_cut": [1, 5],
            "batch_equal": [True, False],
        }
        aug_args = self.makeGrid(aug_args)
        for i in aug_args:
            xaug = aug.crop_and_resize(**i)
            if isinstance(xaug, torch.Tensor):
                self.assertTrue(torch.isnan(xaug).sum() == 0)
                self.assertFalse(torch.equal(i["x"], xaug))
            else:
                self.assertTrue(np.isnan(xaug).sum() == 0)
                self.assertFalse(np.array_equal(i["x"], xaug))
        N = len(aug_args)
        if self.device.type != "cpu":
            aug_args = {
                "x": [self.x2gpu, self.x3gpu, self.x4gpu],
                "segments": [15],
                "N_cut": [1, 5],
                "batch_equal": [True, False],
            }
            aug_args = self.makeGrid(aug_args)
            for i in aug_args:
                xaug = aug.crop_and_resize(**i)
        print("   crop and resize OK: tested", N + len(aug_args), "combinations of input arguments")

    def test_change_ref(self):
        print("Testing change reference...", end="", flush=True)
        aug_args = {
            "x": [self.x2, self.x3, self.x4, self.x2np, self.x3np, self.x4np],
            "mode": ["chan", "avg"],
            "reference": [None, 5],
            "exclude_from_ref": [None, 9, [9, 10]],
        }
        aug_args = self.makeGrid(aug_args)
        for i in aug_args:
            xaug = aug.change_ref(**i)
            if isinstance(xaug, torch.Tensor):
                self.assertTrue(torch.isnan(xaug).sum() == 0)
                self.assertFalse(torch.equal(i["x"], xaug))
            else:
                self.assertTrue(np.isnan(xaug).sum() == 0)
                self.assertFalse(np.array_equal(i["x"], xaug))
        N = len(aug_args)
        if self.device.type != "cpu":
            aug_args = {
                "x": [self.x2gpu, self.x3gpu, self.x4gpu],
                "mode": ["chan", "avg"],
                "reference": [None, 5],
                "exclude_from_ref": [None, 9, [9, 10]],
            }
            aug_args = self.makeGrid(aug_args)
            for i in aug_args:
                xaug = aug.change_ref(**i)

        torch.manual_seed(1234)
        x = torch.zeros(16, 32, 1024)
        x += torch.sin(torch.linspace(0, 8 * torch.pi, 1024))
        x[:, 0, :] = 0.0
        xaug = aug.change_ref(x, "channel", 5)
        self.assertFalse(x[0, 0].max() != 0 and x[0, 0].min() != 0)  # should return False
        self.assertTrue(
            (xaug[0, [i for i in range(1, 32)]].min().item() == 0 and xaug[0, 0].min().item() != 0)
        )  # should return True
        print(
            "   change refeference OK: tested", N + len(aug_args), "combinations of input arguments"
        )

    def test_masking(self):
        print("Testing masking...", end="", flush=True)
        aug_args = {
            "x": [self.x1, self.x2, self.x3, self.x4, self.x1np, self.x2np, self.x3np, self.x4np],
            "mask_number": [1, 2, 4],
            "masked_ratio": [0.1, 0.2, 0.4],
            "batch_equal": [True, False],
        }
        aug_args = self.makeGrid(aug_args)
        for i in aug_args:
            xaug = aug.masking(**i)
            if isinstance(xaug, torch.Tensor):
                self.assertTrue(torch.isnan(xaug).sum() == 0)
                self.assertFalse(torch.equal(i["x"], xaug))
            else:
                self.assertTrue(np.isnan(xaug).sum() == 0)
                self.assertFalse(np.array_equal(i["x"], xaug))
        N = len(aug_args)
        if self.device.type != "cpu":
            aug_args = {
                "x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu],
                "mask_number": [1, 2, 4],
                "masked_ratio": [0.1, 0.2, 0.4],
                "batch_equal": [True, False],
            }
            aug_args = self.makeGrid(aug_args)
            for i in aug_args:
                xaug = aug.masking(**i)

        x = torch.ones(16, 32, 1024) * 2
        x += torch.sin(torch.linspace(0, 8 * torch.pi, 1024))
        xaug = aug.masking(x, 3, 0.5)
        self.assertTrue(
            torch.isclose(
                ((xaug[0, 0] == 0).sum() / len(xaug[0, 0])),
                torch.tensor([0.5]),
                rtol=1e-6,
                atol=1e-8,
            )
        )
        print("   masking OK: tested", N + len(aug_args), "combinations of input arguments")

    def test_channel_dropout(self):
        print("Testing channel dropout...", end="", flush=True)
        aug_args = {
            "x": [self.x2, self.x3, self.x4, self.x2np, self.x3np, self.x4np],
            "Nchan": [None, 2, 3],
            "batch_equal": [True, False],
        }
        aug_args = self.makeGrid(aug_args)
        for i in aug_args:
            xaug = aug.channel_dropout(**i)
            if isinstance(xaug, torch.Tensor):
                self.assertTrue(torch.isnan(xaug).sum() == 0)
                self.assertFalse(torch.equal(i["x"], xaug))
            else:
                self.assertTrue(np.isnan(xaug).sum() == 0)
                self.assertFalse(np.array_equal(i["x"], xaug))
        N = len(aug_args)
        if self.device.type != "cpu":
            aug_args = {
                "x": [self.x2gpu, self.x3gpu, self.x4gpu],
                "Nchan": [None, 2, 3],
                "batch_equal": [True, False],
            }
            aug_args = self.makeGrid(aug_args)
            for i in aug_args:
                xaug = aug.channel_dropout(**i)
        x = torch.ones(32, 1024) * 2 + torch.sin(torch.linspace(0, 8 * torch.pi, 1024))
        xaug = aug.channel_dropout(x, 3)
        self.assertTrue((xaug[:, 10] == 0).sum() == 3)
        print("   channel dropout OK: tested", N + len(aug_args), "combinations of input arguments")

    def test_eeg_artifact(self):
        print("Testing eeg artifact...", end="", flush=True)
        aug_args = {
            "x": [self.x1, self.x2, self.x3, self.x4, self.x1np, self.x2np, self.x3np, self.x4np],
            "Fs": [128],
            "artifact": [None, "white", "line", "eye", "muscle", "drift", "lost"],
            "amplitude": [None, 1],
            "line_at_60Hz": [True, False],
            "lost_time": [0.5, None],
            "drift_slope": [None, 0.2],
            "batch_equal": [True, False],
        }
        aug_args = self.makeGrid(aug_args)
        for i in aug_args:
            xaug = aug.add_eeg_artifact(**i)
            if isinstance(xaug, torch.Tensor):
                self.assertTrue(torch.isnan(xaug).sum() == 0)
                self.assertFalse(torch.equal(i["x"], xaug))
            else:
                self.assertTrue(np.isnan(xaug).sum() == 0)
                self.assertFalse(np.array_equal(i["x"], xaug))
        N = len(aug_args)
        if self.device.type != "cpu":
            aug_args = {
                "x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu],
                "Fs": [128],
                "artifact": [None, "white", "line", "eye", "muscle", "drift", "lost"],
                "amplitude": [None, 1],
                "line_at_60Hz": [True, False],
                "lost_time": [0.5, None],
                "drift_slope": [None, 0.2],
                "batch_equal": [True, False],
            }
            aug_args = self.makeGrid(aug_args)
            for i in aug_args:
                xaug = aug.add_eeg_artifact(**i)
        print("   eeg artifact OK: tested", N + len(aug_args), "combinations of input arguments")


if __name__ == "__main__":
    unittest.main()
