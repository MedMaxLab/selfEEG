import math
import os
import sys
import unittest

import numpy as np
import torch
from scipy.signal import periodogram

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from EEGself._testtools import get_device, make_grid

from selfeeg import augmentation as aug


def _sine(*shape):
    return torch.zeros(*shape) + torch.sin(torch.linspace(0, 8 * torch.pi, shape[-1]))


class TestAugmentationFunctional(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = get_device(
            probe=lambda dev: aug.add_band_noise(torch.randn(1024, device=dev), "theta", 128)
        )
        # Tensors with 1, 2, 3 and 4 dimensions, plus numpy / gpu counterparts.
        shapes = {1: (1024,), 2: (32, 1024), 3: (2, 32, 1024), 4: (32, 2, 32, 1024)}
        cls._t = {k: _sine(*s) for k, s in shapes.items()}
        cls._np = {k: v.numpy() for k, v in cls._t.items()}
        cls._g = {}
        if cls.device.type != "cpu":
            cls._g = {k: v.clone().to(cls.device) for k, v in cls._t.items()}

    def _inputs(self, dims=(1, 2, 3, 4), numpy=True, gpu=True, mps_ok=True):
        xs = [self._t[d] for d in dims]
        if numpy:
            xs += [self._np[d] for d in dims]
        if gpu and self._g and (mps_ok or self.device.type != "mps"):
            xs += [self._g[d] for d in dims]
        return xs

    def _assert_valid(self, xin, xout, changed=True):
        if isinstance(xout, torch.Tensor):
            self.assertEqual(torch.isnan(xout).sum().item(), 0)
            same = torch.equal(xin, xout)
        else:
            self.assertEqual(np.isnan(xout).sum(), 0)
            same = np.array_equal(xin, xout)
        self.assertEqual(same, not changed)

    @staticmethod
    def _descr(args):
        out = {}
        for k, v in args.items():
            if torch.is_tensor(v):
                out["x"] = f"T{tuple(v.shape)}/{v.device.type}"
            elif isinstance(v, np.ndarray):
                out["x"] = f"np{tuple(v.shape)}"
            else:
                out[k] = v
        return out

    def _smoke(self, fn, params, x_list, changed=True, max_comb=32, prep=None, extra=None):
        grid = make_grid({**params, "x": list(x_list)}, max_comb=max_comb)
        for args in grid:
            if prep is not None:
                prep(args)
            with self.subTest(fn=fn.__name__, **self._descr(args)):
                xaug = fn(**args)
                self._assert_valid(args["x"], xaug, changed)
                if extra is not None:
                    extra(args, xaug)

    @staticmethod
    def _bounded(args, xaug):
        assert int((xaug > 1e2).sum()) == 0 and int((xaug < -1e2).sum()) == 0

    # ------------------------------------------------------------------ tests

    def test_identity(self):
        self._smoke(aug.identity, {}, self._inputs(), changed=False)

    def test_shift_vertical(self):
        self._smoke(aug.shift_vertical, {"value": [1, 2.0, 4]}, self._inputs())
        xaug = aug.shift_vertical(self._t[4], 4)
        self.assertTrue(torch.equal(self._t[4] + 4, xaug))

    def test_shift_horizontal(self):
        def prep(a):
            if not a["batch_equal"] and not (a["random_shift"] or a["forward"] is None):
                a["batch_equal"] = True

        self._smoke(
            aug.shift_horizontal,
            {
                "Fs": [128],
                "shift_time": [0.5, 1, 2.0],
                "forward": [None, True, False],
                "random_shift": [False, True],
                "batch_equal": [True, False],
            },
            self._inputs(),
            prep=prep,
            max_comb=48,
        )
        xaug = aug.shift_horizontal(self._t[4], 64, 1, True)
        self.assertEqual(xaug[..., 0:64].sum(), 0)
        self.assertNotEqual(xaug[..., 65].sum(), 0)

    def test_shift_frequency(self):
        def prep(a):
            if not a["batch_equal"] and not (a["random_shift"] or a["forward"] is None):
                a["batch_equal"] = True

        self._smoke(
            aug.shift_frequency,
            {
                "Fs": [128],
                "shift_freq": [1.35, 2, 4.12],
                "forward": [None, True, False],
                "random_shift": [False, True],
                "batch_equal": [True, False],
            },
            self._inputs(mps_ok=False),
            prep=prep,
            max_comb=48,
        )
        Fs = 128
        x = _sine(16, 32, 1024) + torch.sin(torch.linspace(0, 48 * torch.pi, 1024))
        xaug = aug.shift_frequency(x, 10, Fs, True)
        per1 = periodogram(x[0, 0], fs=Fs)[1]
        per2 = periodogram(xaug[0, 0], fs=Fs)[1]
        self.assertTrue(math.isclose(per1[4], per2[84], rel_tol=1e-5))
        self.assertTrue(math.isclose(per1[24], per2[104], rel_tol=1e-5))

    def test_phase_swap(self):
        self._smoke(aug.phase_swap, {}, self._inputs(dims=(3,)))

    def test_flip_vertical(self):
        self._smoke(aug.flip_vertical, {}, self._inputs())
        xaug = aug.flip_vertical(self._t[4])
        self.assertTrue(torch.equal(xaug, self._t[4] * (-1)))

    def test_flip_horizontal(self):
        self._smoke(aug.flip_horizontal, {}, self._inputs())
        xaug = aug.flip_horizontal(self._t[4])
        self.assertTrue(torch.equal(xaug, torch.flip(self._t[4], [self._t[4].ndim - 1])))

    def test_gaussian_noise(self):
        self._smoke(
            aug.add_gaussian_noise,
            {"mean": [0, 1, 2.5], "std": [1.35, 2, 0.72]},
            self._inputs(),
            max_comb=24,
        )
        x = _sine(16, 32, 1024)
        xaug, noise = aug.add_gaussian_noise(x, 0.1, get_noise=True)
        self.assertTrue(math.isclose(noise.std(), 0.1, rel_tol=1e-2))
        self.assertTrue(math.isclose(xaug.mean(), 0, rel_tol=1e-4, abs_tol=1e-3))

    def test_add_noise_SNR(self):
        self._smoke(aug.add_noise_SNR, {"target_snr": [1, 2, 5]}, self._inputs(), max_comb=24)
        x = _sine(16, 32, 512)
        xaug, noise = aug.add_noise_SNR(x, 10, get_noise=True)
        snr = 10 * torch.log10(((x**2).sum().mean()) / ((noise**2).sum().mean()))
        self.assertTrue(math.isclose(snr, 10, rel_tol=1e-1))

    def test_add_band_noise(self):
        self._smoke(
            aug.add_band_noise,
            {
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
            },
            self._inputs(),
            max_comb=40,
        )
        x = _sine(16, 32, 1024)
        xaug, noise = aug.add_band_noise(x, "beta", 128, noise_range=0.2, get_noise=True)
        f, per = periodogram(noise, 128)
        index = np.where(per > 1e-12)[0]
        self.assertEqual(len(np.where(((f[index] < 13) | (f[index] > 30)))[0]), 0)

    def test_scaling(self):
        self._smoke(
            aug.scaling,
            {"value": [None, 1.5, 2, 0.5], "batch_equal": [True, False]},
            self._inputs(),
            max_comb=24,
        )
        x = _sine(16, 32, 1024)
        xaug = aug.scaling(x, 1.5)
        self.assertEqual(xaug.max(), x.max() * 1.5)
        self.assertEqual(xaug.min(), x.min() * 1.5)

    def test_random_slope_scale(self):
        def prep(a):
            if a["batch_equal"] and len(a["x"].shape) < 2:
                a["batch_equal"] = False

        self._smoke(
            aug.random_slope_scale,
            {
                "min_scale": [0.7, 0.9],
                "max_scale": [1.2, 1.5],
                "batch_equal": [True, False],
                "keep_memory": [True, False],
            },
            self._inputs(),
            prep=prep,
            max_comb=24,
        )
        x = _sine(16, 32, 1024)
        xaug = aug.random_slope_scale(x)
        diff1 = torch.abs(xaug[0, 0, 1:] - xaug[0, 0, :-1])
        diff2 = torch.abs(x[0, 0, 1:] - x[0, 0, :-1])
        self.assertEqual(
            torch.logical_or(diff1 <= (diff2 * 1.2), diff1 >= (diff2 * 0.9)).sum(), 1023
        )

    def test_random_FT_phase(self):
        self._smoke(
            aug.random_FT_phase,
            {"value": [0.2, 0.5, 0.75], "batch_equal": [True, False]},
            self._inputs(mps_ok=False),
            max_comb=24,
        )
        x = _sine(16, 32, 1024)
        xaug = aug.random_FT_phase(x, 0.8)
        phase_shift = torch.arccos(2 * ((x[0, 0, 0:512] * xaug[0, 0, :512]).mean()))
        a = torch.sin(torch.linspace(0, 8 * torch.pi, 1024) + phase_shift)
        if (a[0] - xaug[0, 0, 0]).abs() > 0.1:
            a = torch.sin(torch.linspace(0, 8 * torch.pi, 1024) - phase_shift)
        self.assertTrue((a - xaug[0, 0]).mean() < 1e-3)

    def test_moving_average(self):
        self._smoke(aug.moving_avg, {"order": [3, 5, 9]}, self._inputs(), max_comb=24)
        x = torch.randn(16, 32, 1024)
        xaug = aug.moving_avg(x, 5)
        self.assertTrue(math.isclose(x[0, 0, 5 : 5 + 5].sum() / 5, xaug[0, 0, 7], rel_tol=1e-5))

    def test_filter_lowpass(self):
        self._smoke(
            aug.filter_lowpass,
            {
                "Fs": [128, 256],
                "Wp": [30],
                "Ws": [50],
                "rp": [-20 * np.log10(0.90)],
                "rs": [-20 * np.log10(0.15)],
                "filter_type": ["butter", "ellip", "cheby1", "cheby2"],
            },
            self._inputs(),
            max_comb=32,
            extra=self._bounded,
        )
        x = _sine(16, 32, 1024)
        x += torch.sin(torch.linspace(0, 48 * 2 * torch.pi, 1024))
        x += torch.sin(torch.linspace(0, 256 * 2 * torch.pi, 1024))
        xaug = aug.filter_lowpass(x, 128, 20, 30)
        f, per2 = periodogram(xaug[0, 0], 128)
        self.assertTrue(np.isclose(np.max(per2[f > 30]), 0, rtol=1e-04, atol=1e-04))

    def test_filter_highpass(self):
        self._smoke(
            aug.filter_highpass,
            {
                "Fs": [128, 256],
                "Wp": [40],
                "Ws": [20],
                "rp": [-20 * np.log10(0.9)],
                "rs": [-20 * np.log10(0.15)],
                "filter_type": ["butter", "ellip", "cheby1", "cheby2"],
            },
            self._inputs(),
            max_comb=32,
            extra=self._bounded,
        )
        x = _sine(16, 32, 1024)
        x += torch.sin(torch.linspace(0, 48 * 2 * torch.pi, 1024))
        x += torch.sin(torch.linspace(0, 256 * 2 * torch.pi, 1024))
        xaug = aug.filter_highpass(x, 128, 20, 30)
        f, per2 = periodogram(xaug[0, 0], 128)
        self.assertTrue(np.isclose(np.max(per2[f < 20]), 0, rtol=1e-04, atol=1e-04))

    def test_filter_bandpass(self):
        self._smoke(
            aug.filter_bandpass,
            {
                "Fs": [128, 256],
                "eeg_band": ["delta", "alpha", "beta", "gamma", "gamma_low"],
                "filter_type": ["butter", "ellip", "cheby1", "cheby2"],
            },
            self._inputs(),
            max_comb=32,
            extra=self._bounded,
        )
        x = _sine(16, 32, 1024)
        x += torch.sin(torch.linspace(0, 48 * 2 * torch.pi, 1024))
        x += torch.sin(torch.linspace(0, 256 * 2 * torch.pi, 1024))
        xaug = aug.filter_bandpass(x, 128, [13, 22], [5, 27])
        f, per2 = periodogram(xaug[0, 0], 128)
        self.assertTrue(np.isclose(np.max(per2[f < 5]), 0, rtol=1e-04, atol=1e-04))
        self.assertTrue(np.isclose(np.max(per2[f > 27]), 0, rtol=1e-04, atol=1e-04))

    def test_filter_bandstop(self):
        self._smoke(
            aug.filter_bandstop,
            {
                "Fs": [128, 256],
                "eeg_band": [None, "delta", "theta", "alpha", "beta", "gamma", "gamma_low"],
                "filter_type": ["butter", "ellip", "cheby1", "cheby2"],
            },
            self._inputs(),
            max_comb=32,
            extra=self._bounded,
        )
        x = _sine(16, 32, 1024)
        x += torch.sin(torch.linspace(0, 48 * 2 * torch.pi, 1024))
        x += torch.sin(torch.linspace(0, 256 * 2 * torch.pi, 1024))
        xaug = aug.filter_bandpass(x, 128, [13, 22], [5, 27])
        f, per2 = periodogram(xaug[0, 0], 128)
        self.assertTrue(
            np.isclose(np.max(per2[np.logical_and(f > 5, f < 27)]), 0, rtol=1e-04, atol=1e-04)
        )

    def test_permute_channels(self):
        channel_map = [
            "FP1", "AF3", "F1", "F3", "FC5", "FC3", "FC1", "C1", "C5", "TP7", "CP5",
            "CP3", "CP1", "P7", "PO7", "POZ", "PZ", "FPZ", "FP2", "AFZ", "FZ", "F2",
            "F4", "F6", "FT8", "C4", "T8", "TP8", "CP6", "CP4", "CP2", "PO8",
        ]
        # Channels must be distinct for a permutation to change the signal.
        torch.manual_seed(0)
        offset = torch.randn(32, 1)
        inputs = [self._t[d] + offset for d in (2, 3, 4)]
        inputs += [(self._t[d] + offset).numpy() for d in (2, 3, 4)]
        if self._g:
            inputs += [self._g[d] + offset.to(self.device) for d in (2, 3, 4)]

        self._smoke(
            aug.permute_channels,
            {
                "chan2shuf": [-1, 5, 10],
                "mode": ["random", "network"],
                "chan_net": ["DMN", "FPN", ["DMN", "FPN"], "all"],
                "batch_equal": [True, False],
                "channel_map": [channel_map],
            },
            inputs,
            max_comb=40,
        )
        x = torch.zeros(61, 4) + torch.arange(61).reshape(61, 1)
        xaug = aug.permute_channels(x, 10)
        self.assertEqual((x[:, 0] != xaug[:, 0]).sum(), 10)
        eeg1010, chan_net = aug.get_channel_map_and_networks(chan_net=["DMN", "VFN"])
        chan2per = np.union1d(chan_net[0], chan_net[1])
        a = np.intersect1d(eeg1010, chan2per, return_indices=True)[1]
        b = torch.from_numpy(np.delete(np.arange(61), a))
        xaug2 = aug.permute_channels(x, 50, mode="network", chan_net=["DMN", "VFN"])
        self.assertEqual((x[:, 0] != xaug2[:, 0]).sum(), 50)
        self.assertEqual((x[b, 0] == xaug2[b, 0]).sum(), len(b))

    def test_permutation_signal(self):
        self._smoke(
            aug.permutation_signal,
            {"segments": [10, 15, 20], "seg_to_per": [-1, 2, 5, 8], "batch_equal": [True, False]},
            self._inputs(),
            max_comb=32,
        )
        x = torch.ones(16, 32, 1024) * 2 + torch.sin(torch.linspace(0, 8 * torch.pi, 1024))
        xaug = aug.masking(x, 3, 0.5)
        self.assertTrue(
            torch.isclose(
                (xaug[0, 0] == 0).sum() / len(xaug[0, 0]),
                torch.tensor([0.5]),
                rtol=1e-8,
                atol=1e-8,
            )
        )
        a = xaug[0, 0] == 0
        self.assertEqual((a[:-1].ne(a[1:])).sum(), 6)

    def test_warp_signal(self):
        self._smoke(
            aug.warp_signal,
            {
                "segments": [15],
                "stretch_strength": [2, 1.5],
                "squeeze_strength": [0.4, 0.8],
                "batch_equal": [True, False],
            },
            self._inputs(),
            max_comb=24,
        )

    def test_crop_and_resize(self):
        self._smoke(
            aug.crop_and_resize,
            {"segments": [15], "N_cut": [1, 5], "batch_equal": [True, False]},
            self._inputs(dims=(2, 3, 4)),
            max_comb=24,
        )

    def test_change_ref(self):
        self._smoke(
            aug.change_ref,
            {
                "mode": ["chan", "avg"],
                "reference": [None, 5],
                "exclude_from_ref": [None, 9, [9, 10]],
            },
            self._inputs(dims=(2, 3, 4)),
            max_comb=24,
        )
        x = _sine(16, 32, 1024)
        x[:, 0, :] = 0.0
        xaug = aug.change_ref(x, "channel", 5)
        self.assertEqual(xaug[0, list(range(1, 32))].min().item(), 0)
        self.assertNotEqual(xaug[0, 0].min().item(), 0)

    def test_masking(self):
        self._smoke(
            aug.masking,
            {"mask_number": [1, 2, 4], "masked_ratio": [0.1, 0.2, 0.4], "batch_equal": [True, False]},
            self._inputs(),
            max_comb=24,
        )
        x = torch.ones(16, 32, 1024) * 2 + torch.sin(torch.linspace(0, 8 * torch.pi, 1024))
        xaug = aug.masking(x, 3, 0.5)
        self.assertTrue(
            torch.isclose(
                (xaug[0, 0] == 0).sum() / len(xaug[0, 0]),
                torch.tensor([0.5]),
                rtol=1e-6,
                atol=1e-8,
            )
        )

    def test_channel_dropout(self):
        self._smoke(
            aug.channel_dropout,
            {"Nchan": [None, 2, 3], "batch_equal": [True, False]},
            self._inputs(dims=(2, 3, 4)),
            max_comb=24,
        )
        x = torch.ones(32, 1024) * 2 + torch.sin(torch.linspace(0, 8 * torch.pi, 1024))
        xaug = aug.channel_dropout(x, 3)
        self.assertEqual((xaug[:, 10] == 0).sum(), 3)

    def test_eeg_artifact(self):
        self._smoke(
            aug.add_eeg_artifact,
            {
                "Fs": [128],
                "artifact": [None, "white", "line", "eye", "muscle", "drift", "lost"],
                "amplitude": [None, 1],
                "line_at_60Hz": [True, False],
                "lost_time": [0.5, None],
                "drift_slope": [None, 0.2],
                "batch_equal": [True, False],
            },
            self._inputs(),
            max_comb=48,
        )


if __name__ == "__main__":
    unittest.main()
