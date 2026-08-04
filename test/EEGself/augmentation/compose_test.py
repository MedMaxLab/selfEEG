import unittest

import numpy as np
import torch

from selfeeg import augmentation as aug


class TestAugmentationCompose(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.BatchEEG = torch.zeros(16, 32, 1024)
        cls.BatchEEG += torch.sin(torch.linspace(0, 8 * torch.pi, 1024))
        cls.Fs = 128

    def test_StaticSingleAug(self):
        aug_scal = aug.StaticSingleAug(aug.scaling, {"value": 2, "batch_equal": True})
        batch_aug = aug_scal(self.BatchEEG)
        self.assertAlmostEqual(batch_aug.min().item(), -2.0, delta=1e-5)
        self.assertAlmostEqual(batch_aug.max().item(), 2.0, delta=1e-5)

        aug_scal = aug.StaticSingleAug(
            aug.scaling,
            [{"value": 1.5, "batch_equal": False}, {"value": 2.0, "batch_equal": False}],
        )
        batch_aug1 = aug_scal(self.BatchEEG)
        self.assertAlmostEqual(batch_aug1.min().item(), -1.5, delta=1e-4)
        self.assertAlmostEqual(batch_aug1.max().item(), 1.5, delta=1e-4)

        batch_aug2 = aug_scal(self.BatchEEG)
        self.assertAlmostEqual(batch_aug2.min().item(), -2.0, delta=1e-4)
        self.assertAlmostEqual(batch_aug2.max().item(), 2.0, delta=1e-4)

        self.assertFalse(torch.equal(self.BatchEEG, batch_aug))
        self.assertFalse(torch.equal(self.BatchEEG, batch_aug1))
        self.assertFalse(torch.equal(self.BatchEEG, batch_aug2))
        self.assertFalse(torch.equal(batch_aug1, batch_aug2))

    def test_DynamicSingleAug(self):
        aug_warp = aug.DynamicSingleAug(
            aug.warp_signal,
            discrete_arg={"batch_equal": [True, False]},
            range_arg={
                "segments": [5, 15],
                "stretch_strength": [1.5, 2.5],
                "squeeze_strength": [0.4, 2 / 3],
            },
            range_type={"segments": True, "stretch_strength": False, "squeeze_strength": False},
        )
        batch_aug1 = aug_warp(self.BatchEEG)
        batch_aug2 = aug_warp(self.BatchEEG)
        self.assertFalse(torch.equal(self.BatchEEG, batch_aug1))
        self.assertFalse(torch.equal(self.BatchEEG, batch_aug2))
        self.assertFalse(torch.equal(batch_aug1, batch_aug2))

    def test_SequentialAug(self):
        aug_scal = aug.StaticSingleAug(aug.scaling, {"value": 2, "batch_equal": True})
        sequence = aug.SequentialAug(aug_scal, aug.flip_vertical)
        batch_aug1 = sequence(self.BatchEEG)
        batch_aug2 = aug.scaling(self.BatchEEG, 2)

        self.assertFalse(torch.equal(self.BatchEEG, batch_aug1))
        # scaling has been applied
        self.assertAlmostEqual(batch_aug1.min().item(), -2.0, delta=1e-5)
        self.assertAlmostEqual(batch_aug1.max().item(), 2.0, delta=1e-5)
        # vertical flip has been applied on top of the scaling
        self.assertTrue(torch.equal(batch_aug1, batch_aug2 * (-1)))

    def test_CircularAug(self):
        circular = aug.CircularAug(aug.flip_vertical, aug.identity)
        # two full cycles to check the round-robin behaviour
        for _ in range(2):
            self.assertTrue(torch.equal(circular(self.BatchEEG), self.BatchEEG * (-1)))
            self.assertTrue(torch.equal(circular(self.BatchEEG), self.BatchEEG))

    def test_RandomAug(self):
        aug_scal = aug.StaticSingleAug(aug.scaling, {"value": 2, "batch_equal": True})
        sequence = aug.RandomAug(aug_scal, aug.flip_vertical, p=[0.7, 0.3], return_index=True)
        counter = [0, 0]
        N = 3000
        np.random.seed(1234)
        for _ in range(N):
            _, idx = sequence(self.BatchEEG)
            counter[idx] += 1
        self.assertAlmostEqual(counter[0] / N, 0.7, delta=3e-2)
        self.assertAlmostEqual(counter[1] / N, 0.3, delta=3e-2)

    def test_AugmentationComposition(self):
        # First random selection: flip vertical / flip horizontal / identity
        flipv = aug.StaticSingleAug(aug.flip_vertical)
        flipr = aug.StaticSingleAug(aug.flip_horizontal)
        identity = aug.StaticSingleAug(aug.identity)
        block1 = aug.RandomAug(identity, flipv, flipr, p=[0.5, 0.25, 0.25])

        # Second random selection: add band noise or an eye artifact
        band = aug.DynamicSingleAug(
            aug.add_band_noise,
            discrete_arg={
                "bandwidth": ["delta", "theta", "alpha", "beta", (30, 49)],
                "samplerate": self.Fs,
                "noise_range": 0.1,
            },
        )
        eye = aug.DynamicSingleAug(
            aug.add_eeg_artifact,
            discrete_arg={"Fs": self.Fs, "artifact": "eye", "batch_equal": False},
            range_arg={"amplitude": [0.1, 0.5]},
            range_type={"amplitude": False},
        )
        block2 = aug.RandomAug(band, eye, return_index=True)

        # Third random selection: crop-and-resize or warp
        crop = aug.DynamicSingleAug(
            aug.crop_and_resize,
            discrete_arg={"batch_equal": False},
            range_arg={"N_cut": [1, 4], "segments": [10, 15]},
            range_type={"N_cut": True, "segments": True},
        )
        warp = aug.DynamicSingleAug(
            aug.warp_signal,
            discrete_arg={"batch_equal": [True, False]},
            range_arg={
                "segments": [5, 15],
                "stretch_strength": [1.5, 2.5],
                "squeeze_strength": [0.4, 2 / 3],
            },
            range_type={"segments": True, "stretch_strength": False, "squeeze_strength": False},
        )
        block3 = aug.RandomAug(crop, warp, return_index=True)

        augmenter = aug.SequentialAug(block1, block2, block3)
        batch_aug1 = augmenter(self.BatchEEG)
        batch_aug2 = augmenter(self.BatchEEG)
        self.assertFalse(torch.equal(self.BatchEEG, batch_aug1))
        self.assertFalse(torch.equal(self.BatchEEG, batch_aug2))


if __name__ == "__main__":
    unittest.main()
