import unittest

import numpy as np
import torch

from selfeeg import augmentation as aug


class TestAugmentationCompose(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n---------------------------")
        print("TESTING AUGMENTATION.COMPOSE MODULE")
        print("---------------------------")
        cls.BatchEEG = torch.zeros(16, 32, 1024) + torch.sin(torch.linspace(0, 8 * torch.pi, 1024))
        cls.Fs = 128

    def test_StaticSingleAug(self):
        print("Testing static single augmentation...", end="", flush=True)
        Aug_scal = aug.StaticSingleAug(aug.scaling, {"value": 2, "batch_equal": True})

        BatchEEGaug = Aug_scal(self.BatchEEG)
        self.assertTrue(((BatchEEGaug.min() + 2.0) < 1e-5).item())
        self.assertTrue(((BatchEEGaug.max() - 2.0) < 1e-5).item())

        Aug_scal = aug.StaticSingleAug(
            aug.scaling,
            [{"value": 1.5, "batch_equal": False}, {"value": 2.0, "batch_equal": False}],
        )

        BatchEEGaug1 = Aug_scal(self.BatchEEG)
        self.assertTrue(((BatchEEGaug1.min() + 1.5) < 1e-5).item())
        self.assertTrue(((BatchEEGaug1.max() - 1.49999) < 1e-5).item())

        BatchEEGaug2 = Aug_scal(self.BatchEEG)
        self.assertTrue(((BatchEEGaug2.min() + 2.0) < 1e-5).item())
        self.assertTrue(((BatchEEGaug2.max() - 2.0) < 1e-5).item())

        self.assertFalse(torch.equal(self.BatchEEG, BatchEEGaug))
        self.assertFalse(torch.equal(self.BatchEEG, BatchEEGaug1))
        self.assertFalse(torch.equal(self.BatchEEG, BatchEEGaug2))
        self.assertFalse(torch.equal(BatchEEGaug1, BatchEEGaug2))
        print(" static single augmentation OK")

    def test_DynamicSingleAug(self):
        print("Testing dynamic single augmentation...", end="", flush=True)
        Aug_warp = aug.DynamicSingleAug(
            aug.warp_signal,
            discrete_arg={"batch_equal": [True, False]},
            range_arg={
                "segments": [5, 15],
                "stretch_strength": [1.5, 2.5],
                "squeeze_strength": [0.4, 2 / 3],
            },
            range_type={"segments": True, "stretch_strength": False, "squeeze_strength": False},
        )
        BatchEEGaug1 = Aug_warp(self.BatchEEG)
        BatchEEGaug2 = Aug_warp(self.BatchEEG)
        self.assertFalse(torch.equal(self.BatchEEG, BatchEEGaug1))
        self.assertFalse(torch.equal(self.BatchEEG, BatchEEGaug2))
        self.assertFalse(torch.equal(BatchEEGaug1, BatchEEGaug2))
        print("  dynamic single augmentation OK")

    def test_SequentialAug(self):
        print("Testing Sequential augmentation...", end="", flush=True)
        Aug_scal = aug.StaticSingleAug(aug.scaling, {"value": 2, "batch_equal": True})
        Sequence1 = aug.SequentialAug(Aug_scal, aug.flip_vertical)
        BatchEEGaug1 = Sequence1(self.BatchEEG)
        BatchEEGaug2 = aug.scaling(self.BatchEEG, 2)

        # check that augmentation has been performed
        self.assertFalse(torch.equal(self.BatchEEG, BatchEEGaug1))

        # check that scaling has been performed
        self.assertTrue(((BatchEEGaug1.min() + 2.0) < 1e-5).item())
        self.assertTrue(((BatchEEGaug1.max() - 2.0) < 1e-5).item())

        # check that flip has been performed
        self.assertTrue(torch.equal(BatchEEGaug1, BatchEEGaug2 * (-1)))
        print(" Sequential augmentation OK")

    def test_RandomAug(self):
        print("Testing Random augmentation...", end="", flush=True)
        Aug_scal = aug.StaticSingleAug(aug.scaling, {"value": 2, "batch_equal": True})
        Sequence2 = aug.RandomAug(Aug_scal, aug.flip_vertical, p=[0.7, 0.3])

        counter = [0, 0]
        N = 10000
        np.random.seed(1234)
        for i in range(N):
            BatchEEGaug = Sequence2(self.BatchEEG)
            if BatchEEGaug.max() > 1.5:
                counter[0] += 1
            else:
                counter[1] += 1
        counter[0] /= N
        counter[1] /= N
        self.assertTrue(abs(counter[0] - 0.7) < 1e-2)
        self.assertTrue(abs(counter[1] - 0.3) < 1e-2)
        print("   Random augmentation OK")

    def test_UltimateAugmentationComposition(self):
        print(
            "Testing final augmentation composition based on all previous classes...",
            end="",
            flush=True,
        )
        # DEFINE AUGMENTER
        # FIRST RANDOM SELECTION: APPLY FLIP OR CHANGE REFERENCE OR NOTHING
        AUG_flipv = aug.StaticSingleAug(aug.flip_vertical)
        AUG_flipr = aug.StaticSingleAug(aug.flip_horizontal)
        AUG_id = aug.StaticSingleAug(aug.identity)
        Sequence1 = aug.RandomAug(AUG_id, AUG_flipv, AUG_flipr, p=[0.5, 0.25, 0.25])

        # SECOND RANDOM SELECTION: ADD SOME NOISE
        AUG_band = aug.DynamicSingleAug(
            aug.add_band_noise,
            discrete_arg={
                "bandwidth": ["delta", "theta", "alpha", "beta", (30, 49)],
                "samplerate": self.Fs,
                "noise_range": 0.1,
            },
        )
        Aug_eye = aug.DynamicSingleAug(
            aug.add_eeg_artifact,
            discrete_arg={"Fs": self.Fs, "artifact": "eye", "batch_equal": False},
            range_arg={"amplitude": [0.1, 0.5]},
            range_type={"amplitude": False},
        )
        Sequence2 = aug.RandomAug(AUG_band, Aug_eye)
        # THIRD RANDOM SELECTION: CROP OR RANDOM PERMUTATION
        AUG_crop = aug.DynamicSingleAug(
            aug.crop_and_resize,
            discrete_arg={"batch_equal": False},
            range_arg={"N_cut": [1, 4], "segments": [10, 15]},
            range_type={"N_cut": True, "segments": True},
        )
        Aug_warp = aug.DynamicSingleAug(
            aug.warp_signal,
            discrete_arg={"batch_equal": [True, False]},
            range_arg={
                "segments": [5, 15],
                "stretch_strength": [1.5, 2.5],
                "squeeze_strength": [0.4, 2 / 3],
            },
            range_type={"segments": True, "stretch_strength": False, "squeeze_strength": False},
        )
        Sequence3 = aug.RandomAug(AUG_crop, Aug_warp)

        # FINAL AUGMENTER: SEQUENCE OF THE THREE RANDOM LISTS
        Augmenter = aug.SequentialAug(Sequence1, Sequence2, Sequence3)
        BatchEEGaug1 = Augmenter(self.BatchEEG)
        BatchEEGaug2 = Augmenter(self.BatchEEG)
        self.assertFalse(torch.equal(self.BatchEEG, BatchEEGaug1))
        self.assertFalse(torch.equal(self.BatchEEG, BatchEEGaug2))
        print("   final augmentation composition OK")


if __name__ == "__main__":
    unittest.main()
