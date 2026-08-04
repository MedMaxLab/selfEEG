import os
import pickle
import shutil
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from EEGself._testtools import make_grid

from selfeeg import dataloading as dl


class TestDataloading(unittest.TestCase):

    def create_dataset(self, folder_name="Simulated_EEG", data_len=[1024, 4097], seed=1234):
        N = 1000
        if not (os.path.isdir(folder_name)):
            os.mkdir(folder_name)
        np.random.seed(seed=seed)
        for i in range(N):
            x = np.random.randn(2, np.random.randint(data_len[0], data_len[1]))
            y = np.random.randint(1, 5)
            sample = {"data": x, "label": y}
            A = int(i // 200) + 1
            B = int((i - 200 * int(i // 200))) // 5 + 1
            C = i % 5 + 1
            file_name = "Simulated_EEG/" + str(A) + "_" + str(B) + "_" + str(C) + "_1.pickle"
            with open(file_name, "wb") as f:
                pickle.dump(sample, f)

    def loadEEG(self, path, return_label=False, return_label_array=False):
        with open(path, "rb") as handle:
            EEG = pickle.load(handle)
        x = EEG["data"]
        y = EEG["label"]
        if return_label:
            if return_label_array:
                # just to complicate things
                y = np.array([[y] * 100, [y] * 100]).T
                return x, y
            return x, y
        return x

    def transformEEG(self, EEG, value=64):
        return EEG[:, :-value]

    @classmethod
    def setUpClass(cls):
        if not os.path.isdir("tmpsave"):
            os.mkdir("tmpsave")
        cls.eegpath = "Simulated_EEG"
        cls.create_dataset(cls)
        cls.freq = 128  # sampling frequency in [Hz]
        cls.overlap = 0.15  # overlap between partitions
        cls.window = 2  # window length in [seconds]

    def setUp(self):
        self.seed = 1234
        np.random.seed(self.seed)

    def _partition_table(self):
        return dl.get_eeg_partition_number(
            self.eegpath,
            self.freq,
            self.window,
            self.overlap,
            file_format="*.pickle",
            load_function=self.loadEEG,
            optional_load_fun_args=[False],
            transform_function=self.transformEEG,
        )

    def _labels(self, EEGlen):
        labels = np.zeros(EEGlen.shape[0], dtype=int)
        for i in range(EEGlen.shape[0]):
            _, labels[i] = self.loadEEG(EEGlen.iloc[i]["full_path"], True)
        return labels

    def test_get_eeg_partition_number(self):
        grid = make_grid(
            {
                "EEGpath": [self.eegpath],
                "freq": [self.freq],
                "window": [2],
                "overlap": [0, 0.15],
                "includePartial": [True, False],
                "file_format": ["*.pickle"],
                "load_function": [self.loadEEG],
                "optional_load_fun_args": [[False]],
                "transform_function": [None, self.transformEEG],
                "optional_transform_fun_args": [None, [32]],
                "keep_zero_sample": [True, False],
                "save": [True, False],
                "save_path": ["tmpsave/results1.csv"],
                "verbose": [False],
            },
            max_comb=16,
        )
        for args in grid:
            with self.subTest(overlap=args["overlap"], includePartial=args["includePartial"]):
                EEGlen = dl.get_eeg_partition_number(**args)
                self.assertEqual(EEGlen.shape[0], 1000)

    def test_get_eeg_split_table(self):
        EEGlen = self._partition_table()
        Labels = self._labels(EEGlen)

        # fmt: off
        val_dict_id = [
            25, 26, 27, 28, 29, 60, 61, 62, 63, 64, 225, 226, 227, 228, 229, 260,
            261, 262, 263, 264, 425, 426, 427, 428, 429, 460, 461, 462, 463, 464,
            625, 626, 627, 628, 629, 660, 661, 662, 663, 664, 825, 826, 827, 828,
            829, 860, 861, 862, 863, 864,
        ]
        test_dict_id = [
            20, 21, 22, 23, 24, 65, 66, 67, 68, 69, 220, 221, 222, 223, 224, 265,
            266, 267, 268, 269, 420, 421, 422, 423, 424, 465, 466, 467, 468, 469,
            620, 621, 622, 623, 624, 665, 666, 667, 668, 669, 820, 821, 822, 823,
            824, 865, 866, 867, 868, 869,
        ]
        excl_dict_id = [
            15, 16, 17, 18, 19, 70, 71, 72, 73, 74, 215, 216, 217, 218, 219, 270,
            271, 272, 273, 274, 415, 416, 417, 418, 419, 470, 471, 472, 473, 474,
            615, 616, 617, 618, 619, 670, 671, 672, 673, 674, 815, 816, 817, 818,
            819, 870, 871, 872, 873, 874,
        ]
        # fmt: on

        grid = make_grid(
            {
                "partition_table": [EEGlen],
                "test_ratio": [0, 0.2],
                "val_ratio": [0, 0.2],
                "test_split_mode": [0, 1, 2],
                "val_split_mode": [1, 2],
                "exclude_data_id": [None, {x: [13, 23] for x in range(1, 6)}],
                "test_data_id": [None, {x: [14, 22] for x in range(1, 6)}, 4],
                "val_data_id": [None, {x: [15, 21] for x in range(1, 6)}, [3]],
                "val_ratio_on_all_data": [True, False],
                "stratified": [False, True],
                "labels": [Labels],
                "dataset_id_extractor": [lambda x: int(x.split("_")[0])],
                "subject_id_extractor": [None],
                "save": [True],
                "split_tolerance": [0.001],
                "perseverance": [1000],
                "save_path": ["tmpsave/results1.csv"],
                "seed": [self.seed],
            },
            max_comb=48,
        )
        for args in grid:
            with self.subTest(
                test_ratio=args["test_ratio"],
                val_ratio=args["val_ratio"],
                test_split_mode=args["test_split_mode"],
                stratified=args["stratified"],
            ):
                EEGsplit = dl.get_eeg_split_table(**args)
                total_list = EEGsplit[EEGsplit["split_set"] != -1].index.tolist()
                tot = EEGlen.iloc[total_list]["N_samples"].sum()
                self.assertEqual(EEGsplit.shape[0], 1000)

                if isinstance(args["exclude_data_id"], dict):
                    check = EEGsplit.iloc[excl_dict_id]["split_set"].unique()
                    self.assertEqual(len(check), 1)
                    self.assertEqual(check[0], -1)

                if args["test_ratio"] == 0:
                    if isinstance(args["test_data_id"], dict):
                        check = EEGsplit.iloc[test_dict_id]["split_set"].unique()
                        self.assertEqual(len(check), 1)
                        self.assertEqual(check[0], 2)
                    elif isinstance(args["test_data_id"], int):
                        if not isinstance(args["exclude_data_id"], dict):
                            cond = (EEGsplit["file_name"].str[0] == "4").values
                            check = EEGsplit[cond]["split_set"].unique()
                            self.assertEqual(len(check), 1)
                            self.assertEqual(check[0], 2)
                    else:
                        self.assertEqual(EEGlen["N_samples"][EEGsplit["split_set"] == 2].sum(), 0)
                else:
                    if args["test_data_id"] is None:
                        ratio = abs(
                            0.2 - EEGlen["N_samples"][EEGsplit["split_set"] == 2].sum() / tot
                        )
                        self.assertLess(ratio, 2.5e-2)
                        if not args["stratified"] and args["test_split_mode"] == 0:
                            EEGsplit["dataid"] = EEGsplit["file_name"].str[0]
                            group = EEGsplit.groupby(["dataid", "split_set"])
                            lst = list(group.split_set.groups.keys())
                            result = [t for t in lst if t[1] == 2]
                            self.assertEqual(len(result), 1)
                            expected = 200 if args["exclude_data_id"] is None else 190
                            self.assertEqual(group.get_group(result[0]).shape[0], expected)

                if args["val_ratio"] == 0:
                    if isinstance(args["val_data_id"], dict):
                        if isinstance(args["test_data_id"], dict):
                            check = EEGsplit.iloc[val_dict_id]["split_set"].unique()
                            self.assertEqual(len(check), 1)
                            self.assertEqual(check[0], 1)
                    elif args["val_data_id"] is None:
                        self.assertEqual(EEGlen["N_samples"][EEGsplit["split_set"] == 1].sum(), 0)
                else:
                    if args["val_data_id"] is None:
                        thresh = 0.2
                        if not args["val_ratio_on_all_data"]:
                            test_list = EEGsplit[EEGsplit["split_set"] == 2].index.tolist()
                            test_ratio = EEGlen.iloc[test_list]["N_samples"].sum() / tot
                            thresh = 0.2 * (1 - test_ratio)
                        ratio = abs(
                            thresh - EEGlen["N_samples"][EEGsplit["split_set"] == 1].sum() / tot
                        )
                        self.assertLess(ratio, 2.5e-2)

        # stratified split preserves the per-class ratio across sets
        EEGsplit = dl.get_eeg_split_table(
            EEGlen,
            0.2,
            0.2,
            2,
            2,
            val_ratio_on_all_data=True,
            stratified=True,
            labels=Labels,
            split_tolerance=0.001,
            perseverance=10000,
            seed=1234,
        )
        ratio = dl.check_split(EEGlen, EEGsplit, Labels, True, False)["class_ratio"]
        self.assertLess(np.abs(ratio - ratio.mean(0)).max(), 1e-3)

    def test_get_eeg_split_table_kfold(self):
        EEGlen = self._partition_table()
        Labels = self._labels(EEGlen)
        grid = make_grid(
            {
                "partition_table": [EEGlen],
                "test_ratio": [0, 0.2],
                "kfold": [5, 10],
                "test_split_mode": [1, 2],
                "val_split_mode": [1, 2],
                "exclude_data_id": [None, {x: [13, 23] for x in range(1, 6)}],
                "test_data_id": [None, {x: [14, 22] for x in range(1, 6)}, 4],
                "stratified": [False, True],
                "labels": [Labels],
                "save": [True],
                "split_tolerance": [0.01],
                "perseverance": [1000],
                "save_path": ["tmpsave/results1.csv"],
            },
            max_comb=24,
        )
        for args in grid:
            with self.subTest(test_ratio=args["test_ratio"], kfold=args["kfold"], stratified=args["stratified"]):
                EEGsplit = dl.get_eeg_split_table_kfold(**args)
                self.assertEqual(EEGsplit.shape[0], 1000)
                self.assertEqual(EEGsplit.shape[1], args["kfold"] + 1)
                sums = set(EEGsplit.sum(axis=1, numeric_only=True).unique().tolist())
                self.assertTrue(sums.issubset({-1 * args["kfold"], 1, 2 * args["kfold"]}))

    def test_EEGDataset(self):
        EEGlen = self._partition_table()
        EEGsplit = dl.get_eeg_split_table(
            EEGlen,
            test_ratio=0.1,
            val_ratio=0.1,
            test_split_mode="file",
            val_split_mode="file",
            perseverance=5000,
            split_tolerance=0.005,
        )

        dataset_pretrain = dl.EEGDataset(
            EEGlen,
            EEGsplit,
            [self.freq, self.window, self.overlap],
            mode="train",
            load_function=self.loadEEG,
            transform_function=self.transformEEG,
        )
        sample_1 = dataset_pretrain[0]
        self.assertIsInstance(sample_1, torch.Tensor)
        self.assertEqual(sample_1.shape[-1], 256)

        dataset_finetune = dl.EEGDataset(
            EEGlen,
            EEGsplit,
            [self.freq, self.window, self.overlap],
            mode="train",
            supervised=True,
            load_function=self.loadEEG,
            optional_load_fun_args=[True, False],
            transform_function=self.transformEEG,
            label_on_load=True,
        )
        sample_2, label_2 = dataset_finetune[0]
        self.assertIsInstance(sample_2, torch.Tensor)
        self.assertEqual(sample_2.shape[-1], 256)
        self.assertIsInstance(label_2, int)

        # try again with an array of labels
        dataset_finetune = dl.EEGDataset(
            EEGlen,
            EEGsplit,
            [self.freq, self.window, self.overlap],
            mode="train",
            supervised=True,
            load_function=self.loadEEG,
            optional_load_fun_args=[True, True],
            transform_function=self.transformEEG,
            multilabel_on_load=True,
        )
        sample_2, label_2 = dataset_finetune[0]
        self.assertIsInstance(sample_2, torch.Tensor)
        self.assertEqual(sample_2.shape[-1], 256)

        dataset_finetune.preload_dataset()
        sample_3, label_3 = dataset_finetune[0]
        self.assertIsInstance(sample_3, torch.Tensor)
        self.assertEqual(sample_3.shape[-1], 256)

    def test_EEGSampler(self):
        EEGlen = dl.get_eeg_partition_number(
            self.eegpath,
            self.freq,
            self.window,
            self.overlap,
            file_format="*.pickle",
            load_function=self.loadEEG,
            optional_load_fun_args=[False],
            save=True,
            save_path="tmpsave/results1.csv",
        )
        Labels = np.zeros(EEGlen.shape[0])
        for i in range(EEGlen.shape[0]):
            _, Labels[i] = self.loadEEG(EEGlen["full_path"][i], True)
        Labels = Labels.astype(int)
        EEGsplit = dl.get_eeg_split_table(
            EEGlen,
            test_ratio=0.1,
            val_ratio=0.1,
            test_split_mode="file",
            val_split_mode="file",
            stratified=True,
            labels=Labels,
            perseverance=5000,
            split_tolerance=0.005,
        )
        dataset_pretrain = dl.EEGDataset(
            EEGlen,
            EEGsplit,
            [self.freq, self.window, self.overlap],
            mode="train",
            load_function=self.loadEEG,
            transform_function=self.transformEEG,
        )
        # both sampling modalities must build without error
        dl.EEGSampler(dataset_pretrain, Mode=0)
        dl.EEGSampler(dataset_pretrain, 16, 4)

    @classmethod
    def tearDownClass(cls):
        for folder in ("Simulated_EEG", "tmpsave"):
            shutil.rmtree(folder, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
