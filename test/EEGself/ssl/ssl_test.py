import glob
import os
import pickle
import random
import shutil
import sys
import unittest
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from EEGself._testtools import get_device

import selfeeg
from selfeeg import augmentation as aug
from selfeeg import dataloading as dl


def loadEEG(path, return_label=False):
    with open(path, "rb") as handle:
        EEG = pickle.load(handle)
    x = EEG["data"]
    y = EEG["label"]
    if return_label:
        return x, y
    return x


class Decoder(nn.Module):
    def __init__(self, emb, outCh, outT):
        super(Decoder, self).__init__()
        self.lin = nn.Linear(emb, outT)
        self.Conv = nn.Conv1d(1, outCh, 5, padding="same")

    def forward(self, x):
        x = self.lin(x)
        x = torch.unsqueeze(x, 1)
        x = self.Conv(x)
        return x


class TestSSL(unittest.TestCase):

    def loss_finetuning(self, yhat, ytrue):
        return F.binary_cross_entropy_with_logits(torch.squeeze(yhat), ytrue + 0.0)

    def loss_finetuning_val(self, yhat, ytrue):
        return torch.sum((yhat > 0.5) == ytrue) / ytrue.shape[0]

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore", message="Using padding='same'", category=UserWarning)
        cls.seed = 1234
        cls.device = get_device(
            probe=lambda dev: aug.add_band_noise(torch.randn(1024, device=dev), "theta", 128)
        )

        cls.eegpath = "Simulated_EEG"
        cls.freq = 128
        cls.overlap = 0.3
        cls.window = 1
        cls.workers = 0
        cls.batchsize = 16
        cls.Chan = 16

        # A trimmed simulated dataset keeps the training-based tests fast while
        # still exercising the full data-loading / SSL pipeline.
        selfeeg.utils.create_dataset()
        keep = sorted(glob.glob(os.path.join(cls.eegpath, "*.pickle")))[:200]
        keep = set(keep)
        for f in glob.glob(os.path.join(cls.eegpath, "*.pickle")):
            if f not in keep:
                os.remove(f)

        cls.EEGlen = dl.get_eeg_partition_number(
            cls.eegpath, cls.freq, cls.window, cls.overlap, load_function=loadEEG
        )
        cls.EEGsplit = dl.get_eeg_split_table(cls.EEGlen, seed=cls.seed)
        trainset = dl.EEGDataset(
            cls.EEGlen, cls.EEGsplit, [cls.freq, cls.window, cls.overlap], "train", False, loadEEG
        )
        trainsampler = dl.EEGSampler(trainset, cls.batchsize, cls.workers)
        cls.trainloader = DataLoader(
            dataset=trainset,
            batch_size=cls.batchsize,
            sampler=trainsampler,
            num_workers=cls.workers,
        )
        valset = dl.EEGDataset(
            cls.EEGlen,
            cls.EEGsplit,
            [cls.freq, cls.window, cls.overlap],
            "validation",
            False,
            loadEEG,
        )
        cls.valloader = DataLoader(dataset=valset, batch_size=cls.batchsize, shuffle=False)

        band = aug.DynamicSingleAug(
            aug.add_band_noise,
            discrete_arg={
                "bandwidth": ["delta", "theta", "alpha", "beta", (30, 49)],
                "samplerate": cls.freq,
                "noise_range": 0.5,
            },
        )
        mask = aug.DynamicSingleAug(
            aug.masking, discrete_arg={"mask_number": [1, 2, 3, 4], "masked_ratio": 0.25}
        )
        block1 = aug.RandomAug(band, mask, p=[0.7, 0.3])
        block2 = lambda x: selfeeg.utils.scale_range_soft_clip(x, 500, 1.5, "uV", True)
        cls.Augmenter = aug.SequentialAug(block1, block2)

        cls.enc = selfeeg.models.ShallowNetEncoder(8, 8)
        cls.head_size = [16, 32, 32]
        cls.predictor_size = [32, 32]

    def setUp(self):
        self.seed = 1234
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def _fit_and_check(self, model, epochs=2, out_dim=32, **fit_kwargs):
        """Fit an SSL model for a few epochs and run the shared sanity checks."""
        model = model.to(self.device)
        loss_train = model.fit(
            train_dataloader=self.trainloader,
            augmenter=self.Augmenter,
            epochs=epochs,
            validation_dataloader=self.valloader,
            verbose=False,
            device=self.device,
            return_loss_info=True,
            **fit_kwargs,
        )
        model = model.to("cpu")
        self.assertIsInstance(loss_train, dict)
        self.assertEqual(model(torch.randn(32, 8, 128)).shape, torch.Size([32, out_dim]))
        model.test(self.valloader, augmenter=self.Augmenter, verbose=False)
        return loss_train

    def test_evaluate_loss(self):
        y1 = torch.sin(torch.linspace(0, 8 * torch.pi, 1024))
        y2 = torch.sin(torch.linspace(0, 8 * torch.pi, 1024) + torch.pi / 6)
        loss = selfeeg.ssl.evaluate_loss(torch.nn.functional.mse_loss, [y1, y2])
        self.assertAlmostEqual(loss.item(), 0.1341, delta=1e-4)

    def test_EarlyStopping(self):
        # device placement and preservation of the best model
        stopper = selfeeg.ssl.EarlyStopping(device=self.device)
        eegnet = selfeeg.models.EEGNet(2, 16, 256)
        optimizer = torch.optim.SGD(eegnet.parameters(), 0.5)
        ytrue = torch.randn(16)
        yhat = eegnet(torch.randn(16, 16, 256)).squeeze()
        stopper.rec_best_weights(eegnet)

        self.assertEqual(eegnet.Dense.bias.item(), stopper.best_model["Dense.bias"].item())
        self.assertEqual(stopper.best_model["Dense.bias"].device.type, self.device.type)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(yhat, ytrue)
        loss.backward()
        optimizer.step()

        self.assertNotEqual(eegnet.Dense.bias.item(), stopper.best_model["Dense.bias"].item())
        eegnet = selfeeg.models.EEGNet(2, 16, 256)
        stopper.restore_best_weights(eegnet)
        self.assertEqual(eegnet.Dense.bias.item(), stopper.best_model["Dense.bias"].item())
        self.assertEqual(
            stopper.best_model["Dense.bias"].device.type, eegnet.Dense.bias.device.type
        )

        trainset = dl.EEGDataset(
            self.EEGlen,
            self.EEGsplit,
            [128, 2, 0.3],
            "train",
            True,
            loadEEG,
            optional_load_fun_args=[True],
            label_on_load=True,
        )
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=32)
        eegnet = selfeeg.models.EEGNet(2, 8, 256)
        stopper = selfeeg.ssl.EarlyStopping(patience=1, monitored="train", device=self.device)
        stopper.rec_best_weights(eegnet)  # little hack to force early stop correctly
        self.assertEqual(stopper.best_model["Dense.bias"].device.type, self.device.type)
        eegnet = selfeeg.models.EEGNet(2, 8, 256)
        stopper.best_loss = 0  # little hack to force early stop correctly
        selfeeg.ssl.fine_tune(
            eegnet,
            trainloader,
            2,
            EarlyStopper=stopper,
            loss_func=self.loss_finetuning,
            verbose=False,
        )
        self.assertTrue(stopper.earlystop)
        self.assertEqual(eegnet.Dense.bias.device.type, self.device.type)
        self.assertEqual(eegnet.Dense.bias.item(), stopper.best_model["Dense.bias"].item())

    def test_SimCLR(self):
        model = selfeeg.ssl.SimCLR(encoder=self.enc, projection_head=self.head_size)
        self._fit_and_check(model)

    def test_MoCo(self):
        # MoCo v2: momentum encoder + memory bank, custom optimizer/scheduler/loss
        model = selfeeg.ssl.MoCo(
            encoder=self.enc, projection_head=self.head_size, bank_size=1024, m=0.9995
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        self._fit_and_check(
            model,
            optimizer=optimizer,
            loss_func=selfeeg.losses.moco_loss,
            loss_args={"temperature": 0.5},
            lr_scheduler=scheduler,
        )

        # MoCo v3: predictor instead of memory bank
        model = selfeeg.ssl.MoCo(
            encoder=self.enc,
            projection_head=self.head_size,
            predictor=self.predictor_size,
            m=0.9995,
        )
        self._fit_and_check(model)

    def test_BYOL(self):
        model = selfeeg.ssl.BYOL(
            encoder=self.enc,
            projection_head=self.head_size,
            predictor=self.predictor_size,
            m=0.9995,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        earlystop = selfeeg.ssl.EarlyStopping(
            patience=2, min_delta=1e-05, record_best_weights=True, device=self.device
        )
        loss_train = self._fit_and_check(
            model,
            epochs=10,
            EarlyStopper=earlystop,
            optimizer=optimizer,
            loss_func=selfeeg.losses.byol_loss,
            lr_scheduler=scheduler,
        )
        self.assertEqual(loss_train[9][1], earlystop.best_loss)
        self.assertAlmostEqual(
            optimizer.param_groups[-1]["lr"], (0.98 ** len(loss_train)) * 1e-4, delta=1e-4
        )
        earlystop.restore_best_weights(model)

    def test_SimSiam(self):
        model = selfeeg.ssl.SimSiam(
            encoder=self.enc, projection_head=self.head_size, predictor=self.predictor_size
        )
        self._fit_and_check(model)

    def test_VICReg(self):
        model = selfeeg.ssl.VICReg(encoder=self.enc, projection_head=self.head_size)
        self._fit_and_check(model)

    def test_BarlowTwins(self):
        model = selfeeg.ssl.BarlowTwins(encoder=self.enc, projection_head=self.head_size)
        self._fit_and_check(model)

    def test_PredictiveSSL(self):
        model = selfeeg.ssl.PredictiveSSL(self.enc, [16, 1])
        band = aug.DynamicSingleAug(
            aug.add_band_noise,
            discrete_arg={
                "bandwidth": ["delta", "theta", "alpha", "beta", (30, 49)],
                "samplerate": 128,
                "noise_range": 0.5,
            },
        )
        mask = aug.DynamicSingleAug(
            aug.masking, discrete_arg={"mask_number": [1, 2, 3, 4], "masked_ratio": 0.25}
        )
        augment = aug.RandomAug(band, mask, return_index=True)
        loss_train = model.fit(
            train_dataloader=self.trainloader,
            epochs=2,
            loss_func=self.loss_finetuning,
            augmenter=augment,
            augmenter_batch_calls=3,
            validation_dataloader=self.valloader,
            verbose=False,
            device=self.device,
            return_loss_info=True,
        )
        self.assertIsInstance(loss_train, dict)
        model.test(
            self.valloader,
            loss_func=self.loss_finetuning,
            augmenter=augment,
            augmenter_batch_calls=3,
            verbose=False,
            device=self.device,
        )

    def test_ReconstructiveSSL(self):
        dec = Decoder(16, 8, 128)
        gen = selfeeg.ssl.ReconstructiveSSL(self.enc, dec)
        loss_train = gen.fit(
            train_dataloader=self.trainloader,
            epochs=2,
            augmenter=self.Augmenter,
            validation_dataloader=self.valloader,
            verbose=False,
            device=self.device,
            return_loss_info=True,
        )
        self.assertIsInstance(loss_train, dict)
        gen.test(self.valloader, augmenter=self.Augmenter, verbose=False, device=self.device)

    def test_finetuning(self):
        trainset = dl.EEGDataset(
            self.EEGlen,
            self.EEGsplit,
            [self.freq, self.window, self.overlap],
            "train",
            True,
            loadEEG,
            optional_load_fun_args=[True],
            label_on_load=True,
        )
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=32)
        valset = dl.EEGDataset(
            self.EEGlen,
            self.EEGsplit,
            [self.freq, self.window, self.overlap],
            "validation",
            True,
            loadEEG,
            optional_load_fun_args=[True],
            label_on_load=True,
        )
        valloader = torch.utils.data.DataLoader(valset, batch_size=32)
        shanet = selfeeg.models.ShallowNet(2, 8, 128).to(device=self.device)
        loss_info = selfeeg.ssl.fine_tune(
            shanet,
            trainloader,
            device=self.device,
            epochs=10,
            validation_dataloader=valloader,
            loss_func=self.loss_finetuning,
            validation_loss_func=self.loss_finetuning_val,
            verbose=False,
            return_loss_info=True,
        )
        # training must actually reduce the loss (robust to dataset size)
        self.assertLess(loss_info[9][0], loss_info[0][0])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.eegpath, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
