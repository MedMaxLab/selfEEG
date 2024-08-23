import itertools
import os
import pickle
import platform
import random
import unittest
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
    else:
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

    def loss_fineTuning(self, yhat, ytrue):
        return F.binary_cross_entropy_with_logits(torch.squeeze(yhat), ytrue + 0.0)

    def makeGrid(self, pars_dict):
        keys = pars_dict.keys()
        combinations = itertools.product(*pars_dict.values())
        ds = [dict(zip(keys, cc)) for cc in combinations]
        return ds

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore", message="Using padding='same'", category=UserWarning)
        cls.seed = 1234

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

        print("\n---------------------------")
        print("TESTING SSL MODULE")
        if cls.device.type != "cpu":
            print("Found cuda device: testing module on it")
        else:
            print("Didn't found cuda device: testing module on cpu")
        print("---------------------------")
        cls.eegpath = "Simulated_EEG"
        cls.freq = 128  # sampling frequency in [Hz]
        cls.overlap = 0.3  # overlap between partitions
        cls.window = 1  # window length in [seconds]
        cls.workers = 0
        cls.batchsize = 16
        cls.Chan = 16

        selfeeg.utils.create_dataset()

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

        # DEFINE AUGMENTER
        AUG_band = aug.DynamicSingleAug(
            aug.add_band_noise,
            discrete_arg={
                "bandwidth": ["delta", "theta", "alpha", "beta", (30, 49)],
                "samplerate": cls.freq,
                "noise_range": 0.5,
            },
        )
        AUG_mask = aug.DynamicSingleAug(
            aug.masking, discrete_arg={"mask_number": [1, 2, 3, 4], "masked_ratio": 0.25}
        )
        Block1 = aug.RandomAug(AUG_band, AUG_mask, p=[0.7, 0.3])
        Block2 = lambda x: selfeeg.utils.scale_range_soft_clip(x, 500, 1.5, "uV", True)
        cls.Augmenter = aug.SequentialAug(Block1, Block2)

        cls.enc = selfeeg.models.ShallowNetEncoder(8, 8)
        cls.head_size = [16, 32, 32]
        cls.predictor_size = [32, 32]

    def setUp(self):
        self.seed = 1234
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def test_evaluate_loss(self):
        print("testng evaluate loss function...", end="", flush=True)
        y1 = torch.sin(torch.linspace(0, 8 * torch.pi, 1024))
        y2 = torch.sin(torch.linspace(0, 8 * torch.pi, 1024) + torch.pi / 6)
        loss = selfeeg.ssl.evaluate_loss(torch.nn.functional.mse_loss, [y1, y2])
        self.assertTrue(torch.abs(loss - 0.1341).item() < 1e-4)
        print("   evaluate loss ok")

    def test_EarlyStopping(self):
        print("testng EarlyStopper...")

        # first set of assertion on device and preservation of best model
        Stopper = selfeeg.ssl.EarlyStopping(device=self.device)
        eegnet = selfeeg.models.EEGNet(2, 16, 256)
        optimizer = torch.optim.SGD(eegnet.parameters(), 0.5)
        ytrue = torch.randn(16)
        yhat = eegnet(torch.randn(16, 16, 256)).squeeze()
        Stopper.rec_best_weights(eegnet)

        self.assertEqual(eegnet.Dense.bias.item(), Stopper.best_model["Dense.bias"].item())
        self.assertEqual(Stopper.best_model["Dense.bias"].device.type, self.device.type)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(yhat, ytrue)
        loss.backward()
        optimizer.step()

        self.assertNotEqual(eegnet.Dense.bias.item(), Stopper.best_model["Dense.bias"].item())
        eegnet = selfeeg.models.EEGNet(2, 16, 256)
        Stopper.restore_best_weights(eegnet)
        self.assertEqual(eegnet.Dense.bias.item(), Stopper.best_model["Dense.bias"].item())
        self.assertEqual(
            Stopper.best_model["Dense.bias"].device.type, eegnet.Dense.bias.device.type
        )

        TrainSet = dl.EEGDataset(
            self.EEGlen,
            self.EEGsplit,
            [128, 2, 0.3],
            "train",
            True,
            loadEEG,
            optional_load_fun_args=[True],
            label_on_load=True,
        )
        TrainLoader = torch.utils.data.DataLoader(TrainSet, batch_size=32)
        eegnet = selfeeg.models.EEGNet(2, 8, 256)
        Stopper = selfeeg.ssl.EarlyStopping(patience=1, monitored="train", device=self.device)
        Stopper.rec_best_weights(eegnet)  # little hack to force early stop correctly
        self.assertEqual(Stopper.best_model["Dense.bias"].device.type, self.device.type)
        eegnet = selfeeg.models.EEGNet(2, 8, 256)
        Stopper.best_loss = 0  # little hack to force early stop correctly
        loss_info = selfeeg.ssl.fine_tune(
            eegnet,
            TrainLoader,
            2,
            EarlyStopper=Stopper,
            loss_func=self.loss_fineTuning,
            verbose=False,
        )
        self.assertTrue(Stopper.earlystop)
        self.assertEqual(eegnet.Dense.bias.device.type, self.device.type)
        self.assertEqual(eegnet.Dense.bias.item(), Stopper.best_model["Dense.bias"].item())
        print("testng EarlyStopper...   EarlyStopper OK")

    def test_SimCLR(self):
        print("Testing SimCLR (2 epochs)...", end="", flush=True)

        SelfMdl = selfeeg.ssl.SimCLR(encoder=self.enc, projection_head=self.head_size).to(
            device=self.device
        )
        loss_train = SelfMdl.fit(
            train_dataloader=self.trainloader,
            augmenter=self.Augmenter,
            epochs=2,
            validation_dataloader=self.valloader,
            verbose=False,
            device=self.device,
            return_loss_info=True,
        )
        SelfMdl = SelfMdl.to(device="cpu")
        self.assertTrue(isinstance(loss_train, dict))
        self.assertTrue(SelfMdl(torch.randn(32, 8, 128)).shape == torch.Size([32, 32]))
        loss_test = SelfMdl.test(self.valloader, augmenter=self.Augmenter, verbose=False)
        print("   SimCLR OK")

    def test_MoCo(self):
        print("Testing MoCo v2 (2 epochs)...", end="", flush=True)

        SelfMdl = selfeeg.ssl.MoCo(
            encoder=self.enc, projection_head=self.head_size, bank_size=1024, m=0.9995
        ).to(device=self.device)
        loss = selfeeg.losses.moco_loss
        loss_arg = {"temperature": 0.5}
        optimizer = torch.optim.SGD(SelfMdl.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

        loss_train = SelfMdl.fit(
            train_dataloader=self.trainloader,
            augmenter=self.Augmenter,
            epochs=2,
            optimizer=optimizer,
            loss_func=loss,
            loss_args=loss_arg,
            lr_scheduler=scheduler,
            validation_dataloader=self.valloader,
            verbose=False,
            device=self.device,
            return_loss_info=True,
        )
        SelfMdl = SelfMdl.to(device="cpu")
        self.assertTrue(isinstance(loss_train, dict))
        self.assertTrue(SelfMdl(torch.randn(32, 8, 128)).shape == torch.Size([32, 32]))
        loss_test = SelfMdl.test(self.valloader, augmenter=self.Augmenter, verbose=False)
        print("   MoCo v2 OK")

        print("Testing MoCo v3 (2 epochs)...", end="", flush=True)
        SelfMdl = selfeeg.ssl.MoCo(
            encoder=self.enc,
            projection_head=self.head_size,
            predictor=self.predictor_size,
            m=0.9995,
        ).to(device=self.device)
        loss_train = SelfMdl.fit(
            train_dataloader=self.trainloader,
            augmenter=self.Augmenter,
            epochs=2,
            validation_dataloader=self.valloader,
            verbose=False,
            device=self.device,
            return_loss_info=True,
        )
        SelfMdl = SelfMdl.to(device="cpu")
        self.assertTrue(isinstance(loss_train, dict))
        self.assertTrue(SelfMdl(torch.randn(32, 8, 128)).shape == torch.Size([32, 32]))
        loss_test = SelfMdl.test(self.valloader, augmenter=self.Augmenter, verbose=False)
        print("   MoCo v3 OK")

    def test_BYOL(self):
        print("Testing BYOL (10 epochs, Earlystop, Scheduler)...", end="", flush=True)
        SelfMdl = selfeeg.ssl.BYOL(
            encoder=self.enc,
            projection_head=self.head_size,
            predictor=self.predictor_size,
            m=0.9995,
        ).to(device=self.device)

        loss = selfeeg.losses.byol_loss
        optimizer = torch.optim.Adam(SelfMdl.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        earlystop = selfeeg.ssl.EarlyStopping(
            patience=2,
            min_delta=1e-05,
            record_best_weights=True,
            device=self.device,
        )
        loss_train = SelfMdl.fit(
            train_dataloader=self.trainloader,
            augmenter=self.Augmenter,
            epochs=10,
            EarlyStopper=earlystop,
            optimizer=optimizer,
            loss_func=loss,
            lr_scheduler=scheduler,
            validation_dataloader=self.valloader,
            verbose=False,
            device=self.device,
            return_loss_info=True,
        )
        SelfMdl = SelfMdl.to(device="cpu")
        self.assertTrue(isinstance(loss_train, dict))
        self.assertTrue(loss_train[9][1] == earlystop.best_loss)
        self.assertTrue(
            (optimizer.param_groups[-1]["lr"] - (0.98 ** len(loss_train)) * 1e-4) < 1e-4
        )
        self.assertTrue(SelfMdl(torch.randn(32, 8, 128)).shape == torch.Size([32, 32]))
        loss_test = SelfMdl.test(self.valloader, augmenter=self.Augmenter, verbose=False)
        earlystop.restore_best_weights(SelfMdl)
        print("   BYOL OK")

    def test_SimSiam(self):
        print("Testing SimSiam (2 epochs)...", end="", flush=True)

        SelfMdl = selfeeg.ssl.SimSiam(
            encoder=self.enc, projection_head=self.head_size, predictor=self.predictor_size
        ).to(device=self.device)

        loss_train = SelfMdl.fit(
            train_dataloader=self.trainloader,
            augmenter=self.Augmenter,
            epochs=2,
            validation_dataloader=self.valloader,
            verbose=False,
            device=self.device,
            return_loss_info=True,
        )
        SelfMdl = SelfMdl.to(device="cpu")
        self.assertTrue(isinstance(loss_train, dict))
        self.assertTrue(SelfMdl(torch.randn(32, 8, 128)).shape == torch.Size([32, 32]))
        loss_test = SelfMdl.test(self.valloader, augmenter=self.Augmenter, verbose=False)
        print("   SimSiam OK")

    def test_VICReg(self):
        print("Testing VICReg (2 epochs)...", end="", flush=True)

        SelfMdl = selfeeg.ssl.VICReg(encoder=self.enc, projection_head=self.head_size).to(
            device=self.device
        )
        loss_train = SelfMdl.fit(
            train_dataloader=self.trainloader,
            augmenter=self.Augmenter,
            epochs=2,
            validation_dataloader=self.valloader,
            verbose=False,
            device=self.device,
            return_loss_info=True,
        )
        SelfMdl = SelfMdl.to(device="cpu")
        self.assertTrue(isinstance(loss_train, dict))
        self.assertTrue(SelfMdl(torch.randn(32, 8, 128)).shape == torch.Size([32, 32]))
        loss_test = SelfMdl.test(
            self.valloader, augmenter=self.Augmenter, verbose=False
        )  # just to show it works
        print("   VICReg OK")

    def test_BarlowTwins(self):
        print("Testing BarlowTwins (2 epochs)...", end="", flush=True)

        SelfMdl = selfeeg.ssl.BarlowTwins(encoder=self.enc, projection_head=self.head_size).to(
            device=self.device
        )

        loss_train = SelfMdl.fit(
            train_dataloader=self.trainloader,
            augmenter=self.Augmenter,
            epochs=2,
            validation_dataloader=self.valloader,
            verbose=False,
            device=self.device,
            return_loss_info=True,
        )
        SelfMdl = SelfMdl.to(device="cpu")
        self.assertTrue(isinstance(loss_train, dict))
        self.assertTrue(SelfMdl(torch.randn(32, 8, 128)).shape == torch.Size([32, 32]))
        loss_test = SelfMdl.test(
            self.valloader, augmenter=self.Augmenter, verbose=False
        )  # just to show it works
        print("   BarlowTwins OK")

    def test_PredictiveSSL(self):
        print("Testing Predictive SSL (2 epochs)...", end="", flush=True)

        SelfMdl = selfeeg.ssl.PredictiveSSL(self.enc, [16, 1])

        AUG_band = aug.DynamicSingleAug(
            aug.add_band_noise,
            discrete_arg={
                "bandwidth": ["delta", "theta", "alpha", "beta", (30, 49)],
                "samplerate": 128,
                "noise_range": 0.5,
            },
        )
        AUG_mask = aug.DynamicSingleAug(
            aug.masking, discrete_arg={"mask_number": [1, 2, 3, 4], "masked_ratio": 0.25}
        )
        augment = aug.RandomAug(AUG_band, AUG_mask, return_index=True)
        loss_train = SelfMdl.fit(
            train_dataloader=self.trainloader,
            epochs=2,
            loss_func=self.loss_fineTuning,
            augmenter=augment,
            augmenter_batch_calls=3,
            validation_dataloader=self.valloader,
            verbose=False,
            device=self.device,
            return_loss_info=True,
        )
        loss_test = SelfMdl.test(
            self.valloader,
            loss_func=self.loss_fineTuning,
            augmenter=augment,
            augmenter_batch_calls=3,
            verbose=False,
            device=self.device,
        )
        print("   Predictive SSL OK")

    def test_ReconstructiveSSL(self):
        print("Testing Reconstructive SSL (2 epochs)...", end="", flush=True)

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
        loss_train = gen.test(
            self.valloader, augmenter=self.Augmenter, verbose=False, device=self.device
        )
        print("   Reconstructive SSL OK")

    def test_finetuning(self):

        print("testing fine-tuning phase (10 epochs)...", end="", flush=True)
        TrainSet = dl.EEGDataset(
            self.EEGlen,
            self.EEGsplit,
            [self.freq, self.window, self.overlap],
            "train",
            True,
            loadEEG,
            optional_load_fun_args=[True],
            label_on_load=True,
        )
        TrainLoader = torch.utils.data.DataLoader(TrainSet, batch_size=32)
        ValSet = dl.EEGDataset(
            self.EEGlen,
            self.EEGsplit,
            [self.freq, self.window, self.overlap],
            "validation",
            True,
            loadEEG,
            optional_load_fun_args=[True],
            label_on_load=True,
        )
        ValLoader = torch.utils.data.DataLoader(ValSet, batch_size=32)
        shanet = selfeeg.models.ShallowNet(2, 8, 128).to(device=self.device)
        loss_info = selfeeg.ssl.fine_tune(
            shanet,
            TrainLoader,
            device=self.device,
            epochs=10,
            validation_dataloader=ValLoader,
            loss_func=self.loss_fineTuning,
            verbose=False,
            return_loss_info=True,
        )
        self.assertTrue(loss_info[9][0] < 0.015)
        print("   fine-tuning OK")

    @classmethod
    def tearDownClass(cls):
        print("removing generated residual directory (Simulated_EEG)")
        try:
            if platform.system() == "Windows":
                os.system("rmdir /Q /S Simulated_EEG")  # nosec
            else:
                os.system("rm -r Simulated_EEG")  # nosec
        except:
            print('Failed to delete "Simulated_EEG" folder' " Please do it manually")


if __name__ == "__main__":
    unittest.main()
