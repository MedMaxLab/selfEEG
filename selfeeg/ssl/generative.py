from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Callable
import copy
import os
import sys
from typing import Optional, Union

import torch
import torch.nn as nn
import tqdm

from .base import EarlyStopping, SSLBase

__all__ = ["ReconstructiveSSL"]


class ReconstructiveSSL(SSLBase):
    """
    Implementation of a reconstructive pretraining method.

    The task is to reconstruct the original EEG sample from its augmented version.

    Parameters
    ----------
    encoder: nn.Module
        The encoder part of the module. It is the one you wish to pretrain and
        transfer to the new model
    decoder: nn.Module
        The decoder part of the module.

    Warnings
    --------
    This class will not check the compatibility of the encoder's output and
    the decoder's input. Make sure that they have the same size.

    Example
    -------
    >>> import pickle, torch, selfeeg
    >>> import selfeeg.dataloading as dl
    >>> utils.create_dataset()
    >>> def loadEEG(path, return_label=False):
    ...     with open(path, 'rb') as handle:
    ...         EEG = pickle.load(handle)
    ...     x , y= EEG['data'], EEG['label']
    ...     return (x, y) if return_label else x
    >>> EEGlen = dl.get_eeg_partition_number('Simulated_EEG',freq=128, window=1,
    ...                              overlap=0.3, load_function=loadEEG)
    >>> EEGsplit = dl.get_eeg_split_table (EEGlen, seed=1234)
    >>> TrainSet = dl.EEGDataset(EEGlen,EEGsplit,[128,1,0.3],'train',False,loadEEG)
    >>> Loader = torch.utils.data.DataLoader(TrainSet, batch_size=32)
    >>> enc = selfeeg.models.ShallowNetEncoder(8)
    >>> dec = ... # custom decoder
    >>> generative = selfeeg.ssl.ReconstructSSL(enc, dec)
    >>> loss_train = simclr.fit(Loader, 1, return_loss_info=True)

    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: Union[list[int], nn.Module],
    ):

        super(ReconstructiveSSL, self).__init__(encoder)
        self.encoder = encoder
        self.decoder = decoder
        self._sslname = "reconstructive"

    def forward(self, x):
        """
        :meta private:

        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def fit(
        self,
        train_dataloader,
        epochs=1,
        optimizer=None,
        augmenter=None,
        loss_func: Callable = None,
        loss_args: list or dict = [],
        lr_scheduler=None,
        EarlyStopper=None,
        validation_dataloader=None,
        verbose=True,
        device: str or torch.device = None,
        return_loss_info: bool = False,
    ):
        """
        ``fit`` is a custom fit function designed to perform
        pretraining on a given model with the given dataloader.

        Parameters
        ----------
        train_dataloader: Dataloader
            The pytorch Dataloader used to get the training batches. It
            must return a batch as a single tensor X, thus without label tensor Y.
        epochs: int, optional
            The number of training epochs. Must be an integer bigger than 0.

            Default = 1
        optimizer: torch Optimizer, optional
            The optimizer used for weight's update. It can be any optimizer
            provided in the torch.optim module. If not given Adam with default
            parameters will be instantiated.

            Default = torch.optim.Adam
        augmenter: function, optional
            Any function (or callable object) used to perform data augmentation
            on the batch. It is highly suggested to resort to the augmentation
            module, which implements different data augmentation functions and
            classes to combine them.
            If none is given, a default augmentation with random vertical flip +
            random noise is applied.
            Note that in this case, contrary to fully supervised approaches,
            data augmentation is also performed on the validation set,
            since it's part of the SSL algorithm.

            Default = None
        loss_func: Callable, optional
            The custom loss function. It can be any loss function which
            accepts as input only the model's predictions as required arguments
            and loss_args as optional arguments.

            Default = None
        loss_args: Union[list, dict], optional
            The optional arguments to pass to the function. It can be a list
            or a dict.

            Default = None
        lr_scheduler: torch Scheduler
            A pytorch learning rate scheduler used to update the learning rate
            during the fine-tuning.

            Default = None
        EarlyStopper: EarlyStopping, optional
            An instance of the provided EarlyStopping class.

            Default = None

            Note
            ----
            If an EarlyStopping instance is given with monitoring loss set to
            validation loss, but no validation dataloader is given, monitoring
            loss will be automatically set to training loss.

        validation_dataloader: Dataloader, optional
            the pytorch Dataloader used to get the validation batches. It
            must return a batch as a single tensor X, thus without label tensor Y.
            If not given, no validation loss will be calculated

            Default = None
        verbose: bool, optional
            Whether to print a progression bar or not.

            Default = None
        device: torch.device or str, optional
            The device to use for fine-tuning. If given as a string it will
            be converted in a torch.device instance. If not given, 'cpu' device
            will be used.

            Default = None
        return_loss_info: bool, optional
            Whether to return the calculated training validation losses at
            each epoch.

            Default = False

        Returns
        -------
        loss_info: dict, optional
            A dictionary with keys being the epoch number (as integer) and values
            a two element list with the average epoch's training and validation
            loss.

        """
        # Various checks on input parameters.
        # If some arguments weren't given they will be automatically set
        (device, epochs, optimizer, loss_func, perform_validation, loss_info, N_train, N_val) = (
            self._set_fit_args(
                train_dataloader,
                epochs,
                optimizer,
                augmenter,
                loss_func,
                loss_args,
                EarlyStopper,
                validation_dataloader,
                device,
            )
        )

        # training for loop (classical pytorch structure)
        # with some additions
        for epoch in range(epochs):
            print(f"epoch [{epoch+1:6>}/{epochs:6>}]") if verbose else None

            train_loss = 0
            val_loss = 0
            train_loss_tot = 0
            val_loss_tot = 0

            if not (self.training):
                self.train()

            with tqdm.tqdm(
                total=N_train + N_val,
                ncols=100,
                bar_format="{desc}{percentage:3.0f}%|{bar:15}| "
                "{n_fmt}/{total_fmt} [{rate_fmt}{postfix}]",
                disable=not (verbose),
                unit=" Batch",
                file=sys.stdout,
            ) as pbar:
                for batch_idx, X in enumerate(train_dataloader):

                    optimizer.zero_grad()

                    if X.device.type != device.type:
                        X = X.to(device=device)

                    Xaug = augmenter(X)
                    Xrec = self(Xaug)
                    train_loss = self.evaluate_loss(loss_func, [Xrec, X], loss_args)

                    train_loss.backward()
                    optimizer.step()
                    train_loss_tot += train_loss.item()
                    # verbose print
                    if verbose:
                        pbar.set_description(f" train {batch_idx+1:8<}/{N_train:8>}")
                        pbar.set_postfix_str(
                            f"train_loss={train_loss_tot/(batch_idx+1):.5f}, "
                            f"val_loss={val_loss_tot:.5f}"
                        )
                        pbar.update()
                train_loss_tot /= batch_idx + 1

                if lr_scheduler != None:
                    lr_scheduler.step()

                # Perform validation if validation dataloader was given
                if perform_validation:
                    self.eval()
                    with torch.no_grad():
                        val_loss = 0
                        val_loss_tot = 0
                        for batch_idx, X in enumerate(validation_dataloader):

                            if X.device.type != device.type:
                                X = X.to(device=device)

                            Xaug = augmenter(X)
                            Xrec = self(Xaug)

                            val_loss = self.evaluate_loss(loss_func, [Xrec, X], loss_args)
                            val_loss_tot += val_loss.item()
                            if verbose:
                                pbar.set_description(f"   val {batch_idx+1:8<}/{N_val:8>}")
                                pbar.set_postfix_str(
                                    f"train_loss={train_loss_tot:.5f}, "
                                    f"val_loss={val_loss_tot/(batch_idx+1):.5f}"
                                )
                                pbar.update()

                        val_loss_tot /= batch_idx + 1

            # Deal with earlystopper if given
            if EarlyStopper != None:
                updated_mdl = False
                if EarlyStopper.monitored == "validation":
                    curr_monitored = val_loss_tot
                else:
                    train_loss_tot
                EarlyStopper.early_stop(curr_monitored)
                if EarlyStopper.record_best_weights:
                    if EarlyStopper.best_loss == curr_monitored:
                        EarlyStopper.rec_best_weights(self)
                        updated_mdl = True
                if EarlyStopper():
                    if verbose:
                        print(f"no improvement after {EarlyStopper.patience} epochs.")
                        print(f"Training stopped at epoch {epoch}")
                    if EarlyStopper.record_best_weights and not (updated_mdl):
                        EarlyStopper.restore_best_weights(self)
                    if return_loss_info:
                        return loss_info
                    else:
                        return

            if return_loss_info:
                loss_info[epoch] = [train_loss_tot, val_loss_tot]
        if return_loss_info:
            return loss_info

    def test(
        self,
        test_dataloader,
        augmenter=None,
        loss_func: Callable = None,
        loss_args: list or dict = [],
        verbose: bool = True,
        device: str = None,
    ):
        """
        Evaluate the loss on a test dataloader.
        Parameters are the same as described in the fit method, aside for
        those related to model training which are removed.

        It is rare to evaluate the pretraing loss function on a test set.
        Nevertheless this function provides a way to do that.
        An example of usage could be to assess the quality of the learned
        features on the fine-tuning dataset.

        """
        device, augmenter, loss_func, N_test = self._set_test_args(
            test_dataloader, augmenter, loss_func, loss_args, device
        )
        with torch.no_grad():
            test_loss = 0
            test_loss_tot = 0
            with tqdm.tqdm(
                total=N_test,
                ncols=100,
                bar_format="{desc}{percentage:3.0f}%|{bar:15}| "
                "{n_fmt}/{total_fmt} [{rate_fmt}{postfix}]",
                disable=not verbose,
                unit=" Batch",
                file=sys.stdout,
            ) as pbar:
                for batch_idx, X in enumerate(test_dataloader):

                    if X.device.type != device.type:
                        X = X.to(device=device)

                    Xaug = augmenter(X)
                    Xrec = self(Xaug)

                    test_loss = self.evaluate_loss(loss_func, [Xrec, X], loss_args)
                    test_loss_tot += test_loss
                    # verbose print
                    if verbose:
                        pbar.set_description(f"   test {batch_idx+1:8<}/{N_test:8>}")
                        pbar.set_postfix_str(f"test_loss={test_loss_tot/(batch_idx+1):.5f}")
                        pbar.update()

                test_loss_tot /= batch_idx + 1
        return test_loss_tot
