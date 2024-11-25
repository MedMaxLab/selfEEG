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

__all__ = ["PredictiveSSL"]


class PredictiveSSL(SSLBase):
    """
    Implementation of a standard predictive Pretraining.
    Contrary to contrastive, this pretraining performs a classification or
    regression task with a generated pseudo-label.
    A trivial example is the model trying to predict which random augmentation from
    a given set was applied to each sample of the batch.

    Parameters
    ----------
    encoder: nn.Module
        The encoder part of the module. It is the one you wish to pretrain and
        transfer to the new model.
    head: Union[list[int], nn.Module]
        The predictive head to use. It can be:

            1. an nn.Module
            2. a list of ints.

        In case a list of ints is given, an nn.Sequential module with Dense,
        BatchNorm and Relu will be automtically created. The list will be used
        to set input and output dimension of each Dense Layer. For instance, if
        [128, 64, 2] is given, two hidden layers will be created. The first
        with input 128 and output 64, the second with input 64 and output 2.
    return_logits: bool, optional
        Whether to return the output as logit or probability.
        It is suggested to not use False as the pytorch crossentropy loss function
        applies the softmax internally.

        Default = True


    Warnings
    --------
    This class will not check the compatibility of the encoder's output and
    the projection head's input. Make sure that they have the same size.

    Example
    -------
    >>> import pickle, torch, selfeeg
    >>> import selfeeg.dataloading as dl
    >>> import selfeeg.augmentation as aug
    >>> utils.create_dataset()
    >>> def loadEEG(path, return_label=False):
    ...     with open(path, 'rb') as handle:
    ...         EEG = pickle.load(handle)
    ...     x , y= EEG['data'], EEG['label']
    ...     return (x, y) if return_label else x
    >>> def loss_fineTuning(yhat, ytrue):
    ...     return F.binary_cross_entropy_with_logits(torch.squeeze(yhat), ytrue + 0.)
    >>> EEGlen = dl.get_eeg_partition_number('Simulated_EEG',freq=128, window=1,
    ...                              overlap=0.3, load_function=loadEEG)
    >>> EEGsplit = dl.get_eeg_split_table (EEGlen, seed=1234)
    >>> TrainSet = dl.EEGDataset(EEGlen,EEGsplit,[128,1,0.3],'train',False,loadEEG)
    >>> Loader = torch.utils.data.DataLoader(TrainSet, batch_size=32)
    >>> enc = selfeeg.models.ShallowNetEncoder(8)
    >>> pred = selfeeg.ssl.PredictiveSSL(enc, [16,16,2])
    >>> loss_train = pred.fit(Loader, 1, augmenter=augment, return_loss_info=True)

    """

    def __init__(
        self, encoder: nn.Module, head: Union[list[int], nn.Module], return_logits: bool = True
    ):

        super(PredictiveSSL, self).__init__(encoder)
        self.encoder = encoder
        self._sslname = "predictive"

        if isinstance(head, list):
            if len(head) < 2:
                raise ValueError("got a list with only one element")
            else:
                if all(isinstance(i, int) for i in head):
                    DenseList = []
                    for i in range(len(head) - 1):
                        DenseList.append(nn.Linear(head[i], head[i + 1]))

                        # Batch Norm Not applied on output due to BYOL and SimSiam
                        # choices, since those two are more recent SSL algorithms
                        DenseList.append(nn.BatchNorm1d(num_features=head[i + 1]))
                        if i < (len(head) - 2):
                            DenseList.append(nn.ReLU())
                    self.head = nn.Sequential(*DenseList)
                else:
                    raise ValueError("got a list with non integer values")
        else:
            self.head = head
        self.return_logits = return_logits

    def forward(self, x):
        """
        :meta private:

        """
        x = self.encoder(x)
        emb = self.head(x)
        if not (self.return_logits):
            if self.nb_classes <= 2:
                x = torch.sigmoid(x)
            else:
                x = torch.nn.functional.softmax(x, dim=1)
        return emb

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
        augmenter_batch_calls=2,
        labels_on_dataloader=False,
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
            is supposed to return a batch with a single tensor X (no pseudo-labels),
            unless ``labels_on_dataloader`` is set to True.
        epochs: int, optional
            The number of training epochs. Must be an integer bigger than 0.

            Default = 1
        optimizer: torch Optimizer, optional
            The optimizer used for weight's update. It can be any optimizer
            provided in the torch.optim module. If not given, Adam with default
            parameters will be instantiated.

            Default = torch.optim.Adam
        augmenter: Callable, optional
            Any function (or callable object) used to perform data augmentation
            on the batch and generate the pseudo-labels
            (if not provided by the dataloader itself).
            Therefore, unless ``labels_on_dataloader`` is set to True, augmenter is
            expected to take in input a batch tensor X and return both the augmented
            version of X and the pseudo-label tensor Y.
            It is highly suggested to resort to the selfeeg's augmentation module,
            which implements different data augmentation functions and classes
            to combine them. RandomAug, for example, can also return the index of
            the chosen augmentation to be used as a pseudo-label.

            Default = None

            Note
            ----
            This argument is optional because of the alternative way to provide
            pseudo-labels with the ``labels_on_dataloader`` argument,
            but in reality it must be given if the dataloader does not directly
            provide the pseudo-labels.

        loss_func: Callable, optional
            The custom loss function. It can be any loss function that
            accepts as input only the model's predictions as required arguments
            and loss_args as optional arguments.
            If not given, cross entroby loss will be automatically used.

            Default = None
        loss_args: Union[list, dict], optional
            The optional arguments to pass to the function.
            It can be a list or a dict.

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
            is supposed to return a batch with a single tensor X (no pseudo-labels),
            unless ``labels_on_dataloader`` is set to True.
            If not given, no validation loss will be calculated

            Default = None
        augmenter_batch_calls: int, optional
            The number of times the augmenter is called for a single batch.
            Each call selects an equal portion of samples in the batch and gives it to
            the augmenter.

            Default = 2

            Note
            ----
            To better understand how this argument works, suppose to design a task
            where you want the model to predict which augmentation from a predefined
            set was performed on each sample from the batch. Selfeeg classes in the
            compose submodules operate at the batch level, but one might want to
            generate batches with multiple labels and not one with only a single label.
            augmenter_batch_calls solves this problem.

        labels_on_dataloader: boolean, optional
            Set this to True if the dataloader already provides a set of pseudo-labels.
            If ``True`` augmenter and augmenter_batch_calls will be ignored.

            Note
            ----
            if you want to pretrain the model by simply solving another task and you
            need more functionalities, you can consider using the ``fine_tune``
            function, which acts as a generic supervised training.

            Default = False
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
        if (augmenter is None) and not (labels_on_dataloader):
            raise ValueError(
                "at least an augmenter or a dataloader that can output pseudo-labels must be given"
            )
        if augmenter_batch_calls <= 0:
            raise ValueError("augmenter_batch_calls must be an integer greater than 0")

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

                    # pseudo-label already in X, no need for data augmentations
                    if labels_on_dataloader:
                        Ytrue = X[1]
                        X = X[0]
                        if isinstance(X, torch.Tensor):
                            X = X.to(device=device)
                        else:
                            for i in range(len(X)):
                                X[i] = X[i].to(device=device)
                        if isinstance(Y, torch.Tensor):
                            Y = Y.to(device=device)
                        else:
                            for i in range(len(Y)):
                                Y[i] = Y[i].to(device=device)
                    # pseudo-label must be created, need for data augmentation
                    else:
                        X = X.to(device=device)
                        if augmenter_batch_calls == 1:
                            X, Ytrue = augmenter(X)
                        else:
                            permidx = torch.randperm(X.shape[0])
                            piece = X.shape[0] // augmenter_batch_calls
                            samples = permidx[:piece]
                            X[samples], Ytruei = augmenter(X[samples])
                            if isinstance(Ytruei, torch.Tensor):
                                Ytrue = torch.empty(
                                    X.shape[0], *Ytruei.shape[1:], dtype=Ytruei.dtype, device=device
                                )
                            else:
                                Ytrue = torch.empty(X.shape[0], device=device, dtype=type(Ytruei))
                            Ytrue[samples] = Ytruei
                            for i in range(1, augmenter_batch_calls):
                                samples = permidx[piece * i : piece * (i + 1)]
                                X[samples], Ytrue[samples] = augmenter(X[samples])
                            samples = permidx[piece * (i + 1) :]
                            X[samples], Ytrue[samples] = augmenter(X[samples])

                    Yhat = self(X)
                    train_loss = self.evaluate_loss(loss_func, [Yhat, Ytrue], loss_args)

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

                            # pseudo-label already in X, no need for data augmentations
                            if labels_on_dataloader:
                                Ytrue = X[1]
                                X = X[0]
                                if isinstance(X, torch.Tensor):
                                    X = X.to(device=device)
                                else:
                                    for i in range(len(X)):
                                        X[i] = X[i].to(device=device)
                                if isinstance(Y, torch.Tensor):
                                    Y = Y.to(device=device)
                                else:
                                    for i in range(len(Y)):
                                        Y[i] = Y[i].to(device=device)
                            # pseudo-label must be created, need for data augmentation
                            else:
                                X = X.to(device=device)
                                if augmenter_batch_calls == 1:
                                    X, Ytrue = augmenter(X)
                                else:
                                    permidx = torch.randperm(X.shape[0])
                                    piece = X.shape[0] // augmenter_batch_calls
                                    samples = permidx[:piece]
                                    X[samples], Ytruei = augmenter(X[samples])
                                    if isinstance(Ytruei, torch.Tensor):
                                        Ytrue = torch.empty(
                                            X.shape[0],
                                            *Ytruei.shape[1:],
                                            dtype=Ytruei.dtype,
                                            device=device,
                                        )
                                    else:
                                        Ytrue = torch.empty(
                                            X.shape[0], device=device, dtype=type(Ytruei)
                                        )
                                    Ytrue[samples] = Ytruei
                                    for i in range(1, augmenter_batch_calls):
                                        samples = permidx[piece * i : piece * (i + 1)]
                                        X[samples], Ytrue[samples] = augmenter(X[samples])
                                    samples = permidx[piece * (i + 1) :]
                                    X[samples], Ytrue[samples] = augmenter(X[samples])

                            Yhat = self(X)
                            val_loss = self.evaluate_loss(loss_func, [Yhat, Ytrue], loss_args)
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
        loss_func=None,
        loss_args: list or dict = [],
        augmenter_batch_calls=2,
        labels_on_dataloader=False,
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

                    # pseudo-label already in X, no need for data augmentations
                    if labels_on_dataloader:
                        Ytrue = X[1]
                        X = X[0]
                        if isinstance(X, torch.Tensor):
                            X = X.to(device=device)
                        else:
                            for i in range(len(X)):
                                X[i] = X[i].to(device=device)
                        if isinstance(Y, torch.Tensor):
                            Y = Y.to(device=device)
                        else:
                            for i in range(len(Y)):
                                Y[i] = Y[i].to(device=device)
                    # pseudo-label must be created, need for data augmentation
                    else:
                        X = X.to(device=device)
                        if augmenter_batch_calls == 1:
                            X, Ytrue = augmenter(X)
                        else:
                            permidx = torch.randperm(X.shape[0])
                            piece = X.shape[0] // augmenter_batch_calls
                            samples = permidx[:piece]
                            X[samples], Ytruei = augmenter(X[samples])
                            if isinstance(Ytruei, torch.Tensor):
                                Ytrue = torch.empty(
                                    X.shape[0], *Ytruei.shape[1:], dtype=Ytruei.dtype, device=device
                                )
                            else:
                                Ytrue = torch.empty(X.shape[0], device=device, dtype=type(Ytruei))
                            Ytrue[samples] = Ytruei
                            for i in range(1, augmenter_batch_calls):
                                samples = permidx[piece * i : piece * (i + 1)]
                                X[samples], Ytrue[samples] = augmenter(X[samples])
                            samples = permidx[piece * (i + 1) :]
                            X[samples], Ytrue[samples] = augmenter(X[samples])

                    Yhat = self(X)
                    test_loss = self.evaluate_loss(loss_func, [Yhat, Ytrue], loss_args)
                    test_loss_tot += test_loss
                    # verbose print
                    if verbose:
                        pbar.set_description(f"   test {batch_idx+1:8<}/{N_test:8>}")
                        pbar.set_postfix_str(f"test_loss={test_loss_tot/(batch_idx+1):.5f}")
                        pbar.update()

                test_loss_tot /= batch_idx + 1
        return test_loss_tot
