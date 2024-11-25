from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Callable
import copy
import datetime
from itertools import zip_longest
import math
import os
import random
import sys
from typing import Optional, Union

import torch
import torch.nn as nn
import tqdm

from ..dataloading import EEGSampler
from ..losses import losses as Loss

__all__ = [
    "EarlyStopping",
    "evaluate_loss",
    "fine_tune",
    "SSLBase",
]


def _default_augmentation(x):
    """
    simple default augmentation used when no data augmenter is given in SSL
    fit methods. It's just a programming choice to avoid putting the augmenter
    as non optional parameter. No justification
    for the choice of random flip + random noise.
    Just that it can be written in few lines of code.

    :meta private:

    """
    if not (isinstance(x, torch.Tensor)):
        x = torch.Tensor(x)
    x = x * (random.choice([-1, 1]))
    std = torch.std(x)
    noise = std * torch.randn(*x.shape, device=x.device)
    x_noise = x + noise
    return x_noise


def evaluate_loss(
    loss_fun: Callable,
    arguments: torch.Tensor or list[torch.Tensor],
    loss_arg: Union[list, dict] = None,
):
    """
    evaluates a custom loss function.

    It requires `arguments` as required arguments and loss_arg as optional one.
    It is simply the ``SSLBase's evaluate_loss`` method exported as a function.

    Parameters
    ----------
    loss_fun: Callable
        The custom loss function. It can be any Callable object that
        accepts as input:

            1. the model's prediction (or predictions) and the true labels as
               required argument
            2. any element included in loss_args as optional arguments.

        Note that for the ``fine_tune`` method the number of required
        arguments must be 2, i.e. the model's prediction and true labels.
    arguments: torch.Tensor or list[torch.Tensors]
        the required arguments. Based on the way this function is used
        in a training pipeline it can be a single or multiple tensors.
    loss_arg: Union[list, dict], optional
        The optional arguments to pass to the function. It can be a list
        or a dict.

        Default = None

    Returns
    -------
    loss: 'loss_fun output'
        The output of the given loss function. It is expected to be a torch.Tensor.

    Example
    -------
    >>> import torch
    >>> import selfeeg.ssl
    >>> torch.manual_seed(1234)
    >>> ytrue = torch.randn(64, 1)
    >>> yhat  = torch.randn(64, 1)
    >>> loss = ssl.evaluate_loss(torch.nn.functional.mse_loss, [yhat,ytrue])
    >>> print(loss) # will print 1.9893

    """
    # multiple if else to assess which syntax use for loss function call
    if isinstance(arguments, list):
        if loss_arg == None or loss_arg == []:
            loss = loss_fun(*arguments)
        elif isinstance(loss_arg, dict):
            loss = loss_fun(*arguments, **loss_arg)
        else:
            loss = loss_fun(*arguments, *loss_arg)
    else:
        if loss_arg == None or loss_arg == []:
            loss = loss_fun(arguments)
        elif isinstance(loss_arg, dict):
            loss = loss_fun(arguments, **loss_arg)
        else:
            loss = loss_fun(arguments, *loss_arg)
    return loss


def fine_tune(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    epochs=1,
    optimizer=None,
    augmenter=None,
    loss_func: Callable = None,
    loss_args: list or dict = [],
    validation_loss_func: Callable = None,
    validation_loss_args: list or dict = [],
    label_encoder: Callable or list[Callable] = None,
    lr_scheduler=None,
    EarlyStopper=None,
    validation_dataloader: torch.utils.data.DataLoader = None,
    verbose=True,
    device: str or torch.device = None,
    return_loss_info: bool = False,
) -> Optional[dict]:
    """
    performs fine-tuning of a given model.

    Parameters
    ----------
    model: nn.Module
        The pytorch model to fine tune. It must be a nn.Module.
    train_dataloader: Dataloader
        The pytorch Dataloader used to get the training batches. The Dataloar
        must return a batch as a tuple (X, Y), with X the input tensor
        and Y the label tensor.

        Note
        ----
        from version 0.2.0 X and Y can also be lists of Tensors.
        This might be useful for multi-branch or multi-head models.

    epochs: int, optional
        The number of training epochs. It must be an integer bigger than 0.

        Default = 1
    optimizer: torch Optimizer, optional
        The optimizer used for weight's update. It can be any optimizer
        provided in the torch.optim module. If not given, Adam with default
        parameters will be instantiated.

        Default = None
    augmenter: Callable or list of Callables, optional
        Any function (or callable object) used to perform data augmentation
        on the batch. It is highly suggested to resort to the augmentation
        module, which implements different data augmentation functions and
        classes to combine them. Note that data augmentation is not performed
        on the validation set, since its goal is to increase the size of the
        training set and to get more different samples.

        Default = None

        Note
        ----
        from version 0.2.0 augmenter can be a list of Callables.
        This case is specific for scenarios when X is also a list of Tensors
        and you want to apply a specific augmentation for each of its elements.
        Augmentations are performed by using the command
        ``X[i] = augmenter[i](X[i])``.
        It is possible to have ``len(augmenter)<len(X)``.

    loss_func: Callable, optional
        The custom loss function. It can be any loss function which
        accepts as inputs the model's prediction and the true labels
        as required arguments and loss_args as optional arguments.

        Default = None
    loss_args: Union[list, dict], optional
        The optional arguments to pass to the loss function. It can be a list
        or a dict.

        Default = None
    validation_loss_func: Callable, optional
        A custom validation loss function. It can be any loss function which
        accepts as inputs the model's prediction and the true labels
        as required arguments, and loss_args as optional arguments.
        If None, loss_func will be used.

        Default = None
    validation_loss_args: Union[list, dict], optional
        The optional arguments to pass to the validation loss function.
        It can be a list or a dict.
        If None, loss_args will be used.

        Default = None
    label_encoder: callable of list of callables, optional
        A custom function used to encode the returned Dataloaders true labels.
        If None, the Dataloader's true label is used directly. It can be any
        funtion which accept as input the batch label tensor Y.

        Note
        ----
        from version 0.2.0 label_encoder can be a list of Callables.
        This case is specific for scenarios when Y is also a list of Tensors
        and you want to apply a specific encoder for each of its elements.
        label encoding is performed with the command
        ``Y[i] = label_encoder[i](Y[i])``.
        It is possible to have ``len(label_encoder)<len(Y)``.

        Default = None
    lr_scheduler: torch Scheduler
        A pytorch learning rate scheduler used to update the learning rate
        during the fine-tuning.

        Default = None
    EarlyStopper: EarlyStopping, optional
        An instance of the provided EarlyStopping class.

        Default = None
    validation_dataloader: Dataloader, optional
        the pytorch Dataloader used to get the validation batches. It
        must return a batch as a tuple (X, Y), with X the feature tensor
        and Y the label tensor. If not given, no validation loss will be
        calculated

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

    Note
    ----
    If an EarlyStopping instance is given with monitoring loss set to
    validation loss, but no validation dataloader is given, monitoring
    loss will be automatically set to training loss.

    Example
    -------
    >>> import torch, pickle
    >>> import selfeeg.dataloading as dl
    >>> import selfeeg.models
    >>> def loadEEG(path, return_label=False):
    ...     with open(path, 'rb') as handle:
    ...         EEG = pickle.load(handle)
    ...     x , y= EEG['data'], EEG['label']
    ...     return (x, y) if return_label else x
    >>> def loss_fineTuning(yhat, ytrue):
    ...     return F.binary_cross_entropy_with_logits(torch.squeeze(yhat), ytrue + 0.)
    >>> random.seed(1234)
    >>> EEGlen = dl.get_eeg_partition_number(
    ...     'Simulated_EEG', 128, 2, 0.3, load_function=loadEEG)
    >>> EEGsplit = dl.get_eeg_split_table (EEGlen, seed=1234)
    >>> TrainSet = dl.EEGDataset(EEGlen,EEGsplit, [128,2,0.3], 'train', True, loadEEG,
    ...                          optional_load_fun_args=[True], label_on_load=True)
    >>> TrainLoader = torch.utils.data.DataLoader(TrainSet, batch_size=32)
    >>> shanet= models.ShallowNet(2, 8, 256)
    >>> loss_info = ssl.fine_tune(shanet, TrainLoader, loss_func=loss_fineTuning)

    """

    if device is None:
        device = torch.device("cpu")
    else:
        if isinstance(device, str):
            device = torch.device(device.lower())
        elif isinstance(device, torch.device):
            pass
        else:
            raise ValueError("device must be a string or a torch.device instance")
    model.to(device=device)

    if not (isinstance(train_dataloader, torch.utils.data.DataLoader)):
        raise ValueError("train_dataloader must be a pytorch DataLoader")
    if not (isinstance(epochs, int)):
        epochs = int(epochs)
    if epochs < 1:
        raise ValueError("epochs must be bigger than 1")
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())
    if loss_func is None:
        raise ValueError("loss function not given")
    if not (isinstance(loss_args, list) or isinstance(loss_args, dict)):
        raise ValueError("loss_args must be a list or a dict")

    perform_validation = False
    if validation_dataloader != None:
        if not (isinstance(validation_dataloader, torch.utils.data.DataLoader)):
            raise ValueError("validation_dataloader must be a pytorch DataLoader")
        else:
            perform_validation = True
            if validation_loss_func is None:
                validation_loss_func = loss_func
                validation_loss_args = loss_args

    if EarlyStopper is not None:
        if EarlyStopper.monitored == "validation" and not (perform_validation):
            print(
                "Early stopper monitoring is set to validation loss,"
                "but no validation data are given. "
                "Internally changing monitoring to training loss"
            )
            EarlyStopper.monitored = "train"

    loss_info = {i: [None, None] for i in range(epochs)}
    N_train = len(train_dataloader)
    N_val = 0 if validation_dataloader is None else len(validation_dataloader)
    for epoch in range(epochs):
        print(f"epoch [{epoch+1:6>}/{epochs:6>}]") if verbose else None

        train_loss = 0
        val_loss = 0
        train_loss_tot = 0
        val_loss_tot = 0

        if not (model.training):
            model.train()
        with tqdm.tqdm(
            total=N_train + N_val,
            ncols=100,
            bar_format="{desc}{percentage:3.0f}%|{bar:15}| {n_fmt}/{total_fmt}"
            " [{rate_fmt}{postfix}]",
            disable=not (verbose),
            unit=" Batch",
            file=sys.stdout,
        ) as pbar:
            for batch_idx, (X, Ytrue) in enumerate(train_dataloader):

                optimizer.zero_grad()

                # possible cases: X is tensor or not, Augmenter is iterable or not
                if isinstance(X, torch.Tensor):
                    X = X.to(device=device)
                    if augmenter is not None:
                        X = augmenter(X)
                else:
                    if augmenter is not None:
                        if isinstance(augmenter, Iterable):
                            Nmin = min(len(augmenter), len(X))
                            for i in range(Nmin):
                                X[i] = X[i].to(device=device)
                                X[i] = augmenter[i](X[i])
                            for i in range(Nmin, len(X)):
                                X[i] = X[i].to(device=device)
                        else:
                            for i in range(len(X)):
                                X[i] = X[i].to(device=device)
                                X[i] = augmenter(X[i])
                    else:
                        for i in range(len(X)):
                            X[i] = X[i].to(device=device)

                if isinstance(Ytrue, torch.Tensor):
                    if label_encoder is not None:
                        Ytrue = label_encoder(Ytrue)
                    Ytrue = Ytrue.to(device=device)
                else:
                    if label_encoder is not None:
                        if isinstance(label_encoder, Iterable):
                            Nmin = min(len(label_encoder), len(Ytrue))
                            for i in range(Nmin):
                                Ytrue[i] = label_encoder[i](Ytrue[i])
                                Ytrue[i] = Ytrue[i].to(device=device)
                            for i in range(len(Ytrue)):
                                Ytrue[i] = Ytrue[i].to(device=device)
                        else:
                            for i in range(len(Ytrue)):
                                Ytrue[i] = label_encoder(Ytrue[i])
                                Ytrue[i] = Ytrue[i].to(device=device)
                    else:
                        for i in range(len(Ytrue)):
                            Ytrue[i] = Ytrue[i].to(device=device)

                Yhat = model(X)
                train_loss = evaluate_loss(loss_func, [Yhat, Ytrue], loss_args)

                train_loss.backward()
                optimizer.step()
                train_loss_tot += train_loss.item()
                # verbose print
                if verbose:
                    pbar.set_description(f" train {batch_idx+1:8<}/{len(train_dataloader):8>}")
                    pbar.set_postfix_str(
                        f"train_loss={train_loss_tot/(batch_idx+1):.5f}, "
                        f"val_loss={val_loss_tot:.5f}"
                    )
                    pbar.update()
            train_loss_tot /= batch_idx + 1

            if lr_scheduler != None:
                lr_scheduler.step()

            # Perform validation if validation dataloader were given
            if perform_validation:
                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    for batch_idx, (X, Ytrue) in enumerate(validation_dataloader):

                        if isinstance(X, torch.Tensor):
                            X = X.to(device=device)
                        else:
                            for i in range(len(X)):
                                X[i] = X[i].to(device=device)

                        if isinstance(Ytrue, torch.Tensor):
                            if label_encoder is not None:
                                Ytrue = label_encoder(Ytrue)
                            Ytrue = Ytrue.to(device=device)
                        else:
                            if label_encoder is not None:
                                if isinstance(label_encoder, Iterable):
                                    Nmin = min(len(label_encoder), len(Ytrue))
                                    for i in range(Nmin):
                                        Ytrue[i] = label_encoder[i](Ytrue[i])
                                        Ytrue[i] = Ytrue[i].to(device=device)
                                    for i in range(len(Ytrue)):
                                        Ytrue[i] = Ytrue[i].to(device=device)
                                else:
                                    for i in range(len(Ytrue)):
                                        Ytrue[i] = label_encoder(Ytrue[i])
                                        Ytrue[i] = Ytrue[i].to(device=device)
                            else:
                                for i in range(len(Ytrue)):
                                    Ytrue[i] = Ytrue[i].to(device=device)

                        Yhat = model(X)
                        val_loss = evaluate_loss(
                            validation_loss_func,
                            [Yhat, Ytrue],
                            validation_loss_args,
                        )
                        val_loss_tot += val_loss.item()
                        if verbose:
                            pbar.set_description(
                                f"   val {batch_idx+1:8<}/{len(validation_dataloader):8>}"
                            )
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
                curr_monitored = train_loss_tot
            EarlyStopper.early_stop(curr_monitored)
            if EarlyStopper.record_best_weights:
                if EarlyStopper.best_loss == curr_monitored:
                    EarlyStopper.rec_best_weights(model)
                    updated_mdl = True
            if EarlyStopper():
                if verbose:
                    print(f"no improvement after {EarlyStopper.patience} epochs.")
                    print(f"Training stopped at epoch {epoch}")
                if EarlyStopper.record_best_weights and not (updated_mdl):
                    EarlyStopper.restore_best_weights(model)
                if return_loss_info:
                    return loss_info
                else:
                    return

        if return_loss_info:
            loss_info[epoch] = [train_loss_tot, val_loss_tot]
    if return_loss_info:
        return loss_info


class EarlyStopping:
    """
    Pytorch implementation of an early stopper.

    It can monitor the validation or the training loss (no other metrics
    are currently supported).

    Some arguments are similar to Keras EarlyStopping class [early]_ .
    If you want to use other implemented functionalities take a look at
    PyTorch Ignite [ign]_ .

    Parameters
    ----------
    patience: int, optional
        The number of epochs to wait before stopping the training. Can
        be any positive integer.

        Default = 5
    min_delta: float, optional
        The minimum difference between the current best loss and the
        calculated one to consider as an improvement.

        Default = 1e-9
    improvement: str, optional
        Whether to consider an increase or decrease in the best loss
        as an improvement. Accepted strings are:

            - ['d','dec','decrease'] for decrease
            - ['i','inc','increase'] for increase

        Default = "decrease"
    monitored: str, optional
        Whether to monitor the training or validation loss. This
        attribute is used in the ``fine_tuning`` function or
        others class ``fit`` methods to check which calculated loss
        must be given. Accepted values are "train" or "validation".

        Default = "validation"
    record_best_weights: bool, optional
        Whether to record the best weights after every new best loss
        is reached or not. It will be used to restore such weights
        if the training is stopped.

        Default = True
    device: torch.device or str, optional
            The device to use for model record. If given as a string, it will
            be converted in a torch.device instance. If not given, 'cpu' device
            will be used as default.

        Default = None

    Note
    ----
    Like in KERAS the early stopper will not automatically restore the best weights
    if the training ends, i.e., you reach the maximum number of epochs. To get the
    best weights simply call the ``restore_best_weights( model )`` method.

    Example
    -------
    >>> import torch, pickle, selfeeg.losses
    >>> import selfeeg.dataloading as dl
    >>> import selfeeg.models
    >>> utils.create_dataset()
    >>> def loadEEG(path, return_label=False):
    ...     with open(path, 'rb') as handle:
    ...         EEG = pickle.load(handle)
    ...     x , y= EEG['data'], EEG['label']
    ...     return (x, y) if return_label else x
    >>> def loss_fineTuning(yhat, ytrue):
    ...     return F.binary_cross_entropy_with_logits(torch.squeeze(yhat), ytrue + 0.)
    >>> EEGlen = dl.get_eeg_partition_number('Simulated_EEG',128, 2, 0.3, load_function=loadEEG)
    >>> EEGsplit = dl.get_eeg_split_table (EEGlen, seed=1234)
    >>> ratios = dl.check_split(EEGlen,EEGsplit, return_ratio=True)
    >>> TrainSet = dl.EEGDataset(EEGlen,EEGsplit, [128,2,0.3], 'train', True, loadEEG,
    ...                              optional_load_fun_args=[True], label_on_load=True)
    >>> TrainLoader = torch.utils.data.DataLoader(TrainSet, batch_size=32)
    >>> shanet= models.ShallowNet(2, 8, 256)
    >>> Stopper = ssl.EarlyStopping( patience=1, monitored= 'train' )
    >>> Stopper.rec_best_weights(shanet) # little hack to force early stop correctly
    >>> Stopper.best_loss = 0 # little hack to force early stop correctly
    >>> loss_info = ssl.fine_tune(shanet, TrainLoader, 2, EarlyStopper=Stopper,
    ...                           loss_func=loss_fineTuning)
    >>> # it should stop training and print "no improvement after 1 epochs. Training stopped."

    References
    ----------
    .. [early] https://keras.io/api/callbacks/early_stopping/
    .. [ign] https://pytorch.org/ignite/

    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-9,
        improvement: str = "decrease",
        monitored: str = "validation",
        record_best_weights: bool = True,
        device: str or torch.device = None,
    ):
        if device is None:
            self.device = torch.device("cpu")
        else:
            if isinstance(device, str):
                self.device = torch.device(device.lower())
            elif isinstance(device, torch.device):
                self.device = device
            else:
                raise ValueError("device must be a string or a torch.device instance")

        if patience < 0:
            self.patience = 0
        else:
            self.patience = int(patience)

        if isinstance(monitored, str) and (monitored in ["validation", "train"]):
            self.monitored = monitored.lower()
        else:
            raise ValueError("supported monitoring modes are train or validation")

        if min_delta < 0:
            raise ValueError(
                "min_delta must be >= 0. "
                "Use improvement to set if decrease or increase"
                " in the loss must be considered"
            )
        else:
            self.min_delta = min_delta

        if improvement.lower() not in ["d", "i", "dec", "inc", "decrease", "increase"]:
            msgErr = "got " + str(improvement) + " as improvement argument."
            msgErr += " Accepted arguments are "
            msgErr += "d, dec or decrease for decrease; i, inc or increase for increase"
            raise ValueError(msgErr)
        else:
            if improvement.lower() in ["d", "dec", "decrease"]:
                self.improvement = "decrease"
            else:
                self.improvement = "increase"

        self.record_best_weights = record_best_weights
        self.best_loss = 1e12 if improvement.lower() == "decrease" else -1 * 1e12

        self.best_model = None
        self.counter = 0
        self.earlystop = False

    def __call__(self):
        return self.earlystop

    def early_stop(self, loss, count_add=1):
        """
        update the counter and the best loss.

        Parameters
        ----------
        loss: float
            The calculated loss.
        count_add: int, optional
            The number to add to the counter. It can be useful when early stopping
            checks are not performed after each epoch.

        """
        # The function can be compressed with a big if.
        # This expansion is faster and better understandable
        if self.improvement == "decrease":
            # Check if current loss is better than recorded best loss
            if loss < (self.best_loss - self.min_delta):
                self.best_loss = loss
                self.counter = (
                    0  # During train if self.counter==0 record_best_weights will be called
                )
            else:
                self.counter += count_add
                if self.counter >= self.patience:
                    self.earlystop = True

        elif self.improvement == "increase":  # mirror with increase option
            if loss > (self.best_loss + self.min_delta):
                self.best_loss = loss
                self.counter = 0
            else:
                self.counter += count_add
                if self.counter >= self.patience:
                    self.earlystop = True

    def rec_best_weights(self, model):
        """
        Record model's best weights. The copy of the model is sent to
        the device set during EarlyStopping's initialization
        (default is cpu). Original model will retain its device.

        Parameters
        ----------
        model: nn.Module
            The model to record.

        """
        # Need to:
        # 1- not move the original model to another device as this breaks training.
        # 2- find a way to copy the model to another device without creating
        #    a duplicate on the GPU

        # Original command: model is not moved but a copy is created on the gpu
        # before being moved to another device (if set)
        # self.best_model = copy.deepcopy(model).to(device=self.device).state_dict()

        # new command: model is not moved and the OrderedDict is created directly
        # with list comprehension
        self.best_model = OrderedDict(
            [(k, v.to(device=self.device, copy=True)) for k, v in model.state_dict().items()]
        )

    def restore_best_weights(self, model):
        """
        Restore model's best weights.

        Parameters
        ----------
        model: nn.Module
            The model to restore.

        Warnings
        --------
        Before restoring its best weights, the model is moved to the device set
        during EarlyStopping's initialization. Remember to move it again to the
        desired device if EarlyStop's one is not the same.

        """
        model.to(device=self.device)
        model.load_state_dict(self.best_model)

    def reset_counter(self):
        """
        Reset the counter and early stopping flag.
        It might be useful if you want to further train
        your model after the first training is stopped
        (maybe with a lower learning rate).

        """
        self.counter = 0
        self.earlystop = False


class SSLBase(nn.Module):
    """
    Baseline Self-Supervised Learning nn.Module.

    It is used as parent class by the other implemented SSL methods.

    Parameters
    ----------
    encoder: nn.Module
        The encoder part of the module. It is the part of the model you wish
        to pretrain and transfer to the new model.

    Example
    -------
    >>> import selfeeg
    >>> import torch
    >>> enc = selfeeg.models.ShallowNetEncoder(8)
    >>> base = selfeeg.ssl.SSLBase(enc)
    >>> torch.manual_seed(1234)
    >>> a = torch.randn(64,32)
    >>> print(base.evaluate_loss(losses.simclr_loss, [a])) # should return 9.4143
    >>> enc2 = base.get_encoder()
    >>> def check_models(model1, model2):
    ...     for p1, p2 in zip(model1.parameters(), model2.parameters()):
    ...         if p1.data.ne(p2.data).sum() > 0:
    ...             return False
    ...     return True
    >>> print(check_models(base.encoder,enc2)) # should return True
    >>> # assert that they are different objects
    >>> enc2.conv1.weight = torch.nn.Parameter(enc2.conv1.weight*10)
    >>> print(check_models(base.encoder,enc2)) # should return False

    """

    def __init__(self, encoder: nn.Module):
        super(SSLBase, self).__init__()
        self.encoder = encoder
        self._sslname = "base"

    def forward(self, x):
        """
        :meta private:

        """
        pass

    def evaluate_loss(
        self,
        loss_fun: Callable,
        arguments: torch.Tensor or list[torch.Tensors],
        loss_arg: Union[list, dict] = None,
    ) -> torch.Tensor:
        """
        ``evaluate_loss`` evaluate a custom loss function using `arguments`
        as required arguments and loss_arg as optional ones.

        Parameters
        ----------
        loss_fun: function
            The custom loss function. It can be any loss function which
            accepts as input:

                1. the model's prediction (or predictions)
                2. any element included in loss_args as optional arguments.

            Note that the number of required arguments can change based on the
            specific pretraining method used. For example, SimCLR accepts 1 or 2 required
            arguments, while BYOL must take 4.
        arguments: torch.Tensor or list[torch.Tensors]
            the required arguments. Based on the way this function is used
            in a training pipeline it can be a single or multiple tensors.
        loss_arg: Union[list, dict], optional
            The optional arguments to pass to the function. It can be a list
            or a dict.

            Default = None

        Returns
        -------
        loss: torch.Tensor
            The output of the given loss function. It is expected to be
            a torch.Tensor.

        """
        if isinstance(arguments, list):
            if loss_arg == None or loss_arg == []:
                loss = loss_fun(*arguments)
            elif isinstance(loss_arg, dict):
                loss = loss_fun(*arguments, **loss_arg)
            else:
                loss = loss_fun(*arguments, *loss_arg)
        else:
            if loss_arg == None or loss_arg == []:
                loss = loss_fun(arguments)
            elif isinstance(loss_arg, dict):
                loss = loss_fun(arguments, **loss_arg)
            else:
                loss = loss_fun(arguments, *loss_arg)
        return loss

    def get_encoder(self, device="cpu"):
        """
        Returns a copy of the encoder on the selected device.

        Parameters
        ----------
        device: torch.device or str, optional
            The pytorch device where the encoder must be moved.

            Default = 'cpu'

        """
        enc = copy.deepcopy(self.encoder).to(device=device)
        return enc

    def save_encoder(self, path: str = None):
        """
        A method for saving the pretrained encoder.

        Parameters
        ----------
        path: str, optional
            The saving path, that will be given to the ``torch.save()``
            method. If None is given, the encoder will be saved in a created
            SSL_encoders subdirectory. The name will contain the pretraining
            method used (e.g. SimCLR, MoCo etc) and the current time.

            Default = None

        """
        if path is None:
            path = os.getcwd()
            if os.path.isdir(path + "/SSL_encoders"):
                path += "/SSL_encoders"
            else:
                os.mkdir(path + "/SSL_encoders")
                path += "/SSL_encoders"

            dict_names = {
                "simsiam": "_SimSiam",
                "vicreg": "_VICreg",
                "barlowtwins": "_Barlow",
                "byol": "_BYOL",
                "moco": "_MoCo",
                "simclr": "_SimClr",
            }
            sslName = dict_names.get(self._sslname)
            if sslName is None:
                sslName = "_Custom"
            save_path = (
                path
                + "/encoder_"
                + datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
                + sslName
                + ".pt"
            )
        torch.save(self.encoder.state_dict(), save_path)

    def _set_fit_args(
        self,
        train_dataloader,
        epochs=1,
        optimizer=None,
        augmenter=None,
        loss_func: Callable = None,
        loss_args: list or dict = [],
        EarlyStopper=None,
        validation_dataloader=None,
        device: str or torch.device = None,
    ):
        # Various checks on input parameters.
        # If some arguments weren't given they will be automatically set
        if device is None:
            # If device is None simply switch to cpu
            device = torch.device("cpu")
        else:
            if isinstance(device, str):
                device = torch.device(device.lower())
            elif isinstance(device, torch.device):
                pass
            else:
                raise ValueError("device must be a string or a torch.device instance")
        self.to(device=device)

        if not (isinstance(train_dataloader, torch.utils.data.DataLoader)):
            raise ValueError(
                "Current implementation accept only training data as a pytorch DataLoader"
            )
        if not (isinstance(epochs, int)):
            epochs = int(epochs)
        if epochs < 1:
            raise ValueError("epochs must be bigger than 1")
        if optimizer is None:
            if self._sslname == "moco":
                if self.bank_size is not None:
                    optimizer = torch.optim.SGD(self.parameters(), 0.01)
                else:
                    optimizer = torch.optim.Adam(self.parameters(), 0.001)
            else:
                optimizer = torch.optim.Adam(self.parameters())
        if augmenter is None:
            print("augmenter not given. Using a one with with flip + random noise")
            augmenter = _default_augmentation
        if loss_func is None:
            loss_dict = {
                "simsiam": Loss.simsiam_loss,
                "vicreg": Loss.vicreg_loss,
                "barlowtwins": Loss.barlow_loss,
                "byol": Loss.byol_loss,
                "moco": Loss.moco_loss,
                "simclr": Loss.simclr_loss,
                "predictive": torch.nn.functional.cross_entropy,
                "reconstructive": torch.nn.functional.mse_loss,
            }
            loss_func = loss_dict.get(self._sslname)
        if not (isinstance(loss_args, list) or isinstance(loss_args, dict) or loss_args == None):
            raise ValueError(
                "loss_args must be a list or a dict with all"
                " optional arguments of the loss function"
            )
        perform_validation = False
        if validation_dataloader != None:
            if not (isinstance(validation_dataloader, torch.utils.data.DataLoader)):
                raise ValueError(
                    "Current implementation accept only training data as a pytorch DataLoader"
                )
            else:
                perform_validation = True

        if EarlyStopper is not None:
            if EarlyStopper.monitored == "validation" and not (perform_validation):
                print(
                    "Early stopper monitoring is set to validation loss"
                    ", but no validation data are given. "
                    "Internally changing monitoring to training loss"
                )
                EarlyStopper.monitored = "train"

        # calculate some variables for training
        loss_info = {i: [None, None] for i in range(epochs)}
        N_train = len(train_dataloader)
        if isinstance(train_dataloader.sampler, EEGSampler):
            if train_dataloader.sampler.Keep_only_ratio != 1:
                if train_dataloader.drop_last:
                    N_train = math.floor(
                        sum(1 for _ in train_dataloader.sampler.__iter__())
                        / (train_dataloader.batch_size)
                    )
                else:
                    N_train = math.ceil(
                        sum(1 for _ in train_dataloader.sampler.__iter__())
                        / (train_dataloader.batch_size)
                    )

        N_val = 0 if validation_dataloader is None else len(validation_dataloader)
        if perform_validation and isinstance(validation_dataloader.sampler, EEGSampler):
            if validation_dataloader.sampler.Keep_only_ratio != 1:
                if validation_dataloader.drop_last:
                    N_val = math.floor(
                        sum(1 for _ in validation_dataloader.sampler.__iter__())
                        / (validation_dataloader.batch_size)
                    )
                else:
                    N_val = math.ceil(
                        sum(1 for _ in validation_dataloader.sampler.__iter__())
                        / (validation_dataloader.batch_size)
                    )
        return (device, epochs, optimizer, loss_func, perform_validation, loss_info, N_train, N_val)

    def _set_test_args(
        self,
        test_dataloader,
        augmenter=None,
        loss_func: Callable = None,
        loss_args: list or dict = [],
        device: str = None,
    ):
        if device == None:
            # If device is None simply use gpu
            device = torch.device("cpu")
        else:
            if isinstance(device, str):
                device = torch.device(device.lower())
            elif isinstance(device, torch.device):
                pass
            else:
                raise ValueError("device must be a string or a torch.device instance")
        self.to(device=device)
        if not (isinstance(test_dataloader, torch.utils.data.DataLoader)):
            raise ValueError(
                "Current implementation accept only test data" " as a pytorch DataLoader"
            )
        if augmenter == None:
            print("augmenter not given. Using one with with flip + random noise")
            augmenter = _default_augmentation
        if loss_func == None:
            loss_dict = {
                "simsiam": Loss.simsiam_loss,
                "vicreg": Loss.vicreg_loss,
                "barlowtwins": Loss.barlow_loss,
                "byol": Loss.byol_loss,
                "moco": Loss.moco_loss,
                "simclr": Loss.simclr_loss,
                "predictive": torch.nn.functional.cross_entropy,
                "reconstructive": torch.nn.functional.mse_loss,
            }
            loss_func = loss_dict.get(self._sslname)
        if not (isinstance(loss_args, list) or isinstance(loss_args, dict) or loss_args == None):
            raise ValueError(
                "loss_args must be a list or a dict with all optional"
                " arguments of the loss function"
            )

        self.eval()
        N_test = len(test_dataloader)
        if isinstance(test_dataloader.sampler, EEGSampler):
            if test_dataloader.sampler.Keep_only_ratio != 1:
                if test_dataloader.drop_last:
                    N_test = math.floor(
                        sum(1 for _ in test_dataloader.sampler.__iter__())
                        / (test_dataloader.batch_size)
                    )
                else:
                    N_test = math.ceil(
                        sum(1 for _ in test_dataloader.sampler.__iter__())
                        / (test_dataloader.batch_size)
                    )
        return device, augmenter, loss_func, N_test
