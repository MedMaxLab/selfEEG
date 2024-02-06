from __future__ import annotations

import copy
import datetime
import math
import os
import random
import sys
from typing import Optional, Union

import torch
import torch.nn as nn
import tqdm

from ..dataloading import EEGsampler
from ..losses import losses as Loss

__all__ = [
    "Barlow_Twins",
    "BYOL",
    "EarlyStopping",
    "evaluateLoss",
    "fine_tune",
    "MoCo",
    "SimCLR",
    "SimSiam",
    "SSL_Base",
    "VICReg",
]


def Default_augmentation(x):
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


def evaluateLoss(
    loss_fun: "function",
    arguments: torch.Tensor or list[torch.Tensor],
    loss_arg: Union[list, dict] = None,
) -> "loss_fun output":
    """
    ``evaluateLoss`` evaluate a custom loss function using `arguments`
    as required arguments and loss_arg as optional ones. It is simply
    the ``SSL_Base's evaluate_loss`` method exported as a function.

    Parameters
    ----------
    loss_fun: function
        The custom loss function. It can be any loss function which
        accepts as input:

            1. the model's prediction (or predictions) and the true labels as required argument
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
    >>> loss = ssl.evaluateLoss(torch.nn.functional.mse_loss, [yhat,ytrue])
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
    loss_func: "function" = None,
    loss_args: list or dict = [],
    label_encoder: "function" = None,
    lr_scheduler=None,
    EarlyStopper=None,
    validation_dataloader: torch.utils.data.DataLoader = None,
    verbose=True,
    device: str or torch.device = None,
    return_loss_info: bool = False,
) -> Optional[dict]:
    """
    ``fine_tune`` is a custom fit function designed to perform
    fine tuning on a given model with the given dataloader.

    Parameters
    ----------
    model: nn.Module
        The pytorch model to fine tune. It must be a nn.Module.
    train_dataloader: Dataloader
        The pytorch Dataloader used to get the training batches. The Dataloar
        must return a batch as a tuple (X, Y), with X the feature tensor
        and Y the label tensor.
    epochs: int, optional
        The number of training epochs. It must be an integer bigger than 0.

        Default = 1
    optimizer: torch Optimizer, optional
        The optimizer used for weight's update. It can be any optimizer
        provided in the torch.optim module. If not given, Adam with default
        parameters will be instantiated.

        Default = None
    augmenter: function, optional
        Any function (or callable object) used to perform data augmentation
        on the batch. It is highly suggested to resort to the augmentation
        module, which implements different data augmentation functions and
        classes to combine them. Note that data augmentation is not performed
        on the validation set, since its goal is to increase the size of the
        training set and to get more different samples.

        Default = None
    loss_func: function, optional
        The custom loss function. It can be any loss function which
        accepts as input the model's prediction and the true labels
        as required arguments and loss_args as optional arguments.

        Default = None
    loss_args: Union[list, dict], optional
        The optional arguments to pass to the function. It can be a list
        or a dict.

        Default = None
    label_encoder: function, optional
        A custom function used to encode the returned Dataloaders true labels.
        If None, the Dataloader's true label is used directly. It can be any
        funtion which accept as input the batch label tensor Y.

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

        Device = None
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
    >>> import torch, pickle, selfeeg.losses
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
    >>> EEGlen = dl.GetEEGPartitionNumber('Simulated_EEG',128, 2, 0.3, load_function=loadEEG)
    >>> EEGsplit = dl.GetEEGSplitTable (EEGlen, seed=1234)
    >>> ratios = dl.check_split(EEGlen,EEGsplit, return_ratio=True)
    >>> TrainSet = dl.EEGDataset(EEGlen,EEGsplit, [128,2,0.3], 'train', True, loadEEG,
    ...                              optional_load_fun_args=[True], label_on_load=True)
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
        raise ValueError(
            "Current implementation accept only training data" " as a pytorch DataLoader"
        )
    if not (isinstance(epochs, int)):
        epochs = int(epochs)
    if epochs < 1:
        raise ValueError("epochs must be bigger than 1")
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())
    if loss_func is None:
        raise ValueError("loss function not given")
    if not (isinstance(loss_args, list) or isinstance(loss_args, dict)):
        raise ValueError(
            "loss_args must be a list or a dict with " "all optional arguments of the loss function"
        )

    perform_validation = False
    if validation_dataloader != None:
        if not (isinstance(validation_dataloader, torch.utils.data.DataLoader)):
            raise ValueError(
                "Current implementation accept only" " validation data as a pytorch DataLoader"
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
                if X.device.type != device.type:
                    X = X.to(device=device)
                if augmenter is not None:
                    X = augmenter(X)
                if label_encoder is not None:
                    Ytrue = label_encoder(Ytrue)
                if Ytrue.device.type != device.type:
                    Ytrue = Ytrue.to(device=device)

                Yhat = model(X)
                train_loss = evaluateLoss(loss_func, [Yhat, Ytrue], loss_args)

                train_loss.backward()
                optimizer.step()
                train_loss_tot += train_loss.item()
                # verbose print
                if verbose:
                    pbar.set_description(f" train {batch_idx+1:8<}/{len(train_dataloader):8>}")
                    pbar.set_postfix_str(
                        f"train_loss={train_loss_tot/(batch_idx+1):.5f}, val_loss={val_loss_tot:.5f}"
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

                        if X.device.type != device.type:
                            X = X.to(device=device)
                        if label_encoder != None:
                            Ytrue = label_encoder(Ytrue)
                        if Ytrue.device.type != device.type:
                            Ytrue = Ytrue.to(device=device)

                        Yhat = model(X)
                        val_loss = evaluateLoss(loss_func, [Yhat, Ytrue], loss_args)
                        val_loss_tot += val_loss.item()
                        if verbose:
                            pbar.set_description(
                                f"   val {batch_idx+1:8<}/{len(validation_dataloader):8>}"
                            )
                            pbar.set_postfix_str(
                                f"train_loss={train_loss_tot:.5f}, val_loss={val_loss_tot/(batch_idx+1):.5f}"
                            )
                            pbar.update()

                    val_loss_tot /= batch_idx + 1

        # Deal with earlystopper if given
        if EarlyStopper != None:
            updated_mdl = False
            curr_monitored = (
                val_loss_tot if EarlyStopper.monitored == "validation" else train_loss_tot
            )
            EarlyStopper.early_stop(curr_monitored)
            if EarlyStopper.record_best_weights:
                if EarlyStopper.best_loss == curr_monitored:
                    EarlyStopper.rec_best_weights(model)
                    updated_mdl = True
            if EarlyStopper():
                print(
                    "no improvement after {} epochs. Training stopped.".format(
                        EarlyStopper.patience
                    )
                )
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
    Simple implementation of an early stopping class for pytorch.
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

    Note
    ----
    Like in KERAS the early stopper will not automatically restore the best weights if
    the training ends, i.e., you reach the maximum number of epochs. To get the best weights
    simply call the ``restore_best_weights( model )`` method.

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
    >>> EEGlen = dl.GetEEGPartitionNumber('Simulated_EEG',128, 2, 0.3, load_function=loadEEG)
    >>> EEGsplit = dl.GetEEGSplitTable (EEGlen, seed=1234)
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
    ):

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
        the cpu device to avoid increasing the GPU load.

        Parameters
        ----------
        model: nn.Module
            The model to record.

        """
        self.best_model = copy.deepcopy(model).to(device="cpu").state_dict()

    def restore_best_weights(self, model):
        """
        Restore model's best weights.

        Parameters
        ----------
        model: nn.Module
            The model to restore.

        Warnings
        --------
        The model is moved to the CPU device before restoring weights.
        Remember to move it again to the desired device if cpu is not
        the selected one.

        """
        model.to(device="cpu")
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


class SSL_Base(nn.Module):
    """
    Baseline Self-Supervised Learning nn.Module. It is used as parent class
    by the other implemented SSL methods.

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
    >>> base = selfeeg.ssl.SSL_Base(enc)
    >>> torch.manual_seed(1234)
    >>> a = torch.randn(64,32)
    >>> print(base.evaluate_loss(losses.SimCLR_loss, [a])) # should return 9.4143
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
        super(SSL_Base, self).__init__()
        self.encoder = encoder

    def forward(self, x):
        """
        :meta private:

        """
        pass

    def evaluate_loss(
        self,
        loss_fun: "function",
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

            if isinstance(self, SimSiam):
                sslName = "_SimSiam"
            elif isinstance(self, VICReg):
                sslName = "_VICreg"
            elif isinstance(self, Barlow_Twins):
                sslName = "_BarTw"
            elif isinstance(self, BYOL):
                sslName = "_BYOL"
            elif isinstance(self, MoCo):
                sslName = "_MoCo"
            elif isinstance(self, SimCLR):
                sslName = "_SimCLR"
            else:
                sslName = "_Custom"
            save_path = (
                path
                + "/encoder_"
                + datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
                + sslName
                + ".pt"
            )
        torch.save(self.encoder.state_dict(), save_path)


class SimCLR(SSL_Base):
    """
    Implementation of the SimCLR SSL method. To check
    how SimCLR works, read the following paper [NTXent1]_ .
    Official repository at [simgit1]_ .

    Parameters
    ----------
    encoder: nn.Module
        The encoder part of the module. It is the one you wish to pretrain and
        transfer to the new model
    projection_head: Union[list[int], nn.Module]
        The projection head to use. It can be:

            1. an nn.Module
            2. a list of ints.

        In case a list of ints is given, a nn.Sequential module with Dense, BatchNorm
        and Relu will be automtically created. The list will be used to set
        input and output dimension of each Dense Layer. For instance, if
        [64, 128, 64] is given, two hidden layers will be created. The first
        with input 64 and output 128, the second with input 128 and output 64.

    Note
    ----
    BatchNorm is not applied to the last output layer due to reasons explained in
    more recent SSL works (see BYOL and SimSiam).

    Warnings
    --------
    This class will not check the compatibility of the encoder's output and
    the projection head's input. Make sure that they have the same size.

    References
    ----------
    .. [NTXent1] Chen et al. A Simple Framework for Contrastive Learning of Visual
      Representations. (2020). https://doi.org/10.48550/arXiv.2002.05709
    .. [simgit1] To check the original tensorflow implementation visit the following repository:
      https://github.com/google-research/simclr (look at the function add_contrastive_loss
      in objective.py)

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
    >>> def loss_fineTuning(yhat, ytrue):
    ...     return F.binary_cross_entropy_with_logits(torch.squeeze(yhat), ytrue + 0.)
    >>> torch.manual_seed(1234)
    >>> EEGlen = dl.GetEEGPartitionNumber('Simulated_EEG',freq=128, window=1,
    ...                              overlap=0.3, load_function=loadEEG)
    >>> EEGsplit = dl.GetEEGSplitTable (EEGlen, seed=1234)
    >>> TrainSet = dl.EEGDataset(EEGlen,EEGsplit,[128,1,0.3],'train',False,loadEEG)
    >>> Loader = torch.utils.data.DataLoader(TrainSet, batch_size=32)
    >>> enc = selfeeg.models.ShallowNetEncoder(8)
    >>> simclr = selfeeg.ssl.SimCLR(enc, [16,32,32])
    >>> print( simclr(torch.randn(32,8,128)).shape) # should return [32,32])
    >>> loss_train = simclr.fit(Loader, 1, return_loss_info=True)
    >>> print(loss_train[0][0]) # should return 4.39300
    >>> loss_test = simclr.test(Loader) # just to show it works
    >>> print(loss_test) # should return 3.6255

    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_head: Union[list[int], nn.Module],
    ):

        super(SimCLR, self).__init__(encoder)
        self.encoder = encoder

        if isinstance(projection_head, list):
            if len(projection_head) < 2:
                raise ValueError("got a list with only one element")
            else:
                if all(isinstance(i, int) for i in projection_head):
                    DenseList = []
                    for i in range(len(projection_head) - 1):
                        DenseList.append(nn.Linear(projection_head[i], projection_head[i + 1]))
                        # Batch Norm Not applied on output due to BYOL and SimSiam
                        # choices, since those two are more recent SSL implementations
                        DenseList.append(nn.BatchNorm1d(num_features=projection_head[i + 1]))
                        if i < (len(projection_head) - 2):
                            DenseList.append(nn.ReLU())
                    self.projection_head = nn.Sequential(*DenseList)
                else:
                    raise ValueError("got a list with non integer values")
        else:
            self.projection_head = projection_head

    def forward(self, x):
        """
        :meta private:

        """
        x = self.encoder(x)
        emb = self.projection_head(x)
        return emb

    def fit(
        self,
        train_dataloader,
        epochs=1,
        optimizer=None,
        augmenter=None,
        loss_func: "function" = None,
        loss_args: list or dict = [],
        lr_scheduler=None,
        EarlyStopper=None,
        validation_dataloader=None,
        verbose=True,
        device: str or torch.device = None,
        cat_augmentations: bool = False,
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
            classes to combine them. If none is given, a default augmentation with
            random vertical flip + random noise is applied.
            Note that in this case, contrary to fully supervised approaches,
            data augmentation is also performed on the validation set,
            since it's part of the SSL algorithm.

            Default = None
        loss_func: function, optional
            The custom loss function. It can be any loss function which
            accepts as input only the model's predictions as required arguments
            and loss_args as optional arguments.
            If not given SimCLR loss will be automatically used. Check the input
            arguments of ``SimCLR_loss`` to check how to design custom loss functions
            to give to this method

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

            Device = None
        cat_augmentations: bool, optional
            Whether to calculate the loss on the cat version of the two
            projection's or not. It might affect some statistical layers.

            Default = False
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

        """
        # Various checks on input parameters.
        # If some arguments weren't given they will be automatically set
        if device is None:
            # If device is None cannot assume if the model is on gpu and so if to send the batch
            # on an other device, so cpu will be used.
            # If model is sent to another device set the
            # device attribute with a proper string or torch device
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
                "Current implementation accept only training data" " as a pytorch DataLoader"
            )
        if not (isinstance(epochs, int)):
            epochs = int(epochs)
        if epochs < 1:
            raise ValueError("epochs must be bigger than 1")
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters())
        if augmenter is None:
            print("augmenter not given. Using a basic one with with flip + random noise")
            augmenter = Default_augmentation
        if loss_func is None:
            loss_func = Loss.SimCLR_loss
        if not (isinstance(loss_args, list) or isinstance(loss_args, dict) or loss_args == None):
            raise ValueError(
                "loss_args must be a list or a dict with all"
                " optional arguments of the loss function"
            )
        perform_validation = False
        if validation_dataloader != None:
            if not (isinstance(validation_dataloader, torch.utils.data.DataLoader)):
                raise ValueError(
                    "Current implementation accept only " "training data as a pytorch DataLoader"
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
        if isinstance(train_dataloader.sampler, EEGsampler):
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
        if perform_validation and isinstance(validation_dataloader.sampler, EEGsampler):
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
                bar_format="{desc}{percentage:3.0f}%|{bar:15}| {n_fmt}/{total_fmt}"
                " [{rate_fmt}{postfix}]",
                disable=not (verbose),
                unit=" Batch",
                file=sys.stdout,
            ) as pbar:
                for batch_idx, X in enumerate(train_dataloader):

                    optimizer.zero_grad()

                    if X.device.type != device.type:
                        X = X.to(device=device)

                    data_aug1 = augmenter(X)
                    data_aug2 = augmenter(X)

                    if cat_augmentations:
                        data_aug = torch.cat((data_aug1, data_aug2))
                        z = self(data_aug)
                        train_loss = self.evaluate_loss(loss_func, z, loss_args)
                    else:
                        z1 = self(data_aug1)
                        z2 = self(data_aug2)
                        train_loss = self.evaluate_loss(loss_func, torch.cat((z1, z2)), loss_args)

                    train_loss.backward()
                    optimizer.step()
                    train_loss_tot += train_loss.item()
                    # verbose print
                    if verbose:
                        pbar.set_description(f" train {batch_idx+1:8<}/{N_train:8>}")
                        pbar.set_postfix_str(
                            f"train_loss={train_loss_tot/(batch_idx+1):.5f}, val_loss={val_loss_tot:.5f}"
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

                            data_aug = torch.cat((augmenter(X), augmenter(X)), dim=0)
                            z = self(data_aug)
                            val_loss = self.evaluate_loss(loss_func, z, loss_args)
                            val_loss_tot += val_loss.item()
                            if verbose:
                                pbar.set_description(f"   val {batch_idx+1:8<}/{N_val:8>}")
                                pbar.set_postfix_str(
                                    f"train_loss={train_loss_tot:.5f}, val_loss={val_loss_tot/(batch_idx+1):.5f}"
                                )
                                pbar.update()

                        val_loss_tot /= batch_idx + 1

            # Deal with earlystopper if given
            if EarlyStopper != None:
                updated_mdl = False
                curr_monitored = (
                    val_loss_tot if EarlyStopper.monitored == "validation" else train_loss_tot
                )
                EarlyStopper.early_stop(curr_monitored)
                if EarlyStopper.record_best_weights:
                    if EarlyStopper.best_loss == curr_monitored:
                        EarlyStopper.rec_best_weights(self)
                        updated_mdl = True
                if EarlyStopper():
                    print(
                        "no improvement after {} epochs. Training stopped.".format(
                            EarlyStopper.patience
                        )
                    )
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
        loss_func: "function" = None,
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
        if device == None:
            # If device is None cannot assume if the model is on gpu and so if to send the batch
            # on other device, so cpu will be used. If model is sent to another device set the
            # device attribute with a proper string or torch device
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
            print("augmenter not given. Using a basic one with with flip + random noise")
            augmenter = Default_augmentation
        if loss_func == None:
            loss_func = Loss.SimCLR_loss
        if not (isinstance(loss_args, list) or isinstance(loss_args, dict) or loss_args == None):
            raise ValueError(
                "loss_args must be a list or a dict with all optional"
                " arguments of the loss function"
            )

        self.eval()
        N_test = len(test_dataloader)
        if isinstance(test_dataloader.sampler, EEGsampler):
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
        with torch.no_grad():
            test_loss = 0
            test_loss_tot = 0
            with tqdm.tqdm(
                total=N_test,
                ncols=100,
                bar_format="{desc}{percentage:3.0f}%|{bar:15}| {n_fmt}/{total_fmt}"
                " [{rate_fmt}{postfix}]",
                disable=not (verbose),
                unit=" Batch",
                file=sys.stdout,
            ) as pbar:
                for batch_idx, X in enumerate(test_dataloader):

                    if X.device.type != device.type:
                        X = X.to(device=device)

                    # two forward may be slower but uses less memory
                    data_aug1 = augmenter(X)
                    z1 = self(data_aug1)
                    data_aug2 = augmenter(X)
                    z2 = self(data_aug2)
                    test_loss = self.evaluate_loss(loss_func, torch.cat((z1, z2)), loss_args)
                    test_loss_tot += test_loss
                    # verbose print
                    if verbose:
                        pbar.set_description(f"   test {batch_idx+1:8<}/{N_test:8>}")
                        pbar.set_postfix_str(f"test_loss={test_loss_tot/(batch_idx+1):.5f}")
                        pbar.update()

                test_loss_tot /= batch_idx + 1
        return test_loss_tot


class SimSiam(SSL_Base):
    """
    Implementation of the SimSiam SSL method. To check
    how SimSIam works, read the following paper [simsiam1]_ .
    Official repo at [siamgit1]_ .

    Parameters
    ----------
    encoder: nn.Module
        The encoder part of the module. It is the one you wish to pretrain and
        transfer to the new model
    projection_head: Union[list[int], nn.Module]
        The projection head to use. It can be:

            1. an nn.Module
            2. a list of ints.

        In case a list is given, a nn.Sequential module with Dense, BatchNorm
        and Relu will be automtically created. The list will be used to set
        input and output dimension of each Dense Layer. For instance, if
        [64, 128, 64] is given, two hidden layers will be created. The first
        with input 64 and output 128, the second with input 128 and output 64.
    predictor: Union[list[int], nn.Module]
        The predictor to put after the projection head. Accepted arguments
        are the same as for the projection_head.

    Warnings
    --------
    This class will not check the compatibility of the encoder's output and
    the projection head's input (as well as between the projection head and the
    predictor). Make sure that they have the same size.

    References
    ----------
    .. [siamgit1] Original github repo: https://github.com/facebookresearch/simsiam
    .. [simsiam1] Original paper: Chen & He. Exploring Simple Siamese Representation Learning.
      https://arxiv.org/abs/2011.10566

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
    >>> def loss_fineTuning(yhat, ytrue):
    >>>     return F.binary_cross_entropy_with_logits(torch.squeeze(yhat), ytrue + 0.)
    >>> torch.manual_seed(1234)
    >>> EEGlen = dl.GetEEGPartitionNumber('Simulated_EEG',freq=128, window=1,
    ...                              overlap=0.3, load_function=loadEEG)
    >>> EEGsplit = dl.GetEEGSplitTable (EEGlen, seed=1234)
    >>> TrainSet = dl.EEGDataset(EEGlen,EEGsplit,[128,1,0.3],'train',False,loadEEG)
    >>> Loader = torch.utils.data.DataLoader(TrainSet, batch_size=32)
    >>> enc = selfeeg.models.ShallowNetEncoder(8)
    >>> simsiam = selfeeg.ssl.SimSiam(enc, [16,32,32], nn.Sequential(nn.Linear(32,32)))
    >>> print( simsiam(torch.randn(32,8,128)).shape) # should return [32,32])
    >>> loss_train = simsiam.fit(Loader, 1, return_loss_info=True)
    >>> print(loss_train[0][0]) # should return -0.6044
    >>> loss_test = simsiam.test(Loader) # just to show it works
    >>> print(loss_test) # should return -0.9273

    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_head: Union[list[int], nn.Module],
        predictor: Union[list[int], nn.Module],
    ):

        super(SimSiam, self).__init__(encoder)
        self.encoder = encoder
        if isinstance(projection_head, list):
            if len(projection_head) < 2:
                raise ValueError("got a list with only one element")
            else:
                if all(isinstance(i, int) for i in projection_head):
                    DenseList = []
                    for i in range(len(projection_head) - 1):
                        DenseList.append(nn.Linear(projection_head[i], projection_head[i + 1]))
                        DenseList.append(nn.BatchNorm1d(num_features=projection_head[i + 1]))
                        if i < (len(projection_head) - 2):
                            DenseList.append(nn.ReLU())
                    self.projection_head = nn.Sequential(*DenseList)
                else:
                    raise ValueError("got a list with non integer values")
        else:
            self.projection_head = projection_head

        if isinstance(predictor, list):
            if len(predictor) < 2:
                raise ValueError("got a list with only one element")
            else:
                if all(isinstance(i, int) for i in predictor):
                    DenseList = []
                    for i in range(len(predictor) - 1):
                        DenseList.append(nn.Linear(predictor[i], predictor[i + 1]))
                        if i < (len(predictor) - 2):
                            DenseList.append(nn.BatchNorm1d(num_features=predictor[i + 1]))
                            DenseList.append(nn.ReLU())
                    self.predictor = nn.Sequential(*DenseList)
                else:
                    raise ValueError("got a list with non integer values")
        else:
            self.predictor = predictor

    def forward(self, x):
        """
        :meta private:

        """
        x = self.encoder(x)
        x = self.projection_head(x)
        emb = self.predictor(x)
        return emb

    def fit(
        self,
        train_dataloader,
        epochs=1,
        optimizer=None,
        augmenter=None,
        loss_func: "function" = None,
        loss_args: list or dict = [],
        lr_scheduler=None,
        EarlyStopper=None,
        validation_dataloader=None,
        verbose: bool = True,
        device: str = None,
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
            classes to combine them. If none is given, a default augmentation with
            random vertical flip + random noise is applied.
            Note that in this case data augmentation
            is also performed on the validation set, since it is part of the
            SSL algorithm.

            Default = None
        loss_func: function, optional
            The custom loss function. It can be any loss function which
            accepts as input only the model's predictions (4 torch Tensor) as
            required arguments and loss_args as optional arguments. Check the input
            arguments of ``SimSiam_loss`` to check how to design custom loss functions
            to give to this method.

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
        validation_dataloader: Dataloader, optional
            the pytorch Dataloader used to get the validation batches. It
            must return a batch as a single tensor X, thus without label tensor Y.
            If not given, no validation loss will be calculated.

            Default = None
        verbose: bool, optional
            Whether to print a progression bar or not.

            Default = None
        device: torch.device or str, optional
            The device to use for fine-tuning. If given as a string it will
            be converted in a torch.device instance. If not given, 'cpu' device
            will be used.

            Device = None
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

        """

        # Various check on input parameters. If some arguments weren't given
        if device == None:
            # If device is None cannot assume if the model is on gpu and so if to send the batch
            # on other device, so cpu will be used. If model is sent to another device set the
            # device attribute with a proper string or torch device
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
                "Current implementation accept only training " "data as a pytorch DataLoader"
            )
        if not (isinstance(epochs, int)):
            epochs = int(epochs)
        if epochs < 1:
            raise ValueError("epochs must be bigger than 1")
        if optimizer == None:
            optimizer = torch.optim.Adam(self.parameters())
        if augmenter == None:
            print("augmenter not given. Using a basic one with with flip + random noise")
            augmenter = Default_augmentation
        if loss_func == None:
            loss_func = Loss.SimSiam_loss
        if not (isinstance(loss_args, list) or isinstance(loss_args, dict) or loss_args == None):
            raise ValueError(
                "loss_args must be a list or a dict with all"
                " optional arguments of the loss function"
            )
        perform_validation = False
        if validation_dataloader != None:
            if not (isinstance(validation_dataloader, torch.utils.data.DataLoader)):
                raise ValueError(
                    "Current implementation accepts only validation data as" " a pytorch DataLoader"
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

        loss_info = {i: [None, None] for i in range(epochs)}
        N_train = len(train_dataloader)
        if isinstance(train_dataloader.sampler, EEGsampler):
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
        if perform_validation and isinstance(validation_dataloader.sampler, EEGsampler):
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

        # trainin loop (classical pytorch style)
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
                bar_format="{desc}{percentage:3.0f}%|{bar:15}| {n_fmt}/{total_fmt}"
                " [{rate_fmt}{postfix}]",
                disable=not (verbose),
                unit=" Batch",
                file=sys.stdout,
            ) as pbar:
                for batch_idx, X in enumerate(train_dataloader):

                    optimizer.zero_grad()
                    if X.device.type != device.type:
                        X = X.to(device=device)

                    data_aug1 = augmenter(X)
                    data_aug2 = augmenter(X)

                    p1 = self(data_aug1)
                    p2 = self(data_aug2)

                    with torch.no_grad():
                        z1 = self.encoder(data_aug1)
                        z1 = self.projection_head(z1).detach()
                        z2 = self.encoder(data_aug2)
                        z2 = self.projection_head(z2).detach()

                    train_loss = self.evaluate_loss(loss_func, [p1, z1, p2, z2], loss_args)
                    train_loss.backward()
                    optimizer.step()
                    train_loss_tot += train_loss.item()
                    # verbose print
                    if verbose:
                        pbar.set_description(f" train {batch_idx+1:8<}/{N_train:8>}")
                        pbar.set_postfix_str(
                            f"train_loss={train_loss_tot/(batch_idx+1):.5f}, val_loss={val_loss_tot:.5f}"
                        )
                        pbar.update()
                train_loss_tot /= batch_idx + 1

                if lr_scheduler != None:
                    lr_scheduler.step()

                # Perform validation if validation dataloader were given
                if perform_validation:
                    self.eval()
                    with torch.no_grad():
                        val_loss = 0
                        val_loss_tot = 0
                        for batch_idx, X in enumerate(validation_dataloader):
                            if X.device.type != device.type:
                                X = X.to(device=device)

                            data_aug1 = augmenter(X)
                            data_aug2 = augmenter(X)

                            p1 = self(data_aug1)
                            z1 = self.encoder(data_aug1)
                            z1 = self.projection_head(z1).detach()

                            p2 = self(data_aug2)
                            z2 = self.encoder(data_aug2)
                            z2 = self.projection_head(z2).detach()
                            val_loss = self.evaluate_loss(loss_func, [p1, z1, p2, z2], loss_args)
                            val_loss_tot += val_loss.item()
                            if verbose:
                                pbar.set_description(f"   val {batch_idx+1:8<}/{N_val:8>}")
                                pbar.set_postfix_str(
                                    f"train_loss={train_loss_tot:.5f}, val_loss={val_loss_tot/(batch_idx+1):.5f}"
                                )
                                pbar.update()
                        val_loss_tot /= batch_idx + 1

            # Deal with earlystopper if given
            if EarlyStopper != None:
                updated_mdl = False
                curr_monitored = (
                    val_loss_tot if EarlyStopper.monitored == "validation" else train_loss_tot
                )
                EarlyStopper.early_stop(curr_monitored)
                if EarlyStopper.record_best_weights:
                    if EarlyStopper.best_loss == curr_monitored:
                        EarlyStopper.rec_best_weights(self)
                        updated_mdl = True
                if EarlyStopper():
                    print(
                        "no improvement after {} epochs. Training stopped.".format(
                            EarlyStopper.patience
                        )
                    )
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
        loss_func: "function" = None,
        loss_args: list or dict = [],
        verbose: bool = True,
        device: str = None,
    ):
        """
        A method to evaluate the loss on a test dataloader.
        Parameters are the same as described in the fit method, aside for
        those related to model training which are removed.

        It is rare to evaluate the pretraing loss function on a test set.
        Nevertheless this function provides a way to do that.
        An example of usage could be to assess the quality of the learned
        features on the fine-tuning dataset.

        """
        if device == None:
            # If device is None cannot assume if the model is on gpu and so if to send the batch
            # on other device, so cpu will be used. If model is sent to another device set the
            # device attribute with a proper string or torch device
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
            print("augmenter not given. Using a basic one with with flip + random noise")
            augmenter = Default_augmentation
        if loss_func == None:
            loss_func = Loss.SimSiam_loss
        if not (isinstance(loss_args, list) or isinstance(loss_args, dict) or loss_args == None):
            raise ValueError(
                "loss_args must be a list or a dict with all "
                "optional arguments of the loss function"
            )

        N_test = len(test_dataloader)
        if isinstance(test_dataloader.sampler, EEGsampler):
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
        self.eval()
        with torch.no_grad():
            test_loss = 0
            test_loss_tot = 0
            with tqdm.tqdm(
                total=N_test,
                ncols=100,
                bar_format="{desc}{percentage:3.0f}%|{bar:15}| {n_fmt}/{total_fmt}"
                " [{rate_fmt}{postfix}]",
                disable=not (verbose),
                unit=" Batch",
                file=sys.stdout,
            ) as pbar:
                for batch_idx, X in enumerate(test_dataloader):
                    if X.device.type != device.type:
                        X = X.to(device=device)

                    # two forward may be slower but uses less memory
                    data_aug1 = augmenter(X)
                    p1 = self(data_aug1)
                    z1 = self.encoder(data_aug1)
                    z1 = self.projection_head(z1).detach()

                    data_aug2 = augmenter(X)
                    p2 = self(data_aug2)
                    z2 = self.encoder(data_aug2)
                    z2 = self.projection_head(z2).detach()
                    test_loss = self.evaluate_loss(loss_func, [p1, z1, p2, z2], loss_args)
                    test_loss_tot += test_loss
                    # verbose print
                    if verbose:
                        pbar.set_description(f"   test {batch_idx+1:8<}/{N_test:8>}")
                        pbar.set_postfix_str(f"test_loss={test_loss_tot/(batch_idx+1):.5f}")
                        pbar.update()

                test_loss_tot /= batch_idx + 1
        return test_loss_tot


class MoCo(SSL_Base):
    """
    Implementation of the MoCo SSL method. To check
    how MoCo works, read the following paper [moco21]_ [moco31]_ .

    Parameters
    ----------
    encoder: nn.Module
        The encoder part of the module. It is the one you wish to pretrain and
        transfer to the new model.
    projection_head: Union[list[int], nn.Module]
        The projection head to use. It can be:

            1. an nn.Module
            2. a list of ints.

        In case a list is given, a nn.Sequential module with Dense, BatchNorm
        and Relu will be automtically created. The list will be used to set
        input and output dimension of each Dense Layer. For instance, if
        [64, 128, 64] is given, two hidden layers will be created. The first
        with input 64 and output 128, the second with input 128 and output 64.
    predictor: Union[list[int], nn.Module], optional
        The predictor to put after the projection head. Use it with 0 bank size to set
        Moco v3. Accepted arguments are the same as for the projection_head.

        Default = None
    feat_size: int, optional
        The size of the feature vector (projector's output last dim shape).
        It will be used to initialize the queue for MoCo v2. If not given
        the last element of the projection_head list is used. It
        must be given if a custom projection head is used.

        Default = -1
    bank_size: int, optional
        The size of the queue, i.e. the number of projection to keep memory.
        If not given, fit will trigger the calculation of the MoCo v3 loss.

        Default = 0
    m: float, optional
        The value of the momentum coefficient. Suggested values are in the
        range [0.9960, 0.9999].

        Default = 0.999

    Warnings
    --------
    This class will not check the compatibility of the encoder's output and
    the projection head's input (as well as between the projection head and the
    predictor). Make sure that they have the same size.

    Warnings
    --------
    Using ADAM optimizer with MoCo v2 (with bank size) can prevent the training loss
    from decreasing. SGD is highly suggested.

    References
    ----------
    .. [moco21] K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick,
      “Momentum contrast for unsupervised visual representation learning,” in Proceedings of
      the IEEE/CVF conference on computer vision and pattern recognition, pp. 9729–9738, 2020.
    .. [moco31] X. Chen, H. Fan, R. Girshick, and K. He, “Improved base- lines with momentum
      contrastive learning,” arXiv preprint arXiv:2003.04297, 2020.

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
    >>> def loss_fineTuning(yhat, ytrue):
    ...     return F.binary_cross_entropy_with_logits(torch.squeeze(yhat), ytrue + 0.)
    >>> torch.manual_seed(1234)
    >>> EEGlen = dl.GetEEGPartitionNumber('Simulated_EEG',freq=128, window=1,
    ...                              overlap=0.3, load_function=loadEEG)
    >>> EEGsplit = dl.GetEEGSplitTable (EEGlen, seed=1234)
    >>> TrainSet = dl.EEGDataset(EEGlen,EEGsplit,[128,1,0.3],'train',False,loadEEG)
    >>> Loader = torch.utils.data.DataLoader(TrainSet, batch_size=32)
    >>> enc = selfeeg.models.ShallowNetEncoder(8)

    MoCo v2

    >>> moco2 = selfeeg.ssl.MoCo(enc, [16,32,32], bank_size=4096)
    >>> print( moco2(torch.randn(32,8,128)).shape) # should return [32,32])
    >>> loss_train = moco2.fit(Loader, 1, return_loss_info=True)
    >>> print(loss_train[0][0]) # should return 79.2622
    >>> loss_test = moco2.test(Loader) # just to show it works
    >>> print(loss_test) # should return 85.8382

    MoCo v3

    >>> moco3 = selfeeg.ssl.MoCo(enc, [16,32,32], [32,32])
    >>> print( moco3(torch.randn(32,8,128)).shape) # should return [32,32])
    >>> loss_train = moco3.fit(Loader, 1, return_loss_info=True)
    >>> print(loss_train[0][0]) # should return 0.93120
    >>> loss_test = moco3.test(Loader) # just to show it works
    >>> print(loss_test) # should return 0.8531

    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_head: Union[list[int], nn.Module],
        predictor: Union[list[int], nn.Module] = None,
        feat_size: int = -1,
        bank_size: int = 0,
        m: float = 0.999,
    ):

        super(MoCo, self).__init__(encoder)

        self.bank_size = bank_size
        self.m = m

        self.encoder = encoder
        self.momentum_encoder = copy.deepcopy(encoder)

        if isinstance(projection_head, list):
            if len(projection_head) < 2:
                raise ValueError("got a list with only one element")
            else:
                if all(isinstance(i, int) for i in projection_head):
                    DenseList = []
                    for i in range(len(projection_head) - 1):
                        DenseList.append(nn.Linear(projection_head[i], projection_head[i + 1]))
                        DenseList.append(nn.BatchNorm1d(num_features=projection_head[i + 1]))
                        if i < (len(projection_head) - 2):
                            DenseList.append(nn.ReLU())
                    self.projection_head = nn.Sequential(*DenseList)
                    self.momentum_projection_head = nn.Sequential(*DenseList)

                else:
                    raise ValueError("got a list with non integer values")
        else:
            self.projection_head = projection_head
            self.momentum_projection_head = copy.deepcopy(projection_head)

        self.predictor = None
        if predictor != None:
            if isinstance(predictor, list):
                if len(predictor) < 2:
                    raise ValueError("got a list with only one element")
                else:
                    if all(isinstance(i, int) for i in predictor):
                        DenseList = []
                        for i in range(len(predictor) - 1):
                            DenseList.append(nn.Linear(predictor[i], predictor[i + 1]))
                            DenseList.append(nn.BatchNorm1d(num_features=predictor[i + 1]))
                            if i < (len(predictor) - 2):
                                DenseList.append(nn.ReLU())
                        self.predictor = nn.Sequential(*DenseList)
                    else:
                        raise ValueError("got a list with non integer values")
            else:
                self.predictor = predictor

        if (self.predictor is None) and (bank_size <= 0):
            msgWarning = "You are trying to initialize MoCo with only the projection head and "
            msgWarning += "no memory bank. Training will follow MoCo v3 setup for loss"
            msgWarning += " calculation. Training will follow MoCo v3 setup for loss calculation"
            msgWarning += " during training, but it's suggested to set up an 2-hidden layer MLP"
            msgWarning += " predictor as in the original paper"
            print(msgWarning)

        if self.bank_size > 0:
            # need to set feature vector size
            if feat_size > 0:
                self.feat_size = feat_size
            elif isinstance(projection_head, list):
                self.feat_size = projection_head[-1]
            else:
                msgErr = "feature size cannot be extracted from a nn.Module. Please "
                msgErr += "provide the feature size, otherwise memory bank cannot be initialized"
                raise ValueError(msgErr)
            # create the queue
            self.register_buffer("queue", torch.randn(self.feat_size, self.bank_size))
            self.queue = nn.functional.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        for param_base, param_mom in zip(
            self.encoder.parameters(), self.momentum_encoder.parameters()
        ):
            param_mom.requires_grad = False
        for param_base, param_mom in zip(
            self.projection_head.parameters(), self.momentum_projection_head.parameters()
        ):
            param_mom.requires_grad = False

    @torch.no_grad()
    def _update_momentum_encoder(self):
        """
        :meta private:

        """
        for param_b, param_m in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * self.m + param_b.data * (1.0 - self.m)
        for param_b, param_m in zip(
            self.projection_head.parameters(), self.momentum_projection_head.parameters()
        ):
            param_m.data = param_m.data * self.m + param_b.data * (1.0 - self.m)

    @torch.no_grad()
    def _update_queue(self, keys):
        """
        :meta private:

        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if batch_size > self.bank_size:
            raise ValueError("cannot add a batch bigger than bank size")

        if (ptr + batch_size) > self.bank_size:
            diff1 = self.bank_size - ptr
            diff2 = batch_size - diff1
            self.queue[:, ptr:] = keys[:diff1].T
            self.queue[:, :diff2] = keys[diff1:].T
            self.queue_ptr[0] = diff2
        else:
            self.queue[:, ptr : ptr + batch_size] = keys.T
            self.queue_ptr += batch_size

    def forward(self, x):
        """
        :meta private:

        """
        x = self.encoder(x)
        emb = self.projection_head(x)
        if self.predictor != None:
            emb = self.predictor(emb)
        return emb

    def fit(
        self,
        train_dataloader,
        epochs=1,
        optimizer=None,
        augmenter=None,
        loss_func: "function" = None,
        loss_args: list or dict = [],
        lr_scheduler=None,
        EarlyStopper=None,
        validation_dataloader=None,
        verbose: bool = True,
        device: str = None,
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
            provided in the torch.optim module. If not given:

                1. SGD with learning rate 0.01 will be used for moco v2
                2. Adam with default parameters will be used for moco v3.

            Default = torch.optim.Adam
        augmenter: function, optional
            Any function (or callable object) used to perform data augmentation
            on the batch. It is highly suggested to resort to the augmentation
            module, which implements different data augmentation functions and
            classes to combine them. If none is given, a default augmentation with
            random vertical flip + random noise is applied.
            Note that in this case data augmentation
            is also performed on the validation set, since it is part of the
            SSL algorithm.

            Default = None
        loss_func: function, optional
            The custom loss function. It can be any loss function which
            accepts as input only the model's predictions (2 torch Tensor) as
            required arguments and loss_args as optional arguments. Check the input
            arguments of ``Moco_loss`` to check how to design custom loss functions
            to give to this method.

            Default = None
        loss_args: Union[list, dict], optional
            The optional arguments to pass to the function. It can be a list
            or a dict.

            Default = None
        label_encoder: function, optional
            A custom function used to encode the returned Dataloaders true labels.
            If None, the Dataloader's true label is used directly.

            Default = None
        lr_scheduler: torch Scheduler
            A pytorch learning rate scheduler used to update the learning rate
            during fine-tuning.

            Default = None
        EarlyStopper: EarlyStopping, optional
            An instance of the provided EarlyStopping class.

            Default = None
        validation_dataloader: Dataloader, optional
            The pytorch Dataloader used to get the validation batches. It
            must return a batch as a single tensor X, thus without label tensor Y.
            If not given, no validation loss will be calculated.

            Default = None
        verbose: bool, optional
            Whether to print a progression bar or not.

            Default = None
        device: torch.device or str, optional
            The device to use for fine-tuning. If given as a string it will
            be converted in a torch.device instance. If not given, 'cpu' device
            will be used.

            Device = None
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

        """

        # Various check on input parameters. If some arguments weren't given
        if device == None:
            # If device is None cannot assume if the model is on gpu and so if to send the batch
            # on other device, so cpu will be used. If model is sent to another device set the
            # device attribute with a proper string or torch device
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
                "Current implementation accept only training data" " as a pytorch DataLoader"
            )
        if not (isinstance(epochs, int)):
            epochs = int(epochs)
        if epochs < 1:
            raise ValueError("epochs must be bigger than 1")
        if optimizer == None:
            if self.bank_size is not None:
                optimizer = torch.optim.SGD(self.parameters(), 0.01)
            else:
                optimizer = torch.optim.Adam(self.parameters(), 0.001)
        if augmenter == None:
            print("augmenter not given. Using a basic one with with flip + random noise")
            augmenter = Default_augmentation
        if loss_func == None:
            loss_func = Loss.Moco_loss
        if not (isinstance(loss_args, list) or isinstance(loss_args, dict) or loss_args == None):
            raise ValueError(
                "loss_args must be a list or a dict with all"
                " optional arguments of the loss function"
            )
        perform_validation = False
        if validation_dataloader != None:
            if not (isinstance(validation_dataloader, torch.utils.data.DataLoader)):
                raise ValueError(
                    "Current implementation accepts only validation data as" " a pytorch DataLoader"
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

        loss_info = {i: [None, None] for i in range(epochs)}
        N_train = len(train_dataloader)
        if isinstance(train_dataloader.sampler, EEGsampler):
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
        if perform_validation and isinstance(validation_dataloader.sampler, EEGsampler):
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
                bar_format="{desc}{percentage:3.0f}%|{bar:15}| {n_fmt}/{total_fmt}"
                " [{rate_fmt}{postfix}]",
                disable=not (verbose),
                unit=" Batch",
                file=sys.stdout,
            ) as pbar:
                for batch_idx, X in enumerate(train_dataloader):

                    if X.device.type != device.type:
                        X = X.to(device=device)

                    optimizer.zero_grad()
                    self._update_momentum_encoder()

                    data_aug1 = augmenter(X)
                    data_aug2 = augmenter(X)

                    if self.bank_size > 0:
                        # follow moco v2 setup
                        q = self(data_aug1)  # queries
                        with torch.no_grad():
                            k = self.momentum_encoder(data_aug2)
                            k = self.momentum_projection_head(k)
                            k = k.detach()  # keys
                        self._update_queue(k)
                        train_loss = self.evaluate_loss(loss_func, [q, k, self.queue], loss_args)
                    else:
                        # if no memory bank, follow moco v3 setup
                        q1 = self(data_aug1)
                        q2 = self(data_aug2)
                        with torch.no_grad():
                            k1 = self.momentum_encoder(data_aug1)
                            k1 = self.momentum_projection_head(k1)
                            k1 = k1.detach()  # keys
                            k2 = self.momentum_encoder(data_aug2)
                            k2 = self.momentum_projection_head(k2)
                            k2 = k2.detach()  # keys
                        train_loss1 = self.evaluate_loss(loss_func, [q1, k2], loss_args)
                        train_loss2 = self.evaluate_loss(loss_func, [q2, k1], loss_args)
                        train_loss = train_loss1 + train_loss2

                    train_loss.backward()
                    optimizer.step()
                    train_loss_tot += train_loss.item()
                    # verbose print
                    if verbose:
                        pbar.set_description(f" train {batch_idx+1:8<}/{N_train:8>}")
                        pbar.set_postfix_str(
                            f"train_loss={train_loss_tot/(batch_idx+1):.5f}, val_loss={val_loss_tot:.5f}"
                        )
                        pbar.update()
                train_loss_tot /= batch_idx + 1

                if lr_scheduler != None:
                    lr_scheduler.step()

                # Perform validation if validation dataloader were given
                # Note that validation in moco can be misleading if there's a memory_bank
                # since calculated keys cannot be added
                if perform_validation:
                    self.eval()
                    with torch.no_grad():
                        val_loss = 0
                        val_loss_tot = 0
                        for batch_idx, X in enumerate(validation_dataloader):
                            if X.device.type != device.type:
                                X = X.to(device=device)

                            data_aug1 = augmenter(X)
                            data_aug2 = augmenter(X)

                            if self.bank_size > 0:
                                # follow moco v2 setup
                                q = self(data_aug1)  # queries
                                k = self.momentum_encoder(data_aug2)
                                k = self.momentum_projection_head(k)
                                k = k.detach()  # keys
                                val_loss = self.evaluate_loss(
                                    loss_func, [q, k, self.queue], loss_args
                                )
                            else:
                                # if no memory bank, follow moco v3 setup
                                q1 = self(data_aug1)
                                q2 = self(data_aug2)
                                k1 = self.momentum_encoder(data_aug1)
                                k1 = self.momentum_projection_head(k1)
                                k1 = k1.detach()  # keys
                                k2 = self.momentum_encoder(data_aug2)
                                k2 = self.momentum_projection_head(k2)
                                k2 = k2.detach()  # keys
                                val_loss1 = self.evaluate_loss(loss_func, [q1, k2], loss_args)
                                val_loss2 = self.evaluate_loss(loss_func, [q2, k1], loss_args)
                                val_loss = val_loss1 + val_loss2
                            val_loss_tot += val_loss.item()
                            if verbose:
                                pbar.set_description(f"   val {batch_idx+1:8<}/{N_val:8>}")
                                pbar.set_postfix_str(
                                    f"train_loss={train_loss_tot:.5f}, val_loss={val_loss_tot/(batch_idx+1):.5f}"
                                )
                                pbar.update()
                        val_loss_tot /= batch_idx + 1

            # Deal with earlystopper if given
            if EarlyStopper != None:
                updated_mdl = False
                curr_monitored = (
                    val_loss_tot if EarlyStopper.monitored == "validation" else train_loss_tot
                )
                EarlyStopper.early_stop(curr_monitored)
                if EarlyStopper.record_best_weights:
                    if EarlyStopper.best_loss == curr_monitored:
                        EarlyStopper.rec_best_weights(self)
                        updated_mdl = True
                if EarlyStopper():
                    print(
                        "no improvement after {} epochs. Training stopped.".format(
                            EarlyStopper.patience
                        )
                    )
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
        loss_func: "function" = None,
        loss_args: list or dict = [],
        verbose: bool = True,
        device: str = None,
    ):
        """
        A method to evaluate the loss on a test dataloader.
        Parameters are the same as described in the fit method, aside for
        those related to model training which are removed.

        It is rare to evaluate the pretraing loss function on a test set.
        Nevertheless this function provides a way to do that.
        An example of usage could be to assess the quality of the learned
        features on the fine-tuning dataset.

        """

        if device == None:
            # If device is None cannot assume if the model is on gpu and so if to send the batch
            # on other device, so cpu will be used. If model is sent to another device set the
            # device attribute with a proper string or torch device
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
                "Current implementation accept only test" " data as a pytorch DataLoader"
            )
        if augmenter == None:
            print("augmenter not given. Using a basic one with with flip + random noise")
            augmenter = Default_augmentation
        if loss_func == None:
            loss_func = Loss.Moco_loss
        if not (isinstance(loss_args, list) or isinstance(loss_args, dict) or loss_args == None):
            raise ValueError(
                "loss_args must be a list or a dict with all"
                " optional arguments of the loss function"
            )

        N_test = len(test_dataloader)
        if isinstance(test_dataloader.sampler, EEGsampler):
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
        self.eval()
        with torch.no_grad():
            test_loss = 0
            test_loss_tot = 0
            with tqdm.tqdm(
                total=N_test,
                ncols=100,
                bar_format="{desc}{percentage:3.0f}%|{bar:15}| {n_fmt}/{total_fmt}"
                " [{rate_fmt}{postfix}]",
                disable=not (verbose),
                unit=" Batch",
                file=sys.stdout,
            ) as pbar:
                for batch_idx, X in enumerate(test_dataloader):
                    if X.device.type != device.type:
                        X = X.to(device=device)

                    data_aug1 = augmenter(X)
                    data_aug2 = augmenter(X)

                    if self.bank_size > 0:
                        # follow moco v2 setup
                        q = self(data_aug1)  # queries
                        k = self.momentum_encoder(data_aug2)
                        k = self.momentum_projection_head(k)
                        k = k.detach()  # keys
                        test_loss = self.evaluate_loss(loss_func, [q, k, self.queue], loss_args)
                    else:
                        # if no memory bank, follow moco v3 setup
                        q1 = self(data_aug1)
                        q2 = self(data_aug2)
                        k1 = self.momentum_encoder(data_aug1)
                        k1 = self.momentum_projection_head(k1)
                        k1 = k1.detach()  # keys
                        k2 = self.momentum_encoder(data_aug2)
                        k2 = self.momentum_projection_head(k2)
                        k2 = k2.detach()  # keys
                        test_loss1 = self.evaluate_loss(loss_func, [q1, k2], loss_args)
                        test_loss2 = self.evaluate_loss(loss_func, [q2, k1], loss_args)
                        test_loss = test_loss1 + test_loss2
                    test_loss_tot += test_loss
                    # verbose print
                    if verbose:
                        pbar.set_description(f"   test {batch_idx+1:8<}/{N_test:8>}")
                        pbar.set_postfix_str(f"test_loss={test_loss_tot/(batch_idx+1):.5f}")
                        pbar.update()

                test_loss_tot /= batch_idx + 1
        return test_loss_tot


class BYOL(SSL_Base):
    """
    Implementation of the BYOL SSL method. To check
    how BYOL works, read the following paper [BYOL1]_ .

    Parameters
    ----------
    encoder: nn.Module
        The encoder part of the module. It is the one you wish to pretrain and
        transfer to the new model.
    projection_head: Union[list[int], nn.Module]
        The projection head to use. It can be:

            1. an nn.Module
            2. a list of ints.

        In case a list is given, a nn.Sequential module with Dense, BatchNorm
        and Relu will be automtically created. The list will be used to set
        input and output dimension of each Dense Layer. For instance, if
        [64, 128, 64] is given, two hidden layers will be created. The first
        with input 64 and output 128, the second with input 128 and output 64.
    predictor: Union[list[int], nn.Module]
        The predictor to put after the projection head. Accepted arguments
        are the same as for the projection_head.
    m: float, optional
        The value of the momentum coefficient. Suggested values are in the
        range [0.9960, 0.9999].

        Default = 0.999

    Warnings
    --------
    This class will not check the compatibility of the encoder's output and
    the projection head's input (as well as between the projection head and the
    predictor). Make sure that they have the same size.

    References
    ----------
    .. [BYOL1] J.-B. Grill, F. Strub, F. Altché, C. Tallec, P. Richemond, E. Buchatskaya,
      C. Doersch, B. Avila Pires, Z. Guo, M. Gheshlaghi Azar, et al., “Bootstrap your own
      latent- a new approach to self-supervised learning,” Advances in neural information
      processing systems, vol. 33, pp. 21271– 21284, 2020.

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
    >>> def loss_fineTuning(yhat, ytrue):
    ...     return F.binary_cross_entropy_with_logits(torch.squeeze(yhat), ytrue + 0.)
    >>> torch.manual_seed(1234)
    >>> # usual pipeline to construct the dataloader
    >>> EEGlen = dl.GetEEGPartitionNumber('Simulated_EEG',freq=128, window=1,
    ...                              overlap=0.3, load_function=loadEEG)
    >>> EEGsplit = dl.GetEEGSplitTable (EEGlen, seed=1234)
    >>> TrainSet = dl.EEGDataset(EEGlen,EEGsplit,[128,1,0.3],'train',False,loadEEG)
    >>> TrainLoader = torch.utils.data.DataLoader(TrainSet, batch_size=32)
    >>> enc = selfeeg.models.ShallowNetEncoder(8)
    >>> byol = selfeeg.ssl.BYOL(enc, [16,32,32], [32,32])
    >>> print( byol(torch.randn(32,8,128)).shape) # should return [32,32])
    >>> loss_train = byol.fit(TrainLoader, 1, return_loss_info=True)
    >>> print(loss_train[0][0]) # will return 2.16688
    >>> loss_test = byol.test(TrainLoader)
    >>> print(loss_test) # will return 1.2185

    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_head: Union[list[int], nn.Module],
        predictor: Union[list[int], nn.Module] = None,
        m: float = 0.999,
    ):

        super(BYOL, self).__init__(encoder)

        self.m = m
        self.encoder = encoder
        self.momentum_encoder = copy.deepcopy(encoder)

        if isinstance(projection_head, list):
            if len(projection_head) < 2:
                raise ValueError("got a list with only one element")
            else:
                if all(isinstance(i, int) for i in projection_head):
                    DenseList = []
                    for i in range(len(projection_head) - 1):
                        DenseList.append(nn.Linear(projection_head[i], projection_head[i + 1]))
                        if i < (len(projection_head) - 2):
                            DenseList.append(nn.BatchNorm1d(num_features=projection_head[i + 1]))
                            DenseList.append(nn.ReLU())
                    self.projection_head = nn.Sequential(*DenseList)
                    self.momentum_projection_head = nn.Sequential(*DenseList)

                else:
                    raise ValueError("got a list with non integer values")
        else:
            self.projection_head = projection_head
            self.momentum_projection_head = copy.deepcopy(projection_head)

        if isinstance(predictor, list):
            if len(predictor) < 2:
                raise ValueError("got a list with only one element")
            else:
                if all(isinstance(i, int) for i in predictor):
                    DenseList = []
                    for i in range(len(predictor) - 1):
                        DenseList.append(nn.Linear(predictor[i], predictor[i + 1]))
                        if i < (len(predictor) - 2):
                            DenseList.append(nn.BatchNorm1d(num_features=predictor[i + 1]))
                            DenseList.append(nn.ReLU())
                    self.predictor = nn.Sequential(*DenseList)
                else:
                    raise ValueError("got a list with non integer values")
        else:
            self.predictor = predictor

        for param_base, param_mom in zip(
            self.encoder.parameters(), self.momentum_encoder.parameters()
        ):
            param_mom.requires_grad = False
        for param_base, param_mom in zip(
            self.projection_head.parameters(), self.momentum_projection_head.parameters()
        ):
            param_mom.requires_grad = False

    @torch.no_grad()
    def _update_momentum_encoder(self):
        """
        :meta private:

        """
        for param_b, param_m in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * self.m + param_b.data * (1.0 - self.m)
        for param_b, param_m in zip(
            self.projection_head.parameters(), self.momentum_projection_head.parameters()
        ):
            param_m.data = param_m.data * self.m + param_b.data * (1.0 - self.m)

    def forward(self, x):
        """
        :meta private:

        """
        x = self.encoder(x)
        x = self.projection_head(x)
        emb = self.predictor(x)
        return emb

    def fit(
        self,
        train_dataloader,
        epochs=1,
        optimizer=None,
        augmenter=None,
        loss_func: "function" = None,
        loss_args: list or dict = [],
        lr_scheduler=None,
        EarlyStopper=None,
        validation_dataloader=None,
        verbose: bool = True,
        device: str = None,
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
            classes to combine them. If none is given, a default augmentation with
            random vertical flip + random noise is applied.
            Note that in this case data augmentation
            is also performed on the validation set, since it is part of the
            SSL algorithm.

            Default = None
        loss_func: function, optional
            The custom loss function. It can be any loss function which
            accepts as input only the model's predictions (4 torch Tensor) as
            required arguments and loss_args as optional arguments. Check the input
            arguments of ``BYOL_loss`` to check how to design custom loss functions
            to give to this method.

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
        validation_dataloader: Dataloader, optional
            The pytorch Dataloader used to get the validation batches.  It
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

            Device = None
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

        """

        # Various check on input parameters. If some arguments weren't given
        if device == None:
            # If device is None cannot assume if the model is on gpu and so if to send the batch
            # on other device, so cpu will be used. If model is sent to another device set the
            # device attribute with a proper string or torch device
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
                "Current implementation accept only" " training data as a pytorch DataLoader"
            )
        if not (isinstance(epochs, int)):
            epochs = int(epochs)
        if epochs < 1:
            raise ValueError("epochs must be bigger than 1")
        if optimizer == None:
            optimizer = torch.optim.Adam(self.parameters())
        if augmenter == None:
            print("augmenter not given. Using a basic one with with flip + random noise")
            augmenter = Default_augmentation
        if loss_func == None:
            loss_func = Loss.BYOL_loss
        if not (isinstance(loss_args, list) or isinstance(loss_args, dict) or loss_args == None):
            raise ValueError(
                "loss_args must be a list or a dict with all"
                " optional arguments of the loss function"
            )
        perform_validation = False
        if validation_dataloader != None:
            if not (isinstance(validation_dataloader, torch.utils.data.DataLoader)):
                raise ValueError(
                    "Current implementation accepts only validation data as" " a pytorch DataLoader"
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

        loss_info = {i: [None, None] for i in range(epochs)}
        N_train = len(train_dataloader)
        if isinstance(train_dataloader.sampler, EEGsampler):
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
        if perform_validation and isinstance(validation_dataloader.sampler, EEGsampler):
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

        # classical torch training loop with some additions
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
                bar_format="{desc}{percentage:3.0f}%|{bar:15}| {n_fmt}/{total_fmt}"
                " [{rate_fmt}{postfix}]",
                disable=not (verbose),
                unit=" Batch",
                file=sys.stdout,
            ) as pbar:
                for batch_idx, X in enumerate(train_dataloader):

                    if X.device.type != device.type:
                        X = X.to(device=device)

                    optimizer.zero_grad()
                    self._update_momentum_encoder()

                    data_aug1 = augmenter(X)
                    data_aug2 = augmenter(X)

                    p1 = self(data_aug1)
                    p2 = self(data_aug2)
                    with torch.no_grad():
                        z1 = self.momentum_encoder(data_aug1)
                        z1 = self.momentum_projection_head(z1)
                        z1 = z1.detach()  # keys
                        z2 = self.momentum_encoder(data_aug2)
                        z2 = self.momentum_projection_head(z2)
                        z2 = z2.detach()  # keys
                    train_loss = self.evaluate_loss(loss_func, [p1, z1, p2, z2], loss_args)

                    train_loss.backward()
                    optimizer.step()
                    train_loss_tot += train_loss.item()
                    # verbose print
                    if verbose:
                        pbar.set_description(f" train {batch_idx+1:8<}/{N_train:8>}")
                        pbar.set_postfix_str(
                            f"train_loss={train_loss_tot/(batch_idx+1):.5f}, val_loss={val_loss_tot:.5f}"
                        )
                        pbar.update()
                train_loss_tot /= batch_idx + 1

                if lr_scheduler != None:
                    lr_scheduler.step()

                # Perform validation if validation dataloader were given
                # Note that validation in moco can be misleading if there's a memory_bank
                # since calculated keys cannot be added
                if perform_validation:
                    self.eval()
                    with torch.no_grad():
                        for batch_idx, X in enumerate(validation_dataloader):
                            if X.device.type != device.type:
                                X = X.to(device=device)

                            data_aug1 = augmenter(X)
                            data_aug2 = augmenter(X)
                            p1 = self(data_aug1)
                            p2 = self(data_aug2)
                            z1 = self.momentum_encoder(data_aug1)
                            z1 = self.momentum_projection_head(z1)
                            z1 = z1.detach()  # keys
                            z2 = self.momentum_encoder(data_aug2)
                            z2 = self.momentum_projection_head(z2)
                            z2 = z2.detach()  # keys
                            val_loss = self.evaluate_loss(loss_func, [p1, z1, p2, z2], loss_args)

                            val_loss_tot += val_loss.item()
                            if verbose:
                                pbar.set_description(f"   val {batch_idx+1:8<}/{N_val:8>}")
                                pbar.set_postfix_str(
                                    f"train_loss={train_loss_tot:.5f}, val_loss={val_loss_tot/(batch_idx+1):.5f}"
                                )
                                pbar.update()
                        val_loss_tot /= batch_idx + 1

            # Deal with earlystopper if given
            if EarlyStopper != None:
                updated_mdl = False
                curr_monitored = (
                    val_loss_tot if EarlyStopper.monitored == "validation" else train_loss_tot
                )
                EarlyStopper.early_stop(curr_monitored)
                if EarlyStopper.record_best_weights:
                    if EarlyStopper.best_loss == curr_monitored:
                        EarlyStopper.rec_best_weights(self)
                        updated_mdl = True
                if EarlyStopper():
                    print(
                        "no improvement after {} epochs. Training stopped.".format(
                            EarlyStopper.patience
                        )
                    )
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
        loss_func: "function" = None,
        loss_args: list or dict = [],
        verbose: bool = True,
        device: str = None,
    ):
        """
        A method to evaluate the loss on a test dataloader.
        Parameters are the same as in the fit method, apart for the
        ones specific for the training which are removed.

        It is rare to evaluate the pretraing loss function on a test set.
        Nevertheless this function provides a way to do that.
        An example of usage could be to assess the quality of the learned
        features on the fine-tuning dataset.

        """

        if device == None:
            # If device is None cannot assume if the model is on gpu and so if to send the batch
            # on other device, so cpu will be used. If model is sent to another device set the
            # device attribute with a proper string or torch device
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
                "Current implementation accept only training data as a pytorch DataLoader"
            )
        if augmenter == None:
            print("augmenter not given. Using a basic one with with flip + random noise")
            augmenter = Default_augmentation
        if loss_func == None:
            loss_func = Loss.BYOL_loss
        if not (isinstance(loss_args, list) or isinstance(loss_args, dict) or loss_args == None):
            raise ValueError(
                "loss_args must be a list or a dict with all optional arguments of the loss function"
            )

        N_test = len(test_dataloader)
        if isinstance(test_dataloader.sampler, EEGsampler):
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
        self.eval()
        with torch.no_grad():
            test_loss = 0
            test_loss_tot = 0
            with tqdm.tqdm(
                total=N_test,
                ncols=100,
                bar_format="{desc}{percentage:3.0f}%|{bar:15}| {n_fmt}/{total_fmt}"
                " [{rate_fmt}{postfix}]",
                disable=not (verbose),
                unit=" Batch",
                file=sys.stdout,
            ) as pbar:
                for batch_idx, X in enumerate(test_dataloader):
                    if X.device.type != device.type:
                        X = X.to(device=device)

                    data_aug1 = augmenter(X)
                    data_aug2 = augmenter(X)

                    p1 = self(data_aug1)
                    p2 = self(data_aug2)
                    z1 = self.momentum_encoder(data_aug1)
                    z1 = self.momentum_projection_head(z1)
                    z1 = z1.detach()  # keys
                    z2 = self.momentum_encoder(data_aug2)
                    z2 = self.momentum_projection_head(z2)
                    z2 = z2.detach()  # keys
                    test_loss = self.evaluate_loss(loss_func, [p1, z1, p2, z2], loss_args)
                    test_loss_tot += test_loss
                    # verbose print
                    if verbose:
                        pbar.set_description(f"   test {batch_idx+1:8<}/{N_test:8>}")
                        pbar.set_postfix_str(f"test_loss={test_loss_tot/(batch_idx+1):.5f}")
                        pbar.update()

                test_loss_tot /= batch_idx + 1
        return test_loss_tot


class Barlow_Twins(SimCLR):
    """
    Implementation of the Barlow twins SSL method. To check
    how Barlow Twins works, read the following paper [barlow1]_ .

    Parameters
    ----------
    encoder: nn.Module
        The encoder part of the module. It is the one you wish to pretrain and
        transfer to the new model
    projection_head: Union[list[int], nn.Module]
        The projection head to use. It can be:

            1. an nn.Module
            2. a list of ints.

        In case a list is given, a nn.Sequential module with Dense, BatchNorm
        and Relu will be automtically created. The list will be used to set
        input and output dimension of each Dense Layer. For instance, if
        [64, 128, 64] is given, two hidden layers will be created. The first
        with input 64 and output 128, the second with input 128 and output 64.

    Warnings
    --------
    This class will not check the compatibility of the encoder's output and
    the projection head's input (as well as between the projection head).
    Make sure that they have the same size.

    References
    ----------
    .. [barlow1] J. Zbontar, L. Jing, I. Misra, Y. LeCun, and S. Deny,
      “Barlow twins: Self-supervised learning via redundancy re- duction,” in International
      Conference on Machine Learning, pp. 12310–12320, PMLR, 2021.

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
    >>> def loss_fineTuning(yhat, ytrue):
    ...     return F.binary_cross_entropy_with_logits(torch.squeeze(yhat), ytrue + 0.)
    >>> torch.manual_seed(1234)
    >>> # usual pipeline to construct the dataloader
    >>> EEGlen = dl.GetEEGPartitionNumber('Simulated_EEG',freq=128, window=1,
    ...                              overlap=0.3, load_function=loadEEG)
    >>> EEGsplit = dl.GetEEGSplitTable (EEGlen, seed=1234)
    >>> TrainSet = dl.EEGDataset(EEGlen,EEGsplit,[128,1,0.3],'train',False,loadEEG)
    >>> TrainLoader = torch.utils.data.DataLoader(TrainSet, batch_size=32)
    >>> enc = selfeeg.models.ShallowNetEncoder(8)
    >>> barl = selfeeg.ssl.Barlow_Twins(enc, [16,32,32])
    >>> print( barl(torch.randn(32,8,128)).shape) # should return [32,32])
    >>> loss_train = barl.fit(TrainLoader, 1, return_loss_info=True)
    >>> print(loss_train[0][0]) # will return 3.8910
    >>> loss_test = barl.test(TrainLoader)
    >>> print(loss_test) # will return 2.1368

    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_head: Union[list[int], nn.Module],
    ):
        super(Barlow_Twins, self).__init__(encoder, projection_head)

    def fit(
        self,
        train_dataloader,
        epochs=1,
        optimizer=None,
        augmenter=None,
        loss_func: "function" = None,
        loss_args: list or dict = [],
        lr_scheduler=None,
        EarlyStopper=None,
        validation_dataloader=None,
        verbose: bool = True,
        device: str = None,
        cat_augmentations: bool = False,
        return_loss_info: bool = False,
    ):
        """
        ``fit`` is a custom fit function designed to perform
        pretraining on a given model with the given dataloader.

        Parameters
        ----------
        train_dataloader: Dataloader
            The pytorch Dataloader used to get the training batches.  It
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
            classes to combine them. If none is given, a default augmentation with
            random vertical flip + random noise is applied.
            Note that in this case data augmentation
            is also performed on the validation set, since it's part of the
            SSL algorithm.

            Default = None
        loss_func: function, optional
            The custom loss function. It can be any loss function which
            accepts as input only the model's predictions as required arguments
            and loss_args as optional arguments.
            If not given Barlow's loss will be automatically used. Check the input
            arguments of ``Barlow_loss`` to check how to design custom loss functions
            to give to this method.

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
        validation_dataloader: Dataloader, optional
            The pytorch Dataloader used to get the validation batches.  It
            must return a batch as a single tensor X, thus without label tensor Y.
            If not given, no validation loss will be calculated.

            Default = None
        verbose: bool, optional
            Whether to print a progression bar or not.

            Default = None
        device: torch.device or str, optional
            The device to use for fine-tuning. If given as a string it will
            be converted in a torch.device instance. If not given, 'cpu' device
            will be used.

            Device = None
        cat_augmentations: bool, optional
            Whether to calculate the loss on the cat version of the two
            projection's or not. It might affect some statistical layers.

            Default = False
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

        """

        if loss_func == None:
            loss_func = Loss.Barlow_loss
            loss_args = [None]
        loss_info = super().fit(
            train_dataloader,
            epochs,
            optimizer,
            augmenter,
            loss_func,
            loss_args,
            lr_scheduler,
            EarlyStopper,
            validation_dataloader,
            verbose,
            device,
            cat_augmentations,
            return_loss_info,
        )
        if return_loss_info:
            return loss_info

    def test(
        self,
        test_dataloader,
        augmenter=None,
        loss_func: "function" = None,
        loss_args: list or dict = [],
        verbose: bool = True,
        device: str = None,
    ):
        if loss_func == None:
            loss_func = Loss.Barlow_loss
            loss_args = [None]
        loss_info = super().test(test_dataloader, augmenter, loss_func, loss_args, verbose, device)
        return loss_info


class VICReg(SimCLR):
    """
    Implementation of the VICReg SSL method. To check
    how VICReg works, read the following paper [VIC1]_ .

    Parameters
    ----------
    encoder: nn.Module
        The encoder part of the module. It is the one you wish to pretrain and
        transfer to the new model
    projection_head: Union[list[int], nn.Module]
        The projection head to use. It can be:

            1. an nn.Module
            2. a list of ints.

        In case a list is given, a nn.Sequential module with Dense, BatchNorm
        and Relu will be automtically created. The list will be used to set
        input and output dimension of each Dense Layer. For instance, if
        [64, 128, 64] is given, two hidden layers will be created. The first
        with input 64 and output 128, the second with input 128 and output 64.
    predictor: Union[list[int], nn.Module]
        The predictor to put after the projection head. Accepted arguments
        are the same as for the projection_head.

    Warnings
    --------
    This class will not check the compatibility of the encoder's output and
    the projection head's input (as well as between the projection head and the
    predictor). Make sure that they have the same size.

    References
    ----------
    .. [VIC1] A. Bardes, J. Ponce, and Y. LeCun, “Vicreg: Variance- invariance-covariance
      regularization for self-supervised learning,” arXiv preprint arXiv:2105.04906, 2021.

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
    >>> def loss_fineTuning(yhat, ytrue):
    ...     return F.binary_cross_entropy_with_logits(torch.squeeze(yhat), ytrue + 0.)
    >>> torch.manual_seed(1234)
    >>> # usual pipeline to construct the dataloader
    >>> EEGlen = dl.GetEEGPartitionNumber('Simulated_EEG',freq=128, window=1,
    ...                              overlap=0.3, load_function=loadEEG)
    >>> EEGsplit = dl.GetEEGSplitTable (EEGlen, seed=1234)
    >>> TrainSet = dl.EEGDataset(EEGlen,EEGsplit,[128,1,0.3],'train',False,loadEEG)
    >>> TrainLoader = torch.utils.data.DataLoader(TrainSet, batch_size=32)
    >>> enc = selfeeg.models.ShallowNetEncoder(8)
    >>> vic = selfeeg.ssl.VICReg(enc, [16,32,32])
    >>> print( vic(torch.randn(32,8,128)).shape) # should return [32,32])
    >>> loss_train = vic.fit(TrainLoader, 2, return_loss_info=True)
    >>> print(loss_train[0][0]) # will return 21.6086
    >>> loss_test = vic.test(TrainLoader)
    >>> print(loss_test) # will return 21.6785

    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_head: Union[list[int], nn.Module],
    ):
        super(VICReg, self).__init__(encoder, projection_head)

    def fit(
        self,
        train_dataloader,
        epochs=1,
        optimizer=None,
        augmenter=None,
        loss_func: "function" = None,
        loss_args: list or dict = [],
        lr_scheduler=None,
        EarlyStopper=None,
        validation_dataloader=None,
        verbose: bool = True,
        device: str = None,
        cat_augmentations: bool = False,
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
            classes to combine them. If none is given, a default augmentation with
            random vertical flip + random noise is applied.
            Note that in this case data augmentation
            is also performed on the validation set, since it is part of the
            SSL algorithm.

            Default = None
        loss_func: function, optional
            The custom loss function. It can be any loss function which
            accepts as input only the model's predictions as required arguments
            and loss_args as optional arguments.
            If not given VICReg loss will be automatically used. Check the input
            arguments of ``VICReg_loss`` to check how to design custom loss functions
            to give to this method

            Default = None
        loss_args: Union[list, dict], optional
            The optional arguments to pass to the function. it can be a list
            or a dict.

            Default = None
        lr_scheduler: torch Scheduler
            A pytorch learning rate scheduler used to update the learning rate
            during the fine-tuning.

            Default = None
        EarlyStopper: EarlyStopping, optional
            An instance of the provided EarlyStopping class.

            Default = None
        validation_dataloader: Dataloader, optional
            The pytorch Dataloader used to get the validation batches.  It
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

            Device = None
        cat_augmentations: bool, optional
            Whether to calculate the loss on the cat version of the two
            projection's or not. It might affect some statistical layers.

            Default = False
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

        """

        if loss_func == None:
            loss_func = Loss.VICReg_loss
            loss_args = []
        loss_info = super().fit(
            train_dataloader,
            epochs,
            optimizer,
            augmenter,
            loss_func,
            loss_args,
            lr_scheduler,
            EarlyStopper,
            validation_dataloader,
            verbose,
            device,
            cat_augmentations,
            return_loss_info,
        )
        if return_loss_info:
            return loss_info

    def test(
        self,
        test_dataloader,
        augmenter=None,
        loss_func: "function" = None,
        loss_args: list or dict = [],
        verbose: bool = True,
        device: str = None,
    ):

        if loss_func == None:
            loss_func = Loss.VICReg_loss
            loss_args = []
        loss_info = super().test(test_dataloader, augmenter, loss_func, loss_args, verbose, device)
        return loss_info
