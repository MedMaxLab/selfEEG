import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import ConstrainedDense, ConstrainedConv1d, ConstrainedConv2d
from .encoders import (
    BasicBlock1,
    DeepConvNetEncoder,
    EEGInceptionEncoder,
    EEGConformerEncoder,
    EEGNetEncoder,
    EEGSymEncoder,
    FBCNetEncoder,
    ResNet1DEncoder,
    ShallowNetEncoder,
    StagerNetEncoder,
    STNetEncoder,
    TinySleepNetEncoder,
    xEEGNetEncoder,
)
from ..utils.utils import _reset_seed

__all__ = [
    "ATCNet",
    "DeepConvNet",
    "EEGConformer",
    "EEGInception",
    "EEGNet",
    "EEGSym",
    "FBCNet",
    "ResNet1D",
    "ShallowNet",
    "StagerNet",
    "STNet",
    "TinySleepNet",
    "xEEGNet",
]


# ------------------------------
#           EEGNet
# ------------------------------
class EEGNet(nn.Module):
    """
    Pytorch implementation of the EEGNet model.

    For more information see the following paper [EEGnet]_ .
    Keras implementation of the full EEGnet (updated version), more info
    can be found here [eegnetgit]_ .

    The expected **input** is a **3D tensor** with size
    (Batch x Channels x Samples).

    Parameters
    ----------
    nb_classes: int
        The number of classes. If less than 2, a binary classification problem
        is considered (output dimensions will be [batch, 1] in this case).
    Chans: int
        The number of EEG channels.
    Samples: int
        The sample length. It will be used to calculate the embedding size
        (for head initialization).
    kernlength: int, optional
        The length of the temporal convolutional layer.

        Default = 64
    dropRate: float, optional
        The dropout percentage in range [0,1].

        Default = 0.5
    F1: int, optional
        The number of filters in the first layer.

        Default = 8
    D: int, optional
        The depth of the depthwise convolutional layer.

        Default = 16
    dropType: str, optional
        The type of dropout. It can be either 'Dropout' or 'SpatialDropout2D'.

        Default = 'Dropout'
    ELUalpha: float, optional
        The alpha value of the ELU activation function.

        Default = 1
    pool1: int, optional
        The first temporal average pooling kernel size.

        Default = 4
    pool2: int, optional
        The second temporal average pooling kernel size.

        Default = 8
    separable_kernel: int, optional
        The temporal separable conv layer kernel size.

        Default = 16
    depthwise_max_norm: float, optional
        The maximum norm each filter can have in the depthwise block.
        If None no constraint will be included.

        Default = None
    return_logits: bool, optional
        Whether to return the output as logit or probability.
        It is suggested to not use False as the pytorch crossentropy loss function
        applies the softmax internally.

        Default = True
    seed: int, optional
        A custom seed for model initialization. It must be a nonnegative number.
        If None is passed, no custom seed will be set

        Default = None

    Note
    ----
    This implementation refers to the latest version of EEGNet which
    can be found in the official repository (see references).

    References
    ----------
    .. [EEGnet] Lawhern et al., EEGNet: a compact convolutional neural network
      for EEG-based brain–computer interfaces. Journal of Neural Engineering. 2018
    .. [eegnetgit] https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py

    Example
    -------
    >>> import selfeeg.models
    >>> import torch
    >>> x = torch.randn(4,8,64)
    >>> mdl = models.EEGNet(4,8,64)
    >>> out = mdl(x)
    >>> print(out.shape) # shoud return torch.Size([4, 4])
    >>> print(torch.isnan(out).sum()) # shoud return 0

    """

    def __init__(
        self,
        nb_classes: int,
        Chans: int,
        Samples: int,
        kernLength: int = 64,
        dropRate: float = 0.5,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        norm_rate: int = 0.25,
        dropType: str = "Dropout",
        ELUalpha: int = 1,
        pool1: int = 4,
        pool2: int = 8,
        separable_kernel: int = 16,
        depthwise_max_norm: float = 1.0,
        return_logits: bool = True,
        seed: int = None,
    ):

        super(EEGNet, self).__init__()

        self.nb_classes = nb_classes
        self.return_logits = return_logits
        self.encoder = EEGNetEncoder(
            Chans,
            kernLength,
            dropRate,
            F1,
            D,
            F2,
            dropType,
            ELUalpha,
            pool1,
            pool2,
            separable_kernel,
            depthwise_max_norm,
            seed,
        )

        _reset_seed(seed)
        self.Dense = ConstrainedDense(
            F2 * (Samples // int(pool1 * pool2)),
            1 if nb_classes <= 2 else nb_classes,
            max_norm=norm_rate,
        )

    def forward(self, x):
        """
        :meta private:
        """
        x = self.encoder(x)
        x = self.Dense(x)
        if not (self.return_logits):
            if self.nb_classes <= 2:
                x = torch.sigmoid(x)
            else:
                x = F.softmax(x, dim=1)
        return x


# ------------------------------
#          DeepConvNet
# ------------------------------
class DeepConvNet(nn.Module):
    """
    Pytorch Implementation of the DeepConvNet model.

    Official paper can be found here [deepconv]_ .
    A Keras implementation can be found here [deepconvgit]_ .

    The expected **input** is a **3D tensor** with size
    (Batch x Channels x Samples).

    Parameters
    ----------
    nb_classes: int
        The number of classes. If less than 2, a binary classification
        problem is considered (output dimensions will be [batch, 1] in this case).
    Chans: int
        The number of EEG channels.
    Samples: int
        The sample length. It will be used to calculate the embedding size
        (for head initialization).
    kernlength: int, optional
        The length of the temporal convolutional layer.

        Default = 10
    F: int, optional
        The number of filters in the first layer. Next layers
        will continue to double the previous output.

        Default = 25
    Pool: int, optional
        The temporal pooling kernel size.

        Default = 3
    stride: int, optional
        The stride to apply to the convolutional layers.

        Default = 3
    max_norm: int, optional
        A max norm constraint to apply to each filter of the convolutional layer.
        See ``ConstrainedConv2d`` for more info.

        Default = None
    batch_momentum: float, optional
        The batch normalization momentum.

        Default = 0.9
    ELUalpha: float, optional
        The alpha value of the ELU activation function.

        Default = 1
    dropRate: float, optional
        The dropout percentage in range [0,1].

        Default = 0.5
    max_dense_norm: int, optional
        A max norm constraint to apply to the DenseLayer. See
        ``ConstrainedDense`` for more info.

        Default = None
    return_logits: bool, optional
        Whether to return the output as logit or probability.
        It is suggested to not use False as the pytorch crossentropy applies
        the softmax internally.

        Default = True
    seed: int, optional
        A custom seed for model initialization. It must be a nonnegative number.
        If None is passed, no custom seed will be set

        Default = None

    Note
    ----
    This implementation refers to the original implementation of DeepConvNet.
    So, no max norm constraints were applied on the network layers.

    References
    ----------
    .. [deepconv] Schirrmeister, Robin Tibor, et al. "Deep learning with
      convolutional neural networks for EEG decoding and visualization."
      Human brain mapping 38.11 (2017): 5391-5420.
      https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/hbm.23730
    .. [deepconvgit] https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py

    Example
    -------
    >>> import selfeeg.models
    >>> import torch
    >>> x = torch.randn(4,8,512)
    >>> mdl = models.DeepConvNet(4,8,512)
    >>> out = mdl(x)
    >>> print(out.shape) # shoud return torch.Size([4, 4])
    >>> print(torch.isnan(out).sum()) # shoud return 0

    """

    def __init__(
        self,
        nb_classes: int,
        Chans: int,
        Samples: int,
        kernLength: int = 10,
        F: int = 25,
        Pool: int = 3,
        stride: int = 3,
        max_norm: int = None,
        batch_momentum: float = 0.1,
        ELUalpha: int = 1,
        dropRate: float = 0.5,
        max_dense_norm: float = None,
        return_logits: bool = True,
        seed: int = None,
    ):
        super(DeepConvNet, self).__init__()

        self.nb_classes = nb_classes
        self.return_logits = return_logits
        self.encoder = DeepConvNetEncoder(
            Chans, kernLength, F, Pool, stride, max_norm, batch_momentum, ELUalpha, dropRate, seed
        )
        k = kernLength
        Dense_input = [Samples] * 8
        for i in range(4):
            Dense_input[i * 2] = Dense_input[i * 2 - 1] - k + 1
            Dense_input[i * 2 + 1] = (Dense_input[i * 2] - Pool) // stride + 1

        _reset_seed(seed)
        self.Dense = ConstrainedDense(
            F * 8 * Dense_input[-1], 1 if nb_classes <= 2 else nb_classes, max_norm=max_dense_norm
        )

    def forward(self, x):
        """
        :meta private:
        """
        x = self.encoder(x)
        x = self.Dense(x)
        if not (self.return_logits):
            if self.nb_classes <= 2:
                x = torch.sigmoid(x)
            else:
                x = F.softmax(x, dim=1)
        return x


# ------------------------------
#         EEGInception
# ------------------------------
class EEGInception(nn.Module):
    """
    Pytorch Implementation of the EEGInception model.
    Original paper can be found here [eeginc]_ .
    A Keras implementation can be found here [eegincgit]_ .

    The expected **input** is a **3D tensor** with size
    (Batch x Channels x Samples).

    Parameters
    ----------
    nb_classes: int
        The number of classes. If less than 2, a binary classification
        problem is considered (output dimensions will be [batch, 1] in this case).
    Chans: int
        The number of EEG channels.
    Samples: int
        The sample length. It will be used to calculate the embedding size
        (for head initialization).
    F1: int, optional
        The number of filters in the first temporal convolutional layer.
        Other output filters will be calculated according to the paper
        specification.

        Default = 8
    D: int, optional
        The depth of the depthwise convolutional layer.

        Default = 2
    kernel_size: int, optional
        The length of the temporal convolutional layer.

        Default = 64
    pool: int, optional
        The temporal pooling kernel size.

        Default = 4
    dropRate: float, optional
        The dropout percentage in range [0,1].

        Default = 0.5
    ELUalpha: float, optional
        The alpha value of the ELU activation function.

        Default = 1
    bias: bool, optional
        If True, add a learnable bias to the output.

        Default = True
    batch_momentum: float, optional
        The batch normalization momentum.

        Default = 0.9
    max_depth_norm: float, optional
        The maximum norm each filter can have in the depthwise block.
        If None no constraint will be included.

        Default = 1.
    return_logits: bool, optional
        Whether to return the output as logit or probability.
        It is suggested to not use False as
        the pytorch crossentropy applies the softmax internally.

        Default = True
    seed: int, optional
        A custom seed for model initialization. It must be a nonnegative number.
        If None is passed, no custom seed will be set

        Default = None

    References
    ----------
    .. [eeginc] Zhang, Ce, Young-Keun Kim, and Azim Eskandarian.
      "EEG-inception: an accurate and robust end-to-end neural network for
      EEG-based motor imagery classification." Journal of Neural Engineering
      18.4 (2021): 046014.
      https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9311146
    .. [eegincgit] https://github.com/esantamariavazquez/EEG-Inception/blob/main/

    Example
    -------
    >>> import selfeeg.models
    >>> import torch
    >>> x = torch.randn(4,8,64)
    >>> mdl = models.EEGInception(4,8,64)
    >>> out = mdl(x)
    >>> print(out.shape) # shoud return torch.Size([4, 4])
    >>> print(torch.isnan(out).sum()) # shoud return 0

    """

    def __init__(
        self,
        nb_classes: int,
        Chans: int,
        Samples: int,
        F1: int = 8,
        D: int = 2,
        kernel_size: int = 64,
        pool: int = 4,
        dropRate: float = 0.5,
        ELUalpha: float = 1.0,
        bias: bool = True,
        batch_momentum: float = 0.1,
        max_depth_norm: float = 1.0,
        return_logits: bool = True,
        seed: int = None,
    ):
        super(EEGInception, self).__init__()
        self.nb_classes = nb_classes
        self.return_logits = return_logits
        self.encoder = EEGInceptionEncoder(
            Chans,
            F1,
            D,
            kernel_size,
            pool,
            dropRate,
            ELUalpha,
            bias,
            batch_momentum,
            max_depth_norm,
            seed,
        )

        _reset_seed(seed)
        self.Dense = nn.Linear(
            int((F1 * 3) / 4) * int((Samples // (pool * (int(pool // 2) ** 3)))),
            1 if nb_classes <= 2 else nb_classes,
        )

    def forward(self, x):
        """
        :meta private:

        """
        x = self.encoder(x)
        x = self.Dense(x)
        if not (self.return_logits):
            if self.nb_classes <= 2:
                x = torch.sigmoid(x)
            else:
                x = F.softmax(x, dim=1)
        return x


# ------------------------------
#         TinySleepNet
# ------------------------------
class TinySleepNet(nn.Module):
    """
    Pytorch Implementation of the TinySleepNet model.

    TinySleepNet is a minimal but better performing architecture derived from
    DeepSleepNet (proposed by the same authors).
    Paper can be found here [tiny]_ .
    Github repo can be found here [tinygit]_ .

    The expected **input** is a **3D tensor** with size
    (Batch x Channels x Samples).

    Parameters
    ----------
    nb_classes: int
        The number of classes. If less than 2, a binary classification
        problem is considered (output dimensions will be [batch, 1] in this case).
    Chans: int
        The number of EEG channels.
    Fs: float
        The EEG sampling frequency in Hz.
    F1: int, optional
        The number of output filters in the representation learning part.

        Default = 128
    kernlength: int, optional
        The length of the temporal convolutional layer.

        Default = 8
    pool: int, optional
        The temporal pooling kernel size.

        Default = 8
    dropRate: float, optional
        The dropout percentage in range [0,1].

        Default = 0.5
    batch_momentum: float, optional
        The batch normalization momentum.

        Default = 0.9
    max_dense_norm: float, optional
        A value indicating the max norm constraint to
        apply on the final dense layer. If None no constraint will be included.

        Default = 1.
    hidden_lstm: int, optional
        Hidden size of the lstm block.

        Default = 128
    return_logits: bool, optional
        Whether to return the output as logit or probability. It is suggested
        to not use False as the pytorch crossentropy applies the softmax internally.

        Default = True
    seed: int, optional
        A custom seed for model initialization. It must be a nonnegative number.
        If None is passed, no custom seed will be set

        Default = None

    References
    ----------
    .. [tiny] Supratak, Akara, and Yike Guo. "TinySleepNet: An efficient deep
      learning model for sleep stage scoring based on raw single-channel EEG."
      2020 42nd Annual International Conference of the IEEE Engineering in
      Medicine & Biology Society (EMBC). IEEE, 2020.
      https://ieeexplore.ieee.org/abstract/document/9176741
    .. [tinygit] https://github.com/akaraspt/tinysleepnet

    Example
    -------
    >>> import selfeeg.models
    >>> import torch
    >>> x = torch.randn(4,8,1024)
    >>> mdl = models.TinySleepNet(4,8,32)
    >>> out = mdl(x)
    >>> print(out.shape) # shoud return torch.Size([4, 4])
    >>> print(torch.isnan(out).sum()) # shoud return 0

    """

    def __init__(
        self,
        nb_classes: int,
        Chans: int,
        Fs: int,
        F: int = 128,
        kernlength: int = 8,
        pool: int = 8,
        dropRate: float = 0.5,
        batch_momentum: float = 0.1,
        max_dense_norm: float = 2.0,
        hidden_lstm: int = 128,
        return_logits: bool = True,
        seed: int = None,
    ):
        super(TinySleepNet, self).__init__()

        self.nb_classes = nb_classes
        self.return_logits = return_logits
        self.encoder = TinySleepNetEncoder(
            Chans, Fs, F, kernlength, pool, dropRate, batch_momentum, hidden_lstm, seed
        )

        _reset_seed(seed)
        self.drop3 = nn.Dropout1d(dropRate)
        self.Dense = ConstrainedDense(
            hidden_lstm, 1 if nb_classes <= 2 else nb_classes, max_norm=max_dense_norm
        )

    def forward(self, x):
        """
        :meta private:
        """
        x = self.encoder(x)
        x = self.drop3(x)
        x = self.Dense(x)
        if not (self.return_logits):
            if self.nb_classes <= 2:
                x = torch.sigmoid(x)
            else:
                x = F.softmax(x, dim=1)
        return x


# ------------------------------
#          StagerNet
# ------------------------------
class StagerNet(nn.Module):
    """
    Pytorch implementation of the StagerNet model.

    Original paper can be found here [stager]_ .
    The expected **input** is a **3D tensor** with size
    (Batch x Channels x Samples).

    Parameters
    ----------
    nb_classes: int
        The number of classes. If less than 2, a binary classification
        problem is considered (output dimensions will be [batch, 1] in this case).
    Chans: int
        The number of EEG channels.
    Samples: int
        The sample length. It will be used to calculate the embedding size
        (for head initialization).
    dropRate: float, optional
        The dropout percentage in range [0,1].

        Default = 0.5
    kernLength: int, optional
        The length of the temporal convolutional layer.

        Default = 64
    F: int, optional
        The number of output filters in the temporal convolution layer.

        Default = 8
    pool: int, optional
        The temporal pooling kernel size.

        Default = 16
    return_logits: bool, optional
        Whether to return the output as logit or probability.  It is suggested
        to not use False as the pytorch crossentropy applies the softmax internally.

        Default = True
    seed: int, optional
        A custom seed for model initialization. It must be a nonnegative number.
        If None is passed, no custom seed will be set

        Default = None

    References
    ----------
    .. [stager] Chambon et al., A deep learning architecture for temporal
      sleep stage classification using multivariate and multimodal time series,
      arXiv:1707.03321

    Example
    -------
    >>> import selfeeg.models
    >>> import torch
    >>> x = torch.randn(4,8,512)
    >>> mdl = models.StagerNet(4,8,512)
    >>> out = mdl(x)
    >>> print(out.shape) # shoud return torch.Size([4, 4])
    >>> print(torch.isnan(out).sum()) # shoud return 0

    """

    def __init__(
        self,
        nb_classes: int,
        Chans: int,
        Samples: int,
        dropRate: float = 0.5,
        kernLength: int = 64,
        F: int = 8,
        Pool: int = 16,
        return_logits: bool = True,
        seed: int = None,
    ):

        super(StagerNet, self).__init__()
        self.nb_classes = nb_classes
        self.return_logits = return_logits
        self.encoder = StagerNetEncoder(Chans, kernLength=kernLength, F=F, Pool=Pool, seed=seed)

        _reset_seed(seed)
        self.drop = nn.Dropout(p=dropRate)
        self.Dense = nn.Linear(
            Chans * F * (int((int((Samples - Pool) / Pool + 1) - Pool) / Pool + 1)),
            1 if nb_classes <= 2 else nb_classes,
        )

    def forward(self, x):
        """
        :meta private:
        """
        x = self.encoder(x)
        x = self.drop(x)
        x = self.Dense(x)
        if not (self.return_logits):
            if self.nb_classes <= 2:
                x = torch.sigmoid(x)
            else:
                x = F.softmax(x, dim=1)
        return x


# ------------------------------
#         ShallowNet
# ------------------------------
class ShallowNet(nn.Module):
    """
    Pytorch implementation of the ShallowNet model.

    Original paper can be found here [shall]_ .
    The expected **input** is a **3D tensor** with size
    (Batch x Channels x Samples).

    Parameters
    ----------
    nb_classes: int
        The number of classes. If less than 2, a binary classification
        problem is considered (output dimensions will be [batch, 1] in this case).
    Chans: int
        The number of EEG channels.
    Samples: int
        The sample length. It will be used to calculate the embedding size
        (for head initialization).
    F: int, optional
        The number of output filters in the temporal convolution layer.

        Default = 8
    K1: int, optional
        The length of the temporal convolutional layer.

        Default = 25
    Pool: int, optional
        The temporal pooling kernel size.

        Default = 75
    p: float, optional
        The dropout probability. Must be in [0,1)

        Default= 0.2
    return_logits: bool, optional
        Whether to return the output as logit or probability.  It is suggested
        to not use False as the pytorch crossentropy applies the softmax internally.

        Default = True
    seed: int, optional
        A custom seed for model initialization. It must be a nonnegative number.
        If None is passed, no custom seed will be set

        Default = None

    Note
    ----
    In this implementation, the number of channels is an argument.
    However, in the original paper authors preprocess EEG data by selecting
    a subset of only 21 channels. Since the net is very minimalist,
    please follow the authors' notes.

    References
    ----------
    .. [shall] Schirrmeister et al., Deep Learning with convolutional
      neural networks for decoding and visualization of EEG pathology,
      arXiv:1708.08012

    Example
    -------
    >>> import selfeeg.models
    >>> import torch
    >>> x = torch.randn(4,8,512)
    >>> mdl = models.ShallowNet(4,8,512)
    >>> out = mdl(x)
    >>> print(out.shape) # shoud return torch.Size([4, 4])
    >>> print(torch.isnan(out).sum()) # shoud return 0

    """

    def __init__(
        self,
        nb_classes: int,
        Chans: int,
        Samples: int,
        F: int = 40,
        K1: int = 25,
        Pool: int = 75,
        p: float = 0.2,
        return_logits: bool = True,
        seed: int = None,
    ):

        super(ShallowNet, self).__init__()

        self.nb_classes = nb_classes
        self.return_logits = return_logits
        self.encoder = ShallowNetEncoder(Chans, F=F, K1=K1, Pool=Pool, p=p, seed=seed)

        _reset_seed(seed)
        self.Dense = nn.Linear(
            F * ((Samples - K1 + 1 - Pool) // 15 + 1), 1 if nb_classes <= 2 else nb_classes
        )

    def forward(self, x):
        """
        :meta private:
        """
        x = self.encoder(x)
        x = self.Dense(x)
        if not (self.return_logits):
            if self.nb_classes <= 2:
                x = torch.sigmoid(x)
            else:
                x = F.softmax(x, dim=1)
        return x


# ------------------------------
#         ResNet 1D
# ------------------------------
class ResNet1D(nn.Module):
    """
    Pytorch implementation of the Resnet model

    This model adopts temporal convolutional layers, so conv2d layers with
    horizontal kernel.
    Implemented using as reference the following paper [res1]_ .

    The expected **input** is a **3D tensor** with size
    (Batch x Channels x Samples).

    Parameters
    ----------
    nb_classes: int
        The number of classes. If less than 2, a binary classification problem
        is considered (output dimensions will be [batch, 1] in this case).
    Chans: int
        The number of EEG channels.
    Samples: int
        The sample length. It will be used to calculate the embedding size
        (for head initialization).
    block: nn.Module, optional
        An nn.Module defining the resnet block.

        Default: selfeeg.models.BasicBlock1
    Layers: list of 4 int, optional
        A list of integers indicating the number of times the resnet block
        is repeated at each stage.
        It must be a list of length 4 with positive integers.
        Shorter lists are padded to 1 on the right.
        Only the first four elements of longer lists are considered.
        Zeros are changed to 1.

        Default = [2, 2, 2, 2]
    inplane: int, optional
        The number of output filters.

        Default = 16
    kernLength: int, optional
        The length of the temporal convolutional layer.

        Default = 7
    addConnection: bool, optional
        Whether to add a connection from the start of the resnet part to
        the network head. If set to True the output of the following conv2d
        will be concatenate to the postblock
        output:

            1. nn.Conv2d(inplane, 2, kernel_size=(Chans, kernLength),
               stride=(1, int(kernLength//2)), padding=(0, 0), bias=False)

        Default = None
    preBlock: nn.Module, optional
        A custom nn.Module to pass before entering the sequence of resnet blocks.
        If none is left, the following sequence is used:

        1. nn.conv2d(1, self.inplane, kernel_size=(1, kernLength), stride=(1, 2),
           padding=(0, kernLength//2), bias=False)
        2. nn.BatchNorm2d()
        3. nn.ReLU()

        Default = None
    postBlock: nn.Module, optional
        A custom nn.Module to pass after the sequence of resnet blocks
        and before the network head.
        If none is left, the following sequence is used:

            1. nn.conv2d(1, self.inplane, kernel_size=(1, kernLength), bias=False)
            2. nn.BatchNorm2d()
            3. nn.ReLU()

            Default = None
    classifier: nn.Module, optional
        A custom nn.Module defining the network head. If none is left,
        a single dense layer is used.

        Default = None
    return_logits: bool, optional
        Whether to return the output as logit or probability. It is suggested
        to not use False as the pytorch crossentropy applies the softmax internally.

        Default = True
    seed: int, optional
        A custom seed for model initialization. It must be a nonnegative number.
        If None is passed, no custom seed will be set

        Default = None

    Note
    ----
    The compatibility between each custom nn.Module given as argument has
    not been carefully checked. Errors may arise.

    References
    ----------
    .. [res1] Zheng et al., Task-oriented Self-supervised Learning
      for Anomaly Detection in Electroencephalography

    Example
    -------
    >>> import selfeeg.models
    >>> import torch
    >>> x = torch.randn(4,8,512)
    >>> mdl = models.ResNet1D(4,8,512)
    >>> out = mdl(x)
    >>> print(out.shape) # shoud return torch.Size([4, 4])
    >>> print(torch.isnan(out).sum()) # shoud return 0

    """

    def __init__(
        self,
        nb_classes: int,
        Chans: int,
        Samples: int,
        block: nn.Module = BasicBlock1,
        Layers: "list of 4 int" = [2, 2, 2, 2],
        inplane: int = 16,
        kernLength: int = 7,
        addConnection: bool = False,
        preBlock: nn.Module = None,
        postBlock: nn.Module = None,
        classifier: nn.Module = None,
        return_logits: bool = True,
        seed: int = None,
    ):

        super(ResNet1D, self).__init__()
        self.nb_classes = nb_classes
        self.return_logits = return_logits
        # Encoder
        self.encoder = ResNet1DEncoder(
            Chans=Chans,
            block=block,
            Layers=Layers,
            inplane=inplane,
            kernLength=kernLength,
            addConnection=addConnection,
            preBlock=preBlock,
            postBlock=postBlock,
            seed=seed,
        )

        # Classifier
        _reset_seed(seed)
        if classifier is None:
            if addConnection:
                out1 = int((Samples + 2 * (int(kernLength // 2)) - kernLength) // 2) + 1
                self.Dense = nn.Linear(
                    (Chans * inplane + int((out1 - kernLength) / int(kernLength // 2) + 1) * 2),
                    1 if nb_classes <= 2 else nb_classes,
                )
            else:
                self.Dense = nn.Linear(
                    Chans * self.encoder.inplane, 1 if nb_classes <= 2 else nb_classes
                )
        else:
            self.Dense = classifier

    def forward(self, x):
        """
        :meta private:
        """
        x = self.encoder(x)
        x = self.Dense(x)
        if not (self.return_logits):
            if self.nb_classes <= 2:
                x = torch.sigmoid(x)
            else:
                x = F.softmax(x, dim=1)
        return x


# ------------------------------
#            STNet
# ------------------------------
class STNet(nn.Module):
    """
    Pytorch implementation of the STNet model.

    Original paper can be found here [stnet]_ .
    Another implementation can be found here [stnetgit]_ .

    The expected **input** is a **4D tensor** with size
    (Batch x Samples x Grid_width x Grid_width), i.e. the classical 2d matrix
    with rows as channels and columns as samples is rearranged in a 3d tensor where
    the first is the Sample dimension and the last 2 dimensions are the channel
    dim rearranged in a 2d grid. Check the original paper for a better
    understanding of the input.

    Parameters
    ----------
    nb_classes: int
        The number of classes. If less than 2, a binary classification
        problem is considered. (output dimensions will be [batch, 1] in this case)
    Samples: int
        The sample length. It will be used to calculate the embedding size
    grid_size: int, optional
        The grid size, i.e. the size of the EEG channel 2D grid.

        Default = 9
    F: int, optional
        The number of output filters in the convolutional layer.

        Default = 256
    kernLength: int, optional
        The length of the convolutional layer.

        Default = 5
    dropRate: float, optional
        The dropout percentage in range [0,1].

        Default = 0.5
    bias: bool, optional
        If True, adds a learnable bias to the convolutional layers.

        Default = True
    dense_size: int, optional
        The output size of the first dense layer.

        Default = 1024
    return_logits: bool, optional
        Whether to return the output as logit or probability.  It is suggested
        to not use False as the pytorch crossentropy applies the softmax internally.

        Default = True
    seed: int, optional
        A custom seed for model initialization. It must be a nonnegative number.
        If None is passed, no custom seed will be set

        Default = None

    References
    ----------
    .. [stnet] https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9763358
    .. [stnetgit] https://github.com/torcheeg/torcheeg/blob/v1.1.0/torcheeg/models/cnn/stnet.py#L42-L135

    Example
    -------
    >>> import selfeeg.models
    >>> import torch
    >>> x = torch.randn(4,128,9,9)
    >>> mdl = models.STNet(4,128)
    >>> out = mdl(x)
    >>> print(out.shape) # shoud return torch.Size([4, 4])
    >>> print(torch.isnan(out).sum()) # shoud return 0

    """

    def __init__(
        self,
        nb_classes: int,
        Samples: int,
        grid_size: int = 9,
        F: int = 256,
        kernlength: int = 5,
        dropRate: float = 0.5,
        bias: bool = True,
        dense_size: int = 1024,
        return_logits: bool = True,
        seed: int = None,
    ):
        super(STNet, self).__init__()

        self.nb_classes = nb_classes
        self.return_logits = return_logits
        self.encoder = STNetEncoder(Samples, F, kernlength, dropRate, bias, seed=seed)

        _reset_seed(seed)
        self.drop3 = nn.Dropout(dropRate)
        self.Dense = nn.Sequential(
            nn.Linear(int(F / 16) * (grid_size**2), dense_size),
            nn.Dropout(dropRate),
            nn.SELU(),
            nn.Linear(dense_size, 1 if nb_classes <= 2 else nb_classes),
        )

    def forward(self, x):
        """
        :meta private:
        """
        x = self.encoder(x)
        x = self.drop3(x)
        x = self.Dense(x)
        if not (self.return_logits):
            if self.nb_classes <= 2:
                x = torch.sigmoid(x)
            else:
                x = F.softmax(x, dim=1)
        return x


# ------------------------------
#            EEGSym
# ------------------------------
class EEGSym(nn.Module):
    """
    Pytorch implementation of the EEGSym model.

    EEGSym paper can be found here [eegsym]_ .
    Keras implementation can be found here [eegsymgit]_ .

    The expected **input** is a **3D tensor** with size
    (Batch x Channels x Samples). However Channel order is expected to be
    symmetrical along lateral channels to perform the reshaping operation
    correctly. For instance, if the first channel index refers to the FP1 channel,
    then the last must refer to the other hemisphere counterpart, i.e. FP2.

    Parameters
    ----------
    nb_classes: int
        The number of classes. If less than 2, a binary classification
        problem is considered (output dimensions will be [batch, 1] in this case).
    Chans: int
        The number of EEG channels.
    Samples: int
        The number of EEG Samples.
    Fs: float
        The sampling frequency.
    scales_time: tuple of 3 float, optional
        The portion of EEG (in milliseconds) the short, medium and long
        temporal convolutional layers must cover. kernel size will be
        automatically calculated based on the sampling
        rate given.

        Default = (500,250,125)
    lateral_chans: int, optional
        The amount of lateral channels. It will be used to reshape
        the 3D tensor in a 5D tensor with size
        ( batch x filters x hemisphere x channel x samples ).
        See the original paper for more info.

        Default = 3
    first_left: bool, optional
        Whether the first half of the channels are of the left hemisphere or not.

        Default = True
    F: int, optional
        The output filters of each branch of the first inception block.
        Must be a multiple of 8.
        Other outputs will be automatically calculated.

        Default = 8
    pool: int, optional
        The size of the pooling kernel.

        Default = 2
    dropRate: float, optional
        The dropout percentage in range [0,1].

        Default = 0.5
    bias: bool, optional
        If True, adds a learnable bias to the convolutional layers.

        Default = True
    residual: bool, optional
        Whether to add a residual block after the inception block.
        Currently not implemented, will be added in future releases.

        Default = True
    return_logits: bool, optional
        Whether to return the output as logit or probability.  It is suggested
        to not use False as the pytorch crossentropy applies the softmax internally.

        Default = True
    seed: int, optional
        A custom seed for model initialization. It must be a nonnegative number.
        If None is passed, no custom seed will be set

        Default = None

    References
    ----------
    .. [eegsym] https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9807323
    .. [eegsymgit] https://github.com/Serpeve/EEGSym/blob/main/EEGSym_architecture.py

    Example
    -------
    >>> import selfeeg.models
    >>> import torch
    >>> x = torch.randn(4,8,1024)
    >>> mdl = models.EEGSym(4,8,1024,64)
    >>> out = mdl(x)
    >>> print(out.shape) # shoud return torch.Size([4, 4])
    >>> print(torch.isnan(out).sum()) # shoud return 0

    """

    def __init__(
        self,
        nb_classes: int,
        Chans: int,
        Samples: int,
        Fs: int,
        scales_time: tuple = (500, 250, 125),
        lateral_chans: int = 3,
        first_left: bool = True,
        F: int = 8,
        pool: int = 2,
        dropRate: float = 0.5,
        ELUalpha: float = 1.0,
        bias: bool = True,
        residual: bool = True,
        return_logits: bool = True,
        seed: int = None,
    ):
        super(EEGSym, self).__init__()
        self.nb_classes = nb_classes
        self.return_logits = return_logits
        self.encoder = EEGSymEncoder(
            Chans,
            Samples,
            Fs,
            scales_time,
            lateral_chans,
            first_left,
            F,
            pool,
            dropRate,
            ELUalpha,
            bias,
            residual,
            seed=seed,
        )

        _reset_seed(seed)
        self.Dense = nn.Linear(int((F * 9) / 2), 1 if nb_classes <= 2 else nb_classes)

    def forward(self, x):
        """
        :meta private:
        """
        x = self.encoder(x)
        x = self.Dense(x)
        if not (self.return_logits):
            if self.nb_classes <= 2:
                x = torch.sigmoid(x)
            else:
                x = F.softmax(x, dim=1)
        return x


# ------------------------------
#            FBCNet
# ------------------------------
class FBCNet(nn.Module):
    """
    Pytorch implementation of the FBCNet model.

    FBCNet paper can be found here [fbcnet]_ .
    The official implementation can be found here [gitfbc]_ .

    The expected **input** is a **3D tensor** with size
    (Batch x Channels x Samples).

    Filter operation is applied through the torchaudio filtfilt function.
    Do not use too strict filter settings as this might generate nan or
    too high values.

    Parameters
    ----------
    nb_classes: int
        The number of classes. If less than 2, a binary classification
        problem is considered (output dimensions will be [batch, 1] in this case).
    Chans: int
        The number of EEG channels.
    Samples: int
        The number of EEG samples.
    Fs: int or float
        The EEG sampling rate.
    FilterBands: int, optional
        The number of filters to apply to the original signal. It is used by
        the FilterBank layer.

        Default = 9
    FilterRange: int or float, optional
        The passband of each filter, given in Hz.It is used by
        the FilterBank layer.

        Default = 4
    FilterType: str, optional
        The type of filter to use. Allowed arguments are the same as described
        in the `get_filter_coeff()` function of the
        `selfeeg.augmentation.functional` submodule
        ('butter', 'ellip', 'cheby1', 'cheby2'). It is used by
        the FilterBank layer.

        Default = 'cheby1'
    FilterStopRipple: int or float, optional
        Ripple at stopband in decibel. It is used by the FilterBank layer.

        Default = 30
    FilterPassRipple: int or float, optional
        Ripple at passband in decibel. It is used by the FilterBank layer.

        Default = 30
    FilterRangeTol: int or float, optional
        The filter transition bandwidth in Hz. It is used by the FilterBank layer.

        Default = 2
    FilterSkipFirst: bool, optional
        If True, skips the first filter with passband equal to [0, Range] Hz.
        The number of filters specified in Bands will still be preserved.
        It is used by the FilterBank layer.

        Default = True
    D: int, optional
        The depth of the depthwise convolutional layer.

        Default = 2
    TemporalType: str, optional
        The type of temporal feature extraction layer to use.
        Accepted values are 'max', 'mean', 'std', 'var', or 'logvar'.

        Default = 'logvar'
    TemporalStride: int, optional
        The signal length output dimension of the temporal feature
        extraction layer. Kernel length and layer stride will be
        calculated based on the given input. Be sure that Sample
        is a multiple of this attribute.

        Default = 4
    batch_momentum: float, optional
        The batch normalization momentum.

        Default = 0.1
    depthwise_max_norm: float, optional
        The maximum norm each filter can have in the depthwise block.
        If None no constraint will be included.

        Default = None
    linear_max_norm: float, optional
        The maximum norm each filter can have in the final dense layer.
        If None no constraint will be included.

        Default = None
    classifier: nn.Module, optional
        A custom block to apply after the encoder instead of the classical
        linear layer. Must be an istance of an nn.Module. If none a standard
        linear layer is applied.

        Default = None
    return_logits: bool, optional
        Whether to return the output as logit or probability. It is suggested
        to not use False as the pytorch crossentropy applies the softmax internally.

        Default = True
    seed: int, optional
        A custom seed for model initialization. It must be a nonnegative number.
        If None is passed, no custom seed will be set

        Default = None

    References
    ----------
    .. [fbcnet] Mane et al. FBCNet: A Multi-view Convolutional Neural Network
      for Brain-Computer Interface. https://arxiv.org/abs/2104.01233
    .. [gitfbc] https://github.com/ravikiran-mane/FBCNet

    Example
    -------
    >>> import selfeeg.models
    >>> import torch
    >>> x = torch.randn(4,8,512)
    >>> mdl = models.FBCNet(2, 8, 512, 128)
    >>> out = mdl(x)
    >>> print(out.shape) # shoud return torch.Size([4, 2])
    >>> print(torch.isnan(out).sum()) # shoud return 0

    """

    def __init__(
        self,
        nb_classes: int,
        Chans: int,
        Samples: int,
        Fs: int or float,
        FilterBands: int = 9,
        FilterRange: float or int = 4,
        FilterType: str = "Cheby2",
        FilterStopRippple: int or float = 30,
        FilterPassRipple: int or float = 3,
        FilterRangeTol: int or float = 2,
        FilterSkipFirst=True,
        D: int = 32,
        TemporalType: str = "logvar",
        TemporalStride: int = 4,
        batch_momentum: float = 0.1,
        depthwise_max_norm: float = None,
        linear_max_norm: float = None,
        classifier: nn.Module = None,
        return_logits: bool = True,
        seed: int = None,
    ):
        super(FBCNet, self).__init__()

        self.nb_classes = nb_classes
        self.return_logits = return_logits

        # Encoder
        self.encoder = FBCNetEncoder(
            Chans,
            Samples,
            Fs,
            FilterBands,
            FilterRange,
            FilterType,
            FilterStopRippple,
            FilterPassRipple,
            FilterRangeTol,
            FilterSkipFirst,
            D,
            TemporalType,
            TemporalStride,
            batch_momentum,
            depthwise_max_norm,
            seed=seed,
        )

        # Head
        _reset_seed(seed)
        if classifier is None:
            self.head = ConstrainedDense(
                D * FilterBands * TemporalStride,
                1 if nb_classes <= 2 else nb_classes,
                max_norm=linear_max_norm,
            )
        else:
            self.head = classifier

    def forward(self, x):
        """
        :meta private:
        """
        x = self.encoder(x)
        x = self.head(x)
        if not self.return_logits:
            if self.nb_classes <= 2:
                x = torch.sigmoid(x)
            else:
                x = F.softmax(x, dim=1)
        return x


# ------------------------------
#           ATCNet
# ------------------------------
class ATCmha(nn.Module):
    """
    :meta private:
    """

    def __init__(self, seq_len, emb_dim, dropRate=0.5, num_heads=2):
        super(ATCmha, self).__init__()
        self.nrm = nn.LayerNorm([seq_len, emb_dim], eps=1e-06)
        self.mha = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        self.drp = nn.Dropout(dropRate)

    def forward(self, x):
        """
        :meta private:
        """
        x1 = self.nrm(x)
        x1 = self.mha(x1, x1, x1)[0]
        x1 = self.drp(x1)
        x = x + x1
        return x


class ATCtcn(nn.Module):
    """
    :meta private:
    """

    def __init__(
        self,
        chan_dim,
        tcn_depth=2,
        Ftcn=32,
        kernLength=4,
        dropRate=0.3,
        ELUAlpha=0.0,
        batchMomentum=0.99,
        max_norm=0.6,
    ):
        super(ATCtcn, self).__init__()

        self.ELUAlpha = ELUAlpha
        self.tcn_depth = tcn_depth
        self.depth_forward = tcn_depth - 1

        self.tcnBlock_0 = nn.Sequential(
            ConstrainedConv1d(chan_dim, Ftcn, kernLength, max_norm=max_norm),
            nn.BatchNorm1d(Ftcn, batchMomentum),
            nn.ELU(self.ELUAlpha),
            nn.Dropout(dropRate),
            ConstrainedConv1d(Ftcn, Ftcn, kernLength, max_norm=max_norm),
            nn.BatchNorm1d(Ftcn, batchMomentum),
            nn.ELU(self.ELUAlpha),
            nn.Dropout(dropRate),
        )

        if chan_dim == Ftcn:
            self.residual_0 = None
        else:
            self.residual_0 = ConstrainedConv1d(chan_dim, Ftcn, 1, max_norm=max_norm)

        for i in range(self.tcn_depth - 1):
            self.add_module(
                "tcnBlock_" + str(i + 1),
                nn.Sequential(
                    ConstrainedConv1d(
                        Ftcn,
                        Ftcn,
                        kernLength,
                        dilation=2 ** (i + 1),
                        padding="causal",
                        max_norm=max_norm,
                    ),
                    nn.BatchNorm1d(Ftcn, batchMomentum),
                    nn.ELU(self.ELUAlpha),
                    nn.Dropout(dropRate),
                    ConstrainedConv1d(
                        Ftcn,
                        Ftcn,
                        kernLength,
                        dilation=2 ** (i + 1),
                        padding="causal",
                        max_norm=max_norm,
                    ),
                    nn.BatchNorm1d(Ftcn, batchMomentum),
                    nn.ELU(self.ELUAlpha),
                    nn.Dropout(dropRate),
                ),
            )

    def forward(self, x):
        """
        :meta private:
        """
        x1 = self.tcnBlock_0(x)
        if self.residual_0 is not None:
            x = self.residual_0(x)
        x = x + x1
        x = F.elu(x, self.ELUAlpha)
        if self.depth_forward:
            for i in range(self.depth_forward):
                x1 = self.get_submodule("tcnBlock_" + str(i + 1))(x)
                x = x + x1
                x = F.elu(x, self.ELUAlpha)
        return x


class ATCNet(nn.Module):
    """
    Pytorch implementation of the ATCNet model.

    ATCNet paper can be found here [atcnet]_ .
    The official implementation can be found here [gitatc]_ .

    The expected **input** is a **3D tensor** with size
    (Batch x Channels x Samples).

    Warnings
    --------
    Due to the multi-branch nature of the network and the usage of a
    classification head at the end of each branch, this model does not
    have an implementation of the encoder. Keep in mind that the first
    convolutional block is basically an eegnet encoder with a conv2d instead
    of a separable conv2d.

    Parameters
    ----------
    nb_classes: int
        The number of classes. If less than 2, a binary classification
        problem is considered (output dimensions will be [batch, 1] in this case).
    Chans: int
        The number of EEG channels.
    Samples: int
        The number of EEG samples.
    Fs: int or float
        The EEG sampling rate.
    num_windows: int, optonal
        The number of branches to use after the first convolutional block.

        Default = 5
    mha_heads: int, optional
        The number of multi-head attention heads to use in each branch.

        Default = 2
    tcn_depth: int, optional
        The number of temporal convolution blocks to use in each branch.

        Default = 2
    F1: int, optional
        The number of filters to use in the first layer of the first
        convolutional block. It is the same as the F1 argument of the
        EEGNet model

        Default = 16
    D: int, optional
        The depth of the depthwise layer in the first convolutional block.
        It is the same as the D argument of the EEGNet model.

        Default = 2
    pool1: int, optional
        The kernel length of the first pooling layer of the first convolutional
        block. It is the same as the pool1 argument of the EEGNet model.
        If left to None the length is automatically retrieved to cover the same
        portion of EEG as in the original work (regardless of the Sampling Rate).

        Default = None
    pool2: int, optional
        The kernel length of the second pooling layer of the first convolutional
        block. It is the same as the pool2 argument of the EEGNet model.
        If left to None the length is automatically retrieved to cover the same
        portion of EEG as in the original work (regardless of the Sampling Rate).

        Default = None
    dropRate: float, optional
        The dropout rate of the first convolutional layer.
        It is the same as the dropRate argument in the EEGNet model.

        Default = 0.3
    max_norm: float, optional
        The max norm constraint to apply to the convolutional layers
        of the first convolutioal block. If left to None, no constraints
        will be applied

        Default = None
    batchMomentum: float, optional
        The batch normalization momentum of the first convolutional layer.
        It is the same as the batch_momentum argument of the EEGNet model.
        Note that the original paper uses a higher batch momentum (0.9).

        Default = 0.1
    ELUAlpha: float, optional
         the alpha value of the ELU activation function.
         It is the same as the batch_momentum argument of the EEGNet model.

         Default = 1
    mha_dropRate: float, optional
        The dropout rate of the multi head attention block.

        Default = 0.5
    tcn_kernLength: int, optional
        The kernel length of the temporal convolutional block.

        Default = 4
    tcn_F: int, optional
        The number of filters of the temporal convolutioal block.

        Default = 32
    tcn_ELUAlpha: float, optional
        The alpha value for the activation function of the temporal
        convolutioal block.

        Default = 1.0
    tcn_dropRate: float, optional
        The dropout rate of the temporal convolutioal block.

        Default = 0.5
    tcn_max_norm: float, optional
        The max norm constraint to apply to the convolutional layer
        of the temporal convolutioal block. If left to None, no constraints
        will be applied
    tcn_batchMom: float, optional
        The batch normalization momentum of the temporal convolutioal block.
    return_logits: bool, optional
        Whether to return the output as logit or probability.
        It is suggested to not use False as the pytorch crossentropy
        applies the softmax internally.

        Default = True
    seed: int, optional
        A custom seed for model initialization. It must be a nonnegative number.
        If None is passed, no custom seed will be set

        Default = None

    References
    ----------
    .. [atcnet] Altaheri et al. Physics-Informed Attention Temporal Convolutional
      Network for EEG-Based Motor Imagery Classification.
      https://ieeexplore.ieee.org/document/9852687
    .. [gitatc] https://github.com/Altaheri/EEG-ATCNet

    Example
    -------
    >>> import selfeeg.models
    >>> import torch
    >>> x = torch.randn(4,8,512)
    >>> mdl = models.ATCNet(2, 8, 512, 128)
    >>> out = mdl(x)
    >>> print(out.shape) # shoud return torch.Size([4, 2])
    >>> print(torch.isnan(out).sum()) # shoud return 0

    """

    def __init__(
        self,
        nb_classes: int,
        Chans: int,
        Samples: int,
        Fs: float,
        num_windows: int = 5,
        mha_heads: int = 2,
        tcn_depth: int = 2,
        F1: int = 16,
        D: int = 2,
        pool1: int = None,
        pool2: int = None,
        dropRate: float = 0.3,
        max_norm: float = None,
        batchMomentum: float = 0.1,
        ELUAlpha: float = 1.0,
        mha_dropRate: float = 0.5,
        tcn_kernLength: int = 4,
        tcn_F: int = 32,
        tcn_ELUAlpha: float = 0.0,
        tcn_dropRate: float = 0.3,
        tcn_max_norm: float = None,
        tcn_batchMom: float = 0.1,
        return_logits: bool = True,
        seed: int = None,
    ):

        super(ATCNet, self).__init__()
        _reset_seed(seed)

        # important for model construction
        self.return_logits = return_logits
        self.nb_classes = nb_classes
        self.mha_heads = mha_heads
        self.tcn_depth = tcn_depth

        # conv block parameters
        self.pool1 = round(Fs / 31.25) if pool1 is None else pool1
        self.pool2 = round(Samples / (Fs * 0.225)) if pool2 is None else pool2
        self.dropRate = dropRate
        self.batchMomentum = batchMomentum
        self.max_norm = max_norm

        # multi-head parameters
        self.mha_dropRate = mha_dropRate

        # TCN parameters
        self.tcn_kernLength = tcn_kernLength
        self.tcn_F = tcn_F
        self.tcn_ELUAlpha = tcn_ELUAlpha
        self.tcn_dropRate = tcn_dropRate
        self.tcn_max_norm = tcn_max_norm
        self.tcn_batchMom = tcn_batchMom

        # Sliding windows parameters
        self.chan_dim = int(F1 * D)
        self.emb_dim = self.pool2
        self.num_windows = num_windows
        self.win_len = self.emb_dim - self.num_windows + 1

        self.ConvBlock = nn.Sequential(
            ConstrainedConv2d(1, F1, (1, Fs // 4), padding="same", bias=False, max_norm=max_norm),
            nn.BatchNorm2d(F1, batchMomentum),
            ConstrainedConv2d(
                F1, F1 * D, (Chans, 1), padding="valid", bias=False, max_norm=max_norm, groups=F1
            ),
            nn.BatchNorm2d(F1 * D, batchMomentum),
            nn.ELU(ELUAlpha),
            nn.AvgPool2d((1, self.pool1)),
            nn.Dropout(self.dropRate),
            ConstrainedConv2d(
                F1 * D, F1 * D, (1, 16), padding="same", bias=False, max_norm=max_norm
            ),
            nn.BatchNorm2d(F1 * D, batchMomentum),
            nn.ELU(ELUAlpha),
            nn.AdaptiveAvgPool2d((1, self.pool2)),
            nn.Dropout(self.dropRate),
        )

        # Construct each Branch
        _reset_seed(seed)
        for i in range(self.num_windows):
            self.add_multi_head(i)
            self.add_residual_tcn(i)
            self.add_branch_dense(i)

    def add_multi_head(self, idx):
        self.add_module(
            "mha_" + str(idx),
            ATCmha(self.win_len, self.chan_dim, self.mha_dropRate, self.mha_heads),
        )

    def add_residual_tcn(self, idx):
        self.add_module(
            "tcn_" + str(idx),
            ATCtcn(
                self.chan_dim,
                self.tcn_depth,
                self.tcn_F,
                self.tcn_kernLength,
                self.tcn_dropRate,
                self.tcn_ELUAlpha,
                self.tcn_batchMom,
                self.tcn_max_norm,
            ),
        )

    def add_branch_dense(self, idx):
        self.add_module(
            "dns_" + str(idx), nn.Linear(self.tcn_F, 1 if self.nb_classes <= 2 else self.nb_classes)
        )

    def forward(self, x):
        """
        :meta private:
        """
        x = torch.unsqueeze(x, 1)
        x = self.ConvBlock(x)
        x = torch.squeeze(x, 2)
        x = torch.permute(x, (0, 2, 1))
        windows_output = []
        for i in range(self.num_windows):
            xi = x[:, i : i + self.win_len, :]
            xi = self.get_submodule("mha_" + str(i))(xi)
            xi = torch.permute(xi, (0, 2, 1))
            xi = self.get_submodule("tcn_" + str(i))(xi)
            xi = xi[:, :, -1]
            xi = self.get_submodule("dns_" + str(i))(xi)
            windows_output.append(xi)
        x = torch.stack(windows_output)
        x = torch.mean(x, 0)

        if not self.return_logits:
            if self.nb_classes <= 2:
                x = torch.sigmoid(x)
            else:
                x = F.softmax(x, dim=1)
        return x


class EEGConformer(nn.Module):
    """
    Pytorch implementation of EEGConformer.

    For more information see the following paper [EEGcon]_ .
    The original implementation of EEGconformer can be found here [EEGcongit]_ .

    The expected **input** is a **3D tensor** with size
    (Batch x Channels x Samples).

    Parameters
    ----------
    nb_classes: int
        The number of classes. If less than 2, a binary classification problem
        is considered (output dimensions will be [batch, 1] in this case).
    Chans: int
        The number of EEG channels.
    F: int, optional
        The number of output filters in the temporal convolution layer.

        Default = 40
    K1: int, optional
        The length of the temporal convolutional layer.

        Default = 25
    Pool: int, optional
        The temporal pooling kernel size.

        Default = 75
    stride_pool: int, optional
        The temporal pooling stride.

        Default = 15
    d_model: int, optional
        The embedding size. It is the number of expected features in the input of
        the transformer encoder layer.

        Default = 40
    nlayers: int, optional
        The number of transformer encoder layers.

        Default = 6
    nheads: int, optional
        The number of heads in the multi-head attention layers.

        Default = 10
    dim_feedforward: int, optional
        The dimension of the feedforward hidden layer in the transformer encoder.

        Default = 160
    activation_transformer: str or Callabel, optional
        The activation function in the transformer encoder. See the PyTorch
        TransformerEncoderLayer documentation for accepted inputs.

        Default = "gelu"
    p: float, optional
        Dropout probability in the tokenizer. Must be in [0,1)

        Default = 0.2
    p_transformer: float, optional
        Dropout probability in the transformer encoder. Must be in [0,1)

        Default = 0.5
    mlp_dim: list, optional
        A two-element list indicating the output dimensions of the 2 FC
        layers in the final classification head.

        Default = [256, 32]
    return_logits: bool, optional
        Whether to return the output as logit or probability.
        It is suggested to not use False as the pytorch crossentropy loss function
        applies the softmax internally.

        Default = True
    seed: int, optional
        A custom seed for model initialization. It must be a nonnegative number.
        If None is passed, no custom seed will be set

        Default = None

    References
    ----------
    .. [EEGcon] Song et al., EEG Conformer: Convolutional Transformer for EEG Decoding
      and Visualization. IEEE TNSRE. 2023. https://doi.org/10.1109/TNSRE.2022.3230250
    .. [EEGcongit] https://github.com/eeyhsong/EEG-Conformer

    Example
    -------
    >>> import selfeeg.models
    >>> import torch
    >>> x = torch.randn(4,8,512)
    >>> mdl = models.EEGConformer(2, 8, 512)
    >>> out = mdl(x)
    >>> print(out.shape) # shoud return torch.Size([4, 1])

    """

    def __init__(
        self,
        nb_classes: int,
        Chans: int,
        Samples: int,
        F: int = 40,
        K1: int = 25,
        Pool: int = 75,
        stride_pool: int = 15,
        d_model: int = 40,
        nlayers: int = 6,
        nheads: int = 10,
        dim_feedforward: int = 160,
        activation_transformer: str or Callable = "gelu",
        p: float = 0.2,
        p_transformer: float = 0.5,
        mlp_dim: list[int, int] = [256, 32],
        return_logits: bool = True,
        seed: int = None,
    ):

        super(EEGConformer, self).__init__()
        self.return_logits = return_logits
        self.nb_classes = nb_classes

        self.encoder = EEGConformerEncoder(
            Chans,
            F,
            K1,
            Pool,
            stride_pool,
            d_model,
            nlayers,
            nheads,
            dim_feedforward,
            activation_transformer,
            p,
            p_transformer,
            seed,
        )

        _reset_seed(seed)
        self.MLP = nn.Sequential(
            nn.AvgPool1d((Samples - K1 + 1 - Pool) // stride_pool + 1),
            nn.Flatten(start_dim=1),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, mlp_dim[0]),
            nn.ELU(),
            nn.Dropout(p),
            nn.Linear(mlp_dim[0], mlp_dim[1]),
            nn.ELU(),
            nn.Dropout(p),
            nn.Linear(mlp_dim[1], 1 if nb_classes <= 2 else nb_classes),
        )

    def forward(self, x):
        """
        :meta private:
        """
        x = self.encoder(x)
        x = self.MLP(x)
        if not (self.return_logits):
            if self.nb_classes <= 2:
                x = torch.sigmoid(x)
            else:
                x = F.softmax(x, dim=1)
        return x


class xEEGNet(nn.Module):
    """
    Pytorch implementation of xEEGNet.

    For more information see the following paper [xEEG]_ .
    The original implementation of EEGconformer can be found here [xEEGgit]_ .
    The expected **input** is a **3D tensor** with size:
        (Batch x Channels x Samples).

    Parameters
    ----------
    nb_classes: int
        The number of classes. If less than 2, a binary classification problem
        is considered (output dimensions will be [batch, 1] in this case).
    Chans: int
        The number of EEG channels.
    Samples: int
        The sample length (number of time steps).
        It will be used to calculate the embedding size (for head initialization).
    Fs: int
        The sampling rate of the EEG signal in Hz.
        It is used to initialize the weights of the filters.
        Must be specified even if `random_temporal_filter` is False.
    F1: int, optional
        The number of output filters in the temporal convolution layer.

        Default = 7
    K1: int, optional
        The length of the temporal convolutional layer.

        Default = 125
    F2: int, optional
        The number of output filters in the spatial convolution layer.

        Default = 7
    Pool: int, optional
        Kernel size for temporal pooling.

        Default = 75
    p: float, optional
        Dropout probability in [0,1)

        Default = 0.2
    random_temporal_filter: bool, optional
        If True, initialize the temporal filter weights randomly.
        Otherwise, use a passband FIR filter.

        Default = False
    freeze_temporal: int, optional
        Number of forward steps to keep the temporal layer frozen.

        Default = 1e12
    spatial_depthwise: bool, optional
        Whether to apply a depthwise layer in the spatial convolution.

        Default = True
    log_activation_base: str, optional
        Base for the logarithmic activation after pooling.
        Options: "e" (natural log), "10" (logarithm base 10), "dB" (decibel scale).

        Default = "dB"
    norm_type: str, optional
        The type of normalization. Expected values are "batch" or "instance".

        Default = "batchnorm"
    global_pooling: bool, optional
        If True, apply global average pooling instead of flattening.

        Default = True
    bias: list[int, int], optional
        A 2-element list with boolean values.
        If the first element is True, a bias will be added to the temporal
        convolutional layer.
        If the second element is True, a bias will be added to the spatial
        convolutional layer.
        If the third element is True, a bias will be added to the final dense layer.

        Default = [False, False, False]
    return_logits: bool, optional
        If True, return the output as logit.
        It is suggested to not use False as the pytorch crossentropy loss function
        applies the softmax internally.

        Default = True
    seed: int, optional
        A custom seed for model initialization. It must be a nonnegative number.
        If None is passed, no custom seed will be set

        Default = None

    References
    ----------
    .. [xEEG] zanola et al., xEEGNet: Towards Explainable AI in EEG Dementia
      Classification. arXiv preprint. 2025. https://doi.org/10.48550/arXiv.2504.21457
    .. [xEEGgit] https://github.com/MedMaxLab/shallownetXAI

    Example
    -------
    >>> import selfeeg.models
    >>> import torch
    >>> x = torch.randn(4,8,512)
    >>> mdl = models.xEEGNet(3, 8, 512, 125)
    >>> out = mdl(x)
    >>> print(out.shape) # shoud return torch.Size([4, 3])

    """

    def __init__(
        self,
        nb_classes: int,
        Chans: int,
        Samples: int,
        Fs: int,
        F1: int = 7,
        K1: int = 125,
        F2: int = 7,
        Pool: int = 75,
        p: float = 0.2,
        random_temporal_filter=False,
        freeze_temporal: int = 1e12,
        spatial_depthwise: bool = True,
        log_activation_base: str = "dB",
        norm_type: str = "batchnorm",
        global_pooling=True,
        bias: list[int, int, int] = [False, False, False],
        dense_hidden: int = -1,
        return_logits=True,
        seed=None,
    ):

        super(xEEGNet, self).__init__()

        self.nb_classes = nb_classes
        self.return_logits = return_logits
        self.encoder = xEEGNetEncoder(
            Chans,
            Fs,
            F1,
            K1,
            F2,
            Pool,
            p,
            random_temporal_filter,
            freeze_temporal,
            spatial_depthwise,
            log_activation_base,
            norm_type,
            global_pooling,
            bias,
            seed,
        )

        if global_pooling:
            self.emb_size = F2
        else:
            self.emb_size = F2 * ((Samples - K1 + 1 - Pool) // max(1, int(Pool // 5)) + 1)

        _reset_seed(seed)
        if dense_hidden <= 0:
            self.Dense = nn.Linear(
                self.emb_size, 1 if nb_classes <= 2 else nb_classes, bias=bias[2]
            )
        else:
            self.Dense = nn.Sequential(
                nn.Linear(self.emb_size, dense_hidden, bias=True),
                nn.ReLU(),
                nn.Linear(dense_hidden, 1 if nb_classes <= 2 else nb_classes, bias=bias[2]),
            )

    def forward(self, x):
        """
        :meta private:
        """
        x = self.encoder(x)
        x = self.Dense(x)
        if not (self.return_logits):
            if self.nb_classes <= 2:
                x = torch.sigmoid(x)
            else:
                x = F.softmax(x, dim=1)
        return x
