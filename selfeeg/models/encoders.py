import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import (
    ConstrainedConv1d,
    ConstrainedConv2d,
    ConstrainedDense,
    DepthwiseConv2d,
    SeparableConv2d,
    FilterBank,
)

__all__ = [
    "BasicBlock1",
    "DeepConvNetEncoder",
    "EEGInceptionEncoder",
    "EEGNetEncoder",
    "EEGSymEncoder",
    "FBCNetEncoder",
    "ResNet1DEncoder",
    "ShallowNetEncoder",
    "StagerNetEncoder",
    "STNetEncoder",
    "TinySleepNetEncoder",
]


# ------------------------------
#           EEGNet
# ------------------------------
class EEGNetEncoder(nn.Module):
    """
    Pytorch Implementation of the EEGnet Encoder.

    See EEGNet for some references.
    The expected **input** is a **3D tensor** with
    size (Batch x Channels x Samples).

    Parameters
    ----------
    Chans: int
        The number of EEG channels.
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
        The depth of the depthwise conv layer.

        Default = 16
    dropType: str, optional
        The type of dropout. It can be any between 'Dropout' and 'SpatialDropout2D'.

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
        If None no constraint will be applied.

        Default = None


    Note
    ----
    This implementation refers to the latest version of EEGNet which
    can be found in the official repository.

    Example
    -------
    >>> import selfeeg.models
    >>> import torch
    >>> x = torch.randn(4,8,64)
    >>> mdl = models.EEGNetEncoder(8)
    >>> out = mdl(x)
    >>> print(out.shape) # shoud return torch.Size([4, 32])
    >>> print(torch.isnan(out).sum()) # shoud return 0

    """

    def __init__(
        self,
        Chans,
        kernLength=64,
        dropRate=0.5,
        F1=8,
        D=2,
        F2=16,
        dropType="Dropout",
        ELUalpha=1,
        pool1=4,
        pool2=8,
        separable_kernel=16,
        depthwise_max_norm=1.0,
    ):

        if dropType not in ["SpatialDropout2D", "Dropout"]:
            raise ValueError("implemented Dropout types are" " 'Dropout' or 'SpatialDropout2D '")

        super(EEGNetEncoder, self).__init__()

        # Layer 1
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding="same", bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1, False)

        # Layer 2
        self.conv2 = DepthwiseConv2d(
            F1, D, (Chans, 1), padding="valid", bias=False, max_norm=depthwise_max_norm
        )
        self.batchnorm2 = nn.BatchNorm2d(D * F1, False)
        self.elu2 = nn.ELU(alpha=ELUalpha)
        self.pooling2 = nn.AvgPool2d((1, pool1))
        if dropType.lower() == "dropout":
            self.drop2 = nn.Dropout(p=dropRate)
        else:
            self.drop2 = nn.Dropout2d(p=dropRate)

        # Layer 3
        self.sepconv3 = SeparableConv2d(
            D * F1, F2, (1, separable_kernel), bias=False, padding="same"
        )
        self.batchnorm3 = nn.BatchNorm2d(F2, False)
        self.elu3 = nn.ELU(alpha=ELUalpha)
        self.pooling3 = nn.AvgPool2d((1, pool2))
        if dropType.lower() == "dropout":
            self.drop3 = nn.Dropout(p=dropRate)
        else:
            self.drop3 = nn.Dropout2d(p=dropRate)
        self.flatten3 = nn.Flatten()

    def forward(self, x):
        """
        :meta private:
        """
        # Layer 1
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = self.batchnorm1(x)

        # Layer 2
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.elu2(x)
        x = self.pooling2(x)
        x = self.drop2(x)

        # Layer 3
        x = self.sepconv3(x)
        x = self.batchnorm3(x)
        x = self.elu3(x)
        x = self.pooling3(x)
        x = self.drop3(x)
        x = self.flatten3(x)

        return x


# ------------------------------
#          DeepConvNet
# ------------------------------
class DeepConvNetEncoder(nn.Module):
    """
    Pytorch Implementation of the DeepConvNet Encoder.

    See DeepConvNet for some references.
    The expected **input** is a **3D tensor** with size
    (Batch x Channels x Samples).

    Parameters
    ----------
    Chans: int
        The number of EEG channels.
    kernlength: int, optional
        The length of the temporal convolutional layer.

        Default = 10
    F: int, optional
        The number of filters in the first layer. Next layers
        will continue to double the previous output feature size.

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

        Default = 0.1
    ELUalpha: float, optional
        The alpha value of the ELU activation function.

        Default = 1
    dropRate: float, optional
        The dropout percentage in range [0,1].

        Default = 0.5

    Example
    -------
    >>> import selfeeg.models
    >>> import torch
    >>> x = torch.randn(4,8,512)
    >>> mdl = models.DeepConvNetEncoder(8)
    >>> out = mdl(x)
    >>> print(out.shape) # shoud return torch.Size([4, 200])
    >>> print(torch.isnan(out).sum()) # shoud return 0

    """

    def __init__(
        self,
        Chans,
        kernLength=10,
        F=25,
        Pool=3,
        stride=3,
        max_norm=None,
        batch_momentum=0.1,
        ELUalpha=1,
        dropRate=0.5,
    ):

        super(DeepConvNetEncoder, self).__init__()

        self.conv1 = ConstrainedConv2d(
            1, F, (1, kernLength), padding="valid", stride=(1, 1), max_norm=max_norm
        )
        self.conv2 = ConstrainedConv2d(
            F, F, (Chans, 1), stride=(1, 1), padding="valid", max_norm=max_norm
        )
        self.BN1 = nn.BatchNorm2d(F, momentum=batch_momentum)
        self.ELU = nn.ELU(alpha=ELUalpha)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, Pool), stride=(1, stride))
        self.drop1 = nn.Dropout(p=dropRate)

        self.conv3 = ConstrainedConv2d(
            F, F * 2, (1, kernLength), padding="valid", stride=(1, 1), max_norm=max_norm
        )
        self.BN2 = nn.BatchNorm2d(F * 2, momentum=batch_momentum)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, Pool), stride=(1, stride))
        self.drop2 = nn.Dropout(p=dropRate)

        self.conv4 = ConstrainedConv2d(
            F * 2, F * 4, (1, kernLength), padding="valid", stride=(1, 1), max_norm=max_norm
        )
        self.BN3 = nn.BatchNorm2d(F * 4, momentum=batch_momentum)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, Pool), stride=(1, stride))
        self.drop3 = nn.Dropout(p=dropRate)

        self.conv5 = ConstrainedConv2d(
            F * 4, F * 8, (1, kernLength), padding="valid", stride=(1, 1), max_norm=max_norm
        )
        self.BN4 = nn.BatchNorm2d(F * 8, momentum=batch_momentum)
        self.pool4 = nn.MaxPool2d(kernel_size=(1, Pool), stride=(1, stride))
        self.drop4 = nn.Dropout(p=dropRate)

        self.flatten = nn.Flatten()

    def forward(self, x):
        """
        :meta private:
        """
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.BN1(x)
        x = self.ELU(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.conv3(x)
        x = self.BN2(x)
        x = self.ELU(x)
        x = self.pool2(x)
        x = self.drop2(x)
        x = self.conv4(x)
        x = self.BN3(x)
        x = self.ELU(x)
        x = self.pool3(x)
        x = self.drop3(x)
        x = self.conv5(x)
        x = self.BN4(x)
        x = self.ELU(x)
        x = self.pool4(x)
        x = self.drop4(x)
        x = self.flatten(x)
        return x


# ------------------------------
#         EEGInception
# ------------------------------
class EEGInceptionEncoder(nn.Module):
    """
    Pytorch Implementation of the EEGInception Encoder.

    See EEGInception for some references.
    The expected **input** is a **3D tensor** with size
    (Batch x Channels x Samples).

    Parameters
    ----------
    Chans: int
        The number of EEG channels.
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
        If True, adds a learnable bias to the output.

        Default = True
    batch_momentum: float, optional
        The batch normalization momentum.

        Default = 0.9
    max_depth_norm: float, optional
        The maximum norm each filter can have in the depthwise block.
        If None no constraint will be included.

        Default = 1.

    Example
    -------
    >>> import selfeeg.models
    >>> import torch
    >>> x = torch.randn(4,8,64)
    >>> mdl = models.EEGInceptionEncoder(8)
    >>> out = mdl(x)
    >>> print(out.shape) # shoud return torch.Size([4, 12])
    >>> print(torch.isnan(out).sum()) # shoud return 0

    """

    def __init__(
        self,
        Chans,
        F1=8,
        D=2,
        kernel_size=64,
        pool=4,
        dropRate=0.5,
        ELUalpha=1.0,
        bias=True,
        batch_momentum=0.1,
        max_depth_norm=1.0,
    ):

        super(EEGInceptionEncoder, self).__init__()
        self.inc1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_size), padding="same", bias=bias),
            nn.BatchNorm2d(F1, momentum=batch_momentum),
            nn.ELU(alpha=ELUalpha),
            nn.Dropout(dropRate),
            DepthwiseConv2d(
                F1, D, kernel_size=(Chans, 1), padding="valid", max_norm=max_depth_norm
            ),
            nn.BatchNorm2d(F1 * D, momentum=batch_momentum),
            nn.ELU(alpha=ELUalpha),
            nn.Dropout(dropRate),
        )
        self.inc2 = nn.Sequential(
            nn.Conv2d(1, F1, (1, int(kernel_size // 2)), padding="same", bias=bias),
            nn.BatchNorm2d(F1, momentum=batch_momentum),
            nn.ELU(alpha=ELUalpha),
            nn.Dropout(dropRate),
            DepthwiseConv2d(
                F1, D, kernel_size=(Chans, 1), padding="valid", max_norm=max_depth_norm
            ),
            nn.BatchNorm2d(F1 * D, momentum=batch_momentum),
            nn.ELU(alpha=ELUalpha),
            nn.Dropout(dropRate),
        )
        self.inc3 = nn.Sequential(
            nn.Conv2d(1, F1, (1, int(kernel_size // 4)), padding="same", bias=bias),
            nn.BatchNorm2d(F1, momentum=batch_momentum),
            nn.ELU(alpha=ELUalpha),
            nn.Dropout(dropRate),
            DepthwiseConv2d(
                F1, D, kernel_size=(Chans, 1), padding="valid", max_norm=max_depth_norm
            ),
            nn.BatchNorm2d(F1 * D, momentum=batch_momentum),
            nn.ELU(alpha=ELUalpha),
            nn.Dropout(dropRate),
        )
        # concatenate inc1 inc2 e inc3 on filter size in forward
        self.pool1 = nn.AvgPool2d((1, pool))

        self.inc4 = nn.Sequential(
            nn.Conv2d(F1 * D * 3, F1, (1, int(kernel_size // 4)), padding="same", bias=bias),
            nn.BatchNorm2d(F1, momentum=batch_momentum),
            nn.ELU(alpha=ELUalpha),
            nn.Dropout(dropRate),
        )
        self.inc5 = nn.Sequential(
            nn.Conv2d(F1 * D * 3, F1, (1, int(kernel_size // 8)), padding="same", bias=bias),
            nn.BatchNorm2d(F1, momentum=batch_momentum),
            nn.ELU(alpha=ELUalpha),
            nn.Dropout(dropRate),
        )
        self.inc6 = nn.Sequential(
            nn.Conv2d(F1 * D * 3, F1, (1, int(kernel_size // 16)), padding="same", bias=bias),
            nn.BatchNorm2d(F1, momentum=batch_momentum),
            nn.ELU(alpha=ELUalpha),
            nn.Dropout(dropRate),
        )
        # concatenate inc4 inc5 e inc6 on filter size in forward
        self.pool2 = nn.AvgPool2d((1, int(pool // 2)))
        self.out1 = nn.Sequential(
            nn.Conv2d(
                F1 * 3, int((F1 * 3) / 2), (1, int(kernel_size // 8)), padding="same", bias=bias
            ),
            nn.BatchNorm2d(int((F1 * 3) / 2), momentum=batch_momentum),
            nn.ELU(alpha=ELUalpha),
            nn.Dropout(dropRate),
        )
        self.pool3 = nn.AvgPool2d((1, int(pool // 2)))
        self.out2 = nn.Sequential(
            nn.Conv2d(
                int((F1 * 3) / 2),
                int((F1 * 3) / 4),
                (1, int(kernel_size // 16)),
                padding="same",
                bias=bias,
            ),
            nn.BatchNorm2d(int((F1 * 3) / 4), momentum=batch_momentum),
            nn.ELU(alpha=ELUalpha),
            nn.Dropout(dropRate),
        )
        self.pool4 = nn.AvgPool2d((1, int(pool // 2)))
        self.flatten = nn.Flatten()

    def forward(self, x):
        """
        :meta private:
        """
        x = torch.unsqueeze(x, 1)
        x1 = self.inc1(x)
        x2 = self.inc2(x)
        x3 = self.inc3(x)
        x_conc = torch.cat((x1, x2, x3), dim=1)
        x_conc = self.pool1(x_conc)
        x1 = self.inc4(x_conc)
        x2 = self.inc5(x_conc)
        x3 = self.inc6(x_conc)
        xout = torch.cat((x1, x2, x3), dim=1)
        xout = self.pool2(xout)
        xout = self.out1(xout)
        xout = self.pool3(xout)
        xout = self.out2(xout)
        xout = self.pool4(xout)
        xout = self.flatten(xout)
        return xout


# ------------------------------
#         TinySleepNet
# ------------------------------
class TinySleepNetEncoder(nn.Module):
    """
    Pytorch Implementation of the TinySleepNet Encoder.

    TinySleepNet is a minimal but better performing architecture derived from
    DeepSleepNet (proposed by the same authors).
    See TinySleepNet for some references.
    The expected **input** is a **3D tensor** with size
    (Batch x Channels x Samples).

    Parameters
    ----------
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
    hidden_lstm: int, optional
        Hidden size of the lstm block.

        Default = 128

    Example
    -------
    >>> import selfeeg.models
    >>> import torch
    >>> x = torch.randn(4,8,1024)
    >>> mdl = models.TinySleepNetEncoder(8,32)
    >>> out = mdl(x)
    >>> print(out.shape) # shoud return torch.Size([4, 128])
    >>> print(torch.isnan(out).sum()) # shoud return 0

    """

    def __init__(
        self,
        Chans,
        Fs,
        F=128,
        kernlength=8,
        pool=8,
        dropRate=0.5,
        batch_momentum=0.1,
        hidden_lstm=128,
    ):

        super(TinySleepNetEncoder, self).__init__()
        self.conv1 = nn.Conv1d(Chans, F, int(Fs // 2), stride=int(Fs // 16), padding="valid")
        self.BN1 = nn.BatchNorm1d(F, momentum=batch_momentum)
        self.Relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(pool, stride=pool)
        self.drop1 = nn.Dropout1d(dropRate)

        self.conv2 = nn.Conv1d(F, F, 8, stride=1, padding="valid")
        self.BN2 = nn.BatchNorm1d(F, momentum=batch_momentum)
        # ReLU()
        self.conv3 = nn.Conv1d(F, F, 8, stride=1, padding="valid")
        self.BN3 = nn.BatchNorm1d(F, momentum=batch_momentum)
        # ReLU()
        self.conv4 = nn.Conv1d(F, F, 8, stride=1, padding="valid")
        self.BN4 = nn.BatchNorm1d(F, momentum=batch_momentum)
        # ReLU()

        self.pool2 = nn.MaxPool1d(pool // 2, stride=pool // 2)
        self.drop2 = nn.Dropout1d(dropRate)

        self.lstm1 = nn.LSTM(input_size=F, hidden_size=hidden_lstm, num_layers=1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        """
        :meta private:
        """
        x = self.conv1(x)
        x = self.BN1(x)
        x = self.Relu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.BN2(x)
        x = self.Relu(x)
        x = self.conv3(x)
        x = self.BN3(x)
        x = self.Relu(x)
        x = self.conv4(x)
        x = self.BN4(x)
        x = self.Relu(x)

        x = self.pool2(x)
        x = self.drop2(x)

        x = torch.permute(x, (2, 0, 1))

        out, (ht, ct) = self.lstm1(x)

        return ht[-1]


# ------------------------------
#          StagerNet
# ------------------------------
class StagerNetEncoder(nn.Module):
    """
    Pytorch implementation of the StagerNet Encoder.

    See TinySleepNet for some references.
    The expected **input** is a **3D tensor** with size
    (Batch x Channels x Samples).

    Parameters
    ----------
    Chans: int
        The number of EEG channels.
    kernlength: int, optional
        The length of the temporal convolutional layer.

        Default = 8
    F: int, optional
        The number of output filters in the temporal convolution layer.

        Default = 128
    pool: int, optional
        The temporal pooling kernel size.

        Default = 4

    Example
    -------
    >>> import selfeeg.models
    >>> import torch
    >>> x = torch.randn(4,8,512)
    >>> mdl = models.StagerNetEncoder(8)
    >>> out = mdl(x)
    >>> print(out.shape) # shoud return torch.Size([4, 128])
    >>> print(torch.isnan(out).sum()) # shoud return 0

    """

    def __init__(self, Chans, kernLength=64, F=8, Pool=16):

        super(StagerNetEncoder, self).__init__()

        self.conv1 = nn.Conv2d(1, Chans, (Chans, 1), stride=(1, 1), bias=True)
        self.conv2 = nn.Conv2d(1, F, (1, kernLength), stride=(1, 1), padding="same")
        self.pooling2 = nn.MaxPool2d((1, Pool), stride=(1, Pool))
        self.conv3 = nn.Conv2d(F, F, (1, kernLength), stride=(1, 1), padding="same")
        self.pooling3 = nn.MaxPool2d((1, Pool), stride=(1, Pool))
        self.flatten3 = nn.Flatten()

    def forward(self, x):
        """
        :meta private:

        """
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = torch.permute(x, (0, 2, 1, 3))
        x = F.relu(self.conv2(x))
        x = self.pooling2(x)
        x = F.relu(self.conv3(x))
        x = self.pooling3(x)
        x = self.flatten3(x)
        return x


# ------------------------------
#         ShallowNet
# ------------------------------
class ShallowNetEncoder(nn.Module):
    """
    Pytorch implementation of the ShallowNet Encoder.

    See ShallowNet for some references.
    The expected **input** is a **3D tensor** with size
    (Batch x Channels x Samples).

    Parameters
    ----------
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
    p: float, optional
        Dropout probability. Must be in [0,1)

        Default= 0.2

    Note
    ----
    In this implementation, the number of channels is an argument.
    However, in the original paper authors preprocess EEG data by
    selecting a subset of only 21 channels. Since the net is very
    minimalistic, please follow the authors' notes.

    Example
    -------
    >>> import selfeeg.models
    >>> import torch
    >>> x = torch.randn(4,8,512)
    >>> mdl = models.ShallowNetEncoder(8)
    >>> out = mdl(x)
    >>> print(out.shape) # shoud return torch.Size([4, 224])
    >>> print(torch.isnan(out).sum()) # shoud return 0

    """

    def __init__(self, Chans, F=40, K1=25, Pool=75, p=0.2):

        super(ShallowNetEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, F, (1, K1), stride=(1, 1))
        self.conv2 = nn.Conv2d(F, F, (Chans, 1), stride=(1, 1))
        self.batch1 = nn.BatchNorm2d(F)
        self.pool2 = nn.AvgPool2d((1, Pool), stride=(1, 15))
        self.drop1 = nn.Dropout(p)
        self.flatten2 = nn.Flatten()

    def forward(self, x):
        """
        :meta private:
        """
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batch1(x)
        x = torch.square(x)
        x = self.pool2(x)
        x = torch.log(torch.clamp(x, 1e-7, 10000))
        x = self.drop1(x)
        x = self.flatten2(x)
        return x


# ------------------------------
#         ResNet 1D
# ------------------------------
class BasicBlock1(nn.Module):
    """
    :meta private:
    """

    def __init__(self, inplanes, planes, kernLength=7, stride=1):

        super(BasicBlock1, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=(1, kernLength),
            stride=(1, stride),
            padding=(0, kernLength // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=(1, kernLength),
            stride=(1, 1),
            padding=(0, kernLength // 2),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        if inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes,
                    kernel_size=(1, kernLength),
                    stride=(1, stride),
                    padding=(0, kernLength // 2),
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = lambda x: x

    def forward(self, x):
        """
        :meta private:
        """
        residual = self.downsample(x)
        # print('residual: ', residual.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print('out 1: ', out.shape)
        out = self.conv2(out)
        out = self.bn2(out)
        # print('out 2: ', out.shape)

        out += residual
        out = self.relu(out)

        return out


class ResNet1DEncoder(nn.Module):
    """
    Pytorch implementation of the Resnet Encoder

    This version uses temporal convolutional layers
    (so conv2d with horizontal kernel).
    See ResNet for the reference paper which inspired this implementation.

    The expected **input** is a **3D tensor** with size
    (Batch x Channels x Samples).

    Parameters
    ----------
    Chans: int
        The number of EEG channels.
    block: nn.Module, optional
        An nn.Module defining the resnet block.
    Layers: list of 4 int, optional
        A list of integers indicating the number of times
        the resnet block is repeated .

        Default = [2,2,2,2]
    inplane: int, optional
        The number of output filters.
    kernLength: int, optional
        The length of the temporal convolutional layer.

        Default = 25
    addConnection: bool, optional
        Whether to add a connection from the start of the resnet part
        to the network head. If set to True the output of the following
        conv2d will be concatenate to the postblock output:

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
        A custom nn.Module to pass after the sequence of resnet blocks and
        before the network head. If none is left, the following sequence is used:

            1. nn.conv2d(1, self.inplane, kernel_size=(1, kernLength), bias=False)
            2. nn.BatchNorm2d()
            3. nn.ReLU()

            Default = None

    Note
    ----
    The compatibility between each custom nn.Module given as
    argument has not beend checked. Design the network carefully.

    Example
    -------
    >>> import selfeeg.models
    >>> import torch
    >>> x = torch.randn(4,8,512)
    >>> mdl = models.ResNet1DEncoder(8)
    >>> out = mdl(x)
    >>> print(out.shape) # shoud return torch.Size([4, 296])
    >>> print(torch.isnan(out).sum()) # shoud return 0

    """

    def __init__(
        self,
        Chans,
        block: nn.Module = BasicBlock1,
        Layers: "list of 4 ints" = [2, 2, 2, 2],
        inplane: int = 16,
        kernLength: int = 7,
        addConnection: bool = False,
        preBlock: nn.Module = None,
        postBlock: nn.Module = None,
    ):

        super(ResNet1DEncoder, self).__init__()
        self.inplane = inplane
        self.kernLength = kernLength
        self.connection = addConnection

        #   PRE-RESIDUAL
        if preBlock is None:
            self.preBlocks = nn.Sequential(
                nn.Conv2d(
                    1,
                    self.inplane,
                    kernel_size=(1, kernLength),
                    stride=(1, 2),
                    padding=(0, kernLength // 2),
                    bias=False,
                ),
                nn.BatchNorm2d(self.inplane),
                nn.ReLU(inplace=True),
            )
        else:
            self.preBlocks = preBlock

        #  RESIDUAL BLOCKS
        self.layer1 = self._make_layer(
            block, self.inplane, Layers[0], kernLength=kernLength, stride=1
        )
        self.layer2 = self._make_layer(
            block, self.inplane * 2, Layers[1], kernLength=kernLength, stride=2
        )
        self.layer3 = self._make_layer(
            block, self.inplane * 2, Layers[2], kernLength=kernLength, stride=2
        )
        self.layer4 = self._make_layer(
            block, self.inplane * 2, Layers[3], kernLength=kernLength, stride=2
        )

        #  POST-RESIDUAL
        if postBlock is None:
            self.postBlocks = nn.Sequential(
                nn.Conv2d(
                    self.inplane,
                    inplane,
                    kernel_size=(1, kernLength),
                    stride=(1, 1),
                    padding=(0, 0),
                    bias=False,
                ),
                nn.AdaptiveAvgPool2d((Chans, 1)),
            )
        else:
            self.postBlocks = postBlock

        # RESIDUAL SKIP CONNECTION
        if self.connection:
            self.conv3 = nn.Conv2d(
                inplane,
                2,
                kernel_size=(Chans, kernLength),
                stride=(1, int(kernLength // 2)),
                padding="valid",
                bias=False,
            )

        # WEIGHT INITIALIZATION
        self.initialize()

    def initialize(self):
        """
        :meta private:
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, kernLength=7, stride=1, **kwarg):
        """
        :meta private:
        """

        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, kernLength, stride))
            self.inplane = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        :meta private:
        """
        x = torch.unsqueeze(x, 1)

        x1 = self.preBlocks(x)

        x2 = self.layer1(x1)
        x2 = self.layer2(x2)
        x2 = self.layer3(x2)
        x2 = self.layer4(x2)
        x2 = self.postBlocks(x2)
        out1 = x2.view(x2.size(0), -1)

        if self.connection:
            out2 = self.conv3(x1)
            out2 = out2.view(out2.size(0), -1)
            embeddings = torch.cat((out1, out2), dim=-1)
        else:
            embeddings = out1

        return embeddings


# ------------------------------
#            STNet
# ------------------------------
class STNetInceptionBlock(nn.Module):
    """
    :meta private:
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(STNetInceptionBlock, self).__init__()
        self.convBig = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=1, padding="same", bias=bias
        )
        self.convMedium = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size - 2,
            stride=1,
            padding="same",
            bias=bias,
        )
        self.convSmall = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size - 4,
            stride=1,
            padding="same",
            bias=bias,
        )

    def forward(self, x):
        """
        :meta private:
        """
        return self.convBig(x) + self.convMedium(x) + self.convSmall(x)


class STNetEncoder(nn.Module):
    """
    Pytorch implementation of the STNet Encoder.

    See STNet for some references.
    The expected **input** is a **4D tensor** with size
    (Batch x Samples x Grid_width x Grid_width), i.e. the classical 2d matrix
    with rows as channels and columns as samples is rearranged in a 3d tensor where
    the first is the Sample dimension and the last 2 dimensions are the channel
    dim rearrange in a 2d grid. Check the original paper for a better
    understanding of the input.

    Parameters
    ----------
    Sample: int
        The number of EEG Samples.
    F: int, optional
        The number of output filters in the convolutional layer.

        Default = 8
    kernLength: int, optional
        The length of the convolutional layer.

        Default = 5
    dropRate: float, optional
        The dropout percentage in range [0,1].

        Default = 0.5
    bias: bool, optional
        If True, adds a learnable bias to the convolutional layers.

        Default = True

    Example
    -------
    >>> import selfeeg.models
    >>> import torch
    >>> x = torch.randn(4,128,9,9)
    >>> mdl = models.STNetEncoder(128)
    >>> out = mdl(x)
    >>> print(out.shape) # shoud return torch.Size([4, 1296])
    >>> print(torch.isnan(out).sum()) # shoud return 0

    """

    def __init__(self, Samples, F=256, kernlength=5, dropRate=0.5, bias=True):
        super(STNetEncoder, self).__init__()

        self.conv1 = nn.Conv2d(
            Samples, F, kernel_size=kernlength - 2, stride=1, padding="same", bias=bias
        )
        self.selu = nn.SELU()
        self.drop1 = nn.Dropout(dropRate)
        self.conv2 = nn.Conv2d(
            F, int(F / 2), kernel_size=kernlength, stride=1, padding="same", bias=bias
        )
        self.drop2 = nn.Dropout(dropRate)
        self.conv3 = nn.Conv2d(
            int(F / 2), int(F / 4), kernel_size=kernlength, stride=1, padding="same", bias=bias
        )
        self.drop3 = nn.Dropout(dropRate)
        self.sep1 = SeparableConv2d(
            int(F / 4), int(F / 8), kernel_size=kernlength, stride=1, padding="same", bias=bias
        )
        self.drop4 = nn.Dropout(dropRate)
        self.inception = STNetInceptionBlock(int(F / 8), int(F / 16), kernlength, bias)
        self.drop5 = nn.Dropout(dropRate)
        self.flatten = nn.Flatten()

    def forward(self, x):
        """
        :meta private:
        """
        x = self.conv1(x)
        x = self.selu(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.selu(x)
        x = self.drop2(x)
        x = self.conv3(x)
        x = self.selu(x)
        x = self.drop3(x)
        x = self.sep1(x)
        x = self.selu(x)
        x = self.drop4(x)
        x = self.inception(x)
        x = self.selu(x)
        x = self.drop5(x)
        x = self.flatten(x)
        return x


# ------------------------------
#            EEGSym
# ------------------------------
class EEGSymInput(nn.Module):
    """
    :meta private:
    """

    def __init__(self, Chans=8, lateral_chans=3, first_left=True):
        super(EEGSymInput, self).__init__()
        self.lateral = lateral_chans
        self.central = Chans - lateral_chans * 2
        self.hemichan = self.lateral + self.central
        self.left = first_left
        self.right_shuffle = [-i for i in range(1, self.hemichan - self.central + 1)] + [
            i for i in range(self.hemichan - self.central, self.hemichan)
        ]

    def forward(self, x):
        """
        :meta private:
        """
        # expand dimension:
        # new tensor will be 5D with
        # ( batch x filter x hemisphere x channel x samples )
        x = x.unsqueeze(1).unsqueeze(1)
        left = x[..., : self.hemichan, :]
        right = x[..., self.right_shuffle, :]
        if not (self.left):
            left, right = right, left
        x = torch.cat((left, right), 2)
        return x


class EEGSymInception(nn.Module):
    """
    :meta private:
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernels=[16, 32, 64],
        spatial_kernel=5,
        pool=2,
        dropRate=0.5,
        ELUalpha=1.0,
        bias=True,
        residual=True,
    ):
        super(EEGSymInception, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, (1, 1, kernels[0]), padding="same", bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.ELU(alpha=ELUalpha),
            nn.Dropout3d(dropRate),
        )
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, (1, 1, kernels[1]), padding="same", bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.ELU(alpha=ELUalpha),
            nn.Dropout3d(dropRate),
        )
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, (1, 1, kernels[2]), padding="same", bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.ELU(alpha=ELUalpha),
            nn.Dropout3d(dropRate),
        )
        # concatenation
        # add residual
        self.pool = nn.AvgPool3d((1, 1, pool))
        self.spatial = nn.Sequential(
            nn.Conv3d(
                out_channels * 3,
                out_channels * 3,
                (1, spatial_kernel, 1),
                padding="valid",
                groups=out_channels * 3,
                bias=bias,
            ),
            nn.BatchNorm3d(out_channels * 3),
            nn.ELU(alpha=ELUalpha),
            nn.Dropout3d(dropRate),
        )
        # add residual

    def forward(self, x):
        """
        :meta private:
        """
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_conc = torch.cat((x1, x2, x3), dim=1)
        x_conc = x_conc + x
        x_conc = self.pool(x_conc)
        xout = self.spatial(x_conc)
        xout = xout + x_conc
        return xout


class EEGSymResBlock(nn.Module):
    """
    :meta private:
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        spatial_kernel=5,
        pool=2,
        dropRate=0.5,
        ELUalpha=1.0,
        bias=True,
    ):
        super(EEGSymResBlock, self).__init__()
        self.temporal1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, (1, 1, kernel_size), padding="same", bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.ELU(alpha=ELUalpha),
            nn.Dropout3d(dropRate),
        )
        self.temporal2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, (1, 1, 1), padding="same", bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.ELU(alpha=ELUalpha),
            nn.Dropout3d(dropRate),
        )
        # add t1 and t2
        self.pool = nn.AvgPool3d((1, 1, pool))
        self.spatial = nn.Sequential(
            nn.Conv3d(
                out_channels, out_channels, (1, spatial_kernel, 1), padding="valid", bias=bias
            ),
            nn.BatchNorm3d(out_channels),
            nn.ELU(alpha=ELUalpha),
            nn.Dropout3d(dropRate),
        )

    def forward(self, x):
        """
        :meta private:
        """
        x1 = self.temporal1(x)
        x2 = self.temporal2(x)
        xout = x1 + x2
        xout = self.pool(xout)
        x_spa = self.spatial(xout)
        xout = xout + x_spa
        return xout


class EEGSymEncoder(nn.Module):
    """
    Pytorch implementation of the EEGSym Encoder.

    See EEGSym for some references.
    The expected **input** is a **3D tensor** with size
    (Batch x Channels x Samples). However Channel order is expected to be
    symmetrical along lateral channels to perform the reshaping operation
    correctly. For instance, if the first channel index refers to the FP1 channel,
    then the last must refer to the other hemisphere counterpart, i.e. FP2.
    See the original paper to further understand this operation.

    Parameters
    ----------
    Chans: int
        The number of EEG channels.
    Sample: int
        The number of EEG Samples.
    Fs: float
        The sampling frequency.
    scales_time: tuple of 3 float, optional
        The portion of EEG (in milliseconds) the short, medium and long temporal
        convolutional layers must cover. Kernel size will be automatically
        calculated based on the given sampling rate.

        Default = (500,250,125)
    lateral_chans: int, optional
        The amount of lateral channels. It will be used to reshape the 3D tensor
        in a 5D tensor with size
        ( batch x filters x hemisphere x channel x samples ).
        See the original paper for more info.

        Default = 3
    first_left: bool, optional
        Whether the first half of the channels are of the left hemisphere or not.

        Default = True
    F: int, optional
        The output filters of each branch of the first inception block.
        Other output will be automatically calculated.

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

    Example
    -------
    >>> import selfeeg.models
    >>> import torch
    >>> x = torch.randn(4,8,1024)
    >>> mdl = models.EEGSymEncoder(8,1024,64)
    >>> out = mdl(x)
    >>> print(out.shape) # shoud return torch.Size([4, 36])
    >>> print(torch.isnan(out).sum()) # shoud return 0

    """

    def __init__(
        self,
        Chans,
        Samples,
        Fs,
        scales_time=(500, 250, 125),
        lateral_chans=3,
        first_left=True,
        F=8,
        pool=2,
        dropRate=0.5,
        ELUalpha=1.0,
        bias=True,
        residual=True,
    ):

        super(EEGSymEncoder, self).__init__()

        self.input_samples = int(Samples * Fs / 1000)
        self.scales_samples = [int(s * Fs / 1000) for s in scales_time]
        self.scales_samples_2 = [int(s / 4) for s in self.scales_samples]

        self.symInput = EEGSymInput(Chans, lateral_chans, first_left)
        self.inception1 = EEGSymInception(
            1,
            F * 3,
            self.scales_samples,
            self.symInput.hemichan,
            pool,
            dropRate,
            ELUalpha,
            bias,
            residual,
        )
        self.inception2 = EEGSymInception(
            F * 9,
            F * 3,
            self.scales_samples_2,
            self.symInput.hemichan,
            pool,
            dropRate,
            ELUalpha,
            bias,
            residual,
        )

        self.resblock1 = EEGSymResBlock(
            F * 9,
            int((F * 9) / 2),
            self.scales_samples_2[0],
            self.symInput.hemichan,
            pool,
            dropRate,
            ELUalpha,
            bias=True,
        )
        self.resblock2 = EEGSymResBlock(
            int((F * 9) / 2),
            int((F * 9) / 2),
            self.scales_samples_2[1],
            self.symInput.hemichan,
            pool,
            dropRate,
            ELUalpha,
            bias=True,
        )
        self.resblock3 = EEGSymResBlock(
            int((F * 9) / 2),
            int((F * 9) / 4),
            self.scales_samples_2[2],
            self.symInput.hemichan,
            pool,
            dropRate,
            ELUalpha,
            bias=True,
        )

        self.tempend = nn.Sequential(
            nn.Conv3d(
                int((F * 9) / 4),
                int((F * 9) / 4),
                (1, 1, self.scales_samples_2[2]),
                padding="same",
                bias=bias,
            ),
            nn.BatchNorm3d(int((F * 9) / 4)),
            nn.ELU(alpha=ELUalpha),
            nn.Dropout3d(dropRate),
        )
        # add tempend and resblock3
        self.pool1 = nn.AvgPool3d((1, 1, pool))

        self.temp1 = nn.Sequential(
            nn.Conv3d(
                int((F * 9) / 4),
                int((F * 9) / 4),
                (1, 1, self.scales_samples_2[2]),
                padding="same",
                bias=bias,
            ),
            nn.BatchNorm3d(int((F * 9) / 4)),
            nn.ELU(alpha=ELUalpha),
            nn.Dropout3d(dropRate),
        )
        # add temp1 and pool1
        self.temp2 = nn.Sequential(
            nn.Conv3d(
                int((F * 9) / 4),
                int((F * 9) / 4),
                (1, 1, self.scales_samples_2[2]),
                padding="same",
                bias=bias,
            ),
            nn.BatchNorm3d(int((F * 9) / 4)),
            nn.ELU(alpha=ELUalpha),
            nn.Dropout3d(dropRate),
        )
        # add temp2 and temp1
        self.merge1 = nn.Sequential(
            nn.Conv3d(
                int((F * 9) / 4),
                int((F * 9) / 4),
                (2, self.symInput.hemichan, 1),
                groups=int((F * 9) / 8),
                padding="valid",
                bias=bias,
            ),
            nn.BatchNorm3d(int((F * 9) / 4)),
            nn.ELU(alpha=ELUalpha),
            nn.Dropout3d(dropRate),
        )
        self.temp3 = nn.Sequential(
            nn.Conv3d(
                int((F * 9) / 4),
                int((F * 9) / 4),
                (1, 1, self.scales_samples_2[2]),
                padding="same",
                bias=bias,
            ),
            nn.BatchNorm3d(int((F * 9) / 4)),
            nn.ELU(alpha=ELUalpha),
            nn.Dropout3d(dropRate),
        )
        # add temp3 and merge1
        self.merge2 = nn.Sequential(
            nn.Conv3d(
                int((F * 9) / 4),
                int((F * 9) / 2),
                (1, 1, int(Samples // (pool**6))),
                groups=int((F * 9) / 4),
                padding="valid",
                bias=bias,
            ),
            nn.BatchNorm3d(int((F * 9) / 2)),
            nn.ELU(alpha=ELUalpha),
            nn.Dropout3d(dropRate),
        )

        self.out1 = nn.Sequential(
            nn.Conv3d(int((F * 9) / 2), int((F * 9) / 2), (1, 1, 1), padding="same", bias=bias),
            nn.BatchNorm3d(int((F * 9) / 2)),
            nn.ELU(alpha=ELUalpha),
            nn.Dropout3d(dropRate),
        )
        # add out1 and merge2
        self.out2 = nn.Sequential(
            nn.Conv3d(int((F * 9) / 2), int((F * 9) / 2), (1, 1, 1), padding="same", bias=bias),
            nn.BatchNorm3d(int((F * 9) / 2)),
            nn.ELU(alpha=ELUalpha),
            nn.Dropout3d(dropRate),
        )
        # add out2 and out1
        self.out3 = nn.Sequential(
            nn.Conv3d(int((F * 9) / 2), int((F * 9) / 2), (1, 1, 1), padding="same", bias=bias),
            nn.BatchNorm3d(int((F * 9) / 2)),
            nn.ELU(alpha=ELUalpha),
            nn.Dropout3d(dropRate),
        )
        # add out2 and out3
        self.out4 = nn.Sequential(
            nn.Conv3d(int((F * 9) / 2), int((F * 9) / 2), (1, 1, 1), padding="same", bias=bias),
            nn.BatchNorm3d(int((F * 9) / 2)),
            nn.ELU(alpha=ELUalpha),
            nn.Dropout3d(dropRate),
        )
        # add out4 and out3
        self.flatten = nn.Flatten()

    def forward(self, x):
        """
        :meta private:
        """
        x = self.symInput(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x1 = self.tempend(x)
        x = x + x1
        x = self.pool1(x)
        x1 = self.temp1(x)
        x = x + x1
        x1 = self.temp2(x)
        x = x + x1
        x = self.merge1(x)
        x1 = self.temp3(x)
        x = x + x1
        x = self.merge2(x)
        x1 = self.out1(x)
        x = x + x1
        x1 = self.out2(x)
        x = x + x1
        x1 = self.out3(x)
        x = x + x1
        x1 = self.out4(x)
        x = x + x1
        x = self.flatten(x)
        return x


# ------------------------------
#            FBCNet
# ------------------------------
class _ReshapeLayer(nn.Module):
    """
    :meta private:
    """

    def __init__(self, Channels, TemporalStride, Samples):
        super(_ReshapeLayer, self).__init__()

        self.dim1 = Channels
        self.dim2 = TemporalStride
        if Samples % TemporalStride == 0:
            self.dim3 = int(Samples / TemporalStride)
        else:
            raise ValueError("Samples must be a multiple of TemporalStride")

    def forward(self, x):
        return x.reshape([x.shape[0], self.dim1, self.dim2, self.dim3])


class _VarLayer(nn.Module):
    """
    :meta private:
    """

    def __init__(self, dim):
        super(_VarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.var(dim=self.dim, keepdim=True)


class _StdLayer(nn.Module):
    """
    :meta private:
    """

    def __init__(self, dim):
        super(_StdLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.std(dim=self.dim, keepdim=True)


class _LogVarLayer(nn.Module):
    """
    :meta private:
    """

    def __init__(self, dim):
        super(_LogVarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(x.var(dim=self.dim, keepdim=True), 1e-6, 1e6))


class FBCNetEncoder(nn.Module):
    """
    Pytorch implementation of the FBCNet Encoder.

    See FBCNet for some references.
    The expected **input** is a **3D tensor** with size
    (Batch x Channels x Samples).

    Filter operation is applied through the torchaudio filtfilt function.
    Do not use too strict filter settings as this might generate nan or
    too high values.

    Parameters
    ----------
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

    Example
    -------
    >>> import selfeeg.models
    >>> import torch
    >>> x = torch.randn(4,8,512)
    >>> mdl = models.FBCNetEncoder(8, 512, 128)
    >>> out = mdl(x)
    >>> print(out.shape) # shoud return torch.Size([4, 1152])
    >>> print(torch.isnan(out).sum()) # shoud return 0

    """

    def __init__(
        self,
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
        depthwise_max_norm=None,
    ):
        super(FBCNetEncoder, self).__init__()
        self.FilterBands = FilterBands
        self.TemporalStride = TemporalStride
        self.TMBKernel = int(Samples / TemporalStride)

        # Filter Bank
        self.FBank = FilterBank(
            Fs,
            FilterBands,
            FilterRange,
            FilterType,
            FilterStopRippple,
            FilterPassRipple,
            FilterRangeTol,
            FilterSkipFirst,
        )

        # Spatial Convolution
        self.SCB = nn.Sequential(
            DepthwiseConv2d(FilterBands, D, (Chans, 1), max_norm=depthwise_max_norm, padding=0),
            nn.BatchNorm2d(D * FilterBands, momentum=batch_momentum),
            nn.SiLU(),
        )

        if TemporalType.casefold() == "max":
            self.TMB = nn.MaxPool2d((1, self.TMBKernel), self.TMBKernel)
        elif TemporalType.casefold() == "mean":
            self.TMB = nn.AvgPool2d((1, self.TMBKernel), self.TMBKernel)
        elif TemporalType.casefold() == "std":
            self.TMB = nn.Sequential(
                _ReshapeLayer(D * FilterBands, TemporalStride, Samples),
                _StdLayer(dim=3),
            )
        elif TemporalType.casefold() == "var":
            self.TMB = nn.Sequential(
                _ReshapeLayer(D * FilterBands, TemporalStride, Samples),
                _VarLayer(dim=3),
            )
        elif TemporalType.casefold() == "logvar":
            self.TMB = nn.Sequential(
                _ReshapeLayer(D * FilterBands, TemporalStride, Samples),
                _LogVarLayer(dim=3),
            )
        else:
            raise ValueError("wrong temporal layer type.")

    def forward(self, x):
        """
        :meta private:
        """
        x = self.FBank(x)
        x = self.SCB(x)
        x = self.TMB(x)
        x = torch.flatten(x, 1)
        return x
