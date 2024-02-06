import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "BasicBlock1",
    "ConstrainedConv2d",
    "ConstrainedDense",
    "DeepConvNet",
    "DeepConvNetEncoder",
    "DepthwiseConv2d",
    "EEGInception",
    "EEGInceptionEncoder",
    "EEGNet",
    "EEGNetEncoder",
    "EEGSym",
    "EEGSymEncoder",
    "ResNet1D",
    "ResNet1DEncoder",
    "SeparableConv2d",
    "ShallowNet",
    "ShallowNetEncoder",
    "StagerNet",
    "StagerNetEncoder",
    "STNet",
    "STNetEncoder",
    "TinySleepNet",
    "TinySleepNetEncoder",
]


# ### Special Kernels not implemented in pytorch
class DepthwiseConv2d(nn.Conv2d):
    """
    Pytorch implementation of the Depthwise Convolutional layer with the possibility to
    add a norm constraint on the filter (feature) dimension.
    Most of the parameters are the same as described in pytorch conv2D help.

    Parameters
    ----------
    in_channels: int
        Number of input channels.
    depth_multiplier: int
        The depth multiplier. Output channels will be depth_multiplier*in_channels.
    kernel_size: int or tuple
        Size of the convolving kernel.
    stride: int or tuple, optional
        Stride of the convolution.

        Default = 1
    padding: int, tuple or str, optional
        Padding added to all four sides of the input.

        Default = 0
    dilation: int or tuple, optional
        Spacing between kernel elements.

        Default = 1
    bias: bool, optional
        If True, adds a learnable bias to the output.

        Default = True
    max_norm: float, optional
        The maximum norm each filter can have. If None no constraint will be included.

        Default = None

    Example
    -------
    >>> import selfeeg.models
    >>> import torch
    >>> x = torch.randn(4,1,8,64)
    >>> mdl = models.DepthwiseConv2d(1,2,(1,15))
    >>> out = mdl(x)
    >>> print(out.shape) # shoud return torch.Size([4, 2, 8, 64])
    >>> print(torch.isnan(out).sum()) # shoud return 0

    """

    def __init__(
        self,
        in_channels,
        depth_multiplier,
        kernel_size,
        stride=1,
        padding="same",
        dilation=1,
        bias=False,
        max_norm=None,
    ):
        super(DepthwiseConv2d, self).__init__(
            in_channels,
            depth_multiplier * in_channels,
            kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        if max_norm is not None:
            if max_norm <= 0:
                raise ValueError("max_norm can't be lower or equal than 0")
            else:
                self.max_norm = max_norm
        else:
            self.max_norm = max_norm

    @torch.no_grad()
    def scale_norm(self, eps=1e-9):
        """
        Citing the Tensorflow documentation, the implementation tries to replicate this

        integer, axis along which to calculate weight norms.
        For instance, in a Dense layer the weight
        matrix has shape (input_dim, output_dim), set axis to 0 to constrain each
        weight vector of length (input_dim,). In a Conv2D layer with
        data_format="channels_last", the weight tensor has
        shape (rows, cols, input_depth, output_depth), set axis to [0, 1, 2]
        to constrain the weights
        of each filter tensor of size (rows, cols, input_depth).

        :meta private:
        """
        # calculate the norm of each filter of size (row, cols, input_depth), here (1, kernel_size)
        if self.kernel_size[1] > 1:
            norm = self.weight.norm(dim=2, keepdim=True).norm(dim=3, keepdim=True)
        else:
            norm = self.weight.norm(dim=2, keepdim=True)

        # rescale only those filters which have a norm bigger than the maximum allowed
        if (norm > self.max_norm).sum() > 0:
            desired = torch.clamp(norm, 0, self.max_norm)
            self.weight = torch.nn.Parameter(self.weight * desired / (eps + norm))

    def forward(self, input):
        """
        :meta private:
        """
        if self.max_norm is not None:
            self.scale_norm(self.max_norm)
        return self._conv_forward(input, self.weight, self.bias)


class SeparableConv2d(nn.Module):
    """
    Pytorch implementation of the Separable Convolutional layer with the possibility of
    adding a norm constraint on the depthwise filters (feature) dimension.
    The layer applies first a depthwise conv2d, then a pointwise conv2d (kernel size = 1)
    Most of the parameters are the same as described in pytorch conv2D help.

    Parameters
    ----------
    in_channels: int
        Number of input channels.
    out_channels: int
        Number of output channels.
    kernel_size: int or tuple
        Size of the convolving kernel
    stride: int or tuple, optional
        Stride of the convolution.

        Default = 1
    padding: int, tuple or str, optional
        Padding added to all four sides of the input.

        Default = 0
    dilation: int or tuple, optional
        Spacing between kernel elements.

        Default = 1
    bias: bool, optional
        If True, adds a learnable bias to the output.

        Default = True
    depth_multiplier: int, optional
        The depth multiplier of the depthwise block.

        Default = 1
    depth_max_norm: float, optional
        The maximum norm each filter can have in the depthwise block.
        If None no constraint will be included.

        Default = None

    Example
    -------
    >>> import selfeeg.models
    >>> import torch
    >>> x = torch.randn(4,1,8,64)
    >>> mdl = models.SeparableConv2d(1,4,(1,15), depth_multiplier=4)
    >>> out = mdl(x)
    >>> print(out.shape) # shoud return torch.Size([4, 4, 8, 64])
    >>> print(torch.isnan(out).sum()) # shoud return 0

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding="same",
        dilation=1,
        bias=False,
        depth_multiplier=1,
        depth_max_norm=None,
    ):
        super(SeparableConv2d, self).__init__()
        self.depthwise = DepthwiseConv2d(
            in_channels,
            depth_multiplier,
            kernel_size,
            stride,
            padding,
            dilation,
            bias,
            max_norm=None,
        )
        self.pointwise = nn.Conv2d(
            in_channels * depth_multiplier, out_channels, kernel_size=1, bias=bias
        )

    def forward(self, x):
        """
        :meta private:
        """
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class ConstrainedDense(nn.Linear):
    """
    Pytorch implementation of the Dense layer with the possibility of adding a norm constraint.
    Most of the parameters are the same as described in pytorch Linear help.

    Parameters
    ----------
    in_features: int
        Number of input features.
    out_channels: int
        Number of output features.
    bias: bool, optional
        If True, adds a learnable bias to the output.

        Default = True
    device: torch.device or str, optional
        The torch device.
    dtype: torch dtype, optional
        layer dtype, i.e., the data type of the torch.Tensor defining the layer weights.
    max_norm: float, optional
        The maximum norm of the layer. If None no constraint will be included.

        Default = None

    Example
    -------
    >>> import selfeeg.models
    >>> import torch
    >>> x = torch.randn(4,64)
    >>> mdl = models.ConstrainedDense(64,32)
    >>> out = mdl(x)
    >>> print(out.shape) # shoud return torch.Size([4, 32])
    >>> print(torch.isnan(out).sum()) # shoud return 0

    """

    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype=None, max_norm=None
    ):
        super(ConstrainedDense, self).__init__(in_features, out_features, bias, device, dtype)

        if max_norm is not None:
            if max_norm <= 0:
                raise ValueError("max_norm can't be lower or equal than 0")
            else:
                self.max_norm = max_norm
        else:
            self.max_norm = max_norm

    @torch.no_grad()
    def scale_norm(self, eps=1e-9):
        """
        Citing the Tensorflow documentation, the implementation tries to replicate this

        integer, axis along which to calculate weight norms. For instance, in a Dense
        layer the weight matrix has shape
        (input_dim, output_dim), set axis to 0 to constrain each weight vector of length
        (input_dim,).
        In a Conv2D layer with data_format="channels_last", the weight tensor has shape (rows,
        cols, input_depth, output_depth),
        set axis to [0, 1, 2] to constrain the weights of each filter tensor of size (rows,
        cols, input_depth).

        :meta private:
        """
        # calculate the norm of each filter of size (row, cols, input_depth),
        # here (1, kernel_size)
        norm = self.weight.norm(dim=1, keepdim=True)

        # rescale only those filters which have a norm bigger than the maximum allowed
        if (norm > self.max_norm).sum() > 0:
            desired = torch.clamp(norm, 0, self.max_norm)
            self.weight = torch.nn.Parameter(self.weight * desired / (eps + norm))

    def forward(self, input):
        """
        :meta private:
        """
        if self.max_norm is not None:
            self.scale_norm(self.max_norm)
        return F.linear(input, self.weight, self.bias)


class ConstrainedConv2d(nn.Conv2d):
    """
    Pytorch implementation of the Convolutional 2D layer with the possibilty of
    adding a max_norm constraint on the filter (feature) dimension.
    Most of the parameters are the same as described in pytorch conv2D help.

    Parameters
    ----------
    in_channels: int
        Number of input channels.
    out_channels: int
        Number of output channels.
    kernel_size: int or tuple
        Size of the convolving kernel.
    stride: int or tuple, optional
        Stride of the convolution.

        Default = 1
    padding: int, tuple or str, optional
        Padding added to all four sides of the input.

        Default = 0
    dilation: int or tuple, optional
        Spacing between kernel elements.

        Default = 1
    groups: int, optional
        Number of blocked connections from input channels to output channels.

        Default = 1
    bias: bool, optional
        If True, adds a learnable bias to the output.

        Default = True
    padding_mode: str, optional
        Any of 'zeros', 'reflect', 'replicate' or 'circular'.

        Default = 'zeros'
    device: torch.device or str, optional
        The torch device.
    dtype: torch.dtype, optional
        Layer dtype, i.e., the data type of the torch.Tensor defining the layer weights.
    max_norm: float, optional
        The maximum norm each filter can have. If None no constraint will be included.

        Default = None

    Example
    -------
    >>> import selfeeg.models
    >>> import torch
    >>> x = torch.randn(4,1,8,64)
    >>> mdl = models.ConstrainedConv2d(1,4,(1,15))
    >>> out = mdl(x)
    >>> print(out.shape) # shoud return torch.Size([4, 4, 8, 64])
    >>> print(torch.isnan(out).sum()) # shoud return 0

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding="same",
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        max_norm=None,
    ):
        super(ConstrainedConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

        if max_norm is not None:
            if max_norm <= 0:
                raise ValueError("max_norm can't be lower or equal than 0")
            else:
                self.max_norm = max_norm
        else:
            self.max_norm = max_norm

    @torch.no_grad()
    def scale_norm(self, eps=1e-9):
        """
        Citing the Tensorflow documentation, the implementation tries to replicate this
        integer, axis along which to calculate weight norms. For instance, in a Dense
        layer the weight matrix has shape
        (input_dim, output_dim), set axis to 0 to constrain each weight vector of length
        (input_dim,).
        In a Conv2D layer with data_format="channels_last", the weight tensor has shape (rows,
        cols, input_depth, output_depth),
        set axis to [0, 1, 2] to constrain the weights of each filter tensor of size (rows,
        cols, input_depth).

        :meta private:
        """
        # calculate the norm of each filter of size
        # (row, cols, input_depth), here (1, kernel_size)
        if self.kernel_size[1] > 1:
            norm = self.weight.norm(dim=2, keepdim=True).norm(dim=3, keepdim=True)
        else:
            norm = self.weight.norm(dim=2, keepdim=True)

        # rescale only those filters which have a norm bigger than the maximum allowed
        if (norm > self.max_norm).sum() > 0:
            desired = torch.clamp(norm, 0, self.max_norm)
            self.weight = torch.nn.Parameter(self.weight * desired / (eps + norm))

    def forward(self, input):
        """
        :meta private:
        """
        if self.max_norm is not None:
            self.scale_norm(self.max_norm)
        return self._conv_forward(input, self.weight, self.bias)


# ------------------------------
#           EEGNet
# ------------------------------
class EEGNetEncoder(nn.Module):
    """
    Pytorch Implementation of the EEGnet Encoder.
    See EEGNet for some references.

    The expected **input** is a **3D tensor** with size (Batch x Channels x Samples).

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
        If None no constraint will be included.

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
        self.drop2 = (
            nn.Dropout(p=dropRate) if dropType.lower() == "dropout" else nn.Dropout2d(p=dropRate)
        )

        # Layer 3
        self.sepconv3 = SeparableConv2d(
            D * F1, F2, (1, separable_kernel), bias=False, padding="same"
        )
        self.batchnorm3 = nn.BatchNorm2d(F2, False)
        self.elu3 = nn.ELU(alpha=ELUalpha)
        self.pooling3 = nn.AvgPool2d((1, pool2))
        self.drop3 = (
            nn.Dropout(p=dropRate) if dropType.lower() == "dropout" else nn.Dropout2d(p=dropRate)
        )
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


class EEGNet(nn.Module):
    """
    For more information see the following paper [EEGnet]_ .
    Keras implementation of the full EEGnet (updated version), more info
    can be found here [eegnetgit]_ .

    The expected **input** is a **3D tensor** with size (Batch x Channels x Samples).

    Parameters
    ----------
    nb_classes: int
        The number of classes. If less than 2, a binary classification problem is considered
        (output dimensions will be [batch, 1] in this case).
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

    Note
    ----
    This implementation refers to the latest version of EEGNet which
    can be found in the official repository (see references).

    References
    ----------
    .. [EEGnet] Lawhern et al., EEGNet: a compact convolutional neural network for EEG-based
      brain–computer interfaces. Journal of Neural Engineering. 2018
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
        nb_classes,
        Chans,
        Samples,
        kernLength=64,
        dropRate=0.5,
        F1=8,
        D=2,
        F2=16,
        norm_rate=0.25,
        dropType="Dropout",
        ELUalpha=1,
        pool1=4,
        pool2=8,
        separable_kernel=16,
        depthwise_max_norm=1.0,
        return_logits=True,
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
        )
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
class DeepConvNetEncoder(nn.Module):
    """
    Pytorch Implementation of the DeepConvNet Encoder.
    See DeepConvNet for some references.

    The expected **input** is a **3D tensor** with size (Batch x Channels x Samples).

    Parameters
    ----------
    Chans: int
        The number of EEG channels.
    kernlength: int, optional
        The length of the temporal convolutional layer.

        Default = 64
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
        A max norm constraint to apply to each filter of the convolutional layer. See
        ``ConstrainedConv2d`` for more info.

        Default = 2
    batch_momentum: float, optional
        The batch normalization momentum.

        Default = 0.9
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
        max_norm=2.0,
        batch_momentum=0.9,
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


class DeepConvNet(nn.Module):
    """
    Pytorch Implementation of the DeepConvNet neural network.
    Official paper can be found here [deepconv]_ .
    A Keras implementation can be found here [deepconvgit]_ .

    The expected **input** is a **3D tensor** with size (Batch x Channels x Samples).

    Parameters
    ----------
    nb_classes: int
        The number of classes. If less than 2, a binary classification problem is considered
        (output dimensions will be [batch, 1] in this case).
    Chans: int
        The number of EEG channels.
    Samples: int
        The sample length. It will be used to calculate the embedding size
        (for head initialization).
    kernlength: int, optional
        The length of the temporal convolutional layer.

        Default = 64
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
        A max norm constraint to apply to each filter of the convolutional layer. See
        ``ConstrainedConv2d`` for more info.

        Default = 2
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

        Default = 1
    return_logits: bool, optional
        Whether to return the output as logit or probability. It is suggested to not use False as
        the pytorch crossentropy applies the softmax internally.

        Default = True

    References
    ----------
    .. [deepconv] Schirrmeister, Robin Tibor, et al. "Deep learning with convolutional neural
      networks for EEG decoding and visualization." Human brain mapping 38.11 (2017): 5391-5420.
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
        nb_classes,
        Chans,
        Samples,
        kernLength=10,
        F=25,
        Pool=3,
        stride=3,
        max_norm=2.0,
        batch_momentum=0.9,
        ELUalpha=1,
        dropRate=0.5,
        max_dense_norm=1.0,
        return_logits=True,
    ):
        super(DeepConvNet, self).__init__()

        self.nb_classes = nb_classes
        self.return_logits = return_logits
        self.encoder = DeepConvNetEncoder(
            Chans, kernLength, F, Pool, stride, max_norm, batch_momentum, ELUalpha, dropRate
        )
        k = kernLength
        Dense_input = [Samples] * 8
        for i in range(4):
            Dense_input[i * 2] = Dense_input[i * 2 - 1] - k + 1
            Dense_input[i * 2 + 1] = (Dense_input[i * 2] - Pool) // stride + 1
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


class EEGInceptionEncoder(nn.Module):
    """
    Pytorch Implementation of the EEGInception Encoder.
    See EEGInception for some references.

    The expected **input** is a **3D tensor** with size (Batch x Channels x Samples).

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


class EEGInception(nn.Module):
    """
    Pytorch Implementation of the EEGInception model.
    Original paper can be found here [eeginc]_ .
    A Keras implementation can be found here [eegincgit]_ .

    The expected **input** is a **3D tensor** with size (Batch x Channels x Samples).

    Parameters
    ----------
    nb_classes: int
        The number of classes. If less than 2, a binary classification problem is considered
        (output dimensions will be [batch, 1] in this case).
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

    References
    ----------
    .. [eeginc] Zhang, Ce, Young-Keun Kim, and Azim Eskandarian. "EEG-inception: an accurate and robust
      end-to-end neural network for EEG-based motor imagery classification."
      Journal of Neural Engineering 18.4 (2021): 046014.
      https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9311146
    .. [eegincgit] https://github.com/esantamariavazquez/EEG-Inception/blob/main/EEGInception/EEGInception.py

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
        nb_classes,
        Chans,
        Samples,
        F1=8,
        D=2,
        kernel_size=64,
        pool=4,
        dropRate=0.5,
        ELUalpha=1.0,
        bias=True,
        batch_momentum=0.1,
        max_depth_norm=1.0,
        return_logits=True,
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
        )
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
class TinySleepNetEncoder(nn.Module):
    """
    Pytorch Implementation of the TinySleepNet Encoder.
    TinySleepNet is a minimal but better performing architecture derived from
    DeepSleepNet (proposed by the same authors).
    See TinySleepNet for some references.

    The expected **input** is a **3D tensor** with size (Batch x Channels x Samples).

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


class TinySleepNet(nn.Module):
    """
    Pytorch Implementation of the TinySleepNet.
    TinySleepNet is a minimal but better performing architecture derived from
    DeepSleepNet (proposed by the same authors).
    Paper can be found here [tiny]_ .
    Github repo can be found here [tinygit]_ .

    The expected **input** is a **3D tensor** with size (Batch x Channels x Samples).

    Parameters
    ----------
    nb_classes: int
        The number of classes. If less than 2, a binary classification problem is considered
        (output dimensions will be [batch, 1] in this case).
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
        A value indicating the max norm constraint to apply on the final dense layer.
        If None no constraint will be included.

        Default = 1.
    hidden_lstm: int, optional
        Hidden size of the lstm block.

        Default = 128
    return_logits: bool, optional
        Whether to return the output as logit or probability. It is suggested
        to not use False as the pytorch crossentropy applies the softmax internally.

        Default = True

    References
    ----------
    .. [tiny] Supratak, Akara, and Yike Guo. "TinySleepNet: An efficient deep learning model
      for sleep stage scoring based on raw single-channel EEG." 2020 42nd Annual International
      Conference of the IEEE Engineering in Medicine & Biology Society (EMBC). IEEE, 2020.
      https://ieeexplore.ieee.org/abstract/document/9176741?casa_token=wl2VSsbgvq8AAAAA:EAzcLSaXXMzu7ghNoZRaRvEsEEAVH2sqQAi4OdMXVfxDPg296haJXJKIkq_4MVMwr-0rXIgU
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
        nb_classes,
        Chans,
        Fs,
        F=128,
        kernlength=8,
        pool=8,
        dropRate=0.5,
        batch_momentum=0.1,
        max_dense_norm=2.0,
        hidden_lstm=128,
        return_logits=True,
    ):
        super(TinySleepNet, self).__init__()

        self.nb_classes = nb_classes
        self.return_logits = return_logits
        self.encoder = TinySleepNetEncoder(
            Chans, Fs, F, kernlength, pool, dropRate, batch_momentum, hidden_lstm
        )

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
class StagerNetEncoder(nn.Module):
    """
    Pytorch implementation of the StagerNet Encoder.
    See TinySleepNet for some references.

    The expected **input** is a **3D tensor** with size (Batch x Channels x Samples).

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


class StagerNet(nn.Module):
    """
    Pytorch implementation of the StagerNet.
    Original paper can be found here [stager]_ .

    The expected **input** is a **3D tensor** with size (Batch x Channels x Samples).

    Parameters
    ----------
    nb_classes: int
        The number of classes. If less than 2, a binary classification problem is considered
        (output dimensions will be [batch, 1] in this case).
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
        nb_classes,
        Chans,
        Samples,
        dropRate=0.5,
        kernLength=64,
        F=8,
        Pool=16,
        return_logits=True,
    ):

        super(StagerNet, self).__init__()
        self.nb_classes = nb_classes
        self.return_logits = return_logits
        self.encoder = StagerNetEncoder(Chans, kernLength=kernLength, F=F, Pool=Pool)

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
class ShallowNetEncoder(nn.Module):
    """
    Pytorch implementation of the ShallowNet Encoder.
    See ShallowNet for some references.

    The expected **input** is a **3D tensor** with size (Batch x Channels x Samples).

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
    However, in the original paper authors preprocess EEG data by selecting a subset of
    only 21 channels. Since the net is very minimalistic, please follow the authors' notes.

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

    def __init__(self, Chans, F=8, K1=25, Pool=75, p=0.2):

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


class ShallowNet(nn.Module):
    """
    Pytorch implementation of the ShallowNet.
    Original paper can be found here [shall]_ .

    The expected **input** is a **3D tensor** with size (Batch x Channels x Samples).

    Parameters
    ----------
    nb_classes: int
        The number of classes. If less than 2, a binary classification problem is considered
        (output dimensions will be [batch, 1] in this case).
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

    Note
    ----
    In this implementation, the number of channels is an argument.
    However, in the original paper authors preprocess EEG data by selecting a subset of
    only 21 channels. Since the net is very minimalist, please follow the authors' notes.

    References
    ----------
    .. [shall] Schirrmeister et al., Deep Learning with convolutional neural networks
      for decoding and visualization of EEG pathology, arXiv:1708.08012

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

    def __init__(self, nb_classes, Chans, Samples, F=40, K1=25, Pool=75, p=0.2, return_logits=True):

        super(ShallowNet, self).__init__()

        self.nb_classes = nb_classes
        self.return_logits = return_logits
        self.encoder = ShallowNetEncoder(Chans, F=F, K1=K1, Pool=Pool, p=p)
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
class BasicBlock1(nn.Module):
    """
    Basic Resnet block
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
    Pytorch implementation of the Resnet Encoder with temporal convolutional layers.
    See ResNet for the reference paper which inspired this implementation.

    The expected **input** is a **3D tensor** with size (Batch x Channels x Samples).

    Parameters
    ----------
    Chans: int
        The number of EEG channels.
    block: nn.Module, optional
        An nn.Module defining the resnet block.
    Layers: list of 4 int, optional
        A list of integers indicating the number of times the resnet block is repeated .

        Default = [2,2,2,2]
    inplane: int, optional
        The number of output filters.
    kernLength: int, optional
        The length of the temporal convolutional layer.

        Default = 25
    addConnection: bool, optional
        Whether to add a connection from the start of the resnet part to the network head.
        If set to True the output of the following conv2d will be concatenate to the postblock
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
        A custom nn.Module to pass after the sequence of resnet blocks and before the
        network head. If none is left, the following sequence is used:

            1. nn.conv2d(1, self.inplane, kernel_size=(1, kernLength), bias=False)
            2. nn.BatchNorm2d()
            3. nn.ReLU()

            Default = None

    Note
    ----
    The compatibility between each custom nn.Module given as argument has not beend checked.
    Design the network carefully.

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


class ResNet1D(nn.Module):
    """
    Pytorch implementation of the Resnet Encoder with temporal convolutional layers.
    A reference paper [res1]_ .

    The expected **input** is a **3D tensor** with size (Batch x Channels x Samples).

    Parameters
    ----------
    nb_classes: int
        The number of classes. If less than 2, a binary classification problem is considered
        (output dimensions will be [batch, 1] in this case).
    Chans: int
        The number of EEG channels.
    Samples: int
        The sample length. It will be used to calculate the embedding size
        (for head initialization).
    block: nn.Module, optional
        An nn.Module defining the resnet block.

        Default: selfeeg.models.BasicBlock1
    Layers: list of 4 int, optional
        A list of integers indicating the number of times the resnet block is repeated.

        Default = [2,2,2,2]
    inplane: int, optional
        The number of output filters.

        Default = 16
    kernLength: int, optional
        The length of the temporal convolutional layer.

        Default = 7
    addConnection: bool, optional
        Whether to add a connection from the start of the resnet part to the network head.
        If set to True the output of the following conv2d will be concatenate to the postblock
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
        A custom nn.Module to pass after the sequence of resnet blocks and before the
        network head. If none is left, the following sequence is used:

            1. nn.conv2d(1, self.inplane, kernel_size=(1, kernLength), bias=False)
            2. nn.BatchNorm2d()
            3. nn.ReLU()

            Default = None
    classifier: nn.Module, optional
        A custom nn.Module defining the network head. If none is left,
        a single dense layer is used.

        Default = None
    return_logits: bool, optional
        Whether to return the output as logit or probability.  It is suggested
        to not use False as the pytorch crossentropy applies the softmax internally.

        Default = True

    Note
    ----
    The compatibility between each custom nn.Module given as argument has not been checked.
    Design the network carefully.

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
        nb_classes,
        Chans,
        Samples,
        block: nn.Module = BasicBlock1,
        Layers: "list of 4 int" = [0, 0, 0, 0],
        inplane: int = 16,
        kernLength: int = 7,
        addConnection: bool = False,
        preBlock: nn.Module = None,
        postBlock: nn.Module = None,
        classifier: nn.Module = None,
        return_logits=True,
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
        )
        # Classifier
        if classifier is None:
            if addConnection:
                out1 = int((Samples + 2 * (int(kernLength // 2)) - kernLength) // 2) + 1
                self.Dense = nn.Linear(
                    (Chans * inplane + int((out1 - kernLength) / int(kernLength // 2) + 1) * 2),
                    1 if nb_classes <= 2 else nb_classes,
                )
            else:
                self.Dense = nn.Linear(Chans * inplane, 1 if nb_classes <= 2 else nb_classes)
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
    the first is the Sample dimension and the last 2 dimensions are the channel dim rearrange
    in a 2d grid. Check the original paper for a better understanding of the input.

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


class STNet(nn.Module):
    """
    Pytorch implementation of the STNet Encoder.
    Paper can be found here [stnet]_ .
    Another implementation [stnetgit]_ .

    The expected **input** is a **4D tensor** with size
    (Batch x Samples x Grid_width x Grid_width), i.e. the classical 2d matrix
    with rows as channels and columns as samples is rearranged in a 3d tensor where
    the first is the Sample dimension and the last 2 dimensions are the channel dim rearranged
    in a 2d grid. Check the original paper for a better understanding of the input.

    Parameters
    ----------
    nb_classes: int
        The number of classes. If less than 2, a binary classification problem is considered.
        (output dimensions will be [batch, 1] in this case)
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
        nb_classes,
        Samples,
        grid_size=9,
        F=256,
        kernlength=5,
        dropRate=0.5,
        bias=True,
        dense_size=1024,
        return_logits=True,
    ):
        super(STNet, self).__init__()

        self.nb_classes = nb_classes
        self.return_logits = return_logits
        self.encoder = STNetEncoder(Samples, F, kernlength, dropRate, bias)
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
        # new tensor will be 5D with ( batch x filter x hemisphere x channel x samples )
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

    The expected **input** is a **3D tensor** with size (Batch x Channels x Samples).
    However Channel order is expected to be symmetrical along lateral channels to perform
    the reshaping operation correctly. For instance, if the first channel index refers to
    the FP1 channel, then the last must refer to the other hemisphere counterpart, i.e. FP2.
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
        The portion of EEG (in milliseconds) the short, medium and long temporal convolutional
        layers must cover. Kernel size will be automatically calculated based on the given
        sampling rate.

        Default = (500,250,125)
    lateral_chans: int, optional
        The amount of lateral channels. It will be used to reshape the 3D tensor in a
        5D tensor with size ( batch x filters x hemisphere x channel x samples ). See the
        original paper for more info.

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


class EEGSym(nn.Module):
    """
    Pytorch implementation of the EEGSym Encoder.
    EEGSym paper can be found here [eegsym]_ .
    Keras implementation can be found here [eegsymgit]_ .

    The expected **input** is a **3D tensor** with size (Batch x Channels x Samples).
    However Channel order is expected to be symmetrical along lateral channels to perform
    the reshaping operation correctly. For instance, if the first channel index refers to
    the FP1 channel, then the last must refer to the other hemisphere counterpart, i.e. FP2.

    Parameters
    ----------
    nb_classes: int
        The number of classes. If less than 2, a binary classification problem is considered
        (output dimensions will be [batch, 1] in this case).
    Chans: int
        The number of EEG channels.
    Samples: int
        The number of EEG Samples.
    Fs: float
        The sampling frequency.
    scales_time: tuple of 3 float, optional
        The portion of EEG (in milliseconds) the short, medium and long temporal convolutional
        layers must cover. kernel size will be automatically calculated based on the sampling
        rate given.

        Default = (500,250,125)
    lateral_chans: int, optional
        The amount of lateral channels. It will be used to reshape the 3D tensor in a
        5D tensor with size ( batch x filters x hemisphere x channel x samples ). See the
        original paper for more info.

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
        nb_classes,
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
        return_logits=True,
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
        )
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
