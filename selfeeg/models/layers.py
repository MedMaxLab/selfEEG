import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ConstrainedConv1d",
    "ConstrainedConv2d",
    "ConstrainedDense",
    "DepthwiseConv2d",
    "SeparableConv2d",
]


class ConstrainedDense(nn.Linear):
    """
    Pytorch implementation of the Dense layer with the possibility of adding a
    MaxNorm, MinMaxNorm, or a UnitNorm constraint. Most of the parameters are
    the same as described in torch.nn.Linear help.

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
        The maximum norm each hidden unit can have.
        If None no constraint will be added.

        Default = 2.0
    min_norm: float, optional
        The minimum norm each hidden unit can have. Must be a float
        lower than max_norm. If given, MinMaxNorm will be applied in the case
        max_norm is also given. Otherwise, it will be ignored.

        Default = None
    axis_norm: Union[int, list, tuple], optional
        The axis along weights are constrained. It behaves like Keras. So, considering
        that a Conv2D layer has shape (output_depth, input_depth), set axis
        to 1 will constrain the weights of each filter tensor of size
        (input_depth,).

        Default = 1
    minmax_rate: float, optional
        A constraint for MinMaxNorm setting how weights will be rescaled at each step.
        It behaves like Keras `rate` argument of MinMaxNorm contraint. So, using
        minmax_rate = 1 will set a strict enforcement of the constraint,
        while rate<1.0 will slowly rescale layer's hidden units at each step.

        Default = 1.0

    Note
    ----
    To Apply a MaxNorm constraint, set only max_norm. To apply a MinMaxNorm
    constraint, set both min_norm and max_norm. To apply a UnitNorm constraint,
    set both min_norm and max_norm to 1.0.

    Example
    -------
    >>> from selfeeg.models import ConstrainedDense
    >>> import torch
    >>> x = torch.randn(4,64)
    >>> mdl = ConstrainedDense(64,32)
    >>> out = mdl(x)
    >>> norms = torch.sqrt(torch.sum(torch.square(mdl.weight), axis=1))
    >>> print(out.shape) # shoud return torch.Size([4, 32])
    >>> print(torch.isnan(out).sum()) # shoud return 0
    >>> print(torch.sum(norms>(1.4+1e-3)).item() == 0) # should return True

    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
        max_norm=2.0,
        min_norm=None,
        axis_norm=1,
        minmax_rate=1.0,
    ):
        super(ConstrainedDense, self).__init__(in_features, out_features, bias, device, dtype)

        # Check that max_norm is valid
        if max_norm is not None:
            if max_norm <= 0:
                raise ValueError("max_norm can't be lower or equal than 0")
            else:
                self.max_norm = max_norm
        else:
            self.max_norm = max_norm

        # Check that min_norm is valid
        if min_norm is not None:
            if min_norm <= 0:
                raise ValueError("min_norm can't be lower or equal than 0")
            else:
                self.min_norm = min_norm
        else:
            self.min_norm = min_norm

        # If both are given, check that max_norm is bigger than min_norm
        if (self.min_norm is not None) and (self.max_norm is not None):
            if self.min_norm > self.max_norm:
                raise ValueError("max_norm can't be lower than min_norm")

        # Check that minmax_rate is bigger than 0
        if minmax_rate <= 0.0 or minmax_rate > 1.0:
            raise ValueError("minmax_rate must be in (0,1]")
        self.minmax_rate = minmax_rate

        # Check that axis is a valid enter
        if type(axis_norm) not in [tuple, list, int]:
            raise TypeError("axis must be a tuple, list, or int")
        else:
            if type(axis_norm) == int:
                if axis_norm < 0 or axis_norm > 1:
                    raise ValueError("Linear has 2 axis. Values must be in 0 or 1")
            else:
                for i in axis_norm:
                    if i < 0 or i > 1:
                        raise ValueError("Axis values must be in 0 or 1")
        self.axis_norm = axis_norm

        # set the constraint case:
        # 0 --> no contraint
        # 1 --> MaxNorm
        # 2 --> MinMaxNorm
        # 3 --> UnitNorm
        # The division is for computational purpose.
        # MinMaxNorm takes almost twice to execute than other operations.
        if self.max_norm is not None:
            if self.min_norm is not None:
                if self.min_norm == 1 and self.max_norm == 1:
                    self.constraint_type = 3
                else:
                    self.constraint_type = 2
            else:
                self.constraint_type = 1
        else:
            self.constraint_type = 0

    def scale_norm(self, eps=1e-9):
        """
        ``scale_norm`` applies the desired constraint on the Layer.
        It is highly based on the Keras implementation, but here
        MaxNorm, MinMaxNorm and UnitNorm are all implemented inside
        this function.

        """
        if self.constraint_type == 1:
            norms = torch.sqrt(
                torch.sum(torch.square(self.weight), axis=self.axis_norm, keepdims=True)
            )
            desired = torch.clamp(norms, 0, self.max_norm)
            self.weight = torch.nn.Parameter(self.weight * (desired / (eps + norms)))

        elif self.constraint_type == 2:
            norms = torch.sqrt(
                torch.sum(torch.square(self.weight), axis=self.axis_norm, keepdims=True)
            )
            desired = (
                self.minmax_rate * torch.clamp(norms, self.min_norm, self.max_norm)
                + (1 - self.minmax_rate) * norms
            )
            self.weight = torch.nn.Parameter(self.weight * (desired / (eps + norms)))

        elif self.constraint_type == 3:
            norms = torch.sqrt(
                torch.sum(torch.square(self.weight), axis=self.axis_norm, keepdims=True)
            )
            self.weight = torch.nn.Parameter(self.weight / (eps + norms))

    def forward(self, input):
        """
        :meta private:
        """
        if self.constraint_type != 0:
            self.scale_norm()
        return F.linear(input, self.weight, self.bias)


class ConstrainedConv1d(nn.Conv1d):
    """
    Pytorch implementation of the 1D Convolutional layer with the possibility
    to add a MaxNorm, MinMaxNorm, or UnitNorm constraint along the given axis.
    Most of the parameters are the same as described in pytorch Conv2D help.

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
        Padding added to all four sides of the input. This class also accepts the
        string 'causal', which triggers causal convolution like in Wavenet.

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
        The maximum norm each hidden unit can have.
        If None no constraint will be added.

        Default = 2.0
    min_norm: float, optional
        The minimum norm each hidden unit can have. Must be a float
        lower than max_norm. If given, MinMaxNorm will be applied in the case
        max_norm is also given. Otherwise, it will be ignored.

        Default = None
    axis_norm: Union[int, list, tuple], optional
        The axis along weights are constrained. It behaves like Keras. So, considering
        that a Conv2D layer has shape (output_depth, input_depth, length), set axis
        to [1, 2] will constrain the weights of each filter tensor of size
        (input_depth, length).

        Default = [1,2]
    minmax_rate: float, optional
        A constraint for MinMaxNorm setting how weights will be rescaled at each step.
        It behaves like Keras `rate` argument of MinMaxNorm contraint. So, using
        minmax_rate = 1 will set a strict enforcement of the constraint,
        while rate<1.0 will slowly rescale layer's hidden units at each step.

        Default = 1.0

    Note
    ----
    To Apply a MaxNorm constraint, set only max_norm. To apply a MinMaxNorm
    constraint, set both min_norm and max_norm. To apply a UnitNorm constraint,
    set both min_norm and max_norm to 1.0.

    Note
    ----
    When setting ``padding`` to ``"causal"``, padding will be internally changed
    to an integer equal to ``(kernel_size - 1) * dilation``. Then, during forward,
    the extra features are removed. This is preferable over F.pad, which can
    lead to memory allocation or even non-deterministic operations during the
    backboard pass. Additional information can be found at the following link:
    https://github.com/pytorch/pytorch/issues/1333

    Example
    -------
    >>> from import selfeeg.models import ConstrainedConv1d
    >>> import torch
    >>> x = torch.randn(4, 8, 64)
    >>> mdl = ConstrainedConv1d(8, 16, 15, max_norm = 1.4, min_norm = 0.3)
    >>> mdl.weight = torch.nn.Parameter(mdl.weight*10)
    >>> out = mdl(x)
    >>> norms = torch.sqrt(torch.sum(torch.square(mdl.weight), axis=[1,2]))
    >>> print(out.shape) # shoud return torch.Size([4, 16, 64])
    >>> print(torch.isnan(out).sum()) # shoud return 0
    >>> print(torch.sum(norms>(1.4+1e-3)).item() == 0) # should return True

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
        max_norm=2.0,
        min_norm=None,
        axis_norm=[1, 2],
        minmax_rate=1.0,
    ):

        # Check causal Padding
        self.pad = padding
        self.causal_pad = False
        if isinstance(padding, str):
            if padding.casefold() == "causal":
                self.causal_pad = True
                self.pad = (kernel_size - 1) * dilation

        super(ConstrainedConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            self.pad,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

        # Check that max_norm is valid
        if max_norm is not None:
            if max_norm <= 0:
                raise ValueError("max_norm can't be lower or equal than 0")
            else:
                self.max_norm = max_norm
        else:
            self.max_norm = max_norm

        # Check that min_norm is valid
        if min_norm is not None:
            if min_norm <= 0:
                raise ValueError("min_norm can't be lower or equal than 0")
            else:
                self.min_norm = min_norm
        else:
            self.min_norm = min_norm

        # If both are given, check that max_norm is bigger than min_norm
        if (self.min_norm is not None) and (self.max_norm is not None):
            if self.min_norm > self.max_norm:
                raise ValueError("max_norm can't be lower than min_norm")

        # Check that minmax_rate is bigger than 0
        if minmax_rate <= 0.0 or minmax_rate > 1.0:
            raise ValueError("minmax_rate must be in (0,1]")
        self.minmax_rate = minmax_rate

        # Check that axis is a valid enter
        if type(axis_norm) not in [tuple, list, int]:
            raise TypeError("axis must be a tuple, list, or int")
        else:
            if type(axis_norm) == int:
                if axis_norm < 0 or axis_norm > 2:
                    raise ValueError("Conv2D has 4 axis. Values must be in [0, 2]")
            else:
                for i in axis_norm:
                    if i < 0 or i > 2:
                        raise ValueError("Axis values must be in [0, 2]")
        self.axis_norm = axis_norm

        # set the constraint case:
        # 0 --> no contraint
        # 1 --> MaxNorm
        # 2 --> MinMaxNorm
        # 3 --> UnitNorm
        # The division is for computational purpose.
        # MinMaxNorm takes almost twice to execute than other operations.
        if self.max_norm is not None:
            if self.min_norm is not None:
                if self.min_norm == 1 and self.max_norm == 1:
                    self.constraint_type = 3
                else:
                    self.constraint_type = 2
            else:
                self.constraint_type = 1
        else:
            self.constraint_type = 0

    def scale_norm(self, eps=1e-9):
        """
        ``scale_norm`` applies the desired constraint on the Layer.
        It is highly based on the Keras implementation, but here
        MaxNorm, MinMaxNorm and UnitNorm are all implemented inside
        this function.

        """
        if self.constraint_type == 1:
            norms = torch.sqrt(
                torch.sum(torch.square(self.weight), axis=self.axis_norm, keepdims=True)
            )
            desired = torch.clamp(norms, 0, self.max_norm)
            self.weight = torch.nn.Parameter(self.weight * (desired / (eps + norms)))

        elif self.constraint_type == 2:
            norms = torch.sqrt(
                torch.sum(torch.square(self.weight), axis=self.axis_norm, keepdims=True)
            )
            desired = (
                self.minmax_rate * torch.clamp(norms, self.min_norm, self.max_norm)
                + (1 - self.minmax_rate) * norms
            )
            self.weight = torch.nn.Parameter(self.weight * (desired / (eps + norms)))

        elif self.constraint_type == 3:
            norms = torch.sqrt(
                torch.sum(torch.square(self.weight), axis=self.axis_norm, keepdims=True)
            )
            self.weight = torch.nn.Parameter(self.weight / (eps + norms))

    def forward(self, input):
        """
        :meta private:
        """
        if self.constraint_type != 0:
            self.scale_norm()
        if self.causal_pad:
            return self._conv_forward(input, self.weight, self.bias)[:, :, : -self.pad]
        else:
            return self._conv_forward(input, self.weight, self.bias)


class ConstrainedConv2d(nn.Conv2d):
    """
    Pytorch implementation of the 2D Convolutional layer with the possibility
    to add a MaxNorm, MinMaxNorm, or UnitNorm constraint along the given axis.
    Most of the parameters are the same as described in pytorch Conv2D help.

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
        The maximum norm each hidden unit can have.
        If None no constraint will be added.

        Default = 2.0
    min_norm: float, optional
        The minimum norm each hidden unit can have. Must be a float
        lower than max_norm. If given, MinMaxNorm will be applied in the case
        max_norm is also given. Otherwise, it will be ignored.

        Default = None
    axis_norm: Union[int, list, tuple], optional
        The axis along weights are constrained. It behaves like Keras. So, considering
        that a Conv2D layer has shape (output_depth, input_depth, rows, cols), set axis
        to [1, 2, 3] will constrain the weights of each filter tensor of size
        (input_depth, rows, cols).

        Default = [1,2,3]
    minmax_rate: float, optional
        A constraint for MinMaxNorm setting how weights will be rescaled at each step.
        It behaves like Keras `rate` argument of MinMaxNorm contraint. So, using
        minmax_rate = 1 will set a strict enforcement of the constraint,
        while rate<1.0 will slowly rescale layer's hidden units at each step.

        Default = 1.0

    Note
    ----
    To Apply a MaxNorm constraint, set only max_norm. To apply a MinMaxNorm
    constraint, set both min_norm and max_norm. To apply a UnitNorm constraint,
    set both min_norm and max_norm to 1.0.

    Example
    -------
    >>> from import selfeeg.models import ConstrainedConv2d
    >>> import torch
    >>> x = torch.randn(4, 1, 8, 64)
    >>> mdl = ConstrainedConv2d(1, 4, (1, 15), max_norm = 0.5)
    >>> mdl.weight = torch.nn.Parameter(mdl.weight*10)
    >>> out = mdl(x)
    >>> norms = torch.sqrt(torch.sum(torch.square(mdl.weight), axis=[1,2,3]))
    >>> print(out.shape) # shoud return torch.Size([4, 2, 8, 64])
    >>> print(torch.isnan(out).sum()) # shoud return 0
    >>> print(torch.sum(norms>(0.5+1e-3)).item() == 0) # should return True

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
        max_norm=2.0,
        min_norm=None,
        axis_norm=[1, 2, 3],
        minmax_rate=1.0,
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

        # Check that max_norm is valid
        if max_norm is not None:
            if max_norm <= 0:
                raise ValueError("max_norm can't be lower or equal than 0")
            else:
                self.max_norm = max_norm
        else:
            self.max_norm = max_norm

        # Check that min_norm is valid
        if min_norm is not None:
            if min_norm <= 0:
                raise ValueError("min_norm can't be lower or equal than 0")
            else:
                self.min_norm = min_norm
        else:
            self.min_norm = min_norm

        # If both are given, check that max_norm is bigger than min_norm
        if (self.min_norm is not None) and (self.max_norm is not None):
            if self.min_norm > self.max_norm:
                raise ValueError("max_norm can't be lower than min_norm")

        # Check that minmax_rate is bigger than 0
        if minmax_rate <= 0.0 or minmax_rate > 1.0:
            raise ValueError("minmax_rate must be in (0,1]")
        self.minmax_rate = minmax_rate

        # Check that axis is a valid enter
        if type(axis_norm) not in [tuple, list, int]:
            raise TypeError("axis must be a tuple, list, or int")
        else:
            if type(axis_norm) == int:
                if axis_norm < 0 or axis_norm > 3:
                    raise ValueError("Conv2D has 4 axis. Values must be in [0, 3]")
            else:
                for i in axis_norm:
                    if i < 0 or i > 3:
                        raise ValueError("Axis values must be in [0, 3]")
        self.axis_norm = axis_norm

        # set the constraint case:
        # 0 --> no contraint
        # 1 --> MaxNorm
        # 2 --> MinMaxNorm
        # 3 --> UnitNorm
        # The division is for computational purpose.
        # MinMaxNorm takes almost twice to execute than other operations.
        if self.max_norm is not None:
            if self.min_norm is not None:
                if self.min_norm == 1 and self.max_norm == 1:
                    self.constraint_type = 3
                else:
                    self.constraint_type = 2
            else:
                self.constraint_type = 1
        else:
            self.constraint_type = 0

    def scale_norm(self, eps=1e-9):
        """
        ``scale_norm`` applies the desired constraint on the Layer.
        It is highly based on the Keras implementation, but here
        MaxNorm, MinMaxNorm and UnitNorm are all implemented inside
        this function.

        """
        if self.constraint_type == 1:
            norms = torch.sqrt(
                torch.sum(torch.square(self.weight), axis=self.axis_norm, keepdims=True)
            )
            desired = torch.clamp(norms, 0, self.max_norm)
            self.weight = torch.nn.Parameter(self.weight * (desired / (eps + norms)))

        elif self.constraint_type == 2:
            norms = torch.sqrt(
                torch.sum(torch.square(self.weight), axis=self.axis_norm, keepdims=True)
            )
            desired = (
                self.minmax_rate * torch.clamp(norms, self.min_norm, self.max_norm)
                + (1 - self.minmax_rate) * norms
            )
            self.weight = torch.nn.Parameter(self.weight * (desired / (eps + norms)))

        elif self.constraint_type == 3:
            norms = torch.sqrt(
                torch.sum(torch.square(self.weight), axis=self.axis_norm, keepdims=True)
            )
            self.weight = torch.nn.Parameter(self.weight / (eps + norms))

    def forward(self, input):
        """
        :meta private:
        """
        if self.constraint_type != 0:
            self.scale_norm()
        return self._conv_forward(input, self.weight, self.bias)


class DepthwiseConv2d(nn.Conv2d):
    """
    Pytorch implementation of the Depthwise Convolutional layer with
    the possibility to add a MaxNorm, MinMaxNorm, or UnitNorm constraint along
    the given axis. Most of the parameters are the same as described in pytorch
    Conv2D help.

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
        The maximum norm each hidden unit can have.
        If None no constraint will be added.

        Default = 2.0
    min_norm: float, optional
        The minimum norm each hidden unit can have. Must be a float
        lower than max_norm. If given, MinMaxNorm will be applied in the case
        max_norm is also given. Otherwise, it will be ignored.

        Default = None
    axis_norm: Union[int, list, tuple], optional
        The axis along weights are constrained. It behaves like Keras. So, considering
        that a Conv2D layer has shape (output_depth, input_depth, rows, cols), set axis
        to [1, 2, 3] will constrain the weights of each filter tensor of size
        (input_depth, rows, cols).

        Default = [1,2,3]
    minmax_rate: float, optional
        A constraint for MinMaxNorm setting how weights will be rescaled at each step.
        It behaves like Keras `rate` argument of MinMaxNorm contraint. So, using
        minmax_rate = 1 will set a strict enforcement of the constraint,
        while rate<1.0 will slowly rescale layer's hidden units at each step.

        Default = 1.0

    Note
    ----
    To Apply a MaxNorm constraint, set only max_norm. To apply a MinMaxNorm
    constraint, set both min_norm and max_norm. To apply a UnitNorm constraint,
    set both min_norm and max_norm to 1.0.

    Example
    -------
    >>> from import selfeeg.models import DepthwiseConv2d
    >>> import torch
    >>> x = torch.randn(4,1,8,64)
    >>> mdl = DepthwiseConv2d(1, 2, (1, 15), max_norm = 0.5)
    >>> mdl.weight = torch.nn.Parameter(mdl.weight*10)
    >>> out = mdl(x)
    >>> norms = torch.sqrt(torch.sum(torch.square(mdl.weight), axis=[1,2,3]))
    >>> print(out.shape) # shoud return torch.Size([4, 2, 8, 64])
    >>> print(torch.isnan(out).sum()) # shoud return 0
    >>> print(torch.sum(norms>(0.5+1e-3)).item() == 0) # should return True

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
        max_norm=2.0,
        min_norm=None,
        axis_norm=[1, 2, 3],
        minmax_rate=1.0,
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

        # Check that max_norm is valid
        if max_norm is not None:
            if max_norm <= 0:
                raise ValueError("max_norm can't be lower or equal than 0")
            else:
                self.max_norm = max_norm
        else:
            self.max_norm = max_norm

        # Check that min_norm is valid
        if min_norm is not None:
            if min_norm <= 0:
                raise ValueError("min_norm can't be lower or equal than 0")
            else:
                self.min_norm = min_norm
        else:
            self.min_norm = min_norm

        # If both are given, check that max_norm is bigger than min_norm
        if (self.min_norm is not None) and (self.max_norm is not None):
            if self.min_norm > self.max_norm:
                raise ValueError("max_norm can't be lower than min_norm")

        # Check that minmax_rate is bigger than 0
        if minmax_rate <= 0.0 or minmax_rate > 1.0:
            raise ValueError("minmax_rate must be in (0,1]")
        self.minmax_rate = minmax_rate

        # Check that axis is a valid enter
        if type(axis_norm) not in [tuple, list, int]:
            raise TypeError("axis must be a tuple, list, or int")
        else:
            if type(axis_norm) == int:
                if axis_norm < 0 or axis_norm > 3:
                    raise ValueError("Conv2D has 4 axis. Values must be in [0, 3]")
            else:
                for i in axis_norm:
                    if i < 0 or i > 3:
                        raise ValueError("Axis values must be in [0, 3]")
        self.axis_norm = axis_norm

        # set the constraint case:
        # 0 --> no contraint
        # 1 --> MaxNorm
        # 2 --> MinMaxNorm
        # 3 --> UnitNorm
        # The division is for computational purpose.
        # MinMaxNorm takes almost twice to execute than other operations.
        if self.max_norm is not None:
            if self.min_norm is not None:
                if self.min_norm == 1 and self.max_norm == 1:
                    self.constraint_type = 3
                else:
                    self.constraint_type = 2
            else:
                self.constraint_type = 1
        else:
            self.constraint_type = 0

    def scale_norm(self, eps=1e-9):
        """
        ``scale_norm`` applies the desired constraint on the Layer.
        It is highly based on the Keras implementation, but here
        MaxNorm, MinMaxNorm and UnitNorm are all implemented inside
        this function.

        """
        if self.constraint_type == 1:
            norms = torch.sqrt(
                torch.sum(torch.square(self.weight), axis=self.axis_norm, keepdims=True)
            )
            desired = torch.clamp(norms, 0, self.max_norm)
            self.weight = torch.nn.Parameter(self.weight * (desired / (eps + norms)))

        elif self.constraint_type == 2:
            norms = torch.sqrt(
                torch.sum(torch.square(self.weight), axis=self.axis_norm, keepdims=True)
            )
            desired = (
                self.minmax_rate * torch.clamp(norms, self.min_norm, self.max_norm)
                + (1 - self.minmax_rate) * norms
            )
            self.weight = torch.nn.Parameter(self.weight * (desired / (eps + norms)))

        elif self.constraint_type == 3:
            norms = torch.sqrt(
                torch.sum(torch.square(self.weight), axis=self.axis_norm, keepdims=True)
            )
            self.weight = torch.nn.Parameter(self.weight / (eps + norms))

    def forward(self, input):
        """
        :meta private:
        """
        if self.constraint_type != 0:
            self.scale_norm()
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
        The maximum norm each hidden unit in the depthwise layer can have.
        If None no constraint will be added.

        Default = None
    depth_min_norm: float, optional
        The minimum norm each hidden unit in the depthwise layer can have.
        Must be a float lower than max_norm. If given, MinMaxNorm will be applied
        in the case max_norm is also given. Otherwise, it will be ignored.

        Default = None
    depth_minmax_rate: float, optional
        A constraint for depthwise's MinMaxNorm setting how weights will be rescaled
        at each step. It behaves like Keras `rate` argument of MinMaxNorm contraint.
        So, using minmax_rate = 1 will set a strict enforcement of the constraint,
        while rate<1.0 will slowly rescale layer's hidden units at each step.

        Default = 1.0
    axis_norm: Union[int, list, tuple], optional
        The axis along weights are constrained. It behaves like Keras. So, considering
        that a Conv2D layer has shape (output_depth, input_depth), set axis
        to 1 will constrain the weights of each filter tensor of size
        (input_depth,).

        Default = 1
    point_max_norm: float, optional
        Same as depth_max_norm, but applied to the pointwise Convolutional layer.

        Default = None
    point_min_norm: float, optional
        Same as depth_min_norm, but applied to the pointwise Convolutional layer.

        Default = None
    point_minmax_rate: float, optional
        Same as depth_minmax_rate, but applied to the pointwise Convolutional layer.

        Default = 1.0

    Example
    -------
    >>> from selfeeg.models import SeparableConv2d
    >>> import torch
    >>> x = torch.randn(4, 1, 8, 64)
    >>> mdl = SeparableConv2d(1,4, (1,15), depth_multiplier=4)
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
        depth_min_norm=None,
        depth_minmax_rate=1.0,
        point_max_norm=None,
        point_min_norm=None,
        point_minmax_rate=1.0,
        axis_norm=[1, 2, 3],
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
            max_norm=depth_max_norm,
            min_norm=depth_min_norm,
            axis_norm=axis_norm,
            minmax_rate=depth_minmax_rate,
        )
        self.pointwise = ConstrainedConv2d(
            in_channels * depth_multiplier,
            out_channels,
            kernel_size=1,
            bias=bias,
            max_norm=point_max_norm,
            min_norm=point_min_norm,
            axis_norm=axis_norm,
            minmax_rate=point_minmax_rate,
        )

    def forward(self, input):
        """
        :meta private:
        """
        out = self.depthwise(input)
        out = self.pointwise(out)
        return out
