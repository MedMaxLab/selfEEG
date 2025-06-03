from __future__ import annotations

import copy
import os
import pickle
import random
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from numpy.typing import ArrayLike

__all__ = [
    "check_models",
    "count_parameters",
    "create_dataset",
    "get_subarray_closest_sum",
    "RangeScaler",
    "scale_range_soft_clip",
    "torch_pchip",
    "torch_zscore",
    "ZscoreScaler",
]


def subarray_closest_sum(arr: ArrayLike, n: int, k: float) -> tuple(ArrayLike, float, float, float):
    """
    returns a subarray whose element sum is closest to k.

    This function is inspired from [link1]_

    It is important to note that this function returns a subarray and not a
    subset of the array. A subset is a collection of elements in the array taken
    from any index, a subarray here is a slice of the array (arr[start:end]).
    If you are looking for the exact subset with closest sum, which is more
    accurate but more computationally and memory demanding, use another function.

    Parameters
    ----------
    arr: ArrayLike
        The array to search.
    n: int
        The length of the array.
    k: float
        The target value.

    Returns
    -------
    best_arr: ArrayLike
        The subarray whose element sum is closest to k.
    best_start: float
        The starting index of the subarray.
    best_end: float
        The ending index of the subarray.
    min_diff: float
        Absolute difference between the target value and the sum of the subarray's values.

    References
    ----------
    .. [link1] https://www.geeksforgeeks.org/subarray-whose-sum-is-closest-to-k/

    """
    # Initialize start and end pointers, current sum, minimum difference
    # and best start and end pointers
    start = 0
    end = 0
    best_start = 0
    best_end = 0
    curr_sum = arr[0]

    # Initialize the minimum difference between the subarray sum and K
    min_diff = abs(curr_sum - k)

    # Traverse through the array
    while end < n - 1:

        # If the current sum is less than k, move the end pointer to the right
        if curr_sum < k:
            end += 1
            curr_sum += arr[end]
        # Otherwise, move the start pointer to the right
        else:
            curr_sum -= arr[start]
            start += 1

        # Update the minimum difference and store best subarray pointers
        if abs(curr_sum - k) < min_diff:
            min_diff = abs(curr_sum - k)
            best_start = start
            best_end = end
            # if minimum difference is zero, return the optimal subarray
            if min_diff == 0:
                return arr[best_start : best_end + 1], best_start, best_end, min_diff

    return arr[best_start : best_end + 1], best_start, best_end, min_diff


def get_subarray_closest_sum(
    arr: ArrayLike,
    target: float,
    tolerance: float = 0.01,
    perseverance: int = 1000,
    return_subarray: bool = True,
) -> tuple[list, Optional[list]]:
    """
    find the subarray whose values sum is closer to a target.

    The solution found is the first inside a specified tolerance (if possible)
    and return the index of the selected values in the original array.

    To find the subarray, get_subarray_closest_sum calls multiple times the
    ``subarray_closest_sum`` function until the subarray has the sum within
    [target*(1-tolerance), target*(1+tolerance)].
    At each try the array is shuffled in order to get a different solution.
    Keep in mind that the solution is not always the optimal, but rather the first
    which satisfies the requirements given.

    Parameters
    ----------
    arr: ArrayLike
        The array to search.
    target: float
        The target sum.
    tolerance: float, optional
        The tolerance to apply to the sum in percentage, in range [0,1].

        Default = 0.01
    perseverance: int, optional
        The maximum number of tries before stopping searching the subarray
        with closest sum.

        Default = 1000
    return_subarray: bool, optional
        whether to also return the subarray or not.

        Default = True

    Returns
    -------
    final_idx: list
        A list with the index of the identified subarray.
    best_sub_arr: list, optional
        The subarray.

    Example
    -------
    >>> import random
    >>> import selfeeg.utils
    >>> random.seed(1235)
    >>> arr = [i for i in range (1,100)]
    >>> final_idx, best_sub_arr = utils.get_subarray_closest_sum(
    ...     arr, 3251, perseverance=10000)
    >>> print( sum(best_sub_arr)) #should print 3251

    """

    if tolerance < 0 or tolerance > 1:
        raise ValueError("tolerance must be in [0,1]")
    else:
        upper_bound = target * tolerance
    if not isinstance(perseverance, int):
        perseverance = int(perseverance)

    arr_original = arr
    idx = range(len(arr))
    N = len(arr)
    subarr_diff = 0
    best_idx = []
    best_start = 0
    best_end = 0
    best_subarr_diff = float("inf")
    starti = 0
    endi = 0

    # c = np.array([arr,idx]).T
    for _ in range(perseverance):
        c = list(zip(arr, idx))
        random.shuffle(c)
        arr, idx = zip(*c)
        # np.random.shuffle(c)
        # _, starti, endi, subarr_diff = subarray_closest_sum(c[:,0], N, target)
        _, starti, endi, subarr_diff = subarray_closest_sum(arr, N, target)
        if subarr_diff < best_subarr_diff:
            best_subarr_diff = subarr_diff
            best_idx = idx
            best_start = starti
            best_end = endi
            if best_subarr_diff < upper_bound:
                break

    # get final list
    final_idx = list(best_idx[best_start : best_end + 1])
    final_idx.sort()

    if return_subarray:
        best_subarr = list(map(arr_original.__getitem__, final_idx))
        return final_idx, best_subarr
    else:
        return final_idx


def scale_range_soft_clip(
    x: ArrayLike, Range: float = 200, asintote: float = 1.2, scale: str = "mV", exact: bool = True
) -> ArrayLike:
    """
    soft version of the range scaler.

    The function will rescale the data in the following way:

        1. values in Range will be rescaled in the range [-1,1] linearly
        2. values outside the range will be either clipped or soft clipped with
           an exponential saturating curve with first derivative in -1 and 1
           preserved and horizontal asintote (the saturating point) given
           by the user.

    To provide faster computation, this function can also approximate its behaviour
    with a sigmoid function which scales the given input using the specified range
    and asintote. To check the difference in those functions see the geogebra file
    provided in the extra folder of the github repository.

    Parameters
    ----------
    x: ArrayLike
        The array or tensor to rescale. Rescaling can be perfomed along the
        last dimension. Tensors can also be placed in a GPU.
        Computation in this case is much faster
    Range: float, optional
        The range of values to rescale given in microVolt. It rescale linearly the
        values in [-range, range] to [-1, 1]. Must be a positive value. The list
        [-range, range] is created internally.

        Default = 200
    asintote: float, optional
        The horizontal asintote of the soft clipping part.
        Must be a value bigger than 1.

        Default = 1.2
    scale: str, optional
        The scale of the EEG Samples. It can be:

            - 'mV' for milliVolt
            - 'uV' for microVolt
            - 'nV' for nanoVolt

        Default = 'mV'
    exact: bool, optional
        Whether to approximate the composed function (linear + exponential function)
        with a sigmoid. It will make the rescaling much faster but will not preserve
        the linearity in the range [-1, 1].

    Returns
    -------
    x_scaled: ArrayLike
        The rescaled array.

    Example
    -------
    >>> import selfeeg.utils
    >>> import torch
    >>> x = torch.zeros(16,32,1024) + torch.sin(torch.linspace(0, 8*torch.pi,1024))*500
    >>> x_scaled = utils.scale_range_soft_clip(x, 200, 2.5, 'uV' )
    >>> print( x.max()<=2.5 and x.min()>=-2.5) # should return False
    >>> print( x_scaled.max()<=2.5 and x_scaled.min()>=-2.5) # should return True

    """

    if Range < 0:
        raise ValueError("Range argument cannot be lower than 0")
    if asintote is None:
        asintote = 1.0
    elif asintote < 1:
        raise ValueError("asintote must be a value bigger than 1")
    scale = scale.lower()

    Range = Range / 1000
    if scale not in ["mv", "uv", "nv"]:
        raise ValueError("scale must be any of 'mV', 'uV', 'nV'")
    else:
        if scale == "uv":
            x = x / 1.0e3
        elif scale == "nv":
            x = x / 1.0e6

    x_scaled = torch.clone(x) if isinstance(x, torch.Tensor) else np.copy(x)

    # CASE 1: HARD clipping
    if asintote == 1.0:
        mask1 = x > Range
        mask2 = x < -Range
        x_scaled = x / Range
        if isinstance(x, torch.Tensor):
            x_scaled = torch.clamp(x_scaled, min=-1, max=1)
        else:
            x_scaled = np.clip(x_scaled, -1, 1)
        return x_scaled

    # CASE 2: SOFT CLIPPING
    if exact:
        mask1 = x > (Range)
        mask2 = x < (-Range)
        x_scaled = x / Range
        if isinstance(x, torch.Tensor):
            x_scaled[mask2] = (asintote - 1) * torch.exp(
                (x[mask2] + Range) / (Range * (asintote - 1))
            ) - asintote
            x_scaled[mask1] = -(
                (asintote - 1) * torch.exp((-x[mask1] + Range) / (Range * (asintote - 1)))
                - asintote
            )
        else:
            x_scaled[mask2] = (asintote - 1) * np.exp(
                (x[mask2] + Range) / (Range * (asintote - 1))
            ) - asintote
            x_scaled[mask1] = -(
                (asintote - 1) * np.exp((-x[mask1] + Range) / (Range * (asintote - 1))) - asintote
            )
    else:
        # trating c as -coeff
        c = (np.log((2 * asintote) / (1 + asintote) - 1)) / Range
        if isinstance(x, torch.Tensor):
            x_scaled = ((2 * asintote) / (1 + torch.exp(c * x))) - asintote
        else:
            x_scaled = ((2 * asintote) / (1 + np.exp(c * x))) - asintote
    return x_scaled


class RangeScaler:
    """
    class adaptation of the ``scale_range_with_soft_clip`` function.

    Upon call, RangeScaler rescales the given EEG data in the following way:

        1. values in Range will be linearly rescaled in the range [-1,1].
        2. values outside the range will be either clipped or soft clipped with
           an exponential saturating curve with first derivative in -1 and 1
           preserved and horizontal asintote (the saturating point) given
           by the user.

    To provide faster computation, this function can also approximate its
    behaviour with a sigmoid function which scales the given input using the
    specified range and asintote. To check the difference in those functions
    see the geogebra file provided in the extra folder of the github repository.

    Parameters
    ----------
    x: ArrayLike
        The array or tensor to rescale. Rescaling can be perfomed along the last
        dimension. Tensors can also be placed in a GPU.
        Computation in this case is faster.
    Range: float, optional
        The range of values to rescale given in microVolt. It rescale linearly the
        values in [-range, range] to [-1, 1]. Must be a positive value.

        Default = 200
    asintote: float, optional
        The horizontal asintote of the soft clipping part.
        Must be a value bigger than 1.

        Default = 1.2
    scale: str, optional
        The scale of the EEG Samples. It can be:

            - 'mV' for milliVolt
            - 'uV' for microVolt
            - 'nV' for nanoVolt

        Default = 'mV'
    exact: bool, optional
        Whether to approximate the composed function (linear + exponential function)
        with a sigmoid. It will make the rescaling much faster but will not preserve
        the linearity in the range [-1, 1].

    Example
    -------
    >>> import selfeeg.utils
    >>> import torch
    >>> x = torch.zeros(16,32,1024) + torch.sin(torch.linspace(0, 8*torch.pi,1024))*500
    >>> x_scaled = utils.RangeScaler(200, 2.5, 'uV' )(x)
    >>> print( x.max()<=2.5 and x.min()>=-2.5) # should return False
    >>> print( x_scaled.max()<=2.5 and x_scaled.min()>=-2.5) # should return True

    """

    def __init__(
        self, Range: float = 200, asintote: float = 1.2, scale: str = "mV", exact: bool = True
    ):
        if Range < 0:
            raise ValueError("Range cannot be lower than 0")
        if asintote is None:
            asintote = 1.0
        elif asintote < 1:
            raise ValueError("asintote must be a value bigger than 1")
        scale = scale.lower()
        if scale not in ["mv", "uv", "nv"]:
            raise ValueError("scale must be any of 'mV', 'uV', 'nV'")
        self.Range = Range
        self.asintote = asintote
        self.scale = scale
        self.exact = exact

    def __call__(self, x):
        """
        :meta private:
        """
        return scale_range_soft_clip(x, self.Range, self.asintote, self.scale, self.exact)


def torch_zscore(
    x: torch.Tensor,
    axis: int = -2,
    correction: int = 1,
) -> torch.Tensor:
    """
    zscore operator for torch tensors.

    It is heavily based on scipy's zscore in order to provide
    identical results when using numpy arrays. The analogous
    command in scipy is:

        x_zscore = scipy.stats.zscore(x, axis=axis, ddof=correction)

    Parameters
    ----------
    x: torch.Tensor
        The tensor to standardize.
    axis: int
        The axis along which to operate. By default, it assumes that
        the EEG channel dimension is the second to last. If the
        tensor has only one dimension, default value is changed to 0.

        Default = -2
    correction: int
        difference between the sample size and sample degrees of freedom.
        It is applied during the calculation of the standard deviation.
        It is equivalent to the Scipy's zscore `ddof` argument.
        Default is Bessel's correction as used in Pytorch's std function.

        Default = 1

    Returns
    -------
    xz: torch.Tensor
        The tensor standardized along the given dimension.

    """
    dims = len(x.shape)
    if dims == 0:
        raise ValueError("Got a tensor with 0 length")
    elif dims == 1:
        axis = 0

    # get mean and standard deviation
    mn = x.mean(axis, keepdim=True)
    sd = x.std(axis, correction=correction, keepdim=True)

    # a solid solution implemented in scipy's zscore
    # to avoid 0 division or too large values
    x0 = x.min(axis=axis, keepdims=True)[0]
    iszero = torch.eq(x, x0).all(axis=axis, keepdims=True)

    # torch doesn't throw zero division warnings
    sd[iszero] = 1.0
    xz = (x - mn) / sd

    # Put nans
    xz[torch.broadcast_to(iszero, x.shape)] = torch.nan
    return xz


class ZscoreScaler:
    """
    zscore operator callable objects.

    It can accept both torch Tensors and numpy arrays.
    In case of torch Tensors are passed during call,
    ``torch_zscore`` is called.

    Parameters
    ----------
    x: ArrayLike
        The ArrayLike object to standardize.
    axis: int
        The axis along which to operate. By default, it assumes that
        the EEG channel dimension is the second to last. If the
        tensor has only one dimension, default value is changed to 0.

        Default = -2
    correction: int
        difference between the sample size and sample degrees of freedom.
        It is applied during the calculation of the standard deviation.
        It is equivalent to the Scipy's zscore `ddof` argument.
        Default is Bessel's correction as used in Pytorch's std function.

        Default = 1

    """

    def __init__(self, axis: int = -2, correction: int = 1):
        self.axis = axis
        self.correction = correction

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            return torch_zscore(x, self.axis, self.correction)
        else:
            return zscore(x, axis=self.axis, ddof=self.correction)


def torch_pchip(
    x: "1D Tensor",
    y: "ND Tensor",
    xv: "1D Tensor",
    save_memory: bool = True,
    new_y_max_numel: int = 4194304,
) -> torch.Tensor:
    """
    performs the pchip interpolation on the last dimension of the input tensor.

    This function is a pytorch adaptation of the scipy's pchip_interpolate [pchip]_
    . It performs sp-pchip interpolation (Shape Preserving Piecewise Cubic Hermite
    Interpolating Polynomial) on the last dimension of the y tensor.
    x is the original time grid and xv new virtual grid. So, the new values of y at
    time xv are given by the polynomials evaluated at the time grid x.

    This function is compatible with GPU devices.

    Parameters
    ----------
    x: 1D Tensor
        Tensor with the original time grid. Must be the same length as the last
        dimension of y.
    y: ND Tensor
        Tensor to interpolate. The last dimension must be the time dimension of the
        signals to interpolate.
    xv: 1D Tensor
        Tensor with the new virtual grid, i.e. the time points where to interpolate
    save_memory: bool, optional
        Whether to perform the interpolation on subsets of the y tensor by
        recursive function calls or not. Does not apply if y is a 1-D tensor.
        If set to False memory usage can greatly increase (for example with a
        128 MB tensor, the memory usage of the function is 1.2 GB), but it can
        speed up the process. However, this is not the case for all devices and
        performance may also decrease.

        Default = True
    new_y_max_numel: int, optional
        The number of elements which the tensor needs to surpass in order to make
        the function start doing recursive calls. It can be considered as an
        indicator of the maximum allowed memory usage since the lower the number,
        the lower the memory used.

        Default = 256*1024*16 (approximately 16s of recording of a 256 Channel
        EEG sampled at 1024 Hz).

    Returns
    -------
    new_y: torch.Tensor
        The pchip interpolated tensor.

    Note
    ----
    Some technical information and difference with other interpolation can be found
    here: https://blogs.mathworks.com/cleve/2012/07/16/splines-and-pchips/

    Note
    ----
    have a look also at the Scipy's documentation:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html
    Some parts of the code are inspired from:
    https://github.com/scipy/scipy/blob/v1.10.1/scipy/interpolate/_cubic.py#L157-L302

    References
    ----------
    .. [pchip] https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.pchip_interpolate.html

    Example
    -------
    >>> from scipy.interpolate import pchip_interpolate
    >>> import numpy as np
    >>> import selfeeg.utils
    >>> import torch
    >>> x = torch.zeros(16,32,1024) + torch.sin(torch.linspace(0, 8*torch.pi,1024))*500
    >>> xnp = x.numpy()
    >>> x_pchip = utils.torch_pchip(torch.arange(1024), x, torch.linspace(0,1023,475)).numpy()
    >>> xnp_pchip = pchip_interpolate(np.arange(1024),xnp, np.linspace(0,1023,475), axis=-1)
    >>> print(
    ...     np.isclose(x_pchip, xnp_pchip, rtol=1e-3,atol=0.5*1e-3).sum()==16*32*475
    ... ) # Should return True

    """

    if len(x.shape) != 1:
        raise ValueError(
            ["Expected 1D Tensor for x but received a ", str(len(x.shape)), "-D Tensor"]
        )
    if len(xv.shape) != 1:
        raise ValueError(
            ["Expected 1D Tensor for xv but received a ", str(len(xv.shape)), "-D Tensor"]
        )
    if x.shape[0] != y.shape[-1]:
        raise ValueError("x must have the same length than the last dimension of y")

    # Initialize the new interpolated tensor
    Ndim = len(y.shape)
    new_y = torch.empty((*y.shape[: (Ndim - 1)], xv.shape[0]), device=y.device)

    # If save_memory and the new Tensor size is huge, call recursively for
    # each element in the first dimension
    if save_memory:
        if Ndim > 1:
            if ((torch.numel(y) / y.shape[-1]) * xv.shape[0]) > new_y_max_numel:
                for i in range(new_y.shape[0]):
                    new_y[i] = torch_pchip(x, y[i], xv)
                return new_y

    # This is a common part for every channel
    if x.device.type == "mps" or xv.device.type == "mps":
        # torch bucketize is not already implemented in mps unfortunately
        # need to pass in cpu and return to mps. Note that this is very slow
        # like 500 times slower. But at least it doesn't throw an error
        bucket = torch.bucketize(xv.to(device="cpu"), x.to(device="cpu")) - 1
        bucket = bucket.to(device=x.device)
    else:
        bucket = torch.bucketize(xv, x) - 1
    bucket = torch.clamp(bucket, 0, x.shape[0] - 2)
    tv_minus = (xv - x[bucket]).unsqueeze(1)
    infer_tv = torch.cat(
        (tv_minus**3, tv_minus**2, tv_minus, torch.ones(tv_minus.shape, device=tv_minus.device)), 1
    )

    h = x[1:] - x[:-1]
    Delta = (y[..., 1:] - y[..., :-1]) / h
    k = torch.sign(Delta[..., :-1] * Delta[..., 1:]) > 0
    w1 = 2 * h[1:] + h[:-1]
    w2 = h[1:] + 2 * h[:-1]
    whmean = (w1 / Delta[..., :-1] + w2 / Delta[..., 1:]) / (w1 + w2)

    slope = torch.zeros(y.shape, device=y.device)
    slope[..., 1:-1][k] = whmean[k].reciprocal()

    slope[..., 0] = ((2 * h[0] + h[1]) * Delta[..., 0] - h[0] * Delta[..., 1]) / (h[0] + h[1])
    slope_cond = torch.sign(slope[..., 0]) != torch.sign(Delta[..., 0])
    slope[..., 0][slope_cond] = 0
    slope_cond = torch.logical_and(
        torch.sign(Delta[..., 0]) != torch.sign(Delta[..., 1]),
        torch.abs(slope[..., 0]) > torch.abs(3 * Delta[..., 0]),
    )
    slope[..., 0][slope_cond] = 3 * Delta[..., 0][slope_cond]

    slope[..., -1] = ((2 * h[-1] + h[-2]) * Delta[..., -1] - h[-1] * Delta[..., -2]) / (
        h[-1] + h[-2]
    )
    slope_cond = torch.sign(slope[..., -1]) != torch.sign(Delta[..., -1])
    slope[..., -1][slope_cond] = 0
    slope_cond = torch.logical_and(
        torch.sign(Delta[..., -1]) != torch.sign(Delta[..., -1]),
        torch.abs(slope[..., -1]) > torch.abs(3 * Delta[..., 1]),
    )
    slope[..., -1][slope_cond] = 3 * Delta[..., -1][slope_cond]

    t = (slope[..., :-1] + slope[..., 1:] - Delta - Delta) / h
    a = (t) / h
    b = (Delta - slope[..., :-1]) / h - t

    py_coef = torch.stack((a, b, slope[..., :-1], y[..., :-1]), -1)
    new_y = (py_coef[..., bucket, :] * infer_tv).sum(axis=-1)
    return new_y


def create_dataset(
    folder_name: str = "Simulated_EEG",
    Sample_range: list = [512, 1025],
    Chans: int = 8,
    p: list = 0.8,
    return_labels: bool = False,
    seed: int = 1234,
) -> Optional[np.ndarray]:
    """
    creates a simulated EEG dataset for normal abnormal binary classification.

    Samples have random length within a given range.

    Once called, the function will generate 1000 files in a new directory.
    Samples will have name 'A_B_C_D.pickle' with:

        1. A = dataset ID
        2. B = subject ID
        3. C = session ID
        4. D = trial ID.

    In total, ``create_dataset`` will generate files associated to:

        1. 5 datasets (200 files per dataset)
        2. 40 subjects per dataset
        3. 5 sessions per subject
        4. 1 trial per session.

    All files will store a dictionary with two keys:

        1. 'data' = the array with random length and given channels
           (channels in column dimension)
        2. 'label' = an integer with a random binary label (0=normal, 1=abnormal).

    EEG files have values in uV, with range at most in [-550,550] uV.

    Parameters
    ----------
    folder_name: str, optional
        A string with the optional name of the subdirectory to store the
        generated files.

        Default = 'Simulated_EEG'
    Sample_range: list, optional
        A length 2 list with the possible minimum and maximum length of the
        generated EEGs.

        Default = [512, 1025]
    Chans: int, optional
        An integer defining the number of channels each EEG must have.

        Default = 8
    p: float, optional
        A scalar in range [0, 1] with the probability of a sample being normal.

        Default = 0.8
    seed: int, optional
        A seed to set for reproducibility.

        Default = 1234

    Returns
    -------
    classes: ArrayLike
        An array with the generated label. Index association is based on the
        file sorted by names.

    Example
    -------
    >>> import selfeeg.utils
    >>> import glob
    >>> utils.create_dataset()
    >>> print(len(glob.glob('Simulated_EEG/*'))==1000) #shoud return True

    """
    # Various checks
    if not (isinstance(Sample_range, list)):
        raise ValueError("Sample_range must be a list")
    else:
        if len(Sample_range) != 2:
            raise ValueError("Sample_range must have length 2")
    if Chans < 1:
        raise ValueError("Chans must be bigger than 1")
    if (p < 0) or (p > 1):
        raise ValueError("p must be in range [0, 1]")

    # create new sub-folder if that does not exist
    if not (os.path.isdir(folder_name)):
        os.mkdir(folder_name)

    # prepare elements for file generation
    Sample_range.sort()
    N = 1000
    np.random.seed(seed=seed)
    classes = np.zeros(N)
    for i in range(N):
        # get random length and class label
        Sample = np.random.randint(Sample_range[0], Sample_range[1])
        y = np.random.choice([0, 1], p=[p, 1 - p])
        classes[i] = y

        # generate sample while being sure that values will not have
        # strange ranges
        x = 600
        while np.max(x) > 550 or np.min(x) < -550:
            if y == 1:
                stderr = np.sqrt(122.35423)
                F1 = np.random.normal(0.932649, 0.040448)
                F0 = np.random.normal(2.1159355, 2.3523977)
            else:
                stderr = np.sqrt(454.232666)
                F1 = np.random.normal(0.9619603, 0.0301687)
                F0 = np.random.normal(-0.1810323, 3.4712047)
            x = np.zeros((Chans, Sample))
            x[:, 0] = np.random.normal(0, stderr, Chans)
            for k in range(1, Sample):
                x[:, k] = F0 + F1 * x[:, k - 1] + np.random.normal(0, stderr, Chans)

        # store files
        sample = {"data": x, "label": y}
        A = int(i // 200) + 1
        B = int((i - 200 * int(i // 200))) // 5 + 1
        C = i % 5 + 1
        file_name = "Simulated_EEG/" + str(A) + "_" + str(B) + "_" + str(C) + "_1.pickle"
        with open(file_name, "wb") as f:
            pickle.dump(sample, f)
    if return_labels:
        return classes


def check_models(model1: torch.nn.Module, model2: torch.nn.Module) -> bool:
    """
    checks that two nn.Modules are equal.

    Parameters
    ----------
    model1: nn.Module
        The first model to compare.
    model2: nn.Module
        The second model to compare.

    Returns
    -------
    equals: bool
        A boolean stating if the models are equal or not.

    Example
    -------
    >>> import selfeeg.models
    >>> model1 = models.EEGNet(4,8,512)
    >>> model2 = models.EEGNet(4,8,512)
    >>> print( utils.utils.check_models(model1,model2)) # Should return False
    >>> model2.load_state_dict(model1.state_dict())
    >>> utils.check_models(model1,model2)  # Should return False

    """
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


def count_parameters(
    model: torch.nn.Module,
    return_table: bool = False,
    print_table: bool = False,
    add_not_trainable=False,
) -> [int, Optional[pd.DataFrame]]:
    """
    counts the number of **trainable parameters** of a
    Pytorch's nn.Module.

    It can additionally create a two column dataframe
    with module's name and number of trainable parameters.
    Not trainable parameters can be also added to the table if specified.

    The implementation is an enriched implementation
    inspired from [stacko1]_ and [stacko2]_ .

    Parameters
    ----------
    model: nn.Module
        The model to scroll.
    return_table: bool, optional
        Whether to return a with module's name and number of
        trainable parameters or not.

        Default = False
    print_table: bool, optional
        Whether to print the created table or not.

        Default = False
    add_not_trainable: bool, optional
        Whether to add blocks with 0 trainable parameters to the table or not.

        Default = False

    Returns
    -------
    total_params: int
        The number of trainable parameters.
    layer_table: pd.DataFrame, optional
        A two column dataframe with module's name and number of trainable parameters.

    References
    ----------
    .. [stacko1] https://stackoverflow.com/questions/49201236
    .. [stacko2] https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9

    Example
    -------
    >>> import selfeeg.utils
    >>> import selfeeg.models
    >>> mdl = models.ShallowNet(4,8,1024)
    >>> for n, i in enumerate(mdl.parameters()): # bias require grad put to False
    ...     i.requires_grad=False if n in [1,3,5,7] else True
    >>> a,b = utils.count_parameters(mdl, True,True,True)
    >>> print (b == 23760) # should return True

    """
    table = []
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if add_not_trainable:
                params = 0
            else:
                continue
        else:
            params = parameter.numel()
        table.append([name, params])
        total_params += params
    layer_table = pd.DataFrame(table, columns=["Modules", "Parameters"])
    if print_table:
        print(layer_table.to_string())
        print("=" * len(layer_table.to_string().split("\n")[0]))
        char2add = len(layer_table.to_string().split("\n")[0].split("Modules")[0]) - 15
        char2add2 = (
            len(layer_table.to_string().split("\n")[0].split("Modules")[1])
            - len(str(total_params))
            - 1
        )
        print(" " * char2add + "TOTAL TRAINABLE PARAMS" + " " * char2add2, total_params)
    return (layer_table, total_params) if return_table else total_params


def _reset_seed(
    seed: int = None,
    reset_random: bool = True,
    reset_numpy: bool = True,
    reset_torch: bool = True,
) -> None:
    """
    :meta private:
    """
    if seed is not None:
        if seed <= 0:
            raise ValueError("seed must be a nonnegative number")
        if reset_numpy:
            np.random.seed(seed)
        if reset_random:
            random.seed(seed)
        if reset_torch:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
