from __future__ import annotations

import math
import random
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike
from scipy import fft, interpolate, signal
from torchaudio.functional import filtfilt

from selfeeg.utils.utils import torch_pchip

__all__ = [
    "add_band_noise",
    "add_eeg_artifact",
    "add_gaussian_noise",
    "add_noise_SNR",
    "change_ref",
    "channel_dropout",
    "crop_and_resize",
    "filter_bandpass",
    "filter_bandstop",
    "filter_highpass",
    "filter_lowpass",
    "flip_horizontal",
    "flip_vertical",
    "get_channel_map_and_networks",
    "get_eeg_channel_network_names",
    "get_filter_coeff",
    "identity",
    "masking",
    "moving_avg",
    "permutation_signal",
    "permute_channels",
    "phase_swap",
    "random_FT_phase",
    "random_slope_scale",
    "scaling",
    "shift_frequency",
    "shift_horizontal",
    "shift_vertical",
    "warp_signal",
]


# ----- SHIFTS AND FLIPS -------
def identity(x: ArrayLike) -> ArrayLike:
    """
    returns the same array or tensor it was given.

    It can be used during augmentation composition to
    randomly avoid some augmentations.

    Parameters
    ----------
    x: ArrayLike
        the input Tensor or Array.

    Returns
    -------
    x: ArrayLike
        the input Tensor or Array.

    Example
    -------
    >>> import torch
    >>> import selfeeg.augmentation as aug
    >>> x = torch.zeros(16,32,1024) + torch.sin(torch.linspace(0, 8*torch.pi,1024))
    >>> xaug = aug.identity(x)
    >>> print(torch.equal(x,xaug)) # should return True

    """
    return x


def shift_vertical(x: ArrayLike, value: float) -> ArrayLike:
    """
    adds a scalar value to the ArrayLike object.

    Parameters
    ----------
    x: ArrayLike
        the input Tensor or Array.
    value: float
        The scalar to add.

    Returns
    -------
    x: ArrayLike
        the augmented version of the input Tensor or Array.

    Example
    -------
    >>> import torch
    >>> import selfeeg.augmentation as aug
    >>> x = torch.zeros(16,32,1024) + torch.sin(torch.linspace(0, 8*torch.pi,1024))
    >>> xaug = aug.shift_vertical(x, 4)
    >>> print(torch.equal(x+4,xaug)) # should return True

    """
    x_shift = torch.clone(x) if isinstance(x, torch.Tensor) else np.copy(x)
    x_shift = x + value
    return x_shift


def shift_horizontal(
    x: ArrayLike,
    shift_time: float,
    Fs: float,
    forward: bool = None,
    random_shift: bool = False,
    batch_equal: bool = True,
) -> ArrayLike:
    """
    Shifts temporally the elements of the ArrayLike object.

    Shift is applied along the last dimension.
    The empty elements at beginning or the ending part
    after shift are set to zero.

    Parameters
    ----------
    x: ArrayLike
        the input Tensor or Array. Last dimension must be the EEG time dimension.
    shift_time: float
        Desired shift in seconds.
    Fs: float
        The EEG sampling rate in Hz.
    forward: bool, optional
        Whether to shift the EEG forward (True) or backward (False) in time.
        If left to None, a random selection of the shift direction
        will be performed.

        Default = None
    random_shift: bool, optional
        Wheter to choose a random shift length lower than or equal to shift_time,
        i.e., consider shift_time as the exact value to shift or as an upper bound
        for a random selection. In case of True, the random shift value will
        be extracted from a uniform distribution with range [0,shift_time],
        while the forward argument will decide the direction of the shift

        Default = False
    batch_equal: bool, optional
        whether to apply the same shift to all EEG record or not.

        Default = True

    Returns
    -------
    x: ArrayLike
        the augmented version of the input Tensor or Array.

    Note
    ----
    If random shift is set to False and forward is None, then batch_equal
    will be set to `True` since no differences in the shift can be applied.

    Example
    -------
    >>> import torch
    >>> import selfeeg.augmentation as aug
    >>> x = torch.zeros(16,32,1024) + torch.sin(torch.linspace(0, 8*torch.pi,1024))
    >>> xaug = aug.shift_horizontal(x, 64, 1, True)
    >>> print(xaug[...,0:64].sum()==0) # should return True
    >>> print(xaug[...,65].sum()==0 ) # should return False

    """

    if shift_time < 0:
        raise ValueError(
            "shift time must be a positive value. " "To shift backward set forward to False"
        )
    Ndim = len(x.shape)
    x_shift = torch.clone(x) if isinstance(x, torch.Tensor) else np.copy(x)
    if not (batch_equal):
        if not (random_shift or (forward is None)):
            print("set batch equal to true")
            batch_equal = True

    if batch_equal or Ndim < 3:
        if forward is None:
            forward = bool(random.getrandbits(1))

        if random_shift:
            shift = random.randint(1, int(shift_time * Fs))
        else:
            shift = int(shift_time * Fs)

        if forward:
            x_shift[..., :shift] = 0
            x_shift[..., shift:] = x[..., :-shift]
        else:
            x_shift[..., :-shift] = x[..., shift:]
            x_shift[..., -shift:] = 0
    else:
        for i in range(x_shift.shape[0]):
            x_shift[i] = shift_horizontal(
                x[i],
                shift_time=shift_time,
                Fs=Fs,
                forward=forward,
                random_shift=random_shift,
                batch_equal=batch_equal,
            )
    return x_shift


def _UnitStep(x):
    """
    ``_UnitStep`` creates a numpy array or pytorch tensor with shape equal to x
    and with the last dimension filled with the step function values used in
    the Hilbert Transform. This is used to speed up the computation of
    shift_frequency when batch_equal is set to False.
    In short, it avoids initializing the same h multiple times.

    For more info see SciPy's hilbert help and source code:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html

    :meta private:
    """
    N = x.shape[-1]
    if isinstance(x, torch.Tensor):
        h = torch.zeros_like(x, device=x.device)
    else:
        h = np.zeros_like(x)
    if N % 2 == 0:
        h[..., 0] = h[..., N // 2] = 1
        h[..., 1 : N // 2] = 2
    else:
        h[..., 0] = 1
        h[..., 1 : (N + 1) // 2] = 2
    return h


def torch_hilbert(x, h: torch.Tensor = None):
    """
    ``torch_hilbert`` is a minimal version of SciPy's hilbert
    function [hil1]_ adapted for pytorch tensors.

    Parameters
    ----------
    x : torch.Tensor
        Tensor with signal data.
    h : torch.Tensor, optional
        Tensor with the UnitStep function.
        If not given, it will be initialized during call.
        Default: None

    Returns
    -------
    torch.Tensor
        The analytic signal of x calculated along the last dimension of x

    References
    ----------
    .. [hil1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html

    """
    if torch.is_complex(x):
        raise ValueError("x must be real tensor.")

    N = x.shape[-1]
    f = torch.fft.fft(x, N, dim=-1)
    if h is None:
        h = torch.zeros_like(f, device=x.device)
        if N % 2 == 0:
            h[..., 0] = h[..., N // 2] = 1
            h[..., 1 : N // 2] = 2
        else:
            h[..., 0] = 1
            h[..., 1 : (N + 1) // 2] = 2
    Xa = torch.fft.ifft(f * h, dim=-1)
    return Xa


def _shift_frequency(
    x: ArrayLike,
    shift_freq: float,
    Fs: float,
    forward: bool = None,
    random_shift: bool = False,
    batch_equal: bool = True,
    t=None,
    h=None,
) -> ArrayLike:
    """
    See ``shift_frequency`` help.
    """
    if shift_freq < 0:
        raise ValueError(
            "shift freq must be a positive value." " To shift backward set forward to False"
        )
    Ndim = len(x.shape)
    x_shift = torch.clone(x) if isinstance(x, torch.Tensor) else np.copy(x)
    if not (batch_equal):
        if not (random_shift or (forward is None)):
            print("set batch equal to true")
            batch_equal = True

    if batch_equal or Ndim < 3:
        shift = shift_freq
        if forward is None:
            forward = bool(random.getrandbits(1))
        if random_shift:
            shift = np.random.uniform(-shift_freq, shift_freq)
        if not (forward) and shift > 0:
            shift = -shift

        if t is None:
            if isinstance(x, torch.Tensor):
                T = x.shape[-1] / Fs
                t = torch.linspace(0, T - (1 / Fs), int(Fs * T), device=x.device)
            else:
                t = np.r_[0 : T : (1 / Fs)]

        if isinstance(x, torch.Tensor):
            Xa = torch_hilbert(x) if h is None else torch_hilbert(x, h)
            x_shift = torch.real(Xa * torch.exp(1j * 2 * math.pi * shift * t))
        else:
            Xa = signal.hilbert(signal) if h is None else fft.ifft(fft.fft(x) * h)
            x_shift = np.real(Xa * np.exp(1j * 2 * math.pi * shift * t))

    else:
        for i in range(x_shift.shape[0]):
            x_shift[i] = _shift_frequency(
                x[i],
                shift_freq=shift_freq,
                Fs=Fs,
                forward=forward,
                random_shift=random_shift,
                batch_equal=batch_equal,
                t=t,
                h=h[i] if h is not None else None,
            )
    return x_shift


def shift_frequency(
    x: ArrayLike,
    shift_freq: float,
    Fs: float,
    forward: bool = None,
    random_shift: bool = False,
    batch_equal: bool = True,
) -> ArrayLike:
    """
    shifts the frequency components of the signals included
    in the ArrayLike object.

    Shift will be performed as reported in [shiftFT1]_
    (see section 4 of the reference paper).

    Parameters
    ----------
    x: ArrayLike
        The input Tensor or Array. The last two dimensions must refer
        to the EEG (Channels x Samples).
    shift_freq: float
        The desired shift, given in Hz. It must be a positive value.
        If `random_shift` is set to True, shift_freq is used to extract a
        random value from a uniform distribution between
        `[-shift_freq, shift_freq]`, i.e. it becomes
        the maximum value of the distribution.
    Fs: float`
        the EEG sampling rate in Hz.
    forward: bool, optional
        Whether to shift the EEG frequencies forward (True) or backward (False).
        If left to None, a random selection of the shift direction
        will be performed.

        Default = None
    random_shift: bool, optional
        Wheter to choose a random shift from a uniform distribution between
        `[-shift_freq, shift_freq]` or not.

        Default = False
    batch_equal: bool, optional
        Whether to apply the same shift to all EEG record or not.

        Default = True


    Returns
    -------
    x: ArrayLike
        The augmented version of the input Tensor or Array.

    Note
    ----
    If `random shift` is set to False and `forward` is None, then `batch_equal`
    will be set to True since no differences in the shift can be applied.


    References
    ----------
    .. [shiftFT1] Rommel, Cédric, et al. "Data augmentation for learning
      predictive models on EEG: a systematic comparison."
      Journal of Neural Engineering 19.6 (2022): 066020.

    Example
    -------
    >>> import torch
    >>> import math
    >>> import selfeeg.augmentation as aug
    >>> from scipy.signal import periodogram
    >>> Fs= 128
    >>> x = torch.zeros(16,32,1024) + torch.sin(torch.linspace(0, 48*torch.pi,1024))
    >>> x = x + torch.sin(torch.linspace(0, 8*torch.pi,1024))
    >>> xaug = aug.shift_frequency(x, 10, Fs, True)
    >>> f, per1 =periodogram(x[0,0], fs=Fs)
    >>> per2 =periodogram(xaug[0,0], fs=Fs)[1]
    >>> # check if spectrogram is really shfted
    >>> print(math.isclose(per1[4],per2[84], rel_tol=1e-5)) # should return True
    >>> print(math.isclose(per1[24],per2[104],rel_tol=1e-5)) # should return True

    plot the augmentations (require matplotlib to be installed)

    >>> import matplotlib.pyplot as plt
    >>> plt.style.use('seaborn-v0_8-white')
    >>> plt.rcParams['figure.figsize'] = (15.0, 6.0)
    >>> plt.plot(f,per1)
    >>> plt.plot(f,per2)
    >>> plt.tick_params(axis='both', which='major', labelsize=12)
    >>> plt.title(
    ...     'Spectrogram of original and frequency shifted signal', fontsize=15)
    >>> plt.legend(['original sinusoide', 'shifted sinusoide'])
    >>> plt.show()

    """
    T = x.shape[-1] / Fs
    if isinstance(x, torch.Tensor):
        t = torch.linspace(0, T - (1 / Fs), int(Fs * T), device=x.device)
    else:
        t = np.r_[0 : T : (1 / Fs)]
    h = _UnitStep(x)
    return _shift_frequency(x, shift_freq, Fs, forward, random_shift, batch_equal, t, h)


def phase_swap(x: ArrayLike) -> ArrayLike:
    """
    Apply the phase swap data augmentation to the ArrayLike object.

    The phase swap data augmentation consists in merging the amplitude
    and phase components of biosignals from different sources to help
    the model learn their coupling.
    Specifically, the amplitude and phase of two randomly selected EEG samples
    are extracted using the Fourier transform.
    New samples are then generated by applying the inverse Fourier transform,
    combining the amplitude from one sample with the phase from the other.
    See the following paper for more information [phaseswap]_.

    Parameters
    ----------
    x : ArrayLike
        A 3-dimensional torch tensor or numpy array.
        The last two dimensions must refer to the EEG (Channels x Samples).

    Returns
    -------
    x: ArrayLike
        The augmented version of the input Tensor or Array.

    Note
    ----
    `Phase swap` ignores the class of each sample.


    References
    ----------
    .. [phaseswap] Lemkhenter, Abdelhak, and Favaro, Paolo.
      "Boosting Generalization in Bio-signal Classification by
      Learning the Phase-Amplitude Coupling". DAGM GCPR (2020).

    """

    Ndim = len(x.shape)
    if Ndim != 3:
        raise ValueError("x must be a 3-dimensional array or tensor")

    N = x.shape[0]

    if isinstance(x, torch.Tensor):
        # Compute fft, module and phase
        xfft = torch.fft.fft(x)
        amplitude = xfft.abs()
        phase = xfft.angle()
        x_aug = torch.clone(xfft)

        # Random shuffle indeces
        idx_shuffle = torch.randperm(N).to(device=x.device)
        idx_shuffle_1 = idx_shuffle[: (N // 2)]
        idx_shuffle_2 = idx_shuffle[(N // 2) : (N // 2) * 2]

        # Apply phase swap
        x_aug[idx_shuffle_1] = amplitude[idx_shuffle_1] * torch.exp(1j * phase[idx_shuffle_2])
        x_aug[idx_shuffle_2] = amplitude[idx_shuffle_2] * torch.exp(1j * phase[idx_shuffle_1])

        # Reconstruct the signal
        x_aug = (torch.fft.ifft(x_aug)).real.to(device=x.device)

    else:

        xfft = np.fft.fft(x)
        amplitude = np.abs(xfft)
        phase = np.angle(xfft)
        x_aug = np.copy(xfft)

        # Random shuffle indeces
        idx_shuffle = np.random.permutation(N)
        idx_shuffle_1 = idx_shuffle[: (N // 2)]
        idx_shuffle_2 = idx_shuffle[(N // 2) : (N // 2) * 2]

        # Apply phase swap
        x_aug[idx_shuffle_1] = amplitude[idx_shuffle_1] * np.exp(1j * phase[idx_shuffle_2])
        x_aug[idx_shuffle_2] = amplitude[idx_shuffle_2] * np.exp(1j * phase[idx_shuffle_1])

        # Reconstruct the signal
        x_aug = (np.fft.ifft(x_aug)).real

    return x_aug


def flip_vertical(x: ArrayLike) -> ArrayLike:
    """
    changes the sign of all the elements of the input.

    Parameters
    ---------
    x: ArrayLike
        The input Tensor or Array.

    Returns
    -------
    x: ArrayLike
        The augmented version of the input Tensor or Array.

    Example
    -------
    >>> import torch
    >>> import selfeeg.augmentation as aug
    >>> x = torch.randn(64,32,512)
    >>> xaug = aug.phase_swap(x)

    """
    x_flip = x * (-1)
    return x_flip


def flip_horizontal(x: ArrayLike) -> ArrayLike:
    """
    flips the elements of the ArrayLike object along its last dimension.

    Parameters
    ---------
    x: ArrayLike
        The input Tensor or Array.

    Returns
    -------
    x: ArrayLike
        The augmented version of the input Tensor or Array.

    Example
    -------
    >>> import torch
    >>> import selfeeg.augmentation as aug
    >>> x = torch.zeros(16,32,1024) + torch.sin(torch.linspace(0, 8*torch.pi,1024))
    >>> xaug = aug.flip_horizontal(x)
    >>> print(torch.equal(xaug, torch.flip(x, [len(x.shape)-1]))) #should return True

    """
    if isinstance(x, np.ndarray):
        x_flip = np.flip(x, len(x.shape) - 1)
    else:
        x_flip = torch.flip(x, [len(x.shape) - 1])

    return x_flip


# ---- NOISE ADDER -----
def add_gaussian_noise(
    x: ArrayLike, std: float = 1.0, mean: float = 0.0, get_noise: bool = False
) -> tuple[ArrayLike, Optional[ArrayLike]]:
    """
    adds gaussian noise with the desired standard deviation.

    It is also possible to add a vertical shift (mean on the noise).
    Note that in this case the noise will not have 0 mean anymore and it
    cannot be concretely considered white noise.

    Parameters
    ----------
    x: ArrayLike
        The input Tensor or Array.
    std: float, optional
        The std of the gaussian distribution.

        Default =  1
    mean: float, optional
        A constant to add to the generated white noise.

        Default = 0
    get_noise: bool, optional
        Whether to return the generated noise or not.

        Default = False

    Returns
    -------
    x: ArrayLike
        The augmented version of the input Tensor or Array.
    noise: ArrayLike, optional
        The generated noise. Returned only if `get_noise` is set to True.

    Example
    -------
    >>> import torch
    >>> import math
    >>> import selfeeg.augmentation as aug
    >>> x = torch.zeros(16,32,1024) + torch.sin(torch.linspace(0, 8*torch.pi,1024))
    >>> xaug, noise = aug.add_gaussian_noise(x, 0.1, get_noise=True)
    >>> print(math.isclose(noise.std(),0.1,rel_tol=1e-2)) # should return True
    >>> print(math.isclose(xaug.mean(),0,rel_tol=1e-4, abs_tol=1e-3)) # should return True

    """

    if isinstance(x, np.ndarray):
        noise = mean + std * np.random.randn(*x.shape)
    else:
        noise = mean + std * torch.randn(*x.shape, device=x.device)
    x_noise = x + noise
    if get_noise:
        return x_noise, noise
    else:
        return x_noise


def add_noise_SNR(
    x: ArrayLike, target_snr: float = 5.0, get_noise: bool = False
) -> tuple[ArrayLike, Optional[ArrayLike]]:
    """
    add noise to the input ArrayLike object such that its
    SNR (Signal to Noise Ratio) will be the one desired.

    Since the signal is supposed to be already noisy, it makes more sense to
    say that this function scales the SNR by a factor equal to 1/P_noise_new,
    where P_noise_new is the power of the new added noise.
    Check [snr1]_ for more info.

    Parameters
    ----------
    x: ArrayLike
        The input Tensor or Array.
    target_snr: float, optional
        The target SNR.

        Default = 5.
    get_noise: bool, optional
        Whether to return the generated noise or not.

        Default = False

    Returns
    -------
    x: ArrayLike
        The augmented version of the input Tensor or Array.
    noise: ArrayLike, optional
        The generated noise. Returned only if `get_noise` is set to True

    References
    ----------
    .. [snr1] created using the following reference:
           https://stackoverflow.com/questions/14058340/

    Example
    -------
    >>> import torch
    >>> import numpy as np
    >>> import selfeeg.augmentation as aug
    >>> x = torch.zeros(16,32,1024) + torch.sin(torch.linspace(0, 8*np.pi,1024))
    >>> xaug, noise = aug.add_noise_SNR(x, 10, get_noise=True)
    >>> SNR = 10*np.log10(((x**2).sum().mean())/((noise**2).sum().mean()))
    >>> print(math.isclose( SNR ,10,rel_tol=1e-2)) # should return True

    """

    N = len(x.shape)
    if N == 1:
        axis = 0
        new_size = (x.shape[-1],)
    elif N == 2:
        axis = 1
        new_size = (1, x.shape[-1])
    else:
        axis = -1
        new_size = (*x.shape[0:-2], 1, x.shape[-1])

    # get signal power. Not exactly true since we have an already noised signal
    factor = 10 ** (-target_snr / 20)

    if isinstance(x, np.ndarray):
        xpow = np.power(x, 2)
        xpow = np.mean(xpow, keepdims=True)
        xpow = np.sqrt(xpow)
        noise = np.random.randn(*new_size)
    else:
        xpow = torch.pow(x, 2)
        xpow = torch.mean(xpow, axis, keepdim=True)
        xpow = torch.sqrt(xpow)
        noise = torch.randn(size=new_size).to(device=x.device)

    noise = factor * noise
    noise = xpow * noise
    x_noise = x + noise

    if get_noise:
        return x_noise, noise
    else:
        return x_noise


def add_band_noise(
    x: ArrayLike,
    bandwidth: list[tuple[float, float], str, float],
    samplerate: float = 256,
    noise_range: float or list[float, float] = None,
    std: float = None,
    get_noise: bool = False,
) -> tuple[ArrayLike, Optional[ArrayLike]]:
    """
    adds random noise filtered at specific bandwidths.

    Given a set of bandwidths or a set of specific frequencies, `add_band_noise`
    creates a noise whose spectrum is bigger than zero only on the specified bands.
    It can be used to alter only specific frequency components of the original
    signal. By default, the noise generated will have the same standard deviation
    as x, but it can be rescaled so to be within a specific range of values or
    to have a specific standard deviation.

    Parameters
    ----------
    x: ArrayLike
        The input Tensor or Array.
    bandwidth: list
        The frequency components which the noise must have.
        Must be a *list* with the following values:

            - strings: add noise to specific EEG components.
              Can be any of "delta", "theta", "alpha", "beta", "gamma",
              "gamma_low", "gamma_high".
            - scalar: add noise to a specific component.
            - tuple with 2 scalar: add noise to a specific band set with the
              tuple (start_component,end_component).

    samplerate: float, optional
        The sampling rate, given in Hz. Remember to change this value
        according to the signal sampling rate.

        Default = 256
    noise_range: float, optional
        The range within the noise is scaled. Must be a single scalar
        or a two element list. If given as a single scalar, then the range is
        considered the interval [-noise_range, noise_range]. If this parameter
        is given, then std value is ignored, since it is not possible to guarantee
        that the two conditions will be satisfied at the same time.
        To rescale, the following formula is applied::

            noise_new = ((noise - max(noise))/(max(noise)-min(noise)))
            *(target_range_max - target_range_min) + target_range_min

        Default = None
    std: float, optional
        The desired standard deviation of the noise. If noise_range is given,
        this argument is ignored. It simply scale the noise by applying:

            ``noise_new = noise * (target_std / std(noise))``

        Default = None
    get_noise: bool, optional
        Whether to return the generated noise or not.

        Default = False

    Returns
    -------
    x: ArrayLike
        The augmented version of the input Tensor or Array.
    noise: ArrayLike, optional
        The generated noise. Returned only if `get_noise` is set to True.

    Example
    -------
    >>> import torch
    >>> import numpy as np
    >>> from scipy.signal import periodogram
    >>> import selfeeg.augmentation as aug
    >>> x = torch.zeros(16,32,1024) + torch.sin(torch.linspace(0, 8*torch.pi,1024))
    >>> xaug, noise = aug.add_band_noise(x, "beta", 128, noise_range=0.2, get_noise=True)
    >>> f, per = periodogram(noise, 128)
    >>> index= np.where(per>1e-12)[0]
    >>> print( len(np.where( ((f[index]<13) | (f[index]>30)))[0])==0 ) #should return True

    """

    # converting to list if single string or integer is given
    if not (isinstance(bandwidth, list)):
        bandwidth = [bandwidth]
    if noise_range != None:
        if isinstance(noise_range, list):
            if len(noise_range) == 2:
                noise_range.sort()
            else:
                raise ValueError("if given as a list, noise_range must be of length 2")
        else:
            if noise_range > 0:
                noise_range = [-noise_range, noise_range]
            else:
                noise_range = [noise_range, -noise_range]

    # transform all elements in 2 floats tuple
    for i in range(len(bandwidth)):
        # change string bandwidth call to frequency slice
        if isinstance(bandwidth[i], str):
            if bandwidth[i].lower() == "delta":
                bandwidth[i] = (0.5, 4)
            elif bandwidth[i].lower() == "theta":
                bandwidth[i] = (4, 8)
            elif bandwidth[i].lower() == "alpha":
                bandwidth[i] = (8, 13)
            elif bandwidth[i].lower() == "beta":
                bandwidth[i] = (13, 30)
            elif bandwidth[i].lower() == "gamma_low":
                bandwidth[i] = (30, 70)
            elif bandwidth[i].lower() == "gamma_high":
                bandwidth[i] = (70, 150)
            elif bandwidth[i].lower() == "gamma":
                bandwidth[i] = (30, 150)
            else:
                message = 'Brainwave "', bandwidth[i], '" not exist. \n'
                message += "Choose between delta, theta, alpha, beta, gamma, "
                message += "gamma_low, gamma_high"
                raise ValueError(message)

        # change single frequency call to tuple
        elif np.isscalar(bandwidth[i]):
            bandwidth[i] = (bandwidth[i], bandwidth[i])

    N = len(bandwidth)
    samples = x.shape[-1]
    if isinstance(x, np.ndarray):

        f = np.zeros(samples, dtype="complex")
        for i in range(N):
            start = int(bandwidth[i][0] * samples / samplerate)
            end = int(bandwidth[i][1] * samples / samplerate + 1)
            f[start:end] = 1
        Np = (len(f) - 1) // 2
        phases = np.random.rand(Np) * 2 * np.pi
        phases = np.cos(phases) + 1j * np.sin(phases)
        f[1 : Np + 1] *= phases
        f[-1 : -1 - Np : -1] = np.conj(f[1 : Np + 1])
        noise = np.fft.ifft(f).real
        if noise_range == None:
            G = std / np.std(noise) if std != None else np.std(x) / np.std(noise)
            noise *= G
        else:
            max_noise = np.max(noise)
            min_noise = np.min(noise)
            range_noise = max_noise - min_noise
            target_range = noise_range[1] - noise_range[0]
            noise = ((noise - min_noise) / range_noise) * (target_range)
            noise = noise + noise_range[0]
        x_noise = x + noise

    else:
        device = "cpu" if x.device.type == "mps" else x.device
        f = torch.zeros(samples, dtype=torch.complex64, device=device)
        for i in range(N):
            start = int(bandwidth[i][0] * samples / samplerate)
            end = int(bandwidth[i][1] * samples / samplerate + 1)
            f[start:end] = 1
        Np = (samples - 1) // 2
        phases = torch.rand(Np, device=device)
        phases = phases * 2 * math.pi
        phases = torch.cos(phases) + 1j * torch.sin(phases)
        f[1 : Np + 1] *= phases
        f[-Np:] = torch.flip(torch.conj(f[1 : Np + 1]), [0])
        noise = torch.fft.ifft(f).real
        if x.device.type == "mps":
            noise = noise.to(device="mps")
        if noise_range == None:
            if std != None:
                G = std / torch.std(noise)
            else:
                G = torch.std(x) / torch.std(noise)
            noise *= G
        else:
            max_noise = torch.max(noise)
            min_noise = torch.min(noise)
            range_noise = max_noise - min_noise
            target_range = noise_range[1] - noise_range[0]
            noise = ((noise - min_noise) / range_noise) * (target_range)
            noise = noise + noise_range[0]
        x_noise = x + noise

    if get_noise:
        return x_noise, noise
    else:
        return x_noise


def scaling(x: ArrayLike, value: float = None, batch_equal: bool = True) -> ArrayLike:
    """
    rescales the ArrayLike object by a given amplitude.

    Parameters
    ----------
    x: ArrayLike
        The input Tensor or Array.
        The last two dimensions must refer to the EEG recording
        (Channels x Samples).
    value: float, optional
        The rescaling factor. If not given, a random value is extracted from a
        uniform distribution in range [0.5, 2].

        Default: None
    batch_equal: bool, optional
        Whether to apply the same rescaling on all signals or not.
        If False, value must be left to None, otherwise batch_equal
        will be reset to True.

        Default: True

    Returns
    -------
    x: ArrayLike
        The augmented version of the input Tensor or Array.

    Example
    -------
    >>> import torch
    >>> import selfeeg.augmentation as aug
    >>> x = torch.zeros(16,32,1024) + torch.sin(torch.linspace(0, 8*torch.pi,1024))
    >>> xaug = aug.scaling(x, 1.5)
    >>> print( xaug.max()==x.max()*1.5)# should return True
    >>> print( xaug.min()==x.min()*1.5)# should return True

    """
    Ndim = len(x.shape)
    x_scale = torch.clone(x) if isinstance(x, torch.Tensor) else np.copy(x)
    if not (batch_equal) and (value is None):
        batch_equal = True  # speed up computation
    if batch_equal or Ndim < 3:
        if value is None:
            value = random.uniform(0.5, 2)  # nosec
        x_scale *= value
    else:
        for i in range(x_scale.shape[0]):
            x_scale[i] = scaling(x[i], value, batch_equal)
    return x_scale


def random_slope_scale(
    x: ArrayLike,
    min_scale: float = 0.9,
    max_scale: float = 1.2,
    batch_equal: bool = True,
    keep_memory: bool = False,
) -> ArrayLike:
    """
    randomly scales the first derivative of x.

    Given the input `ArrayLike` object **x** where the last two dimensions
    refers to the EEG channels and samples (1D tensor are also accepted),
    random_slope_scale calculates the first derivatives of each EEG records,
    here simplified as the difference between two consecutive values of the last
    dimension, and rescale each of them with a random factor selected from a
    uniform distribution between min_scale and max_scale. This transformation
    is similar to adding a random noise, but with the constraint that the first
    derivatives must keep the same sign of the original EEG (e.g. if a value is
    bigger than the previous one, then this is also true in the transformed data).

    Parameters
    ----------
    x: ArrayLike
        The input Tensor or Array.
        The last two dimensions must refer to the EEG recording (Channels x Samples).
    min_scale: float, optional
        The minimum rescaling factor to be applied. It must be a value bigger than 0.

        Default = 0.9
    max_scale: float, optional
        The maximum rescaling factor to be applied. It must be a value bigger
        than min_scale.

        Default = 1.2
    batch_equal: bool, optional
        Whether to apply the same rescale to all EEGs in the batch or not.
        This apply only if x has more than 2 dimensions, i.e. more than 1 EEG.

        Default: True
    keep_memory: bool, optional
        Whether to keep memory of the previous changes in slope and accumulate
        them during the transformation or not. Basically, instead of using:

            ``x_hat(n)= x(n-1) + scaling*( x(n)-x(n-1) )``

        with n>1, x_hat transformed signal, x original signal,
        keep_memory apply the following:

            ``x_hat(n)= x_hat(n-1) + scaling*( x(n)-x(n-1) )``

        Keep in mind that this may completely change the range of values,
        as consecutive increases in the slopes may cause a strong vertical
        shift of the signal. If set to True, it is suggested to set the scaling
        factor in the range [0.8, 1.2]

        Default: False

    Returns
    -------
    x: ArrayLike
        The augmented version of the input Tensor or Array.

    Example
    -------
    >>> import torch
    >>> import selfeeg.augmentation as aug
    >>> x = torch.zeros(16,32,1024) + torch.sin(torch.linspace(0, 8*torch.pi,1024))
    >>> xaug = aug.random_slope_scale(x)
    >>> diff1=torch.abs(xaug[0,0,1:] - xaug[0,0,:-1])
    >>> diff2=torch.abs(x[0,0,1:] - x[0,0,:-1])
    >>> print(
    ...     torch.logical_or(diff1<=(diff2*1.2),
    ...     diff1>=(diff2*0.9)).sum()
    ... ) # should return 1023

    """

    Ndim = len(x.shape)
    if min_scale < 0:
        raise ValueError("minimum scaling factor can't be lower than 0")
    if max_scale <= min_scale:
        raise ValueError("maximum scaling factor can't be" " lower than minimum scaling factor")

    if batch_equal:
        if isinstance(x, np.ndarray):
            scale_factor = np.random.uniform(
                min_scale, max_scale, size=tuple(x.shape[-2:] + np.array([0, -1]))
            )
        else:
            scale_factor = torch.empty(x.shape[-2], x.shape[-1] - 1, device=x.device).uniform_(
                min_scale, max_scale
            )
    else:
        if isinstance(x, np.ndarray):
            scale_factor = np.random.uniform(min_scale, max_scale, x.shape)[..., 1:]
        else:
            scale_factor = torch.empty_like(x, device=x.device)[..., 1:].uniform_(
                min_scale, max_scale
            )

    x_diff = x[..., 1:] - x[..., :-1]
    x_diff_scaled = x_diff * scale_factor
    if isinstance(x, np.ndarray):
        x_new = np.empty_like(x)
    else:
        x_new = torch.empty_like(x, device=x.device)
    x_new[..., 0] = x[..., 0]
    if keep_memory:
        x_new[..., 1:] = (
            np.cumsum(x_diff_scaled, Ndim - 1)
            if isinstance(x, np.ndarray)
            else torch.cumsum(x_diff_scaled, Ndim - 1)
        )
    else:
        x_new[..., 1:] = x[..., :-1] + x_diff_scaled

    return x_new


def new_random_fft_phase_odd(n, to_torch_tensor: bool = False, device="cpu"):
    """
    Method for random_fft_phase with even length vector.
    See ``random_FT_phase`` help.

    :meta private:

    """
    if to_torch_tensor:
        random_phase = 2j * np.pi * torch.rand((n - 1) // 2)
        new_random_phase = torch.cat(
            (torch.tensor([0.0]), random_phase, -torch.flipud(random_phase))
        ).to(device=device)
    else:
        random_phase = 2j * np.pi * np.random.rand((n - 1) // 2)
        new_random_phase = np.concatenate([[0.0], random_phase, -random_phase[::-1]])
    return new_random_phase


def new_random_fft_phase_even(n, to_torch_tensor: bool = False, device="cpu"):
    """
    Method for random_fft_phase with even length vector.
    See ``random_FT_phase`` help.

    :meta private:

    """
    if to_torch_tensor:
        random_phase = 2j * np.pi * torch.rand(n // 2 - 1)
        new_random_phase = torch.cat(
            (torch.tensor([0.0]), random_phase, torch.tensor([0.0]), -torch.flipud(random_phase))
        ).to(device=device)
    else:
        random_phase = 2j * np.pi * np.random.rand(n // 2 - 1)
        new_random_phase = np.concatenate([[0.0], random_phase, [0.0], -random_phase[::-1]])
    return new_random_phase


def random_FT_phase(x: ArrayLike, value: float = 1, batch_equal: bool = True) -> ArrayLike:
    """
    randomizes the phase of all signals in the input ArrayLike object.

    For more info, see [ftphase1]_.

    Parameters
    ----------
    x: ArrayLike
        the input Tensor or Array.
        The last two dimensions must refer to the EEG recording
        (Channels x Samples).
    value: float, optional
        The magnitude of the phase perturbation. It must be a value between
        (0,1], which will be used to rescale the interval
        [0, 2* 'pi'] in [0, value * 2 * 'pi']

        Default = 1
    batch_equal: bool, optional
        Whether to apply the same perturbation on all signals or not.
        Note that all channels of the same records will be perturbed
        in the same way to preserve cross-channel correlations.

        Default = True

    Returns
    -------
    x: ArrayLike
        The augmented version of the input Tensor or Array.

    References
    ----------
    .. [ftphase1] Rommel, Cédric, et al. "Data augmentation for learning predictive
      models on EEG: a systematic comparison."
      Journal of Neural Engineering 19.6 (2022): 066020.

    Example
    -------
    >>> import torch
    >>> import selfeeg.augmentation as aug
    >>> x = torch.zeros(16,32,1024) + torch.sin(torch.linspace(0, 8*torch.pi,1024))
    >>> xaug = aug.random_FT_phase(x, 0.8)
    >>> # see https://dsp.stackexchange.com/questions/87343/
    >>> phase_shift = torch.arccos( 2*((x[0,0,0:512]*xaug[0,0,:512]).mean()) )
    >>> a=torch.sin(torch.linspace(0, 8*torch.pi,1024) + phase_shift)
    >>> if (a[0] - xaug[0,0,0]).abs()>0.1:
    ...     a=torch.sin(torch.linspace(0, 8*torch.pi,1024) - phase_shift)
    >>> print((a - xaug[0,0]).mean()<1e-3)

    plot the results (required matplotlib to be installed)

    >>> plt.plot(x[0,0])
    >>> plt.plot(xaug[0,0])
    >>> plt.plot(a)
    >>> plt.show()

    """
    if value <= 0 or value > 1:
        raise ValueError("value must be a float in range (0,1]")
    Ndim = len(x.shape)
    x_phase = torch.clone(x) if isinstance(x, torch.Tensor) else np.copy(x)
    if batch_equal or Ndim < 3:
        n = x.shape[-1]
        if isinstance(x, torch.Tensor):
            random_phase = (
                new_random_fft_phase_even(n, True, x.device)
                if n % 2 == 0
                else new_random_fft_phase_odd(n, True, x.device)
            )
            FT_coeff = torch.fft.fft(x)
            x_phase = torch.fft.ifft(FT_coeff * torch.exp(value * random_phase)).real
        else:
            if n % 2 == 0:
                random_phase = new_random_fft_phase_even(n)
            else:
                new_random_fft_phase_odd(n)
            FT_coeff = fft.fft(x)
            x_phase = fft.ifft(FT_coeff * np.exp(value * random_phase)).real
    else:
        for i in range(x_phase.shape[0]):
            x_phase[i] = random_FT_phase(x[i], value, batch_equal)
    return x_phase


# ---- FILTERING -----
def moving_avg(x: ArrayLike, order: int = 5) -> ArrayLike:
    """
    applies a moving average filter to the input ArrayLike object.

    Filter is applied along its last dimension.
    The filter order can be given as function argument.

    Parameters
    ----------
    x: ArrayLike
        The input Tensor or Array. The last two dimensions must refer
        to the EEG recording (Channels x Samples).
    order: int, optional
        The order of the filter.

        Default = 5

    Returns
    -------
    x: ArrayLike
        The augmented version of the input Tensor or Array.

    Example
    -------
    >>> import torch
    >>> import selfeeg.augmentation as aug
    >>> x = torch.randn(16,32,1024)
    >>> xaug = aug.moving_avg(x, 5)
    >>> print(
    ...     math.isclose( x[0,0,5:5+5].sum()/5, xaug[0,0,7],
    ...     rel_tol=1e-8)) #Should output True

    """

    if isinstance(x, np.ndarray):
        x_avg = np.empty_like(x)
        filt = np.ones(order) / order
        Ndim = len(x.shape)

        # call recursively to handle different dimensions
        # (made to handle problem with torch conv2d)
        if Ndim > 1:
            for i in range(x.shape[0]):
                x_avg[i] = moving_avg(x[i], order=order)
        else:
            x_avg = np.convolve(x, filt, "same")

    else:
        Ndim = len(x.shape)
        shape = x.shape
        # adapt to x to conv2d functions
        if Ndim == 1:
            x = x.view(1, 1, 1, *x.shape)
        elif Ndim == 2:
            x = x.view(1, 1, *x.shape)
        elif Ndim == 3:
            x = x.unsqueeze(1)
        x_avg = torch.empty_like(x)
        filt = torch.ones((1, 1, 1, order), device=x.device) / order

        # call recursively if the dimension is larger than 4
        if Ndim > 4:
            for i in range(x.shape[0]):
                x_avg[i] = moving_avg(x[i], order=order)
        else:
            if x.shape[1] > 1:
                # to solve problem of conv channel dimension bigger than 1
                for i in range(x.shape[1]):
                    x_avg[..., i, :, :] = F.conv2d(
                        x[..., i, :, :].unsqueeze(1), filt, padding="same"
                    ).squeeze(1)
            else:
                x_avg = F.conv2d(x, filt, padding="same")
            x_avg = torch.reshape(x_avg, shape)

    return x_avg


def get_filter_coeff(
    Fs: float,
    Wp: float,
    Ws: float,
    rp: float = -20 * np.log10(0.95),
    rs: float = -20 * np.log10(0.15),
    btype: str = "lowpass",
    filter_type: str = "butter",
    order: int = None,
    Wn: Union[float, List[float]] = None,
    eeg_band: str = None,
) -> tuple[ArrayLike, ArrayLike]:
    """
    returns the coefficients a and b necessary to run filtering augmentations.

    It return two arrays a and b [coeff1]_ that are then passed to the scipy's
    [coeff2]_ or torchaudio's [coeff3]_ filtfilt function.

    This function is internally called by other filtering functions when a and b
    coefficients are not given as input argument. It works following this
    priority pipeline:

        - if specific EEG bands are given, set Wp, Ws, rp, rs for
          filter design according to the given band
        - if order and Wn are not given, use previous parameter
          to design the filter
        - use Wn and order to get a and b coefficient to return

    In other words the function will check if the following arguments
    were given using this order:

        `(Wp,Ws,rp,rs) --> (Wn, order) --> (a,b)`


    Parameters
    ----------
    Wp: float or ArrayLike
        Passband edges in Hz. It can be a float for lowpass and highpass filters,
        or a length 2 scalar vector for bandpass and stopband filters.
    Ws: float or ArrayLike
        Stopband edges in Hz. It can be a float for lowpass and highpass filters,
        or a length 2 scalar vector for bandpass and stopband filters.
    rp: float, optional
        Ripple at bandpass in dB.

        Default = -20*log10(0.95)
    rs: float, optional
        Ripple at stopband in dB.

        Default = -20*log10(0.15)
    btype: str, optional
        Filter type. It can be any of the scipy's btype argument.
        (e.g. 'lowpass', 'highpass', 'bandpass')

        Default = 'lowpass'
    filter_type: str, optional
        Which filter to design. The accepted values are 'butter',
        'ellip', 'cheby1', 'cheby2'.

        Default = "butter"
    order: int, optional
        The order of the filter.

        Default = None
    Wn: ArrayLike, optional
        The critical frequency or frequencies.

        Default = None
    eeg_band: str, optional
        Any of the possible EEG bands.
        Accepted values are "delta", "theta", "alpha", "beta",
        "gamma", "gamma_low", "gamma_high".

        Default = None
    Fs: float, optional
        The sampling frequency in Hz. It must be given if eeg_band is also given.

        Default = None

    Returns
    -------
    b: ArrayLike
        Array with the numerator coefficients of rational transfer function.
    a: ArrayLike
        Array with the denominator coefficients of rational transfer function.

    References
    ----------
    .. [coeff1] https://docs.scipy.org/doc/scipy/reference/signal.html
    .. [coeff2] https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html
    .. [coeff3] https://pytorch.org/audio/main/generated/torchaudio.functional.filtfilt.html

    """

    if btype.lower() in ["bandpass", "bandstop"]:
        if eeg_band is not None:
            if eeg_band.lower() == "delta":
                Wp, Ws = 4, 8
                rp, rs = -20 * np.log10(0.95), -20 * np.log10(0.1)
                btype = "highpass" if btype.lower() == "bandstop" else "lowpass"
            elif eeg_band.lower() == "theta":
                Wp, Ws = [4.5, 8.5], [0, 17]
                rp, rs = -20 * np.log10(0.9), -20 * np.log10(0.15)
            elif eeg_band.lower() == "alpha":
                Wp, Ws = [8, 13], [4, 22]
                rp, rs = -20 * np.log10(0.95), -20 * np.log10(0.1)
            elif eeg_band.lower() == "beta":
                Wp, Ws = [13, 30], [8, 40]
                rp, rs = -20 * np.log10(0.95), -20 * np.log10(0.15)
            elif eeg_band.lower() == "gamma_low":
                if Fs >= 78 * 2:
                    Wp, Ws = [30, 70], [20, 80]
                    rp, rs = -20 * np.log10(0.95), -20 * np.log10(0.1)
                else:
                    Wp, Ws = 30, 20
                    rp, rs = -20 * np.log10(0.95), -20 * np.log10(0.1)
                    btype = "lowpass" if btype.lower() == "bandstop" else "highpass"
            elif eeg_band.lower() == "gamma_high":
                if Fs >= 158 * 2:
                    Wp, Ws = [70, 150], [60, 160]
                    rp, rs = -20 * np.log10(0.95), -20 * np.log10(0.1)
                else:
                    Wp, Ws = 70, 60
                    rp, rs = -20 * np.log10(0.95), -20 * np.log10(0.1)
                    btype = "lowpass" if btype.lower() == "bandstop" else "highpass"
            elif eeg_band.lower() == "gamma":
                if Fs >= 158 * 2:
                    Wp, Ws = [30, 150], [20, 160]
                    rp, rs = -20 * np.log10(0.95), -20 * np.log10(0.1)
                else:
                    Wp, Ws = 30, 20
                    rp, rs = -20 * np.log10(0.95), -20 * np.log10(0.1)
                    btype = "highpass"
                    if btype.lower() == "bandstop":
                        btype = "lowpass"
            else:
                message = 'Brainwave "', bandwidth[i], '" not exist. \n'
                message += "Choose between delta, theta, alpha, beta, "
                message += "gamma, gamma_low, gamma_high"
                raise ValueError(message)

            if btype.lower() == "bandstop":
                # simply reverse bandpass and stopband
                Wp, Ws = Ws, Wp

    Wp, Ws = np.array(Wp) / (Fs / 2), np.array(Ws) / (Fs / 2)
    if (order is None) or (Wn is None):
        if filter_type.lower() == "butter":
            order, Wn = signal.buttord(Wp, Ws, rp, rs)
        elif filter_type.lower() == "ellip":
            order, Wn = signal.ellipord(Wp, Ws, rp, rs)
        elif filter_type.lower() == "cheby1":
            order, Wn = signal.cheb1ord(Wp, Ws, rp, rs)
        elif filter_type.lower() == "cheby2":
            order, Wn = signal.cheb2ord(Wp, Ws, rp, rs)

    if filter_type.lower() == "butter":
        b, a = signal.butter(order, Wn, btype)
    elif filter_type.lower() == "ellip":
        b, a = signal.ellip(order, rp, rs, Wn, btype)
    elif filter_type.lower() == "cheby1":
        b, a = signal.cheby1(order, rp, Wn, btype)
    elif filter_type.lower() == "cheby2":
        b, a = signal.cheby2(order, rs, Wn, btype)

    return b, a


def filter_lowpass(
    x: ArrayLike,
    Fs: float,
    Wp: float = 50,
    Ws: float = 70,
    rp: float = -20 * np.log10(0.95),
    rs: float = -20 * np.log10(0.15),
    filter_type: str = "butter",
    order: int = None,
    Wn: float = None,
    a: Union[np.ndarray, float] = None,
    b: Union[np.ndarray, float] = None,
    return_filter_coeff: bool = False,
) -> tuple[ArrayLike, Optional[tuple[ArrayLike, ArrayLike]]]:
    """
    applies a lowpass filter on the last dimension of the ArrayLike.

    If a and b coefficient are not given, internally calls ``get_filter_coeff``
    with the other arguments to get them.
    The filter design follows this hierarchical order:

        `(Wp,Ws,rp,rs) --> (Wn, order) --> (a,b)`

    Therefore the arguments closer to a and b in the scheme are used to get
    the filter coefficients.

    Parameters
    ----------
    x: ArrayLike
        The input Tensor or Array.
        The last two dimensions must refer to the EEG recording
        (Channels x Samples).
    Fs: float
        The sampling frequency in Hz.
    Wp: float, optional
        Passband edge in Hz.

        Default = 50
    Ws: float, optional
        Stopband edge in Hz.

        Default = 70
    rp: float, optional
        Ripple at bandpass in dB.

        Default = -20*log10(0.95)
    rs: float, optional
        Ripple at stopband in dB.

        Default = -20*log10(0.15)
    filter_type: str, optional
        Which filter design.  The accepted values are 'butter',
        'ellip', 'cheby1', 'cheby2'.

        Default = 'butter'
    order: int, optional
        The order of the filter.

        Default = None
    Wn: ArrayLike, optional
        The critical frequency or frequencies.

        Default = None
    a: ArrayLike, optional
        The denominator coefficients of the filter.

        Default = None
    b: ArrayLike, optional
        The numerator coefficients of the filer.

        Default = None
    return_filter_coeff: bool, optional
        Whether to return the filter coefficients or not.

        Default = False

    Returns
    -------
    x: ArrayLike
        the augmented version of the input Tensor or Array.
    b: ArrayLike, optional
        Array with the numerator coefficients of rational transfer function.
    a: ArrayLike, optional
        Array with the denominator coefficients of rational transfer function.

    Note
    ----
    Many parameters are those used in scipy's implementation of Matlab-style
    filters, except for **Wp** and **Ws**, which must be specified directly in Hz.
    The normalization to [0,1] with respect to half-cycles/sample
    (i.e., Nyquist frequency) is done directly inside the ``get_filter_coeff``
    function.

    Note
    ----
    Pytorch filtfilt works differently on edges and is pretty unstable
    with high order filters, so avoid restrictive conditions that can
    increase the order of the filter.

    Example
    -------
    >>> import torch
    >>> import numpy as np
    >>> from scipy.signal import periodogram
    >>> import selfeeg.augmentation as aug
    >>> x = torch.zeros(16,32,1024) + torch.sin(torch.linspace(0, 8*torch.pi,1024))
    >>> x += torch.sin(torch.linspace(0, 48*2*torch.pi,1024))
    >>> x += torch.sin(torch.linspace(0, 256*2*torch.pi,1024))
    >>> f, per1 = periodogram(x[0,0], 128)
    >>> xaug = aug.filter_lowpass(x, 128, 20, 30)
    >>> f, per2 = periodogram(xaug[0,0], 128)
    >>> print(np.isclose(np.max(per2[f>30]), 0, rtol=1e-04, atol=1e-04))

    """

    if filter_type not in ["butter", "ellip", "cheby1", "cheby2"]:
        raise ValueError(
            "filter type not supported. Choose between butter, elliptic, cheby1, cheby2"
        )

    if (a is None) or (b is None):
        b, a = get_filter_coeff(
            Fs=Fs,
            Wp=Wp,
            Ws=Ws,
            rp=rp,
            rs=rs,
            btype="lowpass",
            filter_type=filter_type,
            order=order,
            Wn=Wn,
            eeg_band=None,
        )

    if isinstance(x, np.ndarray):
        x_filt = signal.filtfilt(b, a, x, padtype="constant")
    else:
        a = torch.from_numpy(a).to(dtype=x.dtype, device=x.device)
        b = torch.from_numpy(b).to(dtype=x.dtype, device=x.device)
        x_filt = filtfilt(x, a, b, clamp=False)

    if return_filter_coeff:
        return x_filt, b, a
    else:
        return x_filt


def filter_highpass(
    x: ArrayLike,
    Fs: float,
    Wp: float = 30,
    Ws: float = 13,
    rp: float = -20 * np.log10(0.95),
    rs: float = -20 * np.log10(0.15),
    filter_type: str = "butter",
    order: int = None,
    Wn: float = None,
    a: Union[np.ndarray, float] = None,
    b: Union[np.ndarray, float] = None,
    return_filter_coeff: bool = False,
) -> tuple[ArrayLike, Optional[tuple[ArrayLike, ArrayLike]]]:
    """
    applies an highpass filter on the last dimension of the ArrayLike object.

    If a and b coefficient are not given, internally calls ``get_filter_coeff``
    with the other arguments to get them.
    The filter design follows this hierarchical order:

        `(Wp,Ws,rp,rs) --> (Wn, order) --> (a,b)`

    Therefore the arguments closer to a and b in the scheme are used to get
    the filter coefficients.

    Parameters
    ----------
    x: ArrayLike
        the input Tensor or Array.
        The last two dimensions must refer to the EEG recording
        (Channels x Samples).
    Wp: float, optional
        Passband edge in Hz.

        Default = 30
    Ws: float, optional
        Stopband edge in Hz.

        Default = 13
    rp: float, optional
        Ripple at bandpass in dB.

        Default = -20*log10(0.95)
    rs: float, optional
        Ripple at stopband in dB.

        Default = -20*log10(0.15)
    filter_type: str, optional
        Which filter design.  The accepted values are 'butter',
        'ellip', 'cheby1', 'cheby2'

        Default = 'butter'
    order: int, optional
        The order of the filter.

        Default = None
    Wn: ArrayLike, optional
        The critical frequency or frequencies.

        Default = None
    a: ArrayLike, optional
        The denominator coefficients of the filter.

        Default = None
    b: ArrayLike, optional
        The numerator coefficients of the filter.

        Default = None
    return_filter_coeff: bool, optional
        Whether to return the filter coefficients or not.

        Default = False

    Returns
    -------
    x: ArrayLike
        The augmented version of the input Tensor or Array.
    b: ArrayLike, optional
        Array with the numerator coefficients of rational transfer function.
    a: ArrayLike, optional
        Array with the denominator coefficients of rational transfer function.

    Note
    ----
    Many parameters are those used in scipy's implementation of Matlab-style
    filters, except for **Wp** and **Ws**, which must be specified directly in Hz.
    The normalization to [0,1] with respect to half-cycles/sample
    (i.e., Nyquist frequency) is done directly inside the
    ``get_filter_coeff`` function.

    Note
    ----
    Pytorch filtfilt works differently on edges and is pretty unstable
    with high order filters, so avoid restrictive conditions that can
    increase the order of the filter.

    Example
    -------
    >>> import torch
    >>> import numpy as np
    >>> from scipy.signal import periodogram
    >>> import selfeeg.augmentation as aug
    >>> x = torch.zeros(16,32,1024) + torch.sin(torch.linspace(0, 8*torch.pi,1024))
    >>> x += torch.sin(torch.linspace(0, 48*2*torch.pi,1024))
    >>> x += torch.sin(torch.linspace(0, 256*2*torch.pi,1024))
    >>> f, per1 = periodogram(x[0,0], 128)
    >>> xaug = aug.filter_highpass(x, 128, 20, 30)
    >>> f, per2 = periodogram(xaug[0,0], 128)
    >>> print(np.isclose(np.max(per2[f<20]), 0, rtol=1e-04, atol=1e-04))# should return True

    """

    if filter_type not in ["butter", "ellip", "cheby1", "cheby2"]:
        raise ValueError(
            "filter type not supported. " "Choose between butter, elliptic, cheby1, cheby2"
        )

    if (a is None) or (b is None):
        b, a = get_filter_coeff(
            Fs=Fs,
            Wp=Wp,
            Ws=Ws,
            rp=rp,
            rs=rs,
            btype="highpass",
            filter_type=filter_type,
            order=order,
            Wn=Wn,
            eeg_band=None,
        )

    if isinstance(x, np.ndarray):
        x_filt = signal.filtfilt(b, a, x, padtype="constant")
    else:
        a = torch.from_numpy(a).to(dtype=x.dtype, device=x.device)
        b = torch.from_numpy(b).to(dtype=x.dtype, device=x.device)
        x_filt = filtfilt(x, a, b, clamp=False)

    if return_filter_coeff:
        return x_filt, b, a
    else:
        return x_filt


def filter_bandpass(
    x: ArrayLike,
    Fs: float,
    Wp: list[float] = [8, 35],
    Ws: list[float] = [1, 50],
    rp: float = -20 * np.log10(0.9),
    rs: float = -20 * np.log10(0.15),
    filter_type: str = "butter",
    order: int = None,
    Wn: float = None,
    a: Union[np.ndarray, float] = None,
    b: Union[np.ndarray, float] = None,
    eeg_band: str = None,
    return_filter_coeff: bool = False,
) -> tuple[ArrayLike, Optional[tuple[ArrayLike, ArrayLike]]]:
    """
    applies a bandpass filter on the last dimension of the ArrayLike.

    If a and b coefficient are not given, internally calls ``get_filter_coeff``
    with the other arguments to get them.
    The filter design follows this hierarchical order:

        `(Wp,Ws,rp,rs) --> (Wn, order) --> (a,b)`

    Therefore the arguments closer to a and b in the scheme are used to get
    the filter coefficient.

    If ``eeg_band`` is given, (Wp,Ws,rp,rs) are ignored and instantiated according
    to the EEG band specified. The priority order remains. So, if (Wn, order) or
    (a,b) are given, the filter will be created according to such argument.

    Parameters
    ----------
    x: ArrayLike
        the input Tensor or Array.
        The last two dimensions must refer to the EEG recording
        (Channels x Samples).
    Fs: float
        the sampling frequency in Hz.
    Wp: ArrayLike, optional
        Passband edges in Hz. It must be a length 2 scalar vector.

        Default = None
    Ws: ArrayLike, optional
        Stopband edges in Hz. It must be a length 2 scalar vector.

        Default = None
    rp: float, optional
        Ripple at bandpass in dB.

        Default = -20*log10(0.95)
    rs: float, optional
        Ripple at stopband in dB.

        Default = -20*log10(0.15)
    filter_type: str, optional
        Which filter design. Accepted values are 'butter', 'ellip',
        'cheby1', 'cheby2'

        Default = "butter"
    order: int, optional
        The order of the filter.

        Default = None
    Wn: ArrayLike, optional
        The critical frequency or frequencies.

        Default = None
    a: ArrayLike, optional
        The denominator coefficients of the filter

        Default = None
    b: ArrayLike, optional
        The numerator coefficients of the filer

        Default = None
    eeg_band: str, optional
        Any of the possible EEG bands. Accepted values are "delta",
        "theta", "alpha", "beta", "gamma", "gamma_low", "gamma_high".
        Note that eeg_band bypass any (Wp, Ws, rp, rs)
        if given.

        Default = None
    return_filter_coeff: bool, optional
        whether to return the filter coefficient or not
        Default: False

    Returns
    -------
    x: ArrayLike
        The augmented version of the input Tensor or Array.
    b: ArrayLike, optional
        Array with the numerator coefficients of rational transfer function.
    a: ArrayLike, optional
        Array with the denominator coefficients of rational transfer function.

    Note
    ----
    Many parameters are those used in scipy's implementation of Matlab-style
    filters, except for **Wp** and **Ws**, which must be specified directly in Hz.
    The normalization to [0,1] with respect to half-cycles/sample
    (i.e., Nyquist frequency) is done directly inside the
    ``get_filter_coeff`` function.

    Note
    ----
    Pytorch filtfilt works differently on edges and is pretty unstable
    with high order filters, so avoid restrictive conditions that can
    increase the order
    of the filter.

    Example
    -------
    >>> import torch
    >>> import numpy as np
    >>> from scipy.signal import periodogram
    >>> import selfeeg.augmentation as aug
    >>> x = torch.zeros(16,32,1024) + torch.sin(torch.linspace(0, 8*torch.pi,1024))
    >>> x += torch.sin(torch.linspace(0, 48*2*torch.pi,1024))
    >>> x += torch.sin(torch.linspace(0, 256*2*torch.pi,1024))
    >>> f, per1 = periodogram(x[0,0], 128)
    >>> xaug = aug.filter_bandpass(x, 128, [13,22], [5,27])
    >>> f, per2 = periodogram(xaug[0,0], 128)
    >>> print(np.isclose(np.max(per2[f<5]), 0, rtol=1e-04, atol=1e-04))# should return True
    >>> print(np.isclose(np.max(per2[f>27]), 0, rtol=1e-04, atol=1e-04))# should return True

    """

    if filter_type not in ["butter", "ellip", "cheby1", "cheby2"]:
        raise ValueError(
            "filter type not supported. " "Choose between butter, elliptic, cheby1, cheby2"
        )

    if (a is None) or (b is None):
        b, a = get_filter_coeff(
            Fs=Fs,
            Wp=Wp,
            Ws=Ws,
            rp=rp,
            rs=rs,
            btype="bandpass",
            filter_type=filter_type,
            order=order,
            Wn=Wn,
            eeg_band=eeg_band,
        )

    if isinstance(x, np.ndarray):
        x_filt = signal.filtfilt(b, a, x, padtype="constant")
    else:
        a = torch.from_numpy(a).to(dtype=x.dtype, device=x.device)
        b = torch.from_numpy(b).to(dtype=x.dtype, device=x.device)
        x_filt = filtfilt(x, a, b, clamp=False)

    if return_filter_coeff:
        return x_filt, b, a
    else:
        return x_filt


def filter_bandstop(
    x: ArrayLike,
    Fs: float,
    Wp: list[float] = [0, 15],
    Ws: list[float] = [4, 8],
    rp: float = -20 * np.log10(0.95),
    rs: float = -20 * np.log10(0.1),
    filter_type: str = "butter",
    order: int = None,
    Wn: float = None,
    a: Union[np.ndarray, float] = None,
    b: Union[np.ndarray, float] = None,
    eeg_band: str = None,
    return_filter_coeff: bool = False,
):
    """
    applies a bandstop filter on the last dimension of the ArrayLike object.

    If a and b coefficient are not given, internally calls ``get_filter_coeff``
    with the other arguments to get them.
    The filter design follows this hierarchical order:

        `(Wp,Ws,rp,rs) --> (Wn, order) --> (a,b)`

    Therefore the arguments closer to a and b in the scheme are used to get
    the filter coefficients.

    If ``eeg_band`` is given, (Wp,Ws,rp,rs) are bypassed and instantiated according
    to the eeg band specified. The priority order remain, so if (Wn,order) or
    (a,b) are given, the filter will be created according to such arguments.

    Parameters
    ----------
    x: ArrayLike
        the input Tensor or Array.
        The last two dimensions must refer to the EEG recording
        (Channels x Samples).
    Fs: float
        The sampling frequency in Hz.
    Wp: float, optional
        Passband edges in Hz. It must be a length 2 scalar vector.

        Default = 30
    Ws: float, optional
        Stopband edges in Hz. It must be a length 2 scalar vector.

        Default = 13
    rp: float, optional
        Ripple at bandpass in dB.

        Default = -20*log10(0.95)
    rs: float, optional
        Ripple at stopband in dB.

        Default = -20*log10(0.15)
    filter_type: str, optional
        Which filter design. Accepted values are 'butter',
        'ellip', 'cheby1', 'cheby2'

        Default = "butter"
    order: int, optional
        The order of the filter.

        Default = None
    Wn: ArrayLike, optional
        The critical frequency or frequencies.

        Default = None
    a: ArrayLike, optional
        The denominator coefficients of the filter.

        Default = None
    b: ArrayLike, optional
        The numerator coefficients of the filer.

        Default = None
    eeg_band: str, optional
        Any of the possible EEG bands. Accepted values are "delta",
        "theta", "alpha", "beta", "gamma", "gamma_low", "gamma_high".
        Note that eeg_band bypass any (Wp, Ws, rp, rs), if given.

        Default = None
    return_filter_coeff: bool, optional
        Whether to return the filter coefficients or not.

        Default= False

    Returns
    -------
    x: ArrayLike
        the augmented version of the input Tensor or Array.
    b: ArrayLike, optional
        Array with the numerator coefficients of rational transfer function.
    a: ArrayLike, optional
        Array with the denominator coefficients of rational transfer function.

    Note
    ----
    Many parameters are those used in scipy's implementation of Matlab-style
    filters, except for **Wp** and **Ws**, which must be specified directly in Hz.
    The normalization to [0,1] with respect to half-cycles/sample
    (i.e., Nyquist frequency) is done directly inside the
    ``get_filter_coeff`` function.

    Note
    ----
    Pytorch filtfilt works differently on edges and is pretty unstable
    with high order filters, so avoid restrictive conditions that can
    increase the order

    Example
    -------
    >>> import torch
    >>> import numpy as np
    >>> from scipy.signal import periodogram
    >>> import selfeeg.augmentation as aug
    >>> x = torch.zeros(16,32,1024) + torch.sin(torch.linspace(0, 8*torch.pi,1024))
    >>> x += torch.sin(torch.linspace(0, 48*2*torch.pi,1024))
    >>> x += torch.sin(torch.linspace(0, 256*2*torch.pi,1024))
    >>> f, per1 = periodogram(x[0,0], 128)
    >>> xaug = aug.filter_bandpass(x, 128, [13,22], [5,27])
    >>> f, per2 = periodogram(xaug[0,0], 128)
    >>> print(np.isclose(np.max(per2[np.logical_and(f>5, f<27)]),
    ...                         0, rtol=1e-04, atol=1e-04))
    >>> # should return True

    """

    if filter_type not in ["butter", "ellip", "cheby1", "cheby2"]:
        raise ValueError(
            "filter type not supported. " "Choose between butter, elliptic, cheby1, cheby2"
        )

    if (a is None) or (b is None):
        b, a = get_filter_coeff(
            Fs=Fs,
            Wp=Wp,
            Ws=Ws,
            rp=rp,
            rs=rs,
            btype="bandstop",
            filter_type=filter_type,
            order=order,
            Wn=Wn,
            eeg_band=eeg_band,
        )
    if isinstance(x, np.ndarray):
        x_filt = signal.filtfilt(b, a, x, padtype="constant")
    else:
        a = torch.from_numpy(a).to(dtype=x.dtype, device=x.device)
        b = torch.from_numpy(b).to(dtype=x.dtype, device=x.device)
        x_filt = filtfilt(x, a, b, clamp=False)

    if return_filter_coeff:
        return x_filt, b, a
    else:
        return x_filt


# --- PERMUTATIONS ---
def get_eeg_channel_network_names():
    """
    prints the name of each channel included in the default networks.


    This function clarify the sets of network used in the
    ``permute_channels`` function.

    """

    # fmt: off
    DMN = np.array(
        [
            "AF4", "AF7", "AF8", "AFZ", "CP3", "CP4", "CP5", "F1", "F2", "F3",
            "F4", "F5", "F6", "F7", "F8", "FC1", "FC3", "FC4", "FC5", "FP1",
            "FP2", "FPZ", "FT10", "FT8", "FT9", "FZ", "P3", "P4", "P5", "T7",
            "T8", "TP7", "TP8",
        ],
        dtype="<U4",
    )
    DAN = np.array(
        [
            "C5", "C6", "CP1", "CP2", "CPZ", "FC1", "FC2", "FC5", "P1", "P2",
            "P7", "P8", "PO3", "PO4", "PO7", "PO8", "POZ", "PZ", "T7", "TP8",
        ],
        dtype="<U4",
    )
    VAN = np.array(
        [
            "AF3", "AF4", "AF8", "C5", "C6", "CP1", "CP2", "CP4", "CP5", "CP6",
            "CPZ", "F7", "F8", "FC1", "FC2", "FC5", "FC6", "FT7", "P1", "P2",
            "P7", "P8", "PO3", "PO4", "PO7", "PO8", "POZ", "PZ", "T7", "TP8",
        ],
        dtype="<U4",
    )
    SMN = np.array(
        [
            "C1", "C2", "C3", "C4", "C5", "C6", "CP1", "CP2", "CP5", "CPZ",
            "CZ", "FC5", "FC6", "FCZ", "FT8", "FTZ", "P3", "P5", "P6", "P7",
            "P8", "PO4", "PO7", "PO8", "T7", "T8", "TP7",
        ],
        dtype="<U4",
    )
    VFN = np.array(
        [
            "C1", "C2", "C3", "C4", "C5", "C6", "CP1", "CP2", "CPZ", "FC1", "FC5",
            "FC6", "FT8", "FTZ", "O1", "O2", "OZ", "P7", "P8", "PO3", "PO4", "PO7",
            "PO8", "POZ", "PZ", "T7", "TP8",
        ],
        dtype="<U4",
    )
    FPN = np.array(
        [
            "AF3", "AF4", "AF7", "AF8", "AFZ", "C6", "CP3", "CP4", "CP5", "CP6",
            "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "FC1", "FC2", "FC3",
            "FC4", "FC5", "FC6", "FP1", "FP2", "FPZ", "FT10", "FT7", "FT9", "FZ",
            "P3", "P4", "P5", "T7", "T8", "TP7", "TP8",
        ],
        dtype="<U4",
    )
    # fmt: on

    print("Default Mode Network - DMN")
    print(DMN)
    print("")
    print("Dorsal Attention Network - DAN")
    print(DAN)
    print("")
    print("Ventral Attention Network - VAN")
    print(VAN)
    print("")
    print("SomatoMotor functional Network - SMN")
    print(SMN)
    print("")
    print("Visual Functional Network - VFN")
    print(VFN)
    print("")
    print("FrontoParietal Network - FPN")
    print(FPN)
    print("")


def get_channel_map_and_networks(
    channel_map: list = None,
    chan_net: list[str] = "all",
) -> tuple[ArrayLike, list[ArrayLike]]:
    """
    returns the **channel_map** and the **chan_net** arguments for
    ``permute_channels``.

    See the ``permute_channels`` help for more info.

    Parameters
    ----------
    channel_map: list, optional
        A list with all the EEG channel names, given as strings in upper case
        (e.g. "FP1", "CZ", "C1"). Channel names must refer to the 10-10
        international system. If left to None, the following 61 channel
        map will be initialized:

            - ['FP1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7',
              'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5',
              'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'PO7', 'PO3', 'O1', 'OZ',
              'POZ', 'PZ', 'CPZ', 'FPZ', 'FP2', 'AF8', 'AF4', 'AFZ', 'FZ', 'F2',
              'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCZ', 'CZ', 'C2',
              'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4',
              'CP2', 'P2', 'P4', 'P6', 'P8', 'PO8', 'PO4', 'O2']

        Default = None
    chan_net: list[str], optional
        A list of strings with the brain network acronyms.
        It will be used to select the subset of channels to permute between
        each other if ``permute_channels`` called in **network** mode.

        Default = 'all'

    Returns
    -------
    channel_map: ArrayLike
        The EEG Channel Map as a dtype "U4" numpy array.
    chan_net: list[ArrayLike]
        A list with all the channels network to be used during
        the channel permutation.

    Note
    ----
    This function is internally called by ``permute_channels``.

    """

    # fmt: off
    if channel_map is None:
        channel_map = np.array(
            [
                "FP1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3",
                "FC1", "C1", "C3", "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1",
                "P3", "P5", "P7", "PO7", "PO3", "O1", "OZ", "POZ", "PZ", "CPZ",
                "FPZ", "FP2", "AF8", "AF4", "AFZ", "FZ", "F2", "F4", "F6", "F8",
                "FT8", "FC6", "FC4", "FC2", "FCZ", "CZ", "C2", "C4", "C6", "T8",
                "TP8", "CP6", "CP4", "CP2", "P2", "P4", "P6", "P8", "PO8", "PO4",
                "O2",
            ],
            dtype="<U4",
        )
    elif isinstance(channel_map, list):
        channel_map = np.array(channel_map, dtype="<U4")

    DMN = np.array(
        [
            "AF4", "AF7", "AF8", "AFZ", "CP3", "CP4", "CP5", "F1", "F2", "F3", "F4",
            "F5", "F6", "F7", "F8", "FC1", "FC3", "FC4", "FC5", "FP1", "FP2", "FPZ",
            "FT10", "FT8", "FT9", "FZ", "P3", "P4", "P5", "T7", "T8", "TP7", "TP8",
        ],
        dtype="<U4",
    )
    DAN = np.array(
        [
            "C5", "C6", "CP1", "CP2", "CPZ", "FC1", "FC2", "FC5", "P1", "P2",
            "P7", "P8", "PO3", "PO4", "PO7", "PO8", "POZ", "PZ", "T7", "TP8",
        ],
        dtype="<U4",
    )
    VAN = np.array(
        [
            "AF3", "AF4", "AF8", "C5", "C6", "CP1", "CP2", "CP4", "CP5", "CP6",
            "CPZ", "F7", "F8", "FC1", "FC2", "FC5", "FC6", "FT7", "P1", "P2", "P7",
            "P8", "PO3", "PO4", "PO7", "PO8", "POZ", "PZ", "T7", "TP8",
        ],
        dtype="<U4",
    )
    SMN = np.array(
        [
            "C1", "C2", "C3", "C4", "C5", "C6", "CP1", "CP2", "CP5", "CPZ", "CZ",
            "FC5", "FC6", "FCZ", "FT8", "FTZ", "P3", "P5", "P6", "P7", "P8", "PO4",
            "PO7", "PO8", "T7", "T8", "TP7",
        ],
        dtype="<U4",
    )
    VFN = np.array(
        [
            "C1", "C2", "C3", "C4", "C5", "C6", "CP1", "CP2", "CPZ", "FC1", "FC5",
            "FC6", "FT8", "FTZ", "O1", "O2", "OZ", "P7", "P8", "PO3", "PO4", "PO7",
            "PO8", "POZ", "PZ", "T7", "TP8",
        ],
        dtype="<U4",
    )
    FPN = np.array(
        [
            "AF3", "AF4", "AF7", "AF8", "AFZ", "C6", "CP3", "CP4", "CP5", "CP6",
            "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "FC1", "FC2", "FC3",
            "FC4", "FC5", "FC6", "FP1", "FP2", "FPZ", "FT10", "FT7", "FT9", "FZ",
            "P3", "P4", "P5", "T7", "T8", "TP7", "TP8",
        ],
        dtype="<U4",
    )
    # fmt: on
    networks = [DMN, DAN, VAN, SMN, VFN, FPN]

    if isinstance(chan_net, str):
        chan_net = [chan_net]

    net_idx = []
    for i in range(len(chan_net)):
        if chan_net[i].lower() == "all":
            net_idx = [0, 1, 2, 3, 4, 5]
            break
        elif chan_net[i].lower() == "dmn":
            net_idx.append(0)
        elif chan_net[i].lower() == "dan":
            net_idx.append(1)
        elif chan_net[i].lower() == "van":
            net_idx.append(2)
        elif chan_net[i].lower() == "smn":
            net_idx.append(3)
        elif chan_net[i].lower() == "vfn":
            net_idx.append(4)
        elif chan_net[i].lower() == "fpn":
            net_idx.append(5)
        else:
            msg = "brain network not supported. Use DMN, DAN, VAN, SMN, VFN, FPN"
            raise ValueError(msg)

    for index in sorted(set([0, 1, 2, 3, 4, 5]) - set(net_idx), reverse=True):
        networks.pop(index)
    random.shuffle(networks)

    return channel_map, networks


def permute_channels(
    x: ArrayLike,
    chan2shuf: int = -1,
    mode: str = "random",
    channel_map: list = None,
    chan_net: list[str] = "all",
    batch_equal: bool = True,
) -> ArrayLike:
    """
    permutes the ArrayLike object along the EEG channel dimension.

    Given an input x where the last two dimensions must be
    (EEG_channels x EEG_samples), permutation_channels shuffles
    all or a subset of the EEG's channels. Shuffles can be done randomly
    or using specific networks (based on resting state functional
    connectivity networks).

    Parameters
    ----------
    x: ArrayLike
        The input Tensor or Array.
        The last two dimensions must refer to the EEG recording
        (Channels x Samples). Thus, permutation is applied on the
        second to last dimension.
    chan2shuf: int, optional
        The number of channels to shuffle. It must be greater than 1.
        Only exception is -1, which can be given to permute all the channels.

        Default = -1
    mode: str, optional
        How to permute the channels. Can be:

            - 'random': shuffle channels at random
            - 'network': shuffle channels which belongs to the same network.
              A network is a subset of channels whose activity is
              (with a minumum degree) correlated between each other.
              This mode support only a subset of 61
              channels of the 10-10 system.

        Default = "random"
    channel_map: list[str], optional
        The EEG channel map. Must be a list of strings
        or a numpy array of dtype='<U4' with channel names as elements.
        Channel name must be defined with capital letters (e.g. 'P04', 'FC5').
        If None is left the following 61 channel map is initialized:

            - ['FP1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5',
              'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1',
              'P1', 'P3', 'P5', 'P7', 'PO7', 'PO3', 'O1', 'OZ', 'POZ', 'PZ',
              'CPZ', 'FPZ', 'FP2', 'AF8', 'AF4', 'AFZ', 'FZ', 'F2', 'F4', 'F6',
              'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCZ', 'CZ', 'C2',
              'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4',
              'CP2', 'P2', 'P4', 'P6', 'P8', 'PO8', 'PO4', 'O2']

        Default = None
    chan_net: str or list[str], optional
        The list of networks to use if network mode is selected.
        Must be a list of string or a single string.
        Supported networks are "DMN", "DAN", "VAN", "SMN", "VFN", "FPN".
        Use 'all' to select all networks. To get a list of
        the channel names per network call the
        ``get_eeg_network_channel_names`` function.

        Default = 'all'
    batch_equal: bool, optional
        whether to apply the same permutation to all EEG record or not.
        If True, permute_signal is called recursively in order
        to permute each EEG differently.

        Default = True

    Returns
    -------
    x: ArrayLike
        The augmented version of the input Tensor or Array.

    Warnings
    --------
    Using **chan2shuf = -1** and **mode = 'network'** can result in a lower number
    of channels permuted compared to the whole list of channels included in the
    networks. This is due to the implementation of the permutation block in
    network mode, which iteratively applies the channel permutation at each
    single network, sequentially excluding channels already permuted in previous
    steps (networks are not mutually exclusive, there are overlapping channels).
    At some point a network may remain with only one channel to permute, which
    cannot be permuted since it is alone. This applies only in contexts where
    the number of channel is near the number of the selected ones.

    See Also
    --------
    get_channel_map_and_networks : creates the channel map and networks arrays.
    get_eeg_channel_network_names : prints the channel network arrays.

    Example
    -------
    >>> import torch
    >>> import numpy as np
    >>> import selfeeg.augmentation as aug
    >>> x = torch.zeros(61,4) + torch.arange(61).reshape(61,1)
    >>> xaug = aug.permute_channels(x,10)
    >>> print( (x[:,0]!=xaug[:,0]).sum()) # should output 10
    >>> # Try with network mode and check if channels not in the selected networks were permuted
    >>> eeg1010, chan_net = aug.get_channel_map_and_networks(chan_net=["DMN","VFN"])
    >>> chan2per = np.union1d(chan_net[0], chan_net[1])
    >>> a=np.intersect1d(eeg1010, chan2per, return_indices=True)[1]
    >>> b=torch.from_numpy( np.delete(np.arange(61),a))
    >>> xaug2 = aug.permute_channels(x,50, mode='network', chan_net=["DMN","VFN"])
    >>> print( ((x[:,0]!=xaug2[:,0]).sum())==50) # should output True
    >>> print( ((x[b,0]==xaug2[b,0]).sum())==len(b) ) # should output True

    """
    Nchan = x.shape[-2]
    Ndim = len(x.shape)

    # Check if given input is ok
    if (chan2shuf == 1) or (chan2shuf == 0) or (chan2shuf > Nchan):
        msgLog = """
        chan2shuf must be bigger than 1 and smaller than the number of channels
        in the recorded EEG. \n Default value is -1, which means all EEG channels
        are shuffled
        """
        raise ValueError(msgLog)
    if Ndim == 1:
        msg = "the last two dimensions of x must be [channel]*[time window]"
        raise ValueError(msg)

    chan2shuf = x.shape[-2] if chan2shuf == -1 else chan2shuf
    x2 = np.copy(x) if isinstance(x, np.ndarray) else torch.clone(x)
    if (Ndim < 3) or (batch_equal):

        if mode.lower() == "network":
            # Define or check channel map and channel networks
            channel_map, networks = get_channel_map_and_networks(channel_map, chan_net)
            if channel_map.shape[0] != x.shape[-2]:
                raise ValueError(
                    "channel map does not match the " "number of channels in eeg recording"
                )

            # get index of channels in networks
            is_in_channel_map = np.full(channel_map.shape, False)
            for i in range(len(networks)):
                idx = np.where(np.isin(channel_map, networks[i]))[0]
                is_in_channel_map[idx] = True
            chan_idx = is_in_channel_map.nonzero()[0]

            # randomly select a number of channels equals to chan2shuf
            idxor_full = np.random.permutation(chan_idx)[:chan2shuf]
            idxor_full = np.sort(idxor_full)
            idx_full = np.full(chan2shuf, -1, dtype=int)

            # shuffle according to the selected networks
            for k in range(len(networks)):
                # identify chans idx
                idxor = np.where(np.isin(channel_map[idxor_full], networks[k]))[0]
                idxor = idxor[np.where(idx_full[idxor] == -1)[0]]
                idx = idxor_full[idxor]  # get chans idx
                if idx.shape[0] > 1:
                    while len(np.where(idx == idxor_full[idxor])[0]) > 0:
                        np.random.shuffle(idx)
                idx_full[idxor] = idx

            # final results
            idxor = idxor_full
            idx = idx_full
            if not (isinstance(x, np.ndarray)):
                idxor = torch.from_numpy(idxor_full).to(device=x.device)
                idx = torch.from_numpy(idx_full).to(device=x.device)

        # random mode shuffle channels at random
        elif mode.lower() == "random":
            if isinstance(x, np.ndarray):
                idx = np.arange(Nchan, dtype=int)
                np.random.shuffle(idx)
                idx = idx[:chan2shuf]
                idxor = np.sort(idx)
                if len(idx) > 1:
                    while len(np.where(idx == idxor)[0]) > 0:
                        np.random.shuffle(idx)
            else:
                idx = torch.randperm(Nchan)
                idx = idx[:chan2shuf]
                idxor, _ = torch.sort(idx)
                if len(idx) > 1:
                    while torch.sum(torch.eq(idx, idxor)) != 0:
                        idx = idx[torch.randperm(idx.shape[0])]
                if x.device.type != "cpu":
                    idx = idx.to(device=x.device)
                    idxor = idxor.to(device=x.device)

        # solve problems related to network not having enough channel to permute
        # compared to the desired
        if idxor.shape != idx.shape:
            idx = idx[: idxor.shape[-1]]
        # apply defined shuffle
        xtemp = x[..., idx, :]
        x2[..., idxor, :] = xtemp

    else:
        # call recursively for each dimension until last 2 are reached
        for i in range(x.shape[0]):
            x2[i] = permute_channels(
                x[i],
                chan2shuf=chan2shuf,
                mode=mode,
                channel_map=channel_map,
                chan_net=chan_net,
            )

    return x2


def permutation_signal(
    x: ArrayLike, segments: int = 10, seg_to_per: int = -1, batch_equal: bool = True
) -> ArrayLike:
    """
    permutes some portions of the ArrayLike object along its last dimension.

    Given an input x where last two dimensions must refer to the EEG recording
    (Channels x Samples), ``permutation_signal`` divides the elements of the last
    dimension of x into N segments, then chooses M<=N segments and shuffle them.
    Permutations are equally performed along each Channel of the same EEG.

    Parameters
    ----------
    x: ArrayLike
        The input Tensor or Array.
        The last two dimensions must refer to the EEG recording (Channels x Samples).
    segments: int, optional
        The number of segments in which the last dimension of the input
        `ArrayLike` object **x** must be divided.
        Must be an integer greater than 1.

        Default = 1
    seg_to_per: int, optional
        The number of segments to permute. Must be an integer greater than 1 and
        lower than segments. -1 can be used to permute all the segments.

        Default = -1
    batch_equal: bool, optional
        Whether to apply the same permutation to all EEG record or not.
        If True, the function is called recursively in order to apply a different
        permutation to all EEGs.

        Default = True

    Returns
    -------
    x: ArrayLike
        The augmented version of the input Tensor or Array.

    Example
    -------
    >>> import torch
    >>> import selfeeg.augmentation as aug
    >>> torch.manual_seed(1234)
    >>> x = torch.ones(16,32,1024)*2 + torch.sin(torch.linspace(0, 8*torch.pi,1024))
    >>> xaug = aug.masking(x, 3, 0.5)
    >>> print( torch.isclose(
    ...     ((xaug[0,0]==0).sum()/len(xaug[0,0])) ,
    ...     torch.tensor([0.5]), rtol=1e-8,atol=1e-8) ) # should return True
    >>> a = xaug[0,0]==0
    >>> print( (a[:-1].ne(a[1:])).sum()==6) # should return True


    """

    if segments < 1:
        raise ValueError("segments cannot be less than 2")
    if seg_to_per < 1:
        if seg_to_per >= 0:
            raise ValueError("seg_to_per must be greater than 1 (use -1 to permute all segments)")
        elif seg_to_per == -1:
            seg_to_per = segments
        elif seg_to_per < (-1):
            raise ValueError(
                "got a negative number of segments to permute. "
                "Only -1 to permute all segments is allowed"
            )
    elif seg_to_per > segments:
        raise ValueError("number of segment to permute is bigger than the number of segment")

    Ndim = len(x.shape)
    if (Ndim < 3) or (batch_equal):

        segment_len = x.shape[-1] // segments
        idx1 = np.arange(segments)
        np.random.shuffle(idx1)
        idx2 = np.sort(idx1[:seg_to_per])
        idx1 = np.sort(idx1)
        idx3 = np.copy(idx2)
        while len(np.where(idx2 == idx3)[0]) > 0:
            np.random.shuffle(idx3)
        idx1[idx2] = idx3
        full_idx = np.arange(x.shape[-1])
        for k in range(len(idx1)):
            if idx1[k] != k:
                start = segment_len * idx1[k]
                start2 = segment_len * k
                newidx = np.arange(start, start + segment_len)
                full_idx[start2 : start2 + segment_len] = newidx

        if not (isinstance(x, np.ndarray)):
            full_idx2 = torch.from_numpy(full_idx).to(device=x.device)
            x2 = x[..., full_idx2]
        else:
            x2 = x[..., full_idx]

    else:
        x2 = np.empty_like(x) if isinstance(x, np.ndarray) else torch.empty_like(x)
        for i in range(x.shape[0]):
            x2[i] = permutation_signal(
                x[i], segments=segments, seg_to_per=seg_to_per, batch_equal=batch_equal
            )

    return x2


# --- CROP AND RESIZE ---
def warp_signal(
    x: ArrayLike,
    segments: int = 10,
    stretch_strength: float = 2.0,
    squeeze_strength: float = 0.5,
    batch_equal: bool = True,
) -> ArrayLike:
    """
    stretches and squeezes portions of the ArrayLike object.

    The function is applied along the last dimension of the input ArrayLike object.
    To do that warp_signal:

        1. divides the last dimension of x into N segments
        2. selects at random a subset segments
        3. stretches those segments according to stretch_strength
        4. squeezes other segments according to squeeze_strength
        5. resamples x to the original dimension. For this part pchip
           interpolation with a uniform virtual grid is used

    Parameters
    ----------
    x: ArrayLike
        The input Tensor or Array. The last two dimensions must refer to the
        EEG recording (Channels x Samples).
    segments : int, optional
        The number of segments to consider when dividing the last dimension of x.

        Default = 10
    stretch_strength : float, optional
        The stretch power, i.e. a multiplication factor which determines the number
        of samples the stretched segment must have.

        Default = 2.
    squeeze_strength : float, optional
        The squeeze power. The same as stretch but for the segments to squeeze.

        Default = 0.5
    batch_equal: bool, optional
        Whether to apply the same warp to all records or not.

        Default = True

    Returns
    -------
    x: ArrayLike
        The augmented version of the input Tensor or Array.

    Example
    -------
    >>> import torch
    >>> import selfeeg.augmentation as aug
    >>> x = torch.zeros(16,32,1024) + torch.sin(torch.linspace(0, 8*torch.pi,1024))
    >>> xaug = aug.warp_signal(x,20)

    """
    Ndim = len(x.shape)
    if isinstance(x, np.ndarray):
        x_warped_final = np.empty_like(x)
    else:
        x_warped_final = torch.empty_like(x, device=x.device)

    if batch_equal or Ndim < 3:

        # set segment do stretch or squeeze
        seglen = x.shape[-1] / segments
        seg_range = np.arange(segments)
        stretch = np.random.choice(seg_range, random.randint(1, segments // 2), replace=False)
        squeeze = np.setdiff1d(seg_range, stretch)

        # pre-allocate warped vector to avoid continuous stack call
        Lseg = np.zeros((segments, 2), dtype=int)
        Lseg[:, 0] = (seg_range * seglen).astype(int)
        Lseg[:, 1] = ((seg_range + 1) * seglen).astype(int)
        Lseg = Lseg[:, 1] - Lseg[:, 0]
        Lsegsum = np.zeros(segments + 1)
        Lsegsum[1:] = np.cumsum(Lseg)

        # iterate over segments and stretch or squeeze each segment, then allocate to x_warped
        new_tgrid = [None] * (segments)
        for i in range(segments):
            if i in stretch:
                new_piece_dim = int(np.ceil(Lseg[i] * stretch_strength))
            else:
                new_piece_dim = int(np.ceil(Lseg[i] * squeeze_strength))
            if isinstance(x, torch.Tensor):
                L = torch.arange(math.ceil(new_piece_dim))
            else:
                L = np.arange(math.ceil(new_piece_dim))
            L = L * (Lsegsum[i + 1] - Lsegsum[i]) / (len(L) - 1) + Lsegsum[i]
            new_tgrid[i] = L[:-1] if i < segments - 1 else L

        if isinstance(x, np.ndarray):
            new_tgrid = np.concatenate(new_tgrid)
            old_tgrid = np.arange(x.shape[-1])
            x_warped = interpolate.pchip_interpolate(old_tgrid, x, new_tgrid, axis=-1)
            new_tgrid = np.linspace(0, old_tgrid[-1], x_warped.shape[-1])
            x_warped_final = interpolate.pchip_interpolate(new_tgrid, x_warped, old_tgrid, axis=-1)
        else:
            device = x.device
            new_tgrid = torch.cat(new_tgrid).to(device=device)
            old_tgrid = torch.arange(x.shape[-1]).to(device=device)
            x_warped = torch_pchip(old_tgrid, x, new_tgrid)
            new_tgrid = torch.linspace(0, old_tgrid[-1], x_warped.shape[-1]).to(device=device)
            x_warped_final = torch_pchip(new_tgrid, x_warped, old_tgrid)

    else:
        # Recursively call until second to last dim is reached
        for i in range(x.shape[0]):
            x_warped_final[i] = warp_signal(
                x[i], segments, stretch_strength, squeeze_strength, batch_equal
            )

    return x_warped_final


def crop_and_resize(
    x: ArrayLike,
    segments: int = 10,
    N_cut: int = 1,
    batch_equal: bool = True,
) -> ArrayLike:
    """
    crops some segments of the ArrayLike object.

    Function is applied along the last dimension of the input ArrayLike object and
    is resized to its original dimension. To do that, crop_and_resize:

        1. divides the last dimension of x into N segments
        2. selects at random a subset segments
        3. removes the selected segments from x
        4. creates a new cropped version of x
        5. resamples the new cropped version to the original dimension.
           For this part pchip interpolation with a uniform virtual grid is used.

    Parameters
    ----------
    x: ArrayLike
        the input Tensor or Array.
        The last two dimensions must refer to the EEG recording (Channels x Samples).
    segments : int, optional
        The number of segments to consider when dividing the last dimension of x.
        This is not the number of segments to cut, but the number of segments in
        which the signal is partitioned (a subset of these segments will be
        removed based on N_cut).

        Default = 10
    N_cut : int, optional
        The number of segments to cut, i.e. the number of segments to remove.

        Default = 1
    batch_equal: bool, optional
        Whether to apply the same crop to all EEG record or not.

        Default = True

    Returns
    -------
    x: ArrayLike
        The augmented version of the input Tensor or Array.

    Example
    -------
    >>> import torch
    >>> import selfeeg.augmentation as aug
    >>> x = torch.zeros(16,32,1024) + torch.sin(torch.linspace(0, 8*torch.pi,1024))
    >>> xaug = aug.crop_and_resize(x,32, 15)

    """
    if isinstance(x, np.ndarray):
        x_crop = np.empty_like(x)
    else:
        x_crop = torch.empty_like(x, device=x.device)
    Ndim = len(x.shape)
    if (batch_equal) or (Ndim < 3):

        segment_len = x.shape[-1] // segments
        if isinstance(x, np.ndarray):
            seg_to_rem = np.random.randint(0, segments, N_cut, dtype=int)
            idx_to_rem = np.concatenate(
                [np.arange(segment_len * seg, segment_len * (seg + 1)) for seg in seg_to_rem]
            )
            new_x = np.delete(x, idx_to_rem, axis=-1)
            x_crop = interpolate.pchip_interpolate(
                np.linspace(0, x.shape[-1] - 1, new_x.shape[-1]),
                new_x,
                np.linspace(0, x.shape[-1] - 1, x.shape[-1]),
                axis=-1,
            )
        else:

            seg_to_rem = torch.randperm(segments)[:N_cut]
            idx_to_rem = torch.cat(
                [torch.arange(segment_len * seg, segment_len * (seg + 1)) for seg in seg_to_rem]
            )

            # https://stackoverflow.com/questions/55110047/
            allidx = torch.arange(x.shape[-1])
            combined = torch.cat((allidx, idx_to_rem, idx_to_rem))
            uniques, counts = combined.unique(return_counts=True)
            difference = uniques[counts == 1]
            difference = difference.to(device=x.device)
            new_x = x[..., difference]
            x_crop = torch_pchip(
                torch.linspace(0, x.shape[-1] - 1, new_x.shape[-1], device=x.device),
                new_x,
                torch.linspace(0, x.shape[-1] - 1, x.shape[-1], device=x.device),
            )

    else:
        for i in range(x.shape[0]):
            x_crop[i] = crop_and_resize(x[i], segments, N_cut, batch_equal)

    return x_crop


# RE-REFERENCING
def change_ref(
    x: ArrayLike,
    mode: str or int = "avg",
    reference: int = None,
    exclude_from_ref: int or list[int] = None,
) -> ArrayLike:
    """
    changes the reference of all EEG record in the ArrayLike object.

    Currently, reference can be changed to:

        1. Channel reference (e.g. Cz). Each record of a channel is
           subtracted to the record of the Cz channel. Cz(t) becomes 0 for all t
        2. Common Average Reference (CAR). Each record is subtracted of the
           average of all electrodes. Currently, it does not cover all particular
           cases which can be managed by major libraries like EEGlab or
           MNE as this implementation is minimalist.

    To get a more detailed description about re-referencing, check this brief
    background page of the EEGlab library [eeglab]_ .

    Parameters
    ----------
    x: ArrayLike
        The input Tensor or Array.
        The last two dimensions must refer to the EEG recording
        (Channels x Samples).
    mode: str or int, optional
        The re-reference modality. Accepted arguments are:

            1. 0, 'chan', 'channel'. Single Channel re-referencing
            2. 1, 'avg', 'average', 'car'. Common Avarage re-referencing.

        Default = "avg"
    reference: int, optional
        The reference electrode, given as an integer with the index position of
        the EEG channel in the input `ArrayLike` object **x**.
        Remember that the EEG channel dimension must be the second to last.

        Default = None
    exclude_from_ref: int or list[int], optional
        Argument designed to exclude some channels during average re-referencing.
        This apply for example when x has records from nose tip or ear lobe.

        Default = None

    Returns
    -------
    x: ArrayLike
        The augmented version of the input Tensor or Array.

    References
    ----------
    .. [eeglab] https://eeglab.org/tutorials/ConceptsGuide/rereferencing_background.html

    Example
    -------
    >>> import torch
    >>> import selfeeg.augmentation as aug
    >>> torch.manual_seed(1234)
    >>> x = torch.ones(16,32,1024)*2 + torch.sin(torch.linspace(0, 8*torch.pi,1024))
    >>> x[:,0,:]= 0.
    >>> xaug = aug.change_ref(x, 'channel', 5)
    >>> print(x[0,0].max()!=0 and x[0,0].min()!=0) # should return False
    >>> print( (xaug[0,[i for i in range(1,32)]].min().item()==0 and
    ...         xaug[0,0].min().item()!=0) ) # should return True

    """

    Ndim = len(x.shape)
    if Ndim < 2:
        raise ValueError("x must be at least a 2 dim array")

    if reference != None:
        if isinstance(reference, int):
            if reference >= x.shape[-1]:
                raise ValueError("reference index exceeds the number of channels in x")
    else:
        reference = 0

    if isinstance(mode, int):
        if mode > 1 or mode < 0:
            msgErr = """
            mode not supported. Current available modes are 'avg',
            which use the common average reference and 'Cz', which support
            original vertex reference.
            """
            raise ValueError(msgErr)
    elif isinstance(mode, str):
        mode = mode.lower()
        if mode in ["avg", "average", "car"]:
            mode = 1
        elif mode in ["channel", "chan"]:
            mode = 0
        else:
            msgErr = """
            mode not supported. Current available modes are 'avg',
            which use the common average reference and 'Cz', which support
            original vertex reference.
            """
            raise ValueError(msgErr)
    else:
        raise ValueError("mode must be a string. Use help() to consult all available mode")

    if exclude_from_ref != None:
        if isinstance(exclude_from_ref, int):
            exclude_from_ref = [exclude_from_ref]
        else:
            for i in exclude_from_ref:
                if not (isinstance(i, int)):
                    raise ValueError("exclude_from_ref must be an int or a list of int")
                if i >= x.shape[-1]:
                    raise ValueError("exclude_from_ref indeces exceed the number of channels")

    if Ndim < 3:
        if mode == 1:
            if exclude_from_ref == None:
                Ref = x[[i for i in range(x.shape[0]) if i != reference]].mean(0)
            else:
                Ref = x[
                    [i for i in range(x.shape[0]) if i not in exclude_from_ref + [reference]]
                ].mean(0)
        elif mode == 0:
            Ref = x[reference]
        x_new_ref = x - Ref
    else:
        x_new_ref = np.empty_like(x) if isinstance(x, np.ndarray) else torch.empty_like(x)
        for i in range(x.shape[0]):
            x_new_ref[i] = change_ref(
                x[i], mode=mode, reference=reference, exclude_from_ref=exclude_from_ref
            )

    return x_new_ref


def masking(
    x: ArrayLike, mask_number: int = 1, masked_ratio: float = 0.1, batch_equal: bool = True
) -> ArrayLike:
    """
    puts to zero random portions of the ArrayLike object.

    maskiing is applied along its last dimension.
    The function will apply the same masking operation to all
    Channels of the same EEG. The number of portions to mask and the
    overall masked ratio can be set as input.


    Parameters
    ----------
    x: ArrayLike
        The input Tensor or Array.
        The last two dimensions must refer to the EEG recording
        (Channels x Samples).
    mask_number: int, optional
        The number of masking blocks, i.e., how many portions of the signal
        the function must mask. It must be a positive integer.
        Note that the created portions will have random length, but the
        overall masked ratio will be the one given as input.

        Default = 1
    masked_ratio: float, optional
        The overall percentage of the signal to mask.
        It must be a scalar in range [0,1].

        Default = 0.1
    batch_equal: bool, optional
        Whether to apply the same masking to all elements in the batch or not.
        It does apply only if x has more than 2 dimensions.

        Default = True

    Returns
    -------
    x: ArrayLike
        the augmented version of the input Tensor or Array.

    Note
    ----
    `mask_number` and `mask_ratio` should be used to tune the number and width
    of masked blocks. For example, given masked_ratio = 0.50 and mask_number = 3,
    the number of samples put to 0 will be in total half the signal length with 3
    distinct blocks of consecutive zeroes of random length.

    Example
    -------
    >>> import torch
    >>> import selfeeg.augmentation as aug
    >>> x = torch.ones(16,32,1024)*2 + torch.sin(torch.linspace(0, 8*torch.pi,1024))
    >>> xaug = aug.masking(x, 3, 0.5)
    >>> print( torch.isclose(((xaug[0,0]==0).sum()/len(xaug[0,0])),
    ...                      torch.tensor([0.5]), rtol=1e-8,atol=1e-8) )
    >>> # should return True

    """

    if not (isinstance(mask_number, int)) or mask_number <= 0:
        raise ValueError("mask_number must be a positive integer")
    if masked_ratio <= 0 or masked_ratio >= 1:
        raise ValueError(
            "mask_ratio must be in range (0,1), " "i.e. all values between 0 and 1 excluded"
        )

    Ndim = len(x.shape)
    x_masked = np.copy(x) if isinstance(x, np.ndarray) else torch.clone(x)
    if Ndim < 3 or batch_equal:

        # IDENTIFY LENGTH OF MASKED PIECES
        sample2mask = int(masked_ratio * x.shape[-1])
        pieces = [0] * mask_number
        piece_sum = 0
        for i in range(mask_number - 1):

            left = sample2mask - piece_sum - (mask_number - i + 1)
            minval = max(1, int((left / (mask_number - i) + 1) * 0.75))
            maxval = min(left, int((left / (mask_number - i) + 1) * 1.25))
            pieces[i] = random.randint(minval, maxval)
            piece_sum += pieces[i]
        pieces[-1] = sample2mask - piece_sum

        # IDENTIFY POSITION OF MASKED PIECES
        maxspace = x.shape[-1] - sample2mask
        spaces = [0] * mask_number
        space_sum = 0
        for i in range(mask_number):
            left = maxspace - space_sum - (mask_number - i + 1)
            spaces[i] = random.randint(1, int(left / (mask_number - i) + 1))
            space_sum += spaces[i]

        # APPLYING MASKING
        cnt = 0
        for i in range(mask_number):
            cnt += spaces[i]
            x_masked[..., cnt : cnt + pieces[i]] = 0
            cnt += pieces[i]

    else:
        for i in range(x.shape[0]):
            x_masked[i] = masking(
                x[i], mask_number=mask_number, masked_ratio=masked_ratio, batch_equal=batch_equal
            )

    return x_masked


def channel_dropout(x: ArrayLike, Nchan: int = None, batch_equal: bool = True) -> ArrayLike:
    """
    puts to 0 a given (or random) amount of channels of the ArrayLike object.

    Channels are selected randomly.

    Parameters
    ---------
    x: ArrayLike
        The input Tensor or Array.
        The last two dimensions must refer to the EEG recording
        (Channels x Samples).
    Nchan: int, optional
        Number of channels to drop. If not given, the number of channels is chosen
        at random in the interval [1, (Channel_total // 4) +1 ].

        Default = None
    batch_equal: bool, optional
        Whether to apply the same channel drop to all EEG records or not.

        Default = True

    Returns
    -------
    x: ArrayLike
        the augmented version of the input Tensor or Array.

    Example
    -------
    >>> import torch
    >>> import selfeeg.augmentation as aug
    >>> x = torch.ones(32,1024)*2 + torch.sin(torch.linspace(0, 8*torch.pi,1024))
    >>> xaug = aug.channel_dropout(x, 3)
    >>> print( (xaug[0:,10]==0).sum()==3) # should return True

    """

    Ndim = len(x.shape)
    x_drop = torch.clone(x) if isinstance(x, torch.Tensor) else np.copy(x)

    if batch_equal or Ndim < 3:
        if Nchan is None:
            Nchan = random.randint(1, (x.shape[-2] // 4) + 1)
        else:
            if Nchan > x.shape[-2]:
                raise ValueError(
                    "Nchan can't be higher than the actual number" " of channels in the given EEG"
                )
            else:
                Nchan = int(Nchan)
        drop_chan = np.random.permutation(x.shape[-2])[:Nchan]
        if isinstance(x, torch.Tensor):
            drop_chan = torch.from_numpy(drop_chan).to(device=x.device)
        x_drop[..., drop_chan, :] = 0
    else:
        for i in range(x.shape[0]):
            x_drop[i] = channel_dropout(x[i], Nchan=Nchan, batch_equal=batch_equal)

    return x_drop


def add_eeg_artifact(
    x: ArrayLike,
    Fs: float,
    artifact: str = None,
    amplitude: float = None,
    line_at_60Hz: bool = True,
    lost_time: float = None,
    drift_slope: float = None,
    batch_equal: bool = True,
) -> ArrayLike:
    """
    add common EEG artifacts to the ArrayLike object.

    Given a N-dim tensor or a numpy array, add_eeg_artifact add one of
    the artifact listed in [art]_ along the last dimension of the input element.
    Supported artifacts are:

        1. white: simple white noise
        2. line: noise at 50 Hz or 60 Hz
        3. eye: noise in range [1, 3] Hz
        4. muscle: noise in range [20, 60] Hz
        5. drift: straight line with non-zero slope
        6. lost: cancellation of one portion of the signal

    Line, eye and muscle artifact are generated with the ``add_band_noise``
    function. Lost artifact is generated with the ``masking`` function.
    White and drift are generated internally.

    Parameters
    ----------
    x: ArrayLike
        the input Tensor or Array.
        The last two dimensions must refer to the EEG recording
        (Channels x Samples).
    Fs: float
        The sampling rate of the signal in Hz.
    artifact: str, optional
        The type of artifact to apply.
        If None is left a random artifact will be chosen.

        Default = None
    amplitude: float, optional
        The amplitude of the noise to add. If not given, amplitude=std(x)

        Default = None
    line_at_60Hz: bool, optional
        Whether to apply the line artifact at 60Hz (True) or 50Hz (False).

        Default = True
    lost_time: float, optional
        The amount of time the signal is canceled. Must be given in seconds.
        Internally ``masking`` function is called by converting the given time
        to the percentage of masked signal with the function
        `(Fs*lost_time)/x.shape[-1]`.
        Alternatively, to convert the percentage of the signal masked to the
        amount of time is masked, just revert the formula, so:

            lost_time= (masking_percentage * x.shape[-1]) / Fs.

        If None is given, 20% of the signal is randomly masked.

        Default = None
    drift_slope: float, optional
        The difference between to consecutive points of the straight line to add.
        If None is given, slope is calculated with the amplitude parameter as

            drift_slope = amplitude/x.shape[-1].

        If also amplitude is not given

            drift_slope = std(x)/x.shape[-1]

        Default = None
    batch_equal: bool, optional
        Whether to apply the same masking to all elements in the batch or not.
        Does apply only if x has more than 2 dimensions

        Default = True

    Returns
    -------
    x: ArrayLike
        the augmented version of the input Tensor or Array.

    References
    ----------
    .. [art] Fickling et al., (2019) Good data? The EEG Quality Index for Automated
      Assessment of Signal Quality

    Example
    -------
    >>> import torch
    >>> import selfeeg.augmentation as aug
    >>> x = torch.zeros(16,32,1024) + torch.sin(torch.linspace(0, 8*torch.pi,1024))
    >>> xaug = aug.add_eeg_artifact(x, 128)
    >>> print( torch.equal(x,xaug)) # should return False

    """
    was_random = False  # just for recursive calls
    if artifact != None:
        if artifact not in ["line", "lost", "drift", "eye", "muscle", "white"]:
            raise ValueError(
                "EEG artifact not supported. Accepted any of 'line',"
                "'lost','drift','eye','muscle'"
            )
    else:
        was_random = True
        artifact = random.choice(["line", "lost", "drift", "eye", "muscle", "white"])

    if amplitude != None:
        if not (isinstance(amplitude, list)):
            if amplitude > 0:
                amplitude = [-amplitude, amplitude]
            else:
                amplitude = [amplitude, -amplitude]

    Ndim = len(x.shape)
    if isinstance(x, np.ndarray):
        x_noise = np.empty_like(x)
    else:
        x_noise = torch.empty_like(x, device=x.device)

    if batch_equal or Ndim < 3:

        if artifact == "line":
            freq = 60 if line_at_60Hz else 50
            x_noise, noise = add_band_noise(x, freq, Fs, amplitude, get_noise=True)

        elif artifact == "eye":
            x_noise = add_band_noise(x, (1, 3), Fs, amplitude)

        elif artifact == "muscle":
            x_noise = add_band_noise(x, (20, 60), Fs, amplitude)

        elif artifact == "lost":
            if lost_time == None:
                x_noise = masking(x, 1, 0.2)
            else:
                x_noise = masking(x, 1, (Fs * lost_time) / x.shape[-1])

        elif artifact == "white":
            if isinstance(x, np.ndarray):
                noise = np.random.uniform(-1, 1, size=x.shape)
            else:
                noise = torch.empty_like(x, device=x.device).uniform_(-1, 1)

            if amplitude != None:
                noise *= amplitude[1]
            else:
                if isinstance(x, np.ndarray):
                    G = (
                        amplitude[1] / np.std(noise)
                        if amplitude != None
                        else np.std(x) / np.std(noise)
                    )
                    noise *= G
                else:
                    G = (
                        amplitude[1] / torch.std(noise)
                        if amplitude != None
                        else torch.std(x) / torch.std(noise)
                    )
                    noise *= G
            x_noise = x + noise

        elif artifact == "drift":
            if isinstance(x, np.ndarray):
                if drift_slope != None:
                    noise = np.arange(x.shape[-1]) * drift_slope
                elif amplitude != None:
                    noise = np.linspace(0, amplitude[1], x.shape[-1])
                else:
                    noise = np.linspace(0, np.std(x), x.shape[-1])
            else:
                if drift_slope != None:
                    noise = torch.arange(x.shape[-1], device=x.device) * drift_slope
                elif amplitude != None:
                    noise = torch.linspace(0, amplitude[1], x.shape[-1], device=x.device)
                else:
                    noise = torch.linspace(0, torch.std(x), x.shape[-1], device=x.device)
            x_noise = x + noise

    else:
        artifact_arg = None if was_random else artifact
        for i in range(x.shape[0]):
            x_noise[i] = add_eeg_artifact(
                x[i], Fs, artifact_arg, amplitude, line_at_60Hz, lost_time, drift_slope, batch_equal
            )

    return x_noise
