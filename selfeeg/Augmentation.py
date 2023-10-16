import math
import random
import inspect
import time
from typing import Any, Callable, Iterable, TypeVar, Generic, Sequence, List, Dict, Optional, Union
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import signal
from scipy import interpolate
from scipy import fft
import torch
import torch.nn.functional as F
from torchaudio.functional import lfilter, filtfilt

__all__ = ['identity',
           'shift_vertical', 'shift_horizontal', 'shift_frequency'
           'flip_vertical', 'flip_horizontal',
           'scaling','random_slope_scale', 'random_FT_phase',
           'add_gaussian_noise', 'add_noise_SNR','add_band_noise', 'add_eeg_artifact',
           'moving_avg', 'filter_lowpass', 'filter_highpass', 'filter_bandpass', 'filter_bandstop',
           'get_eeg_channel_network_names', 'get_channel_map_and_networks', 
           'permute_channels', 'permutation_signal', 
           'torch_pchip', 'warp_signal', 'crop_and_resize', 
           'change_ref', 
           'masking', 'channel_dropout',
           'StaticSingleAug', 'DynamicSingleAug', 'SequentialAug'
          ]


# ----- SHIFTS AND FLIPS -------
def identity(x):
    """
    identity return the same array or tensor as it was given.
    It can be used during augmentation composition to randomly avoid some augmentations
    """
    return x

def shift_vertical(x: "N-D Tensor of numpy Array", 
                   value: float):
    """
    
    shift_vertical add a scalar value to the input array x.
    
    Parameters
    ----------
    x: N-D Tensor or numpy array
    value: scalar
        The value to add
    """
    # To do: batch equal and random shift from +- 10 uV random number
    x_shift = x + value
    return x_shift

def shift_horizontal(x: "N-D Tensor of numpy Array",
                     shift_time: float,
                     Fs: float,
                     forward: bool=None,
                     random_shift: bool=False,
                     batch_equal: bool=True
                    ):
    """
    
    shift_horizontal shift temporally the elements of the last dimension of x by a constant.
    
    The empty elements at beginning or the ending part after shift are set to zero.
    
    Paramters
    ---------
    x: N-D Tensor or numpy array
        Array to shift. Last dimension must have the EEG recordings
    shift_time: float
        Shift in seconds, of the desired time shift.
    Fs: float
        the EEG sampling rate in Hz
    forward: bool
        Whether to shift the EEG forward (True) or backward (False) in time. If left to None, a
        random selection of the shift direction will be performed.
        Default: None
    random_shift: bool, optional
        Wheter to choose a random shift length lower than or equal to shift_time, i.e. consider shift_time as
        the exact value to shift or as an upper bound for a random selection
        Default: False
    batch_equal: bool, optional
        whether to apply the same shift to all EEG record or not.
        Note: if random shift is set to False and forward is None, then batch_equal will be equal to True since
              no differences in the shift can be applied.
        Default: True
    """

    if shift_time<0:
        raise ValueError('shift time must be a positive value. To shift backward set forward to False')
    Ndim= len(x.shape)
    x_shift = torch.clone(x) if isinstance(x, torch.Tensor) else np.copy(x)
    if batch_equal:
        if not(random_shift or (forward is None)):
            print('set batch equal to true')
            batch_equal=True
    
    if batch_equal or Ndim<3:
        if forward is None:
            forward= bool(random.getrandbits(1))

        if random_shift:
            shift = random.randint(1, int(shift_time*Fs))
        else:
            shift = int(shift_time*Fs)
        
        if forward:
            x_shift[...,:shift] = 0
            x_shift[...,shift:] = x[...,:-shift]
        else:
            x_shift[...,:-shift] = x[...,shift:]
            x_shift[...,-shift:] = 0
    else:
        for i in range(x_shift.shape[0]):
            x_shift[i] = shift_horizontal(x[i], shift_time=shift_time, Fs= Fs, forward= forward,
                                          random_shift=random_shift, batch_equal=batch_equal)
    return x_shift
    

def _UnitStep(x):
    """
    _UnitStep create a numpy array or pytorch tensor with shape equal to x 
    and with the last dimension filled with the step function values used in the Hilbert Transform.
    This is used to speed up the computation of shift_frequency when batch_equal is set to False.
    In short, it avoids initializing the same h multiple times. 

    For more info see SciPy's hilbert help and source code:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html
    
    """
    N = x.shape[-1]
    h = torch.zeros_like(x, device=x.device) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    if N % 2 == 0:
        h[..., 0] = h[..., N // 2] = 1
        h[..., 1:N // 2] = 2
    else:
        h[..., 0] = 1
        h[..., 1:(N + 1) // 2] = 2
    return h
    

def torch_hilbert(x, h: torch.Tensor=None):
    """
    torch_hilbert is a minimal version of SciPy's hilbert function adapted for pytorch tensors.

    Parameters
    ----------
    x : torch.Tensor
        Tensor with signal data.
    h : torch.Tensor, optional
        Tensor with the UnitStep function. If not given, it will be initialized during call.
        Default: None

    Returns
    -------
    Xa : torch.Tensor
        The analytic signal of x calculated along the last dimension of x

    Notes
    -----
    For more info see SciPy's hilbert help and source code:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html
    """
    if torch.is_complex(x):
        raise ValueError("x must be real tensor.")

    N = x.shape[-1]
    f = torch.fft.fft(x, N, dim=-1)
    if h is None:
        h = torch.zeros_like(f, device=x.device)
        if N % 2 == 0:
            h[..., 0] = h[..., N // 2] = 1
            h[..., 1:N // 2] = 2
        else:
            h[..., 0] = 1
            h[..., 1:(N + 1) // 2] = 2
    Xa = torch.fft.ifft(f * h, dim=-1)


    return Xa
    

def _shift_frequency(x: "N-D Tensor of numpy Array",
                     shift_freq: float,
                     Fs: float,
                     forward: bool=None,
                     random_shift: bool=False,
                     batch_equal: bool=True,
                     t= None,
                     h= None
                    ):
    if shift_freq<0:
        raise ValueError('shift freq must be a positive value.'
                         ' To shift backward set forward to False')
    Ndim= len(x.shape)
    x_shift = torch.clone(x) if isinstance(x, torch.Tensor) else np.copy(x)
    if not(batch_equal):
        if not(random_shift or (forward is None)):
            print('set batch equal to true')
            batch_equal=True
    
    if batch_equal or Ndim<3:
        shift = shift_freq
        if forward is None:
            forward= bool(random.getrandbits(1))
        if random_shift:
            shift = np.random.uniform(-shift_freq, shift_freq)
        if not(forward) and shift>0:
            shift = -shift
        
        if t is None:
            if isinstance(x, torch.Tensor):
                T= x.shape[-1]/Fs
                t = torch.linspace(0, T-(1/Fs), int(Fs*T), device=x.device)
            else:
                t = np.r_[0:T:(1/Fs)] 

        if isinstance(x, torch.Tensor):
            Xa = torch_hilbert(x) if h is None else torch_hilbert(x,h)
            x_shift = torch.real(Xa * torch.exp(1j*2*math.pi*shift*t))
        else:
            Xa = signal.hilbert(signal) if h is None else fft.ifft( fft.fft(x)*h )
            x_shift = np.real(Xa * np.exp(1j*2*math.pi*shift*t))
            
    else:
        for i in range(x_shift.shape[0]):
            x_shift[i] = _shift_frequency(x[i], shift_freq=shift_freq, Fs= Fs, forward= forward,
                                          random_shift=random_shift, batch_equal=batch_equal, 
                                          t=t, h=h[i] if h is not None else None)
    return x_shift


def shift_frequency(x: "N-D Tensor of numpy Array",
                    shift_freq: float,
                    Fs: float,
                    forward: bool=None,
                    random_shift: bool=False,
                    batch_equal: bool=True,
                   ):
    T= x.shape[-1]/Fs
    if isinstance(x, torch.Tensor):
        t = torch.linspace(0, T-(1/Fs), int(Fs*T), device=x.device)
    else:
        t = np.r_[0:T:(1/Fs)] 
    h = _UnitStep(x)
    return _shift_frequency(x, shift_freq, Fs, forward, random_shift, batch_equal, t, h)


def flip_vertical(x: "Array like"):
    """
    flip_vertical change the sign of all the elements of the input array x.
    
    Paramters
    ---------
    x: array like
        Array to flip. Last dimension must have the EEG recordings
    """
    # TO DO: add batch_equal to apply flip only on certain EEGs
    x_flip= x*(-1)
    return x_flip


def flip_horizontal(x: "array like"):
    """
    flip_horizontal flip the elements of the last dimension of x.
    
    Paramters
    ---------
    x: array like
        Array to flip. Last dimension must have the EEG recordings
    """
    # TO DO: add batch_equal to apply flip only on certain EEGs
    if isinstance(x, np.ndarray):
        x_flip = np.flip(x, len(x.shape)-1)
    else:
        x_flip = torch.flip(x, [len(x.shape)-1])
    
    return x_flip


# ---- NOISE ADDER -----
def add_gaussian_noise(x: "N-D Tensor of numpy Array", 
                       mean: float=0., 
                       std: float=1.,
                       get_noise: bool=False
                      ):
    """
    
    add_gaussian_noise add gaussian noise with the desired mean and standard deviation.
    
    Paramters
    ---------
    x: N-D Tensor or numpy array
        array to add noise
    mean: scalar, optional
        the mean of the gaussian distribution
        Default: 0
    std: scalar, optional
        the std of the gaussian distribution
        Default: 1
    get_noise: bool, optional
        whether to return the generated noise or not
        Default: False
    """
    
    if isinstance(x,np.ndarray):
        noise = mean + std * np.random.randn(*x.shape)
    else:
        noise   = mean +  std * torch.randn(*x.shape, device=x.device)
    x_noise = x + noise 
    if get_noise:
        return x_noise, noise
    else:
        return x_noise

    
def add_noise_SNR(x: "N-D Tensor of numpy Array", 
                  target_snr: float=5, 
                  get_noise: bool=False
                 ):
    """
    
    add_noise_SNR add noise such that the SNR (Signal to Noise Ratio) will be the one desired.
    
    Since the signal is supposed to be already noisy, it makes more sense to say that this function reduce 
    the SNR by a factor equal to 1/P_noise_new, where P_noise_new is the power of the new added noise.
    
    Parameters
    ----------
    x: N-D Tensor or numpy array
        array to add noise
    target_SNR: scalar, optional
        the target SNR
        Default: 5
    get_noise: bool, optional
        whether to return the generated noise or not
        Default: False
        
    
    created using the following reference: 
        https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python
    """
    
    # get signal power. Not exactly true since we have an already noised signal
    x_pow = (x ** 2)
    
    if isinstance(x,np.ndarray):
        x_db = 10 * np.log10(x_pow)
        x_pow_avg = np.mean(x_pow)
        x_db_avg = 10 * np.log10(x_pow_avg) 
        noise_db_avg = x_db_avg - target_snr
        noise_pow_avg = 10 ** (noise_db_avg / 10)
        noise = np.random.normal(0, noise_pow_avg**0.5 , size=x.shape) 
        x_noise = x + noise
    
    else:
        x_db = 10 * torch.log10(x_pow)
        x_pow_avg = torch.mean(x_pow)
        x_db_avg = 10 * torch.log10(x_pow_avg)
        noise_db_avg = x_db_avg - target_snr
        noise_pow_avg = 10 ** (noise_db_avg / 10)
        noise = ((noise_pow_avg**0.5)*(torch.randn(*x.shape).to(device=x.device)))
        x_noise = x + noise        
    
    if get_noise:
        return x_noise, noise
    else:
        return x_noise


def add_band_noise(x: "numpy array or tensor",
                   bandwidth: list[tuple[float,float], str, float], 
                   samplerate: float=256,
                   noise_range: float or list[float,float]= None,
                   std: float=None,
                   get_noise: bool=False
                  ):
    
    """
    
    add_band noise add random noise filtered at specific bandwidths.
    
    Given a set of bandwidths or a set of specific frequency, add_band_noise create a noise whose 
    spectrum is bigger than zero only on those bands. It can be used to alter only specific frequency components
    of the original signal. The noise generated will have the same standard deviation as x, but it can be rescaled 
    so to be within a specific range or to have a specific standard deviation.
    
    Parameters
    ----------
    x: N-D Tensor or numpy array
        array to add noise
    bandwidth: list
        The frequency components which the noise must have. Must be a LIST with the following values:
        strings: add noise to specific EEG components. Can be any of "delta", "theta", "alpha", "beta", 
                 "gamma", "gamma_low", "gamma_high"
        scalar: add noise to a specifi component
        tuple with 2 scalar: add noise to a specific band set with the tuple (start_component, end_component)
    samplerate: float, optional
        The sampling rate, given in Hz. Remember to change this value according to the signal sampling rate
        Default: 256
    noise_range: float, optional
        The range within the noise is scaled. Must be a single sclar or a two element list. If given as a single 
        scalar, then the range is considered the interval [-noise_range, noise_range]. If this parameter is given,
        then std value is ignored, since the two conditions cannot be satisfied at the same time. To rescale, 
        the following formula is applied:
        
        noise_new = ( (noise - max(noise))/(max(noise)-min(noise)) )*(target_range_max - target_range_min) + target_range_min
        Default: None
    std: float, optional
        The desired standard deviation of the noise. If noise_range is given, this argument is ignored. 
        It simply scale the noise by applying:
                        noise_new = noise * (target_std / std(noise))
        Default: None
    get_noise: bool, optional
        whether to return the generated noise or not
        Default: False
        
    """
    
    # converting to list if single string or integer is given
    if not(isinstance(bandwidth, list)):
        bandwidth=[bandwidth]
    if noise_range != None:
        if isinstance(noise_range,list):
            if len(noise_range)==2:
                noise_range.sort()
            else:
                raise ValueError('if given as a list, noise_range must be of length 2')
        else:
            noise_range=[-noise_range, noise_range] if noise_range>0 else [noise_range, -noise_range]
    
    # transform all elements in 2 floats tuple
    for i in range(len(bandwidth)):
        # change string bandwidth call to frequency slice
        if isinstance(bandwidth[i],str):
            if bandwidth[i].lower() == 'delta':
                bandwidth[i]=(0.5,4)
            elif bandwidth[i].lower() == 'theta':
                bandwidth[i]=(4,8)
            elif bandwidth[i].lower() == 'alpha':
                bandwidth[i]=(8,13)
            elif bandwidth[i].lower() == 'beta':
                bandwidth[i]=(13,30)
            elif bandwidth[i].lower() == 'gamma_low':
                bandwidth[i]=(30,70)
            elif bandwidth[i].lower() == 'gamma_high':
                bandwidth[i]=(70,150)
            elif bandwidth[i].lower() == 'gamma':
                bandwidth[i]=(30,150)
            else:
                message  = 'Brainwave \"',bandwidth[i], '\" not exist. \n'
                message += 'Choose between delta, theta, alpha, beta, gamma, gamma_low, gamma_high'
                raise ValueError(message)
        
        # change single frequency call
        elif np.isscalar(bandwidth[i]):                
            bandwidth[i]=(bandwidth[i],bandwidth[i])

    N=len(bandwidth)          
    samples = x.shape[-1]
    if isinstance(x,np.ndarray):
        
        f = np.zeros(samples, dtype='complex')
        for i in range(N):
            start = int(bandwidth[i][0]*samples/samplerate)
            end = int(bandwidth[i][1]*samples/samplerate +1)
            f[start:end] = 1
        Np = (len(f) - 1) // 2
        phases = np.random.rand(Np) * 2 * np.pi
        phases = np.cos(phases) + 1j * np.sin(phases)
        f[1:Np+1] *= phases
        f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
        noise = np.fft.ifft(f).real
        if noise_range==None:
            G = std/np.std(noise) if std!=None else np.std(x)/np.std(noise)
            noise *= G
        else:
            max_noise=np.max(noise)
            min_noise=np.min(noise)
            range_noise= max_noise-min_noise
            target_range = (noise_range[1] - noise_range[0])
            noise= ((noise - min_noise)/range_noise)*(target_range) + noise_range[0]
        x_noise= x+noise
               
    else:
        device = 'cpu' if x.device.type=='mps' else x.device
        f = torch.zeros(samples, dtype=torch.complex64, device=device)
        for i in range(N):
            start = int(bandwidth[i][0]*samples/samplerate)
            end = int(bandwidth[i][1]*samples/samplerate +1)
            f[start:end] = 1
        Np = (samples - 1) // 2
        phases = torch.rand(Np, device=device)
        phases =  phases * 2 * math.pi
        phases = torch.cos(phases) + 1j * torch.sin(phases)
        f[1:Np+1] *= phases
        f[-Np:] = torch.flip(torch.conj(f[1:Np+1]), [0])
        noise = torch.fft.ifft(f).real
        if x.device.type=='mps':
            noise = noise.to(device='mps')
        if noise_range==None:
            G = std/torch.std(noise) if std!=None else torch.std(x)/torch.std(noise)
            noise *= G
        else:
            max_noise=torch.max(noise)
            min_noise=torch.min(noise)
            range_noise= max_noise-min_noise
            target_range = (noise_range[1] - noise_range[0])
            noise= ((noise - min_noise)/range_noise)*(target_range) + noise_range[0]
        x_noise= x+noise

    
    if get_noise:
        return x_noise , noise
    else:
        return x_noise

def scaling(x: 'array like',
            value: float=None,
            batch_equal: bool=True
           ):
    """
    scaling rescale the array by a given amplitude.
    
    Parameters
    ----------
    x: array like
        Input array or tensor. Can be of any shape, but the last two dimensions must referres to EEG's Channel x Sample
    value: float, optional
        The rescaling factor. If not given, a random value is extracted from a uniform distribution in range [0.5, 2]
        Default: None
    batch_equal:
        Whether to apply the same rescaling on all signals or not. If False, value must be left to None, otherwise
        batch_equal will be reset to True.
        Default: True
    """
    Ndim = len(x.shape)
    x_scale = torch.clone(x) if isinstance(x, torch.Tensor) else np.copy(x)
    if not(batch_equal) and (value is None):
        batch_equal=True # speed up computation
    if batch_equal or Ndim<3:
        if value is None:
            value = random.uniform(0.5, 2)
        x_scale *= value
    else:
        for i in range(x_scale.shape[0]):
            x_scale[i] = scaling(x[i], value, batch_equal)
    return x_scale

    
def random_slope_scale(x: "N-D Tensor of numpy Array",
                       min_scale: float=0.9,
                       max_scale: float=1.2,
                       batch_equal: bool=False,
                       keep_memory: bool=False
                      ):
    """
    
    random_slope_scale randomly scale the first derivative of x
    
    Given a tensor N-D tensor or numpy array where the last two dimensions represents (EEG_channels x EEG_samples) 
    (1D tensor are also accepted), random_slope_scale calculates the first derivatives of each EEG records, here 
    simplified as the difference between two consecutive values of the last dimension, and rescale each of them with
    a random factor selected from an uniform distribution between min_scale and max_scale. This transformation 
    is similar to adding a random noise, but with the constraint that first derivatise must keep the sign of the 
    original EEG (e.g. if a value is bigger than the previous one, then this is also true in the transformed data, 
    although with the difference is not same).
    
    Parameters
    ----------
    x: N-D Tensor or numpy array
        The element to rescale. x can be one dimensional. If x has more than one dimension, the last two must 
        represent (EEG_channels x EEG_samples) 
    min_scale: float, optional
        The minimum rescaling factor to be applied. Must be a value bigger than 0
        Default: 0.9
    max_scale: float, optional
        The maximum rescaling factor to be applied. Must be a value bigger than min_scale
        Default: 1.2
    batch_equal: bool, optional
        whether to apply the same rescale to all EEGs in the batch or not. This apply if x has more than 2 dimensions.
        Default: False
    keep_memory: bool, optional
        whether to keep memory of the previous changes in slope and accumulate them during transformation or not. 
        Basically, instead of using: 
                x_hat(n)= x(n-1) + scaling*( x(n)-x(n-1) ) ; n>1, x_hat transformed signal, x original signal
        keep_memory apply the following:
                x_hat(n)= x_hat(n-1) + scaling*( x(n)-x(n-1) ) ; n>1, x_hat transformed signal, x original signal
        Keep in mind that this may completely change the range of values, as consecutive increase in the slopes
        may cause a strong vertical shift of the signal. If set to True, set scaling factor in the range [0.8, 1.2]
        Default: False
    
    """
    
    Ndim=len(x.shape)
    if min_scale<0:
        raise ValueError('minimum scaling factor can\'t be lower than 0')
    if max_scale<=min_scale:
        raise ValueError('maximum scaling factor can\'t be lower than minimum scaling factor')
    
    if batch_equal:
        if isinstance(x, np.ndarray):
            scale_factor= np.random.uniform(min_scale, max_scale, size=tuple(x.shape[-2:]+np.array([0,-1])) )
        else:
            scale_factor= torch.empty(x.shape[-2], x.shape[-1]-1, device=x.device).uniform_(min_scale, max_scale)
    else:
        if isinstance(x, np.ndarray):
            scale_factor= np.random.uniform(min_scale, max_scale, x.shape)[...,1:]
        else:
            scale_factor= torch.empty_like(x, device=x.device)[...,1:].uniform_(min_scale, max_scale)
    
    x_diff = x[...,1:] - x[...,:-1]
    x_diff_scaled = x_diff*scale_factor
    x_new = np.empty_like(x) if isinstance(x, np.ndarray) else torch.empty_like(x, device=x.device)
    x_new[...,0] = x[...,0]
    if keep_memory:
        x_new[...,1:] = np.cumsum(x_diff_scaled, Ndim-1) if isinstance(x, np.ndarray) else torch.cumsum(x_diff_scaled, Ndim-1)
    else:
        x_new[...,1:] =  x[...,:-1] + x_diff_scaled
    
    return x_new    


def new_random_fft_phase_odd(n, to_torch_tensor: bool=False, device='cpu'):
    if to_torch_tensor:
        random_phase = 2j*np.pi*torch.rand((n-1)//2)
        new_random_phase = torch.cat((torch.tensor([0.0]), random_phase, -torch.flipud(random_phase))).to(device=device)
    else:
        random_phase = 2j*np.pi*np.random.rand((n-1)//2)
        new_random_phase = np.concatenate([[0.0], random_phase, -random_phase[::-1]])
    return new_random_phase

def new_random_fft_phase_even(n, to_torch_tensor: bool=False, device='cpu'):
    if to_torch_tensor:
        random_phase = 2j*np.pi*torch.rand(n//2-1)
        new_random_phase = torch.cat((torch.tensor([0.0]), random_phase,
                                      torch.tensor([0.0]), -torch.flipud(random_phase))).to(device=device)
    else:
        random_phase = 2j*np.pi*np.random.rand(n//2-1)
        new_random_phase = np.concatenate([[0.0], random_phase, [0.0], -random_phase[::-1]])
    return new_random_phase


def random_FT_phase(x: "array like", 
                    value: float=1,
                    batch_equal: bool=True
                   ):
    """
    random_FT_phase randomize the phase of all signals in tensor/array x.

    Parameters
    ----------
    x: array like
        Input array or tensor. Can be of any shape, but the last two dimensions must referres to EEG's Channel x Sample
    value: float, optional
        The magnitude of the phase perturbation. It must be a value between (0,1], which will be used to rescale the interval
        [0, 2* 'pi'] in [0, value * 2 * 'pi']
        Default: None
    batch_equal:
        Whether to apply the same perturbation on all signals or not. Note that all channels of the same records will be 
        perturbed in the same way to preserve cross-channel correlations.
    """
    if value<=0 or value>1:
        raise ValueError('value must be a float in range (0,1]')
    Ndim = len(x.shape)
    x_phase = torch.clone(x) if isinstance(x, torch.Tensor) else np.copy(x)
    if batch_equal or Ndim<3:
        n = x.shape[-1]
        if isinstance(x, torch.Tensor):
            random_phase = new_random_fft_phase_even(n, True, x.device) if n%2==0 else new_random_fft_phase_odd(n, True, x.device)
            FT_coeff= torch.fft.fft(x)
            x_phase = torch.fft.ifft(FT_coeff*torch.exp(value*random_phase)).real
        else:
            random_phase = new_random_fft_phase_even(n) if n%2==0 else new_random_fft_phase_odd(n)
            FT_coeff = fft.fft(x)
            x_phase =  fft.ifft(FT_coeff*np.exp(value*random_phase)).real      
    else:
        for i in range(x_phase.shape[0]):
            x_phase[i] = random_FT_phase(x[i], value, batch_equal)
    return x_phase

    
    
# ---- FILTERING -----
def moving_avg(x, order: int=5):
    """
    
    moving_avg apply a moving average filter to the signal x.
    
    moving_avg apply a moving average filter to the last dimension of the array or Tensor x. The filter order
    and can be given as function argument.
    
    Parameters
    ----------
    x: N-D Tensor or numpy array
        The element to filter. Signals must be on the last dimension.
    order: int, optional
        The order of the filter.
        Default: 5
    """
    
    if isinstance(x, np.ndarray):
        x_avg = np.empty_like(x)
        filt = np.ones(order)/order
        Ndim= len(x.shape)
        
        # call recursively to handle different dimensions (made to handle problem with torch conv2d)
        if Ndim>1:
            for i in range(x.shape[0]):
                x_avg[i] = moving_avg(x[i], order=order)
        else:
            x_avg = np.convolve( x, filt, 'same')
            
    else:
        Ndim = len(x.shape)
        # adapt to x to conv2d functions
        if Ndim==1:
            x = x.view(1,1,1,*x.shape)
        elif Ndim==2:
            x = x.view(1,1, *x.shape)
        elif Ndim==3:
            x = x.unsqueeze(1)
        x_avg = torch.empty_like(x)
        filt = torch.ones((1,1,1,order), device=x.device)/order
        
        # call recursively if the dimension is larger than 4
        if Ndim > 4:
            for i in range(x.shape[0]):
                x_avg[i] = moving_avg(x[i], order=order)
        else:
            x_avg = F.conv2d(x, filt, padding= 'same')
            x_avg = torch.reshape(x_avg, x.shape)

    
    return x_avg


def get_filter_coeff(Fs: float,
                     Wp: float, 
                     Ws: float,
                     rp: float=-20*np.log10(.95), 
                     rs: float=-20*np.log10(.15), 
                     btype: str='low', 
                     filter_type: str='butter', 
                     order: int=None, 
                     Wn: Union[float,List[float]]=None, 
                     eeg_band: str=None, 
                    ):
    """
    
    get_filter_coeff returns the filter coefficients a and b needed to call the scipy's or torchaudio's 
    filtfilt function.
    
    get_filter_coeff is internally called by other filtering function when a and b coefficients are not given
    as input argument. It works following this priority pipeline:
    1) if specific EEG bands are given, set Wp, Ws, rp, rs for filter design according to the given band
    2) if order and Wn are not given, use previous parameter to design the filter
    3) Use Wn and order to get a and b coefficient to return
    
    In other words (Wp,Ws,rp,rs) ----> (Wn, order) -----> (a,b) 
    
    Parameters
    ----------
    
    Wp: float
        bandpass normalized from 0 to 1
    Ws: float
        stopband normalized from 0 to 1
    rp: float, optional
        ripple at bandpass in decibel. 
        Default: -20*log10(0.95)
    rs: float, optional
        ripple at stopband in decibel. 
        Default: -20*log10(0.15)
    btype: str, optional
        filter type. Can be any of the scipy's btype argument (e.g. 'lowpass', 'highpass', 'bandpass')
        Default: 'low'
    filter_type: str, optional
        which filter design. Accepted values are 'butter', 'ellip', 'cheby1', 'cheby2'
        Default: 'butter'
    order: int, optional
        the order of the filter
        Default: None
    Wn: array_like, optional
        the critical frequency or frequencies.
        Default: None
    eeg_band: str, optional
        any of the possible EEG bands. Accepted values are "delta", "theta", "alpha", "beta", 
        "gamma", "gamma_low", "gamma_high".
        Default: None
    Fs: float, optional
        the sampling frequency. Must be given if eeg_band is also given
        Default: None
    """
    
    if btype.lower() in ['bandpass', 'bandstop']:
        if eeg_band is not None:
            if eeg_band.lower() == 'delta':
                Wp, Ws, rp, rs = 4, 8, -20*np.log10(.95), -20*np.log10(.1)
                btype = 'highpass' if btype.lower()=='bandstop' else 'lowpass'
            elif eeg_band.lower() == 'theta':
                Wp, Ws, rp, rs = [4, 8], [0, 15], -20*np.log10(.95), -20*np.log10(.1)
            elif eeg_band.lower() == 'alpha':
                Wp, Ws, rp, rs = [8, 13], [4, 22], -20*np.log10(.95), -20*np.log10(.1)
            elif eeg_band.lower() == 'beta':
                Wp, Ws, rp, rs = [13, 30], [8, 40], -20*np.log10(.95), -20*np.log10(.15)
            elif eeg_band.lower() == 'gamma_low':
                if Fs>=78*2:
                    Wp, Ws, rp, rs = [30, 70], [20, 80], -20*np.log10(.95), -20*np.log10(.1)
                else:
                    Wp, Ws, rp, rs = 30, 20, -20*np.log10(.95), -20*np.log10(.1)
                    btype = 'lowpass' if btype.lower()=='bandstop' else 'highpass'
            elif eeg_band.lower() == 'gamma_high':
                if Fs>=158*2:
                    Wp, Ws, rp, rs = [70, 150], [60, 160], -20*np.log10(.95), -20*np.log10(.1)
                else:
                    Wp, Ws, rp, rs = 70, 60, -20*np.log10(.95), -20*np.log10(.1)
                    btype = 'lowpass' if btype.lower()=='bandstop' else 'highpass'
            elif eeg_band.lower() == 'gamma':
                if Fs>=158*2:
                    Wp, Ws, rp, rs = [30, 150], [20, 160], -20*np.log10(.95), -20*np.log10(.1)
                else:
                    Wp, Ws, rp, rs, btype = 30, 20, -20*np.log10(.95), -20*np.log10(.1), 'highpass'
                    btype = 'lowpass' if btype.lower()=='bandstop' else 'highpass'
            else:
                message  = 'Brainwave \"',bandwidth[i], '\" not exist. \n'
                message += 'Choose between delta, theta, alpha, beta, gamma, gamma_low, gamma_high'
                raise ValueError(message)

            if btype.lower() == 'bandstop':
                # simply reverse bandpass and stopband
                Wp, Ws = Ws, Wp
    
    Wp, Ws = np.array(Wp)/(Fs/2), np.array(Ws)/(Fs/2)
    if (order is None) or (Wn is None):
        if filter_type.lower()=='butter':
            order, Wn = signal.buttord(Wp, Ws, rp, rs)
        elif filter_type.lower()=='ellip':
            order, Wn = signal.ellipord(Wp, Ws, rp, rs)
        elif filter_type.lower()=='cheby1':
            order, Wn = signal.cheb1ord(Wp, Ws, rp, rs)
        elif filter_type.lower()=='cheby2':
            order, Wn = signal.cheb2ord(Wp, Ws, rp, rs)
    
    if filter_type.lower()=='butter':
        b, a = signal.butter(order, Wn, btype)
    elif filter_type.lower()=='ellip':
        b, a = signal.ellip(order, rp, rs, Wn, btype)
    elif filter_type.lower()=='cheby1':
        b, a = signal.cheby1(order,rp, Wn, btype)
    elif filter_type.lower()=='cheby2':
        b, a = signal.cheby2(order, rs, Wn, btype)
    
    return b, a


# SISTEMA WP E WS (METTI FS PER CAPIRE SE NORMALIZZARE O MENO)
def filter_lowpass(x: "array or tensor",
                   Fs: float,
                   Wp: float=50,
                   Ws: float=70,
                   rp: float=-20*np.log10(.95), 
                   rs: float=-20*np.log10(.15),
                   filter_type: str='butter',
                   order: int=None, 
                   Wn: float=None,
                   a: Union[np.ndarray,float]=None,
                   b: Union[np.ndarray,float]=None,
                   return_filter_coeff: bool=False
                  ):
    """
    
    filter_lowpass apply a lowpass filter on the last dimension of the given input x.
    
    filter_lowpass apply a designed lowpass filter on the last dimension of x. If a and b coefficient are not 
    given, calls get_filter_coeff with the other arguments to get them. The filter dedign follow this order:
                            (Wp,Ws,rp,rs) ----> (Wn, order) -----> (a,b). 
    Therefore the arguments closer to a and b in the scheme are used to get the filter coefficient.

    NOTE 1: lots of parameters are the ones used to call scipy's matlab style filters, ASIDE TO 'Wp' and 'Ws' which
    you must give directly in Hz. The normalization to [0,1] with respect to the half-cycles / sample 
    (i.e. Nyquist frequency) is done directly inside the get_filter_coeff function

    NOTE 2: pytorch filtfilt works differently on edges and is pretty unstable with high order filters, so avoid 
    restrictive condition which can increase the order of the filter.
    
    Parameters
    ----------
    x: N-D array or Tensor
        The element to filter
    Fs: float
        the sampling frequency in Hz
    Wp: float, optional
        bandpass in Hz
        Default: 50
    Ws: float, optional
        stopband in Hz
        Default: 70
    rp: float, optional
        ripple at bandpass in decibel. 
        Default: -20*log10(0.95)
    rs: float, optional
        ripple at stopband in decibel. 
        Default: -20*log10(0.15)
    filter_type: str, optional
        which filter design. Accepted values are 'butter', 'ellip', 'cheby1', 'cheby2'
        Default: 'butter'
    order: int, optional
        the order of the filter
        Default: None
    Wn: array_like, optional
        the critical frequency or frequencies.
        Default: None
    a: array_like, optional
        the denominator coefficient of the filter
        Default: None
    b: array_like, optional
        the numerator coefficient of the filer
        Default: None
    return_filter_coeff: bool, optional
        whether to return the filter coefficient or not
        Default: False
    """
    
    
    if filter_type not in ['butter', 'ellip', 'cheby1', 'cheby2']:
        raise ValueError('filter type not supported. Choose between butter, elliptic, cheby1, cheby2')
    
    if (a is None) or (b is None):
        b, a = get_filter_coeff(Fs=Fs, Wp = Wp, Ws = Ws, rp = rp, rs = rs, btype = 'lowpass', 
                                filter_type = filter_type, order = order, Wn = Wn, eeg_band = None
                               )
         
    if isinstance(x, np.ndarray):
        x_filt = signal.filtfilt(b, a, x, padtype='constant')  
    else:
        a= torch.from_numpy(a).to(dtype=x.dtype, device=x.device)
        b= torch.from_numpy(b).to(dtype=x.dtype, device=x.device)
        x_filt = filtfilt(x, a, b, clamp=False)   
    
    if return_filter_coeff:
        return x_filt, b, a
    else:
        return x_filt

    
def filter_highpass(x: "array or tensor", 
                    Fs: float,
                    Wp: float=30,
                    Ws: float=13,
                    rp: float=-20*np.log10(.95), 
                    rs: float=-20*np.log10(.15),
                    filter_type: str='butter',
                    order: int=None, 
                    Wn: float=None,
                    a: Union[np.ndarray,float]=None,
                    b: Union[np.ndarray,float]=None,
                    return_filter_coeff: bool=False
                   ):
    
    """
    
    filter_highpass apply a highpass filter on the last dimension of the given input x.
    
    filter_highpass apply a designed highpass filter on the last dimension of x. If a and b coefficient are not 
    given, calls get_filter_coeff with the other arguments to get them. The filter dedign follow this order:
                            (Wp,Ws,rp,rs) ----> (Wn, order) -----> (a,b). 
    Therefore the arguments closer to a and b in the scheme are used to get the filter coefficient.

    NOTE 1: lots of parameters are the ones used to call scipy's matlab style filters, ASIDE TO 'Wp' and 'Ws' which
    you must give directly in Hz. The normalization to [0,1] with respect to the half-cycles / sample 
    (i.e. Nyquist frequency) is done directly inside the get_filter_coeff function

    NOTE 2: pytorch filtfilt works differently on edges and is pretty unstable with high order filters, so avoid 
    restrictive condition which can increase the order of the filter.
    
    Parameters
    ----------
    x: N-D array or Tensor
        The element to filter
    Wp: float, optional
        bandpass in Hz
        Default: 30
    Ws: float, optional
        stopband in Hz
        Default: 13
    rp: float, optional
        ripple at bandpass in decibel. 
        Default: -20*log10(0.95)
    rs: float, optional
        ripple at stopband in decibel. 
        Default: -20*log10(0.15)
    filter_type: str, optional
        which filter design. Accepted values are 'butter', 'ellip', 'cheby1', 'cheby2'
        Default: 'butter'
    order: int, optional
        the order of the filter
        Default: None
    Wn: array_like, optional
        the critical frequency or frequencies.
        Default: None
    a: array_like, optional
        the denominator coefficient of the filter
        Default: None
    b: array_like, optional
        the numerator coefficient of the filer
        Default: None
    return_filter_coeff: bool, optional
        whether to return the filter coefficient or not
        Default: False
    """
    
    if filter_type not in ['butter', 'ellip', 'cheby1', 'cheby2']:
        raise ValueError('filter type not supported. Choose between butter, elliptic, cheby1, cheby2')
    
    if (a is None) or (b is None):
        b, a = get_filter_coeff(Fs=Fs, Wp = Wp, Ws = Ws, rp = rp, rs = rs, btype = 'highpass', 
                                filter_type = filter_type, order = order, Wn = Wn,eeg_band = None,
                               )
         
    if isinstance(x, np.ndarray):
        x_filt = signal.filtfilt(b, a, x, padtype='constant' )  
    else:
        a= torch.from_numpy(a).to(dtype=x.dtype, device=x.device)
        b= torch.from_numpy(b).to(dtype=x.dtype, device=x.device)
        x_filt = filtfilt(x, a, b, clamp=False)   
    
    if return_filter_coeff:
        return x_filt, b, a
    else:
        return x_filt


def filter_bandpass(x: "array or tensor",
                    Fs: float,
                    Wp: list[float]=None,
                    Ws: list[float]=None,
                    rp: float=-20*np.log10(.95), 
                    rs: float=-20*np.log10(.05),
                    filter_type: str='butter',
                    order: int=None, 
                    Wn: float=None,
                    a: Union[np.ndarray,float]=None,
                    b: Union[np.ndarray,float]=None,
                    eeg_band: str=None,
                    return_filter_coeff: bool=False
                   ):
    """
    
    filter_bandpass apply a bandpass filter on the last dimension of the given input x.
    
    filter_bandpass apply a designed bandpass filter on the last dimension of x. If a and b coefficient are not 
    given, calls get_filter_coeff with the other arguments to get them. The filter dedign follow this order:
                            (Wp,Ws,rp,rs) ----> (Wn, order) -----> (a,b). 
    Therefore the arguments closer to a and b in the scheme are used to get the filter coefficient.
    
    If eeg_band are given, (Wp,Ws,rp,rs) are bypassed and instantiated according to the eeg band specified. The
    priority order remain, so if (Wn,order) or (a,b) are given, filter is created according to the described priority
    order

    NOTE 1: lots of parameters are the ones used to call scipy's matlab style filters, ASIDE TO 'Wp' and 'Ws' which
    you must give directly in Hz. The normalization to [0,1] with respect to the half-cycles / sample 
    (i.e. Nyquist frequency) is done directly inside the get_filter_coeff function

    NOTE 2: pytorch filtfilt works differently on edges and is pretty unstable with high order filters, so avoid 
    restrictive condition which can increase the order of the filter.
    
    Parameters
    ----------
    x: N-D array or Tensor
        The element to filter
    Fs: float
        the sampling frequency in Hz
    Wp: float, optional
        bandpass in Hz.
        Default: 30
    Ws: float, optional
        stopband in Hz
        Default: 13
    rp: float, optional
        ripple at bandpass in decibel. 
        Default: -20*log10(0.95)
    rs: float, optional
        ripple at stopband in decibel. 
        Default: -20*log10(0.15)
    filter_type: str, optional
        which filter design. Accepted values are 'butter', 'ellip', 'cheby1', 'cheby2'
        Default: 'butter'
    order: int, optional
        the order of the filter
        Default: None
    Wn: array_like, optional
        the critical frequency or frequencies.
        Default: None
    a: array_like, optional
        the denominator coefficient of the filter
        Default: None
    b: array_like, optional
        the numerator coefficient of the filer
        Default: None
    eeg_band: str, optional
        any of the possible EEG bands. Accepted values are "delta", "theta", "alpha", "beta", 
        "gamma", "gamma_low", "gamma_high". Note: eeg_band bypass any Wp and Ws, if given
        Default: None
    return_filter_coeff: bool, optional
        whether to return the filter coefficient or not
        Default: False
        
    NOTE: pytorch filtfilt works differently on edges and is pretty unstable with high order filters, so avoid 
    restrictive condition which can increase the order of the filter.
    """
    
    if filter_type not in ['butter', 'ellip', 'cheby1', 'cheby2']:
        raise ValueError('filter type not supported. Choose between butter, elliptic, cheby1, cheby2')
    
    if (a is None) or (b is None):
        b, a = get_filter_coeff(Fs=Fs, Wp = Wp, Ws = Ws, rp = rp, rs = rs, btype = 'bandpass', 
                                filter_type = filter_type, order = order, Wn = Wn,eeg_band = eeg_band, 
                               )
         
    if isinstance(x, np.ndarray):
        x_filt = signal.filtfilt(b, a, x, padtype='constant' )  
    else:
        a= torch.from_numpy(a).to(dtype=x.dtype, device=x.device)
        b= torch.from_numpy(b).to(dtype=x.dtype, device=x.device)
        x_filt = filtfilt(x, a, b, clamp=False) 
    
    if return_filter_coeff:
        return x_filt, b, a
    else:
        return x_filt


def filter_bandstop(x: "array or tensor",
                    Fs: float,
                    Wp: list[float]=None,
                    Ws: list[float]=None,
                    rp: float=-20*np.log10(.95), 
                    rs: float=-20*np.log10(.05),
                    filter_type: str='butter',
                    order: int=None, 
                    Wn: float=None,
                    a: Union[np.ndarray,float]=None,
                    b: Union[np.ndarray,float]=None,
                    eeg_band: str=None,
                    return_filter_coeff: bool=False
                   ):
    """
    
    filter_bandstop apply a bandstop filter on the last dimension of the given input x.
    
    filter_bandstop apply a designed bandstop filter on the last dimension of x. If a and b coefficient are not 
    given, calls get_filter_coeff with the other arguments to get them. The filter dedign follow this order:
                            (Wp,Ws,rp,rs) ----> (Wn, order) -----> (a,b). 
    Therefore the arguments closer to a and b in the scheme are used to get the filter coefficient.
    
    If eeg_band are given, (Wp,Ws,rp,rs) are bypassed and instantiated according to the eeg band specified. The
    priority order remain, so if (Wn,order) or (a,b) are given, filter is created according to such parameters.

    NOTE 1: lots of parameters are the ones used to call scipy's matlab style filters, ASIDE TO 'Wp' and 'Ws' which
    you must give directly in Hz. The normalization to [0,1] with respect to the half-cycles / sample 
    (i.e. Nyquist frequency) is done directly inside the get_filter_coeff function.

    NOTE 2: pytorch filtfilt works differently on edges and is pretty unstable with high order filters, so avoid 
    restrictive condition which can increase the order of the filter.
    
    Parameters
    ----------
    x: N-D array or Tensor
        The element to filter
    Fs: float
        the sampling frequency in Hz
    Wp: float, optional
        bandpass in Hz.
        Default: 30
    Ws: float, optional
        stopband in Hz
        Default: 13
    rp: float, optional
        ripple at bandpass in decibel. 
        Default: -20*log10(0.95)
    rs: float, optional
        ripple at stopband in decibel. 
        Default: -20*log10(0.15)
    filter_type: str, optional
        which filter design. Accepted values are 'butter', 'ellip', 'cheby1', 'cheby2'
        Default: 'butter'
    order: int, optional
        the order of the filter
        Default: None
    Wn: array_like, optional
        the critical frequency or frequencies.
        Default: None
    a: array_like, optional
        the denominator coefficient of the filter
        Default: None
    b: array_like, optional
        the numerator coefficient of the filer
        Default: None
    eeg_band: str, optional
        any of the possible EEG bands. Accepted values are "delta", "theta", "alpha", "beta", 
        "gamma", "gamma_low", "gamma_high". Note: eeg_band bypass any Wp and Ws, if given
        Default: None
    return_filter_coeff: bool, optional
        whether to return the filter coefficient or not
        Default: False
        
    NOTE: pytorch filtfilt works differently on edges and is pretty unstable with high order filters, so avoid 
    restrictive condition which can increase the order of the filter.
    """
    
    if filter_type not in ['butter', 'ellip', 'cheby1', 'cheby2']:
        raise ValueError('filter type not supported. Choose between butter, elliptic, cheby1, cheby2')
    
    if (a is None) or (b is None):
        b, a = get_filter_coeff(Fs=Fs, Wp = Wp, Ws = Ws, rp = rp, rs = rs, btype = 'bandstop', 
                                filter_type = filter_type, order = order, Wn = Wn, eeg_band = eeg_band 
                               )
    print(b, a)    
    if isinstance(x, np.ndarray):
        x_filt = signal.filtfilt(b, a, x, padtype='constant' )  
    else:
        a= torch.from_numpy(a).to(dtype=x.dtype, device=x.device)
        b= torch.from_numpy(b).to(dtype=x.dtype, device=x.device)
        x_filt = filtfilt(x, a, b, clamp=False) 
    
    if return_filter_coeff:
        return x_filt, b, a
    else:
        return x_filt

# --- PERMUTATIONS ---
def get_eeg_channel_network_names():
    
    DMN= np.array(['AF4', 'AF7', 'AF8', 'AFZ', 'CP3', 'CP4', 'CP5', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6',
                   'F7', 'F8', 'FC1', 'FC3', 'FC4', 'FC5', 'FP1', 'FP2', 'FPZ', 'FT10', 'FT8', 'FT9',
                   'FZ', 'P3', 'P4', 'P5', 'T7', 'T8', 'TP7', 'TP8'], dtype='<U4')
    DAN= np.array(['C5', 'C6', 'CP1', 'CP2', 'CPZ', 'FC1', 'FC2', 'FC5', 'P1', 'P2',
                   'P7', 'P8', 'PO3', 'PO4', 'PO7', 'PO8', 'POZ', 'PZ', 'T7', 'TP8'], dtype='<U4')
    VAN= np.array(['AF3', 'AF4', 'AF8', 'C5', 'C6', 'CP1', 'CP2', 'CP4', 'CP5', 'CP6',
                   'CPZ', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6', 'FT7', 'P1', 'P2',
                   'P7', 'P8', 'PO3', 'PO4', 'PO7', 'PO8', 'POZ', 'PZ', 'T7', 'TP8'], dtype='<U4')
    SMN= np.array(['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CP1', 'CP2', 'CP5', 'CPZ',
                   'CZ', 'FC5', 'FC6', 'FCZ', 'FT8', 'FTZ', 'P3', 'P5','P6', 'P7', 'P8',
                   'PO4', 'PO7', 'PO8', 'T7', 'T8', 'TP7'], dtype='<U4')
    VFN= np.array(['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CP1', 'CP2', 'CPZ', 'FC1',
                   'FC5', 'FC6', 'FT8', 'FTZ', 'O1', 'O2', 'OZ', 'P7', 'P8', 'PO3',
                   'PO4', 'PO7', 'PO8', 'POZ', 'PZ', 'T7', 'TP8'], dtype='<U4') 
    FPN= np.array(['AF3', 'AF4', 'AF7', 'AF8', 'AFZ', 'C6', 'CP3', 'CP4', 'CP5',
                   'CP6', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FC1',
                   'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FP1', 'FP2', 'FPZ', 'FT10',
                   'FT7', 'FT9', 'FZ', 'P3', 'P4', 'P5', 'T7', 'T8', 'TP7', 'TP8'], dtype='<U4')
    print('Default Mode Network - DMN')
    print(DMN)
    print('')
    print('Dorsal Attention Network - DAN')
    print(DAN)
    print('')
    print('Ventral Attention Network - VAN')
    print(VAN)
    print('')
    print('SomatoMotor functional Network - SMN')
    print(SMN)
    print('')
    print('Visual Functional Network - VFN')
    print(VFN)
    print('')
    print('FrontoParietal Network - FPN')
    print(FPN)
    print('')

    
def get_channel_map_and_networks(channel_map: list=None,
                                 chan_net: list[str]='all',
                                ):
    """
    
    get_channel_map_and_networks simply return the channel_map and chan_net argument for permute_channels.
    Run help(permute_channels) to get more informations.
    """
    
    
    if channel_map is None:
        channel_map = np.array(['FP1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5',
                                'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3',
                                'CP1', 'P1', 'P3', 'P5', 'P7', 'PO7', 'PO3', 'O1', 'OZ',
                                'POZ', 'PZ', 'CPZ', 'FPZ', 'FP2', 'AF8', 'AF4', 'AFZ', 'FZ',
                                'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCZ',
                                'CZ', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2',
                                'P2', 'P4', 'P6', 'P8', 'PO8', 'PO4', 'O2'], dtype='<U4')
    elif isinstance(channel_map, list):
        channel_map = np.array(channel_map, dtype='<U4')
    
    # define networks (according to rojas et al. 2018)
    DMN= np.array(['AF4', 'AF7', 'AF8', 'AFZ', 'CP3', 'CP4', 'CP5', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6',
                   'F7', 'F8', 'FC1', 'FC3', 'FC4', 'FC5', 'FP1', 'FP2', 'FPZ', 'FT10', 'FT8', 'FT9',
                   'FZ', 'P3', 'P4', 'P5', 'T7', 'T8', 'TP7', 'TP8'], dtype='<U4')
    DAN= np.array(['C5', 'C6', 'CP1', 'CP2', 'CPZ', 'FC1', 'FC2', 'FC5', 'P1', 'P2',
                   'P7', 'P8', 'PO3', 'PO4', 'PO7', 'PO8', 'POZ', 'PZ', 'T7', 'TP8'], dtype='<U4')
    VAN= np.array(['AF3', 'AF4', 'AF8', 'C5', 'C6', 'CP1', 'CP2', 'CP4', 'CP5', 'CP6',
                   'CPZ', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6', 'FT7', 'P1', 'P2',
                   'P7', 'P8', 'PO3', 'PO4', 'PO7', 'PO8', 'POZ', 'PZ', 'T7', 'TP8'], dtype='<U4')
    SMN= np.array(['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CP1', 'CP2', 'CP5', 'CPZ',
                   'CZ', 'FC5', 'FC6', 'FCZ', 'FT8', 'FTZ', 'P3', 'P5','P6', 'P7', 'P8',
                   'PO4', 'PO7', 'PO8', 'T7', 'T8', 'TP7'], dtype='<U4')
    VFN= np.array(['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CP1', 'CP2', 'CPZ', 'FC1',
                   'FC5', 'FC6', 'FT8', 'FTZ', 'O1', 'O2', 'OZ', 'P7', 'P8', 'PO3',
                   'PO4', 'PO7', 'PO8', 'POZ', 'PZ', 'T7', 'TP8'], dtype='<U4') 
    FPN= np.array(['AF3', 'AF4', 'AF7', 'AF8', 'AFZ', 'C6', 'CP3', 'CP4', 'CP5',
                   'CP6', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FC1',
                   'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FP1', 'FP2', 'FPZ', 'FT10',
                   'FT7', 'FT9', 'FZ', 'P3', 'P4', 'P5', 'T7', 'T8', 'TP7', 'TP8'], dtype='<U4')
    networks =[DMN, DAN, VAN, SMN, VFN, FPN]

    if isinstance(chan_net, str):
        chan_net = [chan_net]

    net_idx=[]
    for i in range(len(chan_net)):
        if chan_net[i].lower() == 'all':
            net_idx = [0,1,2,3,4,5]
            break
        elif chan_net[i].lower() == 'dmn':
            net_idx.append(0)
        elif chan_net[i].lower() == 'dan':
            net_idx.append(1)
        elif chan_net[i].lower() == 'van':
            net_idx.append(2)
        elif chan_net[i].lower() == 'smn':
            net_idx.append(3) 
        elif chan_net[i].lower() == 'vfn':
            net_idx.append(4)
        elif chan_net[i].lower() == 'fpn':
            net_idx.append(5)
        else:
            raise ValueError('brain network not supported. Can be any of DMN, DAN, VAN, SMN, VFN, FPN')

    for index in sorted( set([0,1,2,3,4,5])-set(net_idx) , reverse=True):
        networks.pop(index)
    random.shuffle(networks)

    return channel_map, networks
    

def permute_channels(x, 
                     chan2shuf: int=-1,
                     mode: str="random",
                     channel_map: list=None,
                     chan_net: list[str]='all',
                     batch_equal: bool=False
                    ):
    """
    
    permutation_channels permute the input tensor EEG signals x along the channel dimension (second to last).
    
    Given an input x where the last two dimension must be (EEG_channels x EEG_samples), permutation_channels 
    shuffles all or a subset of the eeg along its channels. Shuffles can be done randomly or using specific
    networks (based on resting state functional connectivity networks).
    If batch_equal is set to False, call the function recursively along each of the (N_{1}*N_{2}...*N{-3}) 
    dimensions of the tensor.
    
    Parameter
    ---------
    
    x: N-D Tensor or numpy array
        The element to shuffle. The last two dimensions must be (EEG_channel x EEG_samples), which means that the 
        permutation is applied on the second to last dimension. 
    chan2shuf: int, optional
        The number of channels to shuffle. Must be greater than 1. -1 is the only accepted negative number and 
        means permute all the segments.
        Default: -1
    mode: str, optional
        How to permute the channels. Can be any of:
            'random': shuffle channels at random
            'network': shuffle channels which belongs to the same network. A network is a subset of channels whose
                       activity is (with a minumum degree) between each other. This mode support only a subset of
                       61 channels of the 10-10 system
        Default: 'random'
    channel_map: list of str, optional
        The channel map of EEG acquisitions. Must be a list of string or a numpy array of dtype='<U4' with channel 
        names as elements. Channel name must be defined with capital letters (e.g. 'P04', 'FC5').
        Default: np.array(['FP1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 
                            'C5', 'T7', 'TP7', 'CP5', 'CP3','CP1', 'P1', 'P3', 'P5', 'P7', 'PO7', 'PO3', 'O1', 
                            'OZ', 'POZ', 'PZ', 'CPZ', 'FPZ', 'FP2', 'AF8', 'AF4', 'AFZ', 'FZ', 'F2', 'F4', 'F6', 
                            'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCZ', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 
                            'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'PO8', 'PO4'],dtype='<U4')
    chan_net: str or list of str, optional
        The list of networks to use if network mode is selected. Must be a list of string or a single string.
        Supported networks are DMN, DAN, VAN, SMN, VFN, FPN. Use 'all' to select all networks. To get a list of
        the channel names per network use get_eeg_network_channel_names()
        Default: 'all'
    batch_equal: bool, optional
        whether to apply the same permutation to all EEG record or not. If True, permute_signal is called 
        recursively for each dimension of the batch until the last two are reached (e.g. given a tensor x
        of dimension (16,8,64,512) the function permute each of the 16*8 EEG signals (64 channels of 512 samples)
        individually).
        Default: False
    
    """
    Nchan=x.shape[-2]
    Ndim= len(x.shape)
    
    # Check if given input is ok 
    if (chan2shuf==1) or (chan2shuf==0) or (chan2shuf>Nchan):
        msgLog='chan2shuf must be bigger than 1 and smaller than the number of channels in the recorded EEG. \n '
        msgLog += 'Default value is -1, which means all EEG channels are shuffled'
        raise ValueError(msgLog)
    if Ndim==1:
        raise ValueError('x must be an array or tensor with the last two dimensions [channel]*[time window]')
    
    chan2shuf = x.shape[-2] if chan2shuf==-1 else chan2shuf 
    x2= np.copy(x) if isinstance(x, np.ndarray) else torch.clone(x)
    if (Ndim<3) or (batch_equal):
        
        if mode.lower()=='network':
            # Define or check channel map and channel networks
            channel_map , networks = get_channel_map_and_networks(channel_map, chan_net)
            if channel_map.shape[0] != x.shape[-2]:
                raise ValueError('channel map does not match the number of channels in eeg recording')
            
            # get index of channels in networks
            is_in_channel_map=np.full(channel_map.shape, False)
            for i in range(len(networks)):
                idx= np.where(np.in1d(channel_map, networks[i]))[0]
                is_in_channel_map[idx]=True
            chan_idx=is_in_channel_map.nonzero()[0]

            # randomly select a number of channels equals to chan2shuf
            idxor_full = np.random.permutation(chan_idx)[:chan2shuf]
            idxor_full = np.sort(idxor_full)
            idx_full=np.full(chan2shuf, -1, dtype=int)

            # shuffle according to the selected networks
            for k in range(len(networks)):
                idxor = np.where(np.in1d(channel_map[idxor_full], networks[k]))[0] #identify chans idx
                idxor = idxor[np.where(idx_full[idxor]==-1)[0]] # keep only non shuffled channels
                idx = (idxor_full[idxor]) #get chans idx
                if idx.shape[0]>1:
                    while len(np.where(idx==idxor_full[idxor])[0])>0:
                        np.random.shuffle(idx)
                idx_full[idxor]=idx

            # final results
            idxor = idxor_full
            idx = idx_full   
            if not(isinstance(x,np.ndarray)):
                idxor = torch.from_numpy(idxor_full).to(device=x.device)
                idx = torch.from_numpy(idx_full).to(device=x.device)

        
        # random mode shuffle channels at random
        elif mode.lower()=='random':
            if isinstance(x, np.ndarray):
                idx = np.arange(Nchan, dtype=int)
                np.random.shuffle(idx)
                idx = idx[:chan2shuf]
                idxor = np.sort(idx)
                if len(idx)>1:
                    while len(np.where(idx==idxor)[0])>0:
                        np.random.shuffle(idx)
            else:
                idx = torch.randperm(Nchan)
                idx = idx[:chan2shuf]
                idxor, _ = torch.sort(idx)
                if len(idx)>1:
                    while torch.sum(torch.eq(idx,idxor))!=0:
                        idx = idx[torch.randperm(idx.shape[0])]
                if x.device.type!='cpu':
                    idx = idx.to(device=x.device)
                    idxor= idxor.to(device=x.device)

        # apply defined shuffle
        xtemp = x[..., idx, :]
        x2[...,idxor,:] = xtemp
    
    else:
        # call recursively for each dimension until last 2 are reached
        for i in range(x.shape[0]):
            x2[i] = permute_channels(x[i], chan2shuf= chan2shuf, mode=mode, 
                                     channel_map=channel_map, chan_net=chan_net)
               
    return x2


def permutation_signal(x, 
                       segments: int=10, 
                       seg_to_per: int=-1,
                       batch_equal: bool=False
                      ):
    """
    
    permutation_signal permute some portion of the last dimension of the input N-D array_like x
    
    Given an input x where the last two dimension must be (EEG_channels x EEG_samples), permutation_signal 
    divides the elements of the last dimension of x into N segments, then chooses M<=N segments and shuffle it. 
    Permutations are equally performed along each Channel of the same EEG. 
    If batch_equal is set to False, call the function recursively along each of the (N_{1}*N_{2}...*N{-3}) 
    dimensions of the tensor.
    
    Parameter
    ---------
    
    x: N-D Tensor or numpy array
        The element to shuffle. The last two dimensions must be (EEG_channel x EEG_samples), which means that the 
        same permutation is applied to all the channels of the EEG signal.
    segments: int, optional
        The number of segments in which the last dimension of x must be divided. Must be greater than 1
        Default: 1
    seg_to_per: int, optional
        The number of segments to permute. Must be greater than 1 and lower than segments. -1 is the only
        accepted negative number and means permute all the segments.
        Default: -1
    batch_equal: bool, optional
        whether to apply the same permutation to all EEG record or not. If True, permute_signal is called 
        recursively for each dimension of the batch until the last two are reached (e.g. given a tensor x
        of dimension (16,8,64,512) the function permute each of the 16*8 EEG signals (64 channels of 512 samples)
        individually).
        Default: False
    """
    
    if segments<1:
        raise ValueError('segments cannot be less than 2')     
    if seg_to_per<1:
        if seg_to_per>=0:
            raise ValueError('seg_to_per must be bigger than 1 (put -1 to permute all segments)')
        elif seg_to_per==-1:
            seg_to_per=segments
        elif (seg_to_per<(-1)):
            msgError='got a negative number of segments to permute. Only -1 to permute all segments is allowed'
            raise ValueError(msgError)
    elif seg_to_per>segments:
        raise ValueError('number of segment to permute is bigger than the number of segment')
    
    Ndim=len(x.shape)
    if (Ndim<3) or (batch_equal):
        
        segment_len= x.shape[-1] // segments
        idx1=np.arange(segments)
        np.random.shuffle(idx1)
        idx2 = np.sort(idx1[:seg_to_per])
        idx1=np.sort(idx1)
        idx3 = np.copy(idx2)
        while len(np.where(idx2==idx3)[0])>0:
            np.random.shuffle(idx3)
        idx1[idx2]=idx3
        full_idx = np.arange(x.shape[-1])
        for k in range(len(idx1)):
            if idx1[k]!= k:
                start=segment_len*idx1[k]
                start2 = segment_len*k
                newidx = np.arange(start, start+segment_len)
                full_idx[start2: start2+segment_len]= newidx
                
        if not(isinstance(x, np.ndarray)):
            full_idx2=torch.from_numpy(full_idx).to(device=x.device)
            x2 = x[..., full_idx2]
        else:
            x2 = x[..., full_idx]
    
    else:
        x2 = np.empty_like(x) if isinstance(x, np.ndarray) else torch.empty_like(x)
        for i in range(x.shape[0]):
            x2[i] = permutation_signal(x[i], segments=segments, seg_to_per=seg_to_per, batch_equal=batch_equal)
            
    return x2
    
    

# --- CROP AND RESIZE ---
def torch_pchip(x: "1D Tensor", 
                y: "ND Tensor", 
                xv: "1D Tensor",
                save_memory: bool=True,
                new_y_max_numel: int=4194304
               ):
    """
    
    torch_pchip perform pchip interpolation on the last dimension of the input tensor y.
    
    This function is a pytorch adaptation of the scipy's pchip_interpolate. It performs sp-pchip interpolation
    (Shape Preserving Piecewise Cubic Hermite Interpolating Polynomial) on the last dimension of the y tensor.
    x is the original time grid and xv new virtual grid. So, the new values of y at time xv are given by the 
    polynomials evaluated at the time grid x.
    
    Parameter
    ---------
    
    x: 1D Tensor
        Tensor with the original time grid. Must be the same length as the last dimension of y
    y: ND Tensor
        Tensor to interpolate. The last dimension must have the signals to interpolate
    xv: 1D Tensor
        Tensor with the new virtual grid, i.e. the time points where to interpolate
    save_memory: bool, optional
        whether to perform the interpolation on subsets of the y tensor by recursively function calls or not.
        Does not apply if y is a 1-D tensor. If set to False memory usage can drastically increase 
        (for example with a 128 MB tensor, the memory usage of the function is 1.2 GB), but in some devices 
        it can speed up the process. However, this is not the case for all devices and performance may increase
        (see example below run on an old computer).
        Default: True
    new_y_max_numel: int, optional
        The number of elements which the tensor needs to surpass to make the function starting recursive calls.
        It can be considered as an indicator of the maximum allowed memory usage since slower the number, slower
        the memory used. 
        Default: 256*1024*16 (approximately 16s of recording of a 256 Channel EEG sampled at 1024 Hz)
    
    
    Some technical information and difference with other interpolation:
        https://blogs.mathworks.com/cleve/2012/07/16/splines-and-pchips/
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html
    Some parts of the code are inspired from: 
        https://github.com/scipy/scipy/blob/v1.10.1/scipy/interpolate/_cubic.py#L157-L302
    
    Example
    -------
    # result on my (old) computer with a 4D tensor/array with dim (1024,1,96,512) and xv with dim (1024,)
    #         torch_pchip no save memory | torch_pchip save memory | scipy pchip_interpolate
    #                   15.48s                     5.33s                     21.96s
    
    """
    
    if len(x.shape)!= 1:
        raise ValueError(['Expected 1D Tensor for x but received a ', str(len(x.shape)), '-D Tensor']) 
    if len(xv.shape)!= 1:
        raise ValueError(['Expected 1D Tensor for xv but received a ', str(len(xv.shape)), '-D Tensor'])
    if x.shape[0] != y.shape[-1]:
        raise ValueError('x must have the same length than the last dimension of y')

    # Initialize the new interpolated tensor
    Ndim=len(y.shape)
    new_y=torch.empty(( *y.shape[:(Ndim-1)], xv.shape[0]), device=y.device)
    
    # If save_memory and the new Tensor size is huge, call recursively for each element in the first dimension 
    if save_memory:
        if Ndim>1:
            if ((torch.numel(y)/y.shape[-1])*xv.shape[0])>new_y_max_numel:
                for i in range(new_y.shape[0]):
                    new_y[i] = torch_pchip(x, y[i], xv)
                return new_y
    
    
    # This is a common part for every channel
    if x.device.type=='mps' or xv.device.type=='mps':
        # torch bucketize is not already implemented in mps unfortunately
        # need to pass in cpu and return to mps. Note that this is very slow
        # like 500 times slower. But at least it doesn't throw an error 
        bucket=torch.bucketize(xv.to(device='cpu'), x.to(device='cpu')) -1
        bucket= bucket.to(device=x.device)
    else:
        bucket = torch.bucketize(xv, x) -1
    bucket = torch.clamp(bucket, 0, x.shape[0]-2)
    tv_minus = (xv - x[bucket]).unsqueeze(1)
    infer_tv = torch.cat(( tv_minus**3, tv_minus**2, tv_minus, torch.ones(tv_minus.shape, device=tv_minus.device)), 1) 
    
    h = (x[1:]-x[:-1])
    Delta = (y[...,1:] - y[...,:-1]) /h
    k = (torch.sign(Delta[...,:-1]*Delta[...,1:]) > 0)
    w1 = 2*h[1:] + h[:-1]
    w2 = h[1:] + 2*h[:-1]
    whmean = (w1/Delta[...,:-1] + w2/Delta[...,1:]) / (w1 + w2)
    
    slope = torch.zeros(y.shape, device=y.device)
    slope[...,1:-1][k] = whmean[k].reciprocal()

    slope[...,0] = ((2*h[0]+h[1])*Delta[...,0] - h[0]*Delta[...,1])/(h[0]+h[1])
    slope_cond = torch.sign(slope[...,0]) != torch.sign(Delta[...,0])
    slope[...,0][slope_cond] = 0
    slope_cond = torch.logical_and( torch.sign(Delta[...,0]) != torch.sign(Delta[...,1]), 
                                   torch.abs(slope[...,0]) > torch.abs(3*Delta[...,0]) )
    slope[...,0][ slope_cond ] = 3*Delta[...,0][slope_cond]
    
    slope[...,-1] = ((2*h[-1]+h[-2])*Delta[...,-1] - h[-1]*Delta[...,-2])/(h[-1]+h[-2])
    slope_cond = torch.sign(slope[...,-1]) != torch.sign(Delta[...,-1])
    slope[...,-1][ slope_cond ] = 0
    slope_cond = torch.logical_and( torch.sign(Delta[...,-1]) != torch.sign(Delta[...,-1]), 
                                   torch.abs(slope[...,-1]) > torch.abs(3*Delta[...,1]) )
    slope[...,-1][ slope_cond ] = 3*Delta[...,-1][slope_cond]


    t = (slope[...,:-1] + slope[...,1:] - Delta - Delta)  / h 
    a = ( t )/ h
    b = (Delta - slope[...,:-1]) / h - t
    
    

    py_coef = torch.stack((a, b, slope[...,:-1], y[...,:-1]),-1)
    new_y = (py_coef[...,bucket,:] * infer_tv ).sum(axis=-1)
    return new_y


def warp_signal(x,
                segments: int=10,
                stretch_strength: float=2.,
                squeeze_strength: float=0.5,
                batch_equal: bool=False,
               ):
    
    """
    
    warp_signal stretch and squeeze portion of the last dimension of the input x 
    and resize to the original size.
    
    Given x a N-D Tensor where the last two dim are Channel x Signal, warp_signal:
    1) divide the last dimension of x into N segments
    2) select at random a subset segments
    3) stretch those segments according to stretch_strength
    4) squeeze other segments according to squeeze_strength
    5) resample x to the original dimension. For this part pchip interpolation 
       with a uniform virtual grid is used
    If batch_equal is set to False, call the function recursively and repeat step 1 to 5 for each 
    (N_{1}*N_{2}...*N{-3}) elements of the tensor, until the recursion gets a 2-dimensional sub-tensor.
    
    Parameters
    ----------
    x : N-D Tensor or numpy array
        The array to warp. It accepts any dimensions but the last 2 must be Channel x Signal.
        For example, if x is 4-D, it must be in the form (N1 x N2 x Channels x Signal), where N1 and N2 are usually
        the batch size and channels
    segments : int, optional
        The number of segments to consider when dividing the last dimension of x.
        Default: 10
    stretch_strength : float, optional
        The stretch power, i.e. a multiplication factor which determines the number of samples the stretched 
        segment must have. 
        Default: 2.
    squeeze_strength : float, optional
        The squeeze power. The same as stretch but for the segments to squeeze
        Default: 0.5
    batch_equal: bool, optional
        whether to apply the same warp to all records or not. True means faster computation but more memory
        consuption and less variability, False the opposite.
        Default: False
    """
    
    Ndim=len(x.shape)
    x_warped_final= np.empty_like(x) if isinstance(x, np.ndarray) else torch.empty_like(x, device=x.device)
    
    if batch_equal or Ndim<3:

        # set segment do stretch squeeze
        seglen= x.shape[-1] / segments
        seg_range = np.arange(segments)
        stretch = np.random.choice(seg_range, random.randint(1, segments//2), replace=False)
        squeeze = np.setdiff1d(seg_range, stretch)

        # pre-allocate warped vector to avoid continuous stack call
        Lseg = np.zeros((segments,2), dtype=int)
        Lseg[:,0] = (seg_range*seglen).astype(int)
        Lseg[:,1] = ( (seg_range+1)*seglen).astype(int)
        Lseg = Lseg[:,1] - Lseg[:,0]
        Lsegsum = np.cumsum(Lseg)

        x_size= [int(i) for i in x.shape]
        warped_len = int(np.sum(np.ceil(Lseg[stretch]*stretch_strength)) + 
                         np.sum(np.ceil(Lseg[squeeze]*squeeze_strength)) )
        x_size[-1]=warped_len

        # initialize warped array (i.e. the array where to allocate stretched and squeezed segments)
        x_warped = np.empty(x_size) if isinstance(x, np.ndarray) else torch.empty(x_size, device=x.device)
        
        # iterate over segments and stretch or squeeze each segment, then allocate to x_warped
        idx_cnt=0
        for i in range(segments):

            piece = x[..., int(i * seglen):int( (i + 1) * seglen)]
            if i in stretch:
                new_piece_dim = int(np.ceil(piece.shape[-1] * stretch_strength))
            else:
                new_piece_dim = int(np.ceil(piece.shape[-1] * squeeze_strength))

            if isinstance(x, np.ndarray):
                warped_piece = interpolate.pchip_interpolate(np.linspace(0, seglen-1, piece.shape[-1]), piece, 
                                                             np.linspace(0, seglen-1, new_piece_dim), axis=-1)
            else:
                warped_piece = torch_pchip( torch.linspace(0, seglen-1, piece.shape[-1], device=x.device), 
                                            piece, 
                                            torch.linspace(0, seglen-1, new_piece_dim, device=x.device))

            x_warped[..., idx_cnt : idx_cnt+new_piece_dim]=warped_piece
            idx_cnt += new_piece_dim
            
        # resample x_warped to fit original size
        if isinstance(x_warped, np.ndarray):
            x_warped_final = interpolate.pchip_interpolate(np.linspace(0, warped_len-1, warped_len), x_warped, 
                                                             np.linspace(0, warped_len-1, x.shape[-1]), axis=-1)
        else:
            x_warped_final = torch_pchip(torch.linspace(0, warped_len-1, warped_len, device=x.device), 
                                         x_warped, 
                                         torch.linspace(0, warped_len, x.shape[-1], device=x.device))
    
    
    else:
        # Recursively call until second to last dim is reached
        for i in range(x.shape[0]):
            x_warped_final[i] =  warp_signal(x[i] ,segments, stretch_strength,squeeze_strength, batch_equal)
     
    return x_warped_final


def crop_and_resize(x: 'N-D Tensor or array',
                    segments: int=10,
                    N_cut: int=1,
                    batch_equal: bool=False,
                   ):
    """
    
    crop_and_resize crop some segments of the last dimension of the input x and resize to the original dimension.
    
    Given x a N-D Tensor where the last two dim are EEG_Channel x EEG_Signal, crop_and_resize:
    1) divide the last dimension of x into N segments
    2) select at random a subset segments
    3) remove the selected segments from x
    4) create a new cropped version of x
    5) resample the new cropped version to the original dimension. For this part pchip interpolation 
       with a uniform virtual grid is used
    If batch_equal is set to False, call the function recursively and repeat step 1 to 5 for each 
    (N_{1}*N_{2}...*N{-3}) elements of the tensor.
    
    Parameters
    ----------
    x : N-D Tensor or numpy array
        The array to crop and resize. It accepts any dimensions but the last 2 must be EEG_Channel x EEG_Signal.
        For example, if x is 4-D, it must be in the form (N1 x N2 x Channels x Signal), where N1 and N2 are usually
        the batch size and channels
    segments : int, optional
        The number of segments to consider when dividing the last dimension of x.
        Default: 10
    N_cut : int, optional
        The number of segments to cut. 
        Default: 1
    batch_equal: bool, optional
        whether to apply the same crop to all EEG record or not. True means faster computation but more memory
        consuption and less variability, False the opposite
        Default: False
        
        
    Example
    -------
    dim = (16,1,64,512)
    segments=15
    N_cut=6
    
    x = torch.sin(torch.linspace(0,20*math.pi, dim[-1]))
    zero_tensor = torch.zeros(dim)
    x = zero_tensor + x
    # x = x.numpy() #the result won't change if x is a numpy array
    
    x_crop = crop_and_resize(x, segments= segments, N_cut= N_cut, batch_equal=True)
    print(torch.equal(x_crop[1], x_crop[2])) # True
    x_crop = crop_and_resize(x, segments= segments, N_cut= N_cut, batch_equal=False)
    print(torch.equal(x_crop[1], x_crop[2])) # False
    
    # plot the results
    plt.plot(xnp[0,0,0,:])
    plt.show()
    plt.plot(x_crop[0,0,0,:])
    plt.plot(x_crop[2,0,0,:])
    plt.show()
    
    """
    x_crop=np.empty_like(x) if isinstance(x, np.ndarray) else torch.empty_like(x, device=x.device)
    Ndim = len(x.shape)
    if (batch_equal) or (Ndim<3):
        
        segment_len= x.shape[-1] // segments
        if isinstance(x, np.ndarray):
            seg_to_rem = np.random.randint(0,segments, N_cut, dtype=int)
            idx_to_rem = np.empty(segment_len*N_cut, dtype=int)
            for i in range(seg_to_rem.shape[0]):
                start=segment_len*(seg_to_rem[i])
                idx1 = segment_len*i
                idx_to_rem[idx1 : idx1+segment_len]= np.linspace(start, start+segment_len-1, segment_len)

            new_x= np.delete(x, idx_to_rem, axis=-1)
            x_crop = interpolate.pchip_interpolate(np.linspace(0, x.shape[-1]-1, new_x.shape[-1]), 
                                                   new_x, np.linspace(0,x.shape[-1]-1,x.shape[-1]), axis=-1)
        else:

            seg_to_rem = torch.randperm(segments)[:N_cut]
            idx_to_rem = torch.empty(segment_len*N_cut, dtype=torch.int)
            for i in range(seg_to_rem.shape[0]):
                start=segment_len*(seg_to_rem[i])
                idx1 = segment_len*i
                idx_to_rem[idx1 : idx1+segment_len]= torch.linspace(start, start+segment_len-1, segment_len)

            # https://stackoverflow.com/questions/55110047/finding-non-intersection-of-two-pytorch-tensors
            allidx = torch.arange(x.shape[-1])
            combined = torch.cat( (allidx, idx_to_rem, idx_to_rem) )
            uniques, counts = combined.unique(return_counts=True)
            difference = uniques[counts == 1]
            difference = difference.to(device=x.device)
            new_x= x[...,difference]
            x_crop = torch_pchip(torch.linspace(0, x.shape[-1]-1, new_x.shape[-1], device=x.device), 
                                 new_x, 
                                 torch.linspace(0,x.shape[-1]-1,x.shape[-1], device=x.device))
    
    else:
        for i in range(x.shape[0]):
            x_crop[i] = crop_and_resize(x[i], segments, N_cut, batch_equal)

        
    return x_crop



# RE-REFERENCING
def change_ref(x: "N-D Tensor of numpy Array",
               mode: str or int='avg',
               reference: int=None,
               exclude_from_ref: int or list[int]=None,
              ):
    """
    
    change_ref change the reference of all EEG record in x.
    
    Given an input tensor or array x where the last two dimension must be (EEG_channels x EEG_samples),
    change_ref change the reference of the all the EEG records according to the modality specified.
    Currently, reference can be changed to:
    
    1) Channel reference (e.g. Cz). Each record of a channel is subtracted to the record of the Cz channel. 
       Cz(t) becomes 0 for all t
    2) Common Average Reference (CAR). Each record is subtracted with the average of all electrodes. Currently,
       it doesn't cover all particular cases as this implementation is minimalist.
    
    To get a more detailed description about re-referencing, check this brief background page of the EEGlab library
    at the following link: https://eeglab.org/tutorials/ConceptsGuide/rereferencing_background.html
    
    Parameter
    ---------
    
    x: N-D Tensor or numpy array
        The element to re-reference. It can be a batch of EEGs, but the last two dimensions 
        must be (EEG_channel x EEG_samples)
    mode: str or int, optional
        The re-reference modality. Accepted arguments:
            1) 0, 'chan', 'channel'. Single Channel re-referencing.
            2) 1, 'avg', 'average', 'car'. Common Avarage re-referencing.
        Default: 'avg'
    reference: int, optional
        The reference electrode, given as an int referring to the position in the EEG_channels dimension of X
        (remember that this means the first dim if x is a matrix or the second to last if x is N-dim)
        Default: None
    exclude_from_ref: int or list[int]
        Argument designed to exclude some channels during average re-referencing. This apply for example when x
        has records from nose tip or ear lobe.
        Default: None
        
    """
    
    Ndim=len(x.shape)
    if Ndim<2:
        raise ValueError('x must be at least a 2 dim array')
    
    if reference!=None:
        if isinstance(reference,int):
            if reference>=x.shape[-1]:
                raise ValueError('reference index exceeds the number of channels in x')
    else:
        reference=0
    
    if isinstance(mode,int):
        if mode>1 or mode<0:
            msgErr = 'mode not supported. Current available modes are \'avg\', which use the common average reference '
            msgErr += 'and \'Cz\', which support original vertex reference.'
            raise ValueError(msgErr)
    elif isinstance(mode,str):
        mode=mode.lower()
        if mode in ['avg', 'average','car']:
            mode=1
        elif mode in ['channel','chan']:
            mode=0
        else:
            msgErr = 'mode not supported. Current available modes are \'avg\', which use the common average reference '
            msgErr += 'and \'Cz\', which support original vertex reference.'
            raise ValueError(msgErr)
    else:
        raise ValueError('mode must be a string. Use help() to consult all available mode')
    
    if exclude_from_ref != None:
        if isinstance(exclude_from_ref,int):
            exclude_from_ref = [exclude_from_ref]
        else:
            for i in exclude_from_ref:
                if not(isinstance(i,int)):
                    raise ValueError('exclude_from_ref must be an int or a list of int')
                if i >= x.shape[-1]:
                    raise ValueError('exclude_from_ref indeces exceed the number of channels in x')
    
    if Ndim<3:
        if mode==1:
            if exclude_from_ref == None:
                Ref= x[[i for i in range(x.shape[0]) if i!=reference]].mean(0)
            else:
                Ref = x[[i for i in range(x.shape[0]) if i not in exclude_from_ref+[reference]]].mean(0)
        elif mode==0:
            Ref= x[reference]
        x_new_ref= x-Ref
        #if mode==0:
        #    x_new_ref[reference]= -Ref     
    else:
        x_new_ref= np.empty_like(x) if isinstance(x, np.ndarray) else torch.empty_like(x)
        for i in range(x.shape[0]):
            x_new_ref[i]=change_ref(x[i], mode=mode, reference=reference, exclude_from_ref=exclude_from_ref)
    
    return x_new_ref



def masking(x: "N-D Tensor of numpy Array",
            mask_number: int=1,
            masked_ratio: float=0.1,
            batch_equal: bool=False
           ):
    """
    
    masking put portions of the input argument to 0.
    
    Given a N-dim tensor or a numpy array, masking put a portion of the tensor/array to 0 along the last dimension.
    If the tensor has more than the one dimension, the last two dimensions must be (Channel x Signal) and the function 
    will mask each channel of the signal equally.
    
    
    Parameters
    ----------
    x : N-D Tensor or numpy array
        The element to mask. It accepts any dimensions but the last 2 must be Channel x Signal, since the function is 
        performed along the last dimension but equally for each channel of the signal.
    mask_number: int, optional
        The number of portion to mask. It must be a positive integer
        Default: 1
    masked_ratio: float, optional
        The percentage of the signal to mask. It must be a scalar in range 0<maskef_ratio<1.
        Default: 0.1
    batch_equal: bool, optional
        Whether to apply the same masking to all elements in the batch or not. Does apply only if x has more than 2 dimensions
        Default: False
        
    """
    
    if not(isinstance(mask_number,int)) or mask_number<=0:
        raise ValueError('mask_number must be a positive integer')
    if masked_ratio<=0 or masked_ratio>=1:
        raise ValueError('mask_ratio must be in range (0,1), i.e. all values between 0 and 1 excluded')
    
    Ndim=len(x.shape)
    x_masked= np.copy(x) if isinstance(x, np.ndarray) else torch.clone(x)
    if Ndim<3 or batch_equal: 
        
        # IDENTIFY LENGTH OF MASKED PIECES
        sample2mask=int(masked_ratio*x.shape[-1])
        pieces= [0]*mask_number
        piece_sum=0
        for i in range(mask_number-1):
            
            left= sample2mask-piece_sum-(mask_number-i+1)
            minval= max(1, int( (left/(mask_number-i) +1)*0.75)  )
            maxval= min( left, int( (left/(mask_number-i) +1)*1.25) )
            pieces[i] = random.randint(minval, maxval)
            piece_sum += pieces[i]
        pieces[-1]=sample2mask-piece_sum
        
        # IDENTIFY POSITION OF MASKED PIECES
        maxspace=x.shape[-1]-sample2mask
        spaces= [0]*mask_number
        space_sum=0
        for i in range(mask_number):
            left= maxspace-space_sum-(mask_number-i+1)
            spaces[i] = random.randint(1, int( left/(mask_number-i) +1) )
            space_sum += spaces[i]
        
        # APPLYING MASKING
        cnt=0
        for i in range(mask_number):
            cnt += spaces[i]
            x_masked[...,cnt:cnt+pieces[i]]=0
            cnt += pieces[i]
        
    else:
        for i in range(x.shape[0]):
            x_masked[i]= masking(x[i], mask_number=mask_number, masked_ratio=masked_ratio, batch_equal=batch_equal)
    
    return x_masked

def channel_dropout(x: "N-D Tensor of numpy Array",
                    Nchan: int= None,
                    batch_equal: bool=True
                   ):
    """
    
    channel_dropout put to 0 a given (or random) amount of channels selected at random.
    
    Given a N-D tensor or numpy array, where the last two dimensions refers to Channels x Samples,
    channel_dropout will choose a random a subset of Channels based on Nchan and put them to 0.
    
    If batch_equal is set to True, the function will be called recursively producing a different
    augmentation for each EEG in the tensor.
    
    Paramters
    ---------
    x: N-D Tensor or numpy array
        Array or Tensor to augment. Last two dimensions must be Channels x Samples
    Nchan: int, optional
        Number of channels to drop. If not given, the number of channels is chosen at random in the 
        interval [1, (Channel_total // 4) +1 ]
        Defult: None
    batch_equal: bool, optional
        whether to apply the same channel drop to all EEG records or not.
        Default: True
        
    """
    
    Ndim= len(x.shape)
    x_drop = torch.clone(x) if isinstance(x, torch.Tensor) else np.copy(x)
    
    if batch_equal or Ndim<3:
        if Nchan is None:
            Nchan = random.randint(1, (x.shape[-2]//4)+1)
        else:
            if Nchan > x.shape[-2]:
                raise ValueError('Nchan can\'t be higher than the actual number' 
                                 ' of channels in the given EEG')
            else:
                Nchan = int(Nchan)
        drop_chan= np.random.permutation(x.shape[-2])[:Nchan]
        if isinstance(x, torch.Tensor):
            drop_chan= torch.from_numpy(drop_chan).to(device=x.device)
        x_drop[...,drop_chan,:] = 0 
    else:
        for i in range(x.shape[0]):
            x_drop[i] = channel_dropout(x[i], Nchan=Nchan, batch_equal=batch_equal)
        
    return x_drop


def add_eeg_artifact(x,
                     Fs: float,
                     artifact: str= None,
                     amplitude: float=None,
                     line_at_60Hz: bool=True,
                     lost_time: float=None,
                     drift_slope: float= None,
                     batch_equal: bool= False
                    ):
    """
    add_eeg_artifact add one common artifacts of EEG signals.
    
    Given a N-dim tensor or a numpy array, add_eeg_artifact add one of the artifact listed in
    -- Fickling et al., (2019) Good data? The EEG Quality Index for Automated Assessment of Signal Quality --
    along the last dimension of the input element. Supported artifacts are:
    
        1) white: simple white noise
        2) line: noise at 50 Hz or 60 Hz
        3) eye: noise in range [1, 3] Hz
        4) muscle: noise in range [20, 60] Hz
        5) drift: straight line with non-zero slope
        6) lost: cancellation of one portion of the signal
    
    Line, eye and muscle artifact are generated with the add_band_noise function. 
    Lost artifact is generated with the masking function.
    White and drift are generated inside this function.
    
    Parameters
    ----------
    x : N-D Tensor or numpy array
        The element to add the artifact. It accepts any dimensions but the last 2 must be Channel x Signal, 
        since the function is performed along the last dimension but equally for each channel of the signal.
    Fs: float
        The sampling rate of the signal in Hz
    amplitude: float, optional
        The amplitude of the noise to add. If not given, amplitude=std(x)
        Default: None
    line_at_60Hz: bool, optional
        Whether to apply the line artifact at 60Hz (True) or 50Hz (False).
        Default: True
    lost_time: float, optional
        The amount of time the signal is canceled. Must be given in seconds. Internally masking function is called
        by converting the given time to the percentage of masked signal with the function
        (Fs*lost_time)/x.shape[-1]. Alternatively, to convert the percentage of the signal masked to the amount of 
        time is masked, just revert the formula, so lost_time= (masking_percentage * x.shape[-1]) / Fs.
        If None is given, 20% of the signal is masked
        Default: None
    drift_slope: float, optional
        The difference between to consecutive points of the straight line to add. If None is given, slope is
        calculated with the amplitude parameter as drift_slope = amplitude/x.shape[-1]. If also amplitude is not
        given drift_slope = std(x)/x.shape[-1]
    batch_equal: bool, optional
        Whether to apply the same masking to all elements in the batch or not. Does apply only if x has more than 2 dimensions
        Default: False
    
    """
    was_random=False #just for recursive calls
    if artifact != None:
        if artifact not in ['line','lost','drift' ,'eye', 'muscle', 'white']:
            raise ValueError('EEG artifact not supported. Accepted any of \'line\',\'lost\',\'drift\',\'eye\',\'muscle\'')
    else:
        was_random=True
        artifact = random.choice(['line','lost','drift' ,'eye', 'muscle', 'white'])
    
    if amplitude != None:
        if not(isinstance(amplitude,list)):
            amplitude=[-amplitude, amplitude] if amplitude>0 else [amplitude, -amplitude]
    
    Ndim=len(x.shape)
    x_noise = np.empty_like(x) if isinstance(x,np.ndarray) else torch.empty_like(x, device=x.device)
    if batch_equal or Ndim<3:
        
        if artifact == 'line':
            freq= 60 if line_at_60Hz else 50
            x_noise, noise = add_band_noise( x, freq, Fs, amplitude, get_noise=True)
        
        elif artifact == 'eye':
            x_noise = add_band_noise( x, (1,3), Fs, amplitude)
        
        elif artifact == 'muscle':
            x_noise = add_band_noise( x, (20,60), Fs, amplitude)
        
        elif artifact == 'lost':
            if lost_time==None:
                x_noise = masking( x, 1, 0.2)
            else:
                x_noise = masking( x, 1, (Fs*lost_time)/x.shape[-1])
        
        elif artifact == 'white':
            if isinstance(x,np.ndarray):
                noise = np.random.uniform(-1,1, size=x.shape)
            else:
                noise = torch.empty_like(x, device=x.device).uniform_(-1,1)
            
            if amplitude !=None:
                noise *= amplitude[1]
            else:
                if isinstance(x,np.ndarray):
                    G = amplitude[1]/np.std(noise) if amplitude!=None else np.std(x)/np.std(noise)
                    noise *= G
                else:
                    G = amplitude[1]/torch.std(noise) if amplitude!=None else torch.std(x)/torch.std(noise)
                    noise *= G
            x_noise = x + noise 
                    
        elif artifact == 'drift':
            if isinstance(x, np.ndarray):
                if drift_slope!=None:
                    noise = np.arange(x.shape[-1])*drift_slope
                elif amplitude!=None:
                    noise = np.linspace(0, amplitude[1], x.shape[-1])
                else:
                    noise = np.linspace(0, np.std(x), x.shape[-1])
            else:
                if drift_slope!=None:
                    noise = torch.arange(x.shape[-1], device=x.device)*drift_slope
                elif amplitude!=None:
                    noise = torch.linspace(0, amplitude[1], x.shape[-1], device=x.device)
                else:
                    noise = torch.linspace(0, torch.std(x), x.shape[-1], device=x.device)
            x_noise = x + noise
     
    else:
        artifact_arg= None if was_random else artifact
        for i in range(x.shape[0]):
            x_noise[i] = add_eeg_artifact(x[i], Fs, artifact_arg, amplitude, line_at_60Hz, lost_time,
                                          drift_slope, batch_equal)
    
    return x_noise


# -----------------------
# AUGMENTATION CLASSES
# -----------------------
class StaticSingleAug():
    """
    
    StaticSingleAug is a class designed to perform a single data augmentation where the optional arguments
    are previously set and given during initialization. No random choice of the arguments is performed. The 
    class accepts multiple set of optional arguments. In this case they are called individually at each class
    call, in a sequential manner. This means that the first call use the first set of arguments, the second 
    will use the second set of arguments, and so on. When the last set is used, the class will restart from the
    first set of arguments.
    """
    
    def __init__(self, augmentation, arguments: list or dict or list[list or dict]=None):
        """
        
        Parameters
        ----------
        
        augmentation: function
            The augmentation function to apply. It can be a custom function, but the first 
            argument must be the element to augment
        arguments: list, dict, list[list or dict], optional
            The arguments to give to the augmentation function. It can be:
                1) None. In this case the default parameters of the function are used. Remember that if there are
                   other non optional arguments, the call will throw an error.
                2) a simple list. In this case the function is called with the sintax "augmentation(x, *arguments)"
                3) a simple dict. In this case the function is called with the sintax "augmentation(x, **arguments)"
                4) a list of dict or list. This is a particular case where multiple combination of arguments are given.
                   Each element of the list must be a list or a dict with the specific argument combination. Every time
                   the class is called, one of the given combination is used to perform the data augmentation. The list
                   is followed sequentially with repetion, meaning that the first call use the first set of arguments of the 
                   list, the second the second set of arguments, and so on. When the last element of the list is used, the 
                   function will restart scrolling the given list
            Default: None       
        """
        
        if not(inspect.isfunction(augmentation) or inspect.isbuiltin(augmentation)):
            raise ValueError('augmentation must be a function to call')
        else:
            self.augmentation=augmentation
        
        self.arguments=arguments
        self.counter=0
        self.maxcounter=0
        self.multipleStaticArguments=False
        if arguments !=None:
            if all(isinstance(i,list) or isinstance(i,dict) for i in arguments):
                self.multipleStaticArguments=True
                self.maxcounter=len(arguments)
        
    def PerformAugmentation(self, X):
        
        if self.multipleStaticArguments:
            argument=self.arguments[self.counter]
            if isinstance(argument, list):
                Xaug = self.augmentation(X, *argument)
            else:
                Xaug = self.augmentation(X, **argument)
            
            self.counter +=1
            if self.counter == self.maxcounter:
                self.counter=0 
        else:
            if self.arguments==None:
                Xaug= self.augmentation(X)
            elif isinstance(self.arguments, list):
                Xaug = self.augmentation(X, *self.arguments)
            else:
                Xaug = self.augmentation(X, **self.arguments)
        
        return Xaug
        
    def __call__(self, X):
        return self.PerformAugmentation(X)
        
        
        
class DynamicSingleAug():
    """
    
    DynamicSingleAug is a class designed to perform a single data augmentation where the optional arguments
    are chosen at random from a given discrete set or a given range. Random choice of the arguments is performed
    at each call.
    """
    def __init__(self, 
                 augmentation, 
                 discrete_arg: Dict[str, Any]=None, 
                 range_arg: Dict[str, list[ int or float, int or float]]=None,
                 range_type: Dict[str, str or bool] or list[str or bool]=None
                ):
        """
        At least one of discrete_arg or range_arg arguments must be given, otherwise simply use a StaticSingleAug
        
        Parameters
        ----------
        
        augmentation: function
            The augmentation function to apply. It can be a custom function, but the first 
            argument must be the element to augment
        discrete_arg: dict, optional
            A dictionary specifying arguments whose value must be chosen within a discrete set. The dict must have:
                - Keys as string with the name of one of the optional arguments
                - Values as lists of elements to be randomly chosen. Single elements are accepted if a specific 
                  value for an argument needs to be set. In this case it's not mandatory to give it as list, as 
                  automatic conversion will be performed.
            Default: None
        range_arg: dict, optional
            A dictionary specifying arguments whose value must be chosen within a continuous range. The dict must have:
                - Keys as string with the name of one of the optional arguments
                - Values as two element lists specifying the range of values where to randomly select the argument value. 
            Default: None
        range_type: dict or list, optional
            A dictionary or a list specifying if values in range_arg must be given to the augmentation function as integers:
            If given as a dict, keys must be the same as the one of range_arg (no more or less, the same). If given as a list,
            the length must be the same of range_arg. In particular:
                1) if range_type is a dict. 
                    - Keys must be the ones in range_arg
                    - Values must be single element specifying if the argument must be an integer. In this case, use a boolean
                      True or a string 'int' to specify if the argument must be converted to an int.
                2) if range_arg is a list.
                    - Values must be set as the values in the dict. The order is the one used when iterating along the
                      range_arg dict
                3) if None is given, a list of True with length equal to range_arg is automatically created, since int
                   arguments are more compatible with float ones.
            Default: None     
        """
        
        # set augmentation function
        if not(inspect.isfunction(augmentation) or inspect.isbuiltin(augmentation)):
            raise ValueError('augmentation must be a function to call')
        else:
            self.augmentation=augmentation
        
        # get function argument name
        self.argnames= inspect.getfullargspec(augmentation)[0][1:]
        
        # check if given discrete_arg keys are actually augmentation arguments
        self.discrete_arg=None
        if discrete_arg != None:
            if isinstance(discrete_arg, dict):
                if all(i in self.argnames for i in discrete_arg):
                    self.discrete_arg=discrete_arg
                else:
                    raise ValueError('keys of discrete_arg argument must be the argument of the augmentation fun')
            else:
                raise ValueError('discrete_arg must be a dictionary')
        
        # check if given range_arg keys are actually augmentation arguments 
        # also check if values are two element list
        self.range_arg=None
        if range_arg != None:
            if isinstance(range_arg, dict):
                if all(i in self.argnames for i in range_arg):
                    if all( (isinstance(i,list) and len(i)==2) for i in range_arg.values()):
                        self.range_arg=range_arg
                    else:
                        raise ValueError('range_arg values must be a len 2 list with min and max range')
                else:
                    raise ValueError('keys of range_arg argument must be the argument of the augmentation fun')
                for i in range_arg:
                    if not(isinstance(range_arg[i],list)):
                        range_arg[i] = [range_arg[i]]
            else:
                raise ValueError('range_arg must be a dictionary')
        
        # check if range_types keys are the same as range_args
        self.range_type=None
        if range_type!=None:
            if isinstance(range_type, dict):
                if range_type.keys() == range_arg.keys():
                    self.range_type=range_type
                else:
                    raise ValueError('keys of range_type must be the same as range_arg')
            elif isinstance(range_type, list):
                if len(range_type)==len(self.range_arg):
                    self.range_type=range_type
                else:
                    raise ValueError('range_type must have the same length as range_args')
            else:
                raise ValueError('discrete_arg must be a dictionary or a list')
        else:
            if self.range_arg!=None:
                self.range_type=[True]*len(self.range_arg)
        
        self.is_range_type_dict= True if isinstance(self.range_type, dict) else False
        
        self.given_arg = list(self.discrete_arg) if self.discrete_arg!=None else []
        self.given_arg += list(self.range_arg) if self.range_arg!=None else []
        
    
    def PerformAugmentation(self, X):    
        arguments={i:None for i in self.given_arg}
        if self.discrete_arg!=None:
            for i in self.discrete_arg:
                if isinstance(self.discrete_arg[i],list): 
                    arguments[i] = random.choice(self.discrete_arg[i]) 
                else:
                    arguments[i]= self.discrete_arg[i] 
        
        cnt=0 # counter if range_type is a list, it's a sort of enumerate
        if self.range_arg!=None:
            for i in self.range_arg.keys():
                arguments[i]=random.uniform(self.range_arg[i][0], self.range_arg[i][1])
                if self.is_range_type_dict:
                    if self.range_type[i] in ['int', True]:
                        arguments[i] = int(arguments[i])
                else:
                    if self.range_type[cnt] in ['int', True]:
                        arguments[i] = int(arguments[i])
                    cnt+=1
        
        Xaug = self.augmentation(X, **arguments)
        return Xaug
        
    def __call__(self, X):
        return self.PerformAugmentation(X)


    
class SequentialAug():
    
    def __init__(self,*augmentations):

        self.augs=[item for item in augmentations]
     
    def PerformAugmentation(self, X): 
        
        Xaugs = self.augs[0](X)
        for i in range(1,len(self.augs)):
            Xaugs = self.augs[i](Xaugs)
        return Xaugs
            
    def __call__(self, X):
        return self.PerformAugmentation(X)

class RandomAug():
    """
    RandomAug perform a random augmentation from a list of arguments.
    Class must be initialized giving a sequence 
    """
    def __init__(self,*augmentations):
        
        self.augs = [item for item in augmentations]
        self.N = len(self.augs)
     
    def PerformAugmentation(self, X): 

        idx=random.randint(0,self.N-1)
        Xaugs = self.augs[idx](X)
        return Xaugs
            
    def __call__(self, X):
        return self.PerformAugmentation(X)



