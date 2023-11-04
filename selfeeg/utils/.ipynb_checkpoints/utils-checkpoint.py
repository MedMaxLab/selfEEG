from __future__ import annotations
from typing import Union, Sequence, Optional
import copy
import random
import numpy as np
from numpy.typing import ArrayLike
import torch

__all__ = ['get_subarray_closest_sum',
           'scale_range_soft_clip','RangeScaler',
           'torch_pchip'
          ]

def subarray_closest_sum(arr: list, 
                         n: int, 
                         k: float
                        ) -> list:
    """
    ``subarrat_closest_sum`` returns a subarray whose element sum is closest to k.
    
    This function is taken from geeksforgeeks at the following link [link1]_
    
    It is important to note that this function return a subarray and not a subset of the array.
    A subset is a collection of elements in the array taken from any index, a subarray here is 
    a slice of the array (arr[start:end]). If you are looking for a subset with closest sum,
    which is more accurate but more computationally and memory demanding, 
    search for another function.
    
    Parameters
    ----------
    arr: list
        the array to search
    n: int
        the length of the array
    k: float
        the target value

    Returns
    -------
    best_arr: list
        The subarray whose element sum is closest to k

    References
    ----------
    .. [link1] https://www.geeksforgeeks.org/subarray-whose-sum-is-closest-to-k/
    
    """
    
    # Initialize start and end pointers, current sum, and minimum difference
    best_arr=[]
    start = 0
    end = 0
    curr_sum = arr[0]
    min_diff = float('inf')
    # Initialize the minimum difference between the subarray sum and K
    min_diff = abs(curr_sum - k)
    # Traverse through the array
    while end < n - 1:
        # If the current sum is less than K, move the end pointer to the right
        if curr_sum < k:
            end += 1
            curr_sum += arr[end]
        # If the current sum is greater than or equal to K, move the start pointer to the right
        else:
            curr_sum -= arr[start]
            start += 1
 
        # Update the minimum difference between the subarray sum and K
        if abs(curr_sum - k) < min_diff:
            min_diff = abs(curr_sum - k)
    
    # Print the subarray with the sum closest to K
    start = 0
    end = 0
    curr_sum = arr[0]
 
    while end < n - 1:
        if curr_sum < k:
            end += 1
            curr_sum += arr[end]
        else:
            curr_sum -= arr[start]
            start += 1 
        # Print the subarray with the sum closest to K
        if abs(curr_sum - k) == min_diff:
            for i in range(start, end+1):
                best_arr.append(arr[i])
            break
    return best_arr


def get_subarray_closest_sum(arr: Sequence[int], 
                             target: float, 
                             tolerance: float=0.01,
                             perseverance: int=1000,
                             return_subarray: bool=True 
                            ) -> tuple[list, Optional[list]]:
    """
    ``get_subarray_closest_sum`` find the subarray of whose values sum is 
    closer to a target up to a specified tolerance (if possible) and return the index of the 
    selected values in the original array.
    
    To find the subarray, get_subarray_closest_sum calls multiple times the 
    ``subarray_closest_sum`` function until the subarray has the sum within 
    [target*(1-tolerance), target*(1+tolerance)].
    At each try the array is shuffled in order to get a different solution. Keep in mind that 
    the solution is not always the optimal, but is the first which satisfy the requirements 
    given.
    
    Parameters
    ----------
    arr: list
        the array to search
    target: float
        the target sum
    tolerance: float, optional
        the tolerance to apply to the sum in percentage. It must be a value between 0 and 1.
        
        Default = 0.01
    return_subarray: bool, optional
        whether to also return the subarray or not
        
        Default = True
    perseverance: int, optional
        The maximum number of tries before stopping searching the subarray with closest sum.
        
        Default = 1000

    Returns
    -------
    final_idx: list
        a list with the index of the identified subarray
    best_sub_arr: list, optional
        the identified subarray
    
    """
    
    
    if tolerance<0 or tolerance>1:
        raise ValueError('tolerance must be in [0,1]')
    if not(isinstance(perseverance, int)):
        perseverance=int(perseverance)
    
    # np.argsort
    idx = range(len(arr))
    N=len(arr)
    best_sub_arr=[]
    for _ in range(perseverance):
        
        c = list(zip(arr, idx))
        random.shuffle(c)
        arr, idx = zip(*c)
        
        sub_arr = subarray_closest_sum(arr, N, target)
        #print(sub_arr, abs(sum(sub_arr)- target), abs(sum(best_sub_arr)-target))
        if (abs(sum(sub_arr)- target)) < (abs(sum(best_sub_arr)-target)):
            best_sub_arr=sub_arr
        if (target*(1-tolerance))<sum(sub_arr)<(target*(1+tolerance)):
            best_sub_arr=sub_arr
            break
    # get final list
    best_sub2=copy.deepcopy(best_sub_arr)
    final_idx=[]
    for i in range(len(arr)):
        if arr[i] in best_sub2:
            final_idx.append(idx[i])
            best_sub2.remove(arr[i])
    
    if return_subarray:
        return final_idx, best_sub_arr
    else:
        return final_idx


def scale_range_soft_clip(x: ArrayLike, 
                          Range: float=200, 
                          asintote: float=1.2, 
                          scale: str='mV', 
                          exact: bool=True
                         ) -> ArrayLike:
    """
    ``scale_range_soft_clip`` rescale the EEG data.
    The function will rescale the data in the following way:

        1. Values in Range will be rescaled in the range [-1,1] linearly
        2. Values outside the range will be either clipped or soft clipped with 
           an exponential saturating curve with first derivative in -1 and 1
           preserved and horizontal asintote (the saturating point) given by the user

    To provide faster computation, this function can also be approximated with a 
    sigmoid scaled with the given input range and asintote. To check the difference
    in those functions see the geogebra file provided in the extra folder of the github
    repository.

    Parameters
    ----------
    x: ArrayLike
        The array or tensor to rescale. Rescaling can be perfomed along the last dimension.
        Tensors can also be placed in a GPU. Computation in this case is much faster
    Range: float, optional
        The range of values to rescale given in microVolt. It rescale linearly the 
        values in [-range, range] to [-1, 1]. Must be a positive value.

        Default = 200
    asintote: float, optional
        The horizontal asintote of the soft clipping part. Must be a value bigger than 1

        Default = 1.2
    scale: str, optional
        The scale of the EEG Samples. It can be:

            - 'mV' for milliVolt
            - 'uV' for microVolt
            - 'nV' for nanoVolt

        Default = 'mV'
    exact: bool, optional
        whether to approximate the composed function (linear + exponential function) with 
        a sigmoid. It will make the rescaling much faster but won't preserve the linearity 
        in the range [-1, 1]

    Returns
    -------
    x_scaled: ArrayLike
        The rescaled array
    
    """
    
    if Range<0:
        raise ValueError('Range cannot be lower than 0')
    if asintote is None:
        asintote = 1.0
    elif asintote<1:
        raise ValueError('asintote must be a value bigger than 1')
    scale=scale.lower()
    if scale not in ['mv','uv','nv']:
        raise ValueError('scale must be any of \'mV\', \'uV\', \'nV\'')
    else:
        if scale=='uv':
            x /= 1.0e3
        elif scale=='nv':
            x /= 1.0e6
    Range=Range/1000
    x_scaled = torch.clone(x) if isinstance(x, torch.Tensor) else np.copy(x)

    # CASE 1: HARD clipping
    if asintote==1.0:
        mask1 = x>Range
        mask2 = x<-Range
        x_scaled = x/Range
        if isinstance (x, torch.Tensor):
            x_scaled = torch.clamp(x_scaled, min=-1, max=1)
        else:
            x_scaled = np.clip(x_scaled, -1 ,1)
        return x_scaled

    # CASE 2: SOFT CLIPPING
    if exact:
        mask1 = x > (Range)
        mask2 = x < (-Range)
        x_scaled = x/Range
        if isinstance(x, torch.Tensor):
            x_scaled[mask2]  = (asintote-1)*torch.exp((x[mask2]+Range)/(Range*(asintote-1))) -asintote
            x_scaled[mask1]  = -((asintote-1)*torch.exp((-x[mask1]+Range)/(Range*(asintote-1))) -asintote)
        else:
            x_scaled[mask2]  = (asintote-1)*np.exp((x[mask2]+Range)/(Range*(asintote-1))) -asintote
            x_scaled[mask1]  = -((asintote-1)*np.exp((-x[mask1]+Range)/(Range*(asintote-1))) -asintote)       
    else:
        # trating c as -coeff
        c= (np.log((2*asintote)/(1+asintote) -1 ))/Range
        if isinstance(x, torch.Tensor):
            x_scaled = ((2*asintote)/(1+torch.exp(c*x))) - asintote
        else:
            x_scaled = ((2*asintote)/(1+np.exp(c*x))) - asintote
    return x_scaled


class RangeScaler():
    """
    ``RangeScaler`` is the class adaptation of the 
    ``scale_range_with_soft_clip`` function. 
    It rescale the EEG data.
    The class will rescale the data in the following way:

        1. Values in Range will be rescaled in the range [-1,1] linearly
        2. Values outside the range will be either clipped or soft clipped with 
           an exponential saturating curve with first derivative in -1 and 1
           preserved and horizontal asintote (the saturating point) given by the user

    To provide faster computation, this function can also be approximated with a 
    sigmoid scaled with the given input range and asintote. To check the difference
    in those functions see the geogebra file provided in the extra folder of the github
    repository.

    Parameters
    ----------
    x: ArrayLike
        The array or tensor to rescale. Rescaling can be perfomed along the last dimension.
        Tensors can also be placed in a GPU. Computation in this case is much faster
    Range: float, optional
        The range of values to rescale given in microVolt. It rescale linearly the 
        values in [-range, range] to [-1, 1]. Must be a positive value.

        Default = 200
    asintote: float, optional
        The horizontal asintote of the soft clipping part. Must be a value bigger than 1

        Default = 1.2
    scale: str, optional
        The scale of the EEG Samples. It can be:

            - 'mV' for milliVolt
            - 'uV' for microVolt
            - 'nV' for nanoVolt

        Default = 'mV'
    exact: bool, optional
        whether to approximate the composed function (linear + exponential function) with 
        a sigmoid. It will make the rescaling much faster but won't preserve the linearity 
        in the range [-1, 1]

    """

    def __init__(self, Range=200, asintote=1.2, scale='mV', exact=True):
        if Range<0:
            raise ValueError('Range cannot be lower than 0')
        if asintote is None:
            asintote = 1.0
        elif asintote<1:
            raise ValueError('asintote must be a value bigger than 1')
        scale=scale.lower()
        if scale not in ['mv','uv','nv']:
            raise ValueError('scale must be any of \'mV\', \'uV\', \'nV\'')
        self.Range    = Range
        self.asintote = asintote
        self.scale    = scale
        self.exact    = exact
    
    def __call__(self,x):
        """
        :meta private:
        """
        print('called function')
        return scale_range_soft_clip(x, self.Range, self.asintote, self.scale, self.exact)



def torch_pchip(x: "1D Tensor", 
                y: "ND Tensor", 
                xv: "1D Tensor",
                save_memory: bool=True,
                new_y_max_numel: int=4194304
               ):
    """
    ``torch_pchip`` performs the pchip interpolation on 
    the last dimension of the input tensor y.
    
    This function is a pytorch adaptation of the scipy's pchip_interpolate [pchip]_ . 
    It performs sp-pchip interpolation (Shape Preserving Piecewise Cubic Hermite Interpolating 
    Polynomial) on the last dimension of the y tensor.
    x is the original time grid and xv new virtual grid. So, the new values of y at time xv are 
    given by the polynomials evaluated at the time grid x.
    
    Parameters
    ----------
    x: 1D Tensor
        Tensor with the original time grid. Must be the same length as the last dimension of y
    y: ND Tensor
        Tensor to interpolate. The last dimension must have the signals to interpolate
    xv: 1D Tensor
        Tensor with the new virtual grid, i.e. the time points where to interpolate
    save_memory: bool, optional
        whether to perform the interpolation on subsets of the y tensor by recursively function 
        calls or not. Does not apply if y is a 1-D tensor. If set to False memory usage can 
        drastically increase (for example with a 128 MB tensor, the memory usage of the 
        function is 1.2 GB), but in some devices it can speed up the process. 
        However, this is not the case for all devices and performance may also decrease.
        
        Default = True
    new_y_max_numel: int, optional
        The number of elements which the tensor needs to surpass in order to make the function 
        start doing recursive calls. It can be considered as an indicator of the maximum 
        allowed memory usage since lower the number, lower the memory used. 
        
        Default = 256*1024*16 (approximately 16s of recording of a 256 Channel 
        EEG sampled at 1024 Hz)
    
    Note
    ----
    Some technical information and difference with other interpolation can be found here:
    https://blogs.mathworks.com/cleve/2012/07/16/splines-and-pchips/
    
    Note
    ----
    have a look also at the Scipy's documentation: 
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html
    Some parts of the code are inspired from:
    https://github.com/scipy/scipy/blob/v1.10.1/scipy/interpolate/_cubic.py#L157-L302

    References
    ----------
    .. [pchip] https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.pchip_interpolate.html
    
    """
    
    if len(x.shape)!= 1:
        raise ValueError(['Expected 1D Tensor for x but received a ', str(len(x.shape)), '-D Tensor']) 
    if len(xv.shape)!= 1:
        raise ValueError(['Expected 1D Tensor for xv but received a ', str(len(xv.shape)),'-D Tensor'])
    if x.shape[0] != y.shape[-1]:
        raise ValueError('x must have the same length than the last dimension of y')

    # Initialize the new interpolated tensor
    Ndim=len(y.shape)
    new_y=torch.empty(( *y.shape[:(Ndim-1)], xv.shape[0]), device=y.device)
    
    # If save_memory and the new Tensor size is huge, call recursively for each element in 
    # the first dimension 
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
    infer_tv = torch.cat(( tv_minus**3, tv_minus**2, tv_minus, 
                          torch.ones(tv_minus.shape, device=tv_minus.device)), 1) 
    
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
