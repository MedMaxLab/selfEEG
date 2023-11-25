#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
sys.path.append(os.getcwd().split('/test')[0])
import itertools
import random
import numpy as np
import torch
from selfeeg import augmentation as aug

def makeGrid(pars_dict):  
    keys=pars_dict.keys()
    combinations=itertools.product(*pars_dict.values())
    ds=[dict(zip(keys,cc)) for cc in combinations]
    return ds

device  = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dims = (32,2,32,512)
x1  = torch.randn(*dims[-1:])
x2  = torch.randn(*dims[-2:])
x3  = torch.randn(*dims[-3:])
x4  = torch.randn(*dims)
x1np  = x1.numpy()
x2np  = x2.numpy()
x3np  = x3.numpy()
x4np  = x4.numpy()


if device.type != 'cpu':
    x1gpu  = torch.clone(x1).to(device=device)
    x2gpu  = torch.clone(x2).to(device=device)
    x3gpu  = torch.clone(x3).to(device=device)
    x4gpu  = torch.clone(x4).to(device=device)


# In[2]:


print('\n---------------------------')
print('TESTING AUGMENTATION MODULE')
if device.type != 'cpu':
    print('Found cuda device: testing nn.modules with both cpu and gpu')
else:
    print('Didn\'t found cuda device: testing nn.modules with only cpu')
print('---------------------------')


# In[3]:


print('Testing identity...', end="", flush=True)
aug_args = { 'x': [x1,x2,x3,x4,x1np,x2np,x3np,x4np] }
aug_args = makeGrid(aug_args)
for i in aug_args:
    xaug = aug.identity(**i)
    if isinstance(xaug, torch.Tensor):
        assert torch.equal(i['x'],xaug)
    else:
        assert np.array_equal(i['x'],xaug)
N = len(aug_args)
if device.type != 'cpu':
    aug_args = { 'x': [x1gpu, x2gpu, x3gpu, x4gpu] }
    aug_args = makeGrid(aug_args)
    for i in aug_args:
        xaug = aug.identity(**i)
        assert torch.equal(i['x'],xaug)
print('   identity OK: tested', N+len(aug_args), 'combinations of input arguments')


# In[4]:


print('Testing shift vertical...', end="", flush=True)
aug_args = { 'x': [x1,x2,x3,x4,x1np,x2np,x3np,x4np], 'value': [1, 2.0, 4] }
aug_args = makeGrid(aug_args)
for i in aug_args:
    xaug = aug.shift_vertical(**i)
    # 1e-5 is added due to conversion of dtype in numpy array
    assert xaug[*[0]*len(xaug.shape)]<=(i['x'][*[0]*len(xaug.shape)]+i['value']+1e-5) 
N = len(aug_args)
if device.type != 'cpu':
    aug_args = { 'x': [x1gpu, x2gpu, x3gpu, x4gpu], 'value': [1, 2.0, 4] }
    aug_args = makeGrid(aug_args)
    for i in aug_args:
        xaug = aug.shift_vertical(**i)
        assert xaug[*[0]*len(xaug.shape)]<=(i['x'][*[0]*len(xaug.shape)]+i['value'])
print('   shift vertical OK: tested', N+len(aug_args), 'combinations of input arguments')


# In[5]:


print('Testing shift horizontal...', end="", flush=True)

aug_args = { 'x': [x1,x2,x3,x4,x1np,x2np,x3np,x4np], 'Fs':[128], 
            'shift_time': [0.5, 1, 2.0], 'forward':[None, True, False],
            'random_shift':[False,True], 'batch_equal':[True,False]
           }
aug_args = makeGrid(aug_args)
for i in aug_args:
    # change batch equal to avoid function print
    if not(i['batch_equal']):
        if not( i['random_shift'] or (i['forward'] is None) ):
            i['batch_equal']=True
    xaug = aug.shift_horizontal(**i)
N = len(aug_args)
if device.type != 'cpu':
    aug_args = { 'x': [x1gpu, x2gpu, x3gpu, x4gpu], 'Fs':[128], 
                'shift_time': [0.5, 1, 2.0], 'forward':[None, True, False],
                'random_shift':[False,True], 'batch_equal':[True,False]
               }
    aug_args = makeGrid(aug_args)
    for i in aug_args:
        if not(i['batch_equal']):
            if not( i['random_shift'] or (i['forward'] is None) ):
                i['batch_equal']=True
        xaug = aug.shift_horizontal(**i)
        
print('   shift vertical OK: tested', N+len(aug_args), 'combinations of input arguments')


# In[6]:


print('Testing shift frequency...', end="", flush=True)
aug_args = { 'x': [x1,x2,x3,x4,x1np,x2np,x3np,x4np], 'Fs':[128], 
            'shift_freq': [1.35, 2, 4.12], 'forward':[None, True, False],
            'random_shift':[False,True], 'batch_equal':[True,False]
           }
aug_args = makeGrid(aug_args)
for i in aug_args:
    if not(i['batch_equal']):
        if not( i['random_shift'] or (i['forward'] is None) ):
            i['batch_equal']=True
    xaug = aug.shift_frequency(**i)
N = len(aug_args)    
if device.type != 'cpu':
    aug_args = { 'x': [x1gpu, x2gpu, x3gpu, x4gpu], 'Fs':[128], 
                'shift_freq': [1.35, 2, 4.12], 'forward':[None, True, False],
                'random_shift':[False,True], 'batch_equal':[True,False]
               }
    aug_args = makeGrid(aug_args)
    for i in aug_args:
        if not(i['batch_equal']):
            if not( i['random_shift'] or (i['forward'] is None) ):
                i['batch_equal']=True
        xaug = aug.shift_frequency(**i)
print('   shift frequency OK: tested', N+len(aug_args), 'combinations of input arguments')


# In[7]:


print('Testing flip vertical...', end="", flush=True)
aug_args = { 'x': [x1,x2,x3,x4,x1np,x2np,x3np,x4np] }
aug_args = makeGrid(aug_args)
for i in aug_args:
    xaug = aug.flip_vertical(**i)
N = len(aug_args)
if device.type != 'cpu':
    aug_args = { 'x': [x1gpu, x2gpu, x3gpu, x4gpu] }
    aug_args = makeGrid(aug_args)
    for i in aug_args:
        xaug = aug.flip_vertical(**i)
print('   flip vertical OK: tested', N+len(aug_args), 'combinations of input arguments')


# In[8]:


print('Testing flip horizontal...', end="", flush=True)
aug_args = { 'x': [x1,x2,x3,x4,x1np,x2np,x3np,x4np] }
aug_args = makeGrid(aug_args)
for i in aug_args:
    xaug = aug.flip_horizontal(**i)
N = len(aug_args)
if device.type != 'cpu':
    aug_args = { 'x': [x1gpu, x2gpu, x3gpu, x4gpu] }
    aug_args = makeGrid(aug_args)
    for i in aug_args:
        xaug = aug.flip_horizontal(**i)
print('   flip vertical OK: tested', N+len(aug_args), 'combinations of input arguments')


# In[9]:


print('Testing gaussian noise...', end="", flush=True)
aug_args = { 'x': [x1,x2,x3,x4,x1np,x2np,x3np,x4np], 'mean':[0, 1, 2.5], 
            'std': [1.35, 2, 0.72]
           }
aug_args = makeGrid(aug_args)
for i in aug_args:
    xaug = aug.add_gaussian_noise(**i)
N = len(aug_args)    
if device.type != 'cpu':
    aug_args = { 'x': [x1gpu, x2gpu, x3gpu, x4gpu],  'mean':[0, 1, 2.5], 
                'std': [1.35, 2, 0.72]}
    aug_args = makeGrid(aug_args)
    for i in aug_args:
        xaug = aug.add_gaussian_noise(**i)
print('   gaussian noise OK: tested', N+len(aug_args), 'combinations of input arguments')


# In[10]:


print('Testing gaussian noise...', end="", flush=True)
aug_args = { 'x': [x1,x2,x3,x4,x1np,x2np,x3np,x4np], 'target_snr':[1,2,5] }
aug_args = makeGrid(aug_args)
for i in aug_args:
    xaug = aug.add_noise_SNR(**i)
N = len(aug_args)    
if device.type != 'cpu':
    aug_args = { 'x': [x1gpu, x2gpu, x3gpu, x4gpu], 'target_snr':[1,2,5] }
    aug_args = makeGrid(aug_args)
    for i in aug_args:
        xaug = aug.add_noise_SNR(**i)
print('   gaussian noise OK: tested', N+len(aug_args), 'combinations of input arguments')


# In[11]:


print('Testing band noise...', end="", flush=True)
aug_args = { 'x': [x1,x2,x3,x4,x1np,x2np,x3np,x4np], 
            'bandwidth':[['theta','gamma'],[(1,10),(15,18)],[4,50],50,['theta',(10,20),50]],
            'samplerate':[128],'noise_range':[None,2,1.5], 'std':[None,1.4,1.23]
           }
aug_args = makeGrid(aug_args)
for i in aug_args:
    xaug = aug.add_band_noise(**i)
N = len(aug_args)    
if device.type != 'cpu':
    aug_args = { 'x': [x1gpu, x2gpu, x3gpu, x4gpu], 
                'bandwidth':[['theta','gamma'],[(1,10),(15,18)],[4,50],50,['theta',(10,20),50]],
                'samplerate':[128],'noise_range':[None,2,1.5], 'std':[None,1.4,1.23]
               }
    aug_args = makeGrid(aug_args)
    for i in aug_args:
        xaug = aug.add_band_noise(**i)
print('   band noise OK: tested', N+len(aug_args), 'combinations of input arguments')


# In[12]:


print('Testing scaling...', end="", flush=True)
aug_args = { 'x': [x1,x2,x3,x4,x1np,x2np,x3np,x4np], 
            'value':[None,1.5,2,0.5], 'batch_equal':[True,False]
           }
aug_args = makeGrid(aug_args)
for i in aug_args:
    xaug = aug.scaling(**i)
N = len(aug_args)    
if device.type != 'cpu':
    aug_args = { 'x': [x1gpu, x2gpu, x3gpu, x4gpu], 
                'value':[None,1.5,2,0.5], 'batch_equal':[True,False]
               }
    aug_args = makeGrid(aug_args)
    for i in aug_args:
        xaug = aug.scaling(**i)
print('   scaling OK: tested', N+len(aug_args), 'combinations of input arguments')


# In[13]:


print('Testing random slope scale...', end="", flush=True)
aug_args = { 'x': [x1,x2,x3,x4,x1np,x2np,x3np,x4np], 
            'min_scale':[0.7,0.9], 'max_scale':[1.2,1.5], 
            'batch_equal':[True,False], 'keep_memory':[True,False]
           }
aug_args = makeGrid(aug_args)
for i in aug_args:
    if i['batch_equal'] and len(i['x'].shape)<2:
        i['batch_equal']=False
    xaug = aug.random_slope_scale(**i)
N = len(aug_args)    
if device.type != 'cpu':
    aug_args = { 'x': [x1gpu, x2gpu, x3gpu, x4gpu], 
            'min_scale':[0.7,0.9], 'max_scale':[1.2,1.5], 
            'batch_equal':[True,False], 'keep_memory':[True,False]
               }
    aug_args = makeGrid(aug_args)
    for i in aug_args:
        if i['batch_equal'] and len(i['x'].shape)<2:
            i['batch_equal']=False
        xaug = aug.random_slope_scale(**i)
print('   random slope scale OK: tested', N+len(aug_args), 'combinations of input arguments')


# In[14]:


print('Testing random FT phase...', end="", flush=True)
aug_args = { 'x': [x1,x2,x3,x4,x1np,x2np,x3np,x4np], 
            'value':[0.2,0.5,0.75], 'batch_equal':[True,False]
           }
aug_args = makeGrid(aug_args)
for i in aug_args:
    xaug = aug.random_FT_phase(**i)
N = len(aug_args)    
if device.type != 'cpu':
    aug_args = { 'x': [x1gpu, x2gpu, x3gpu, x4gpu], 
                'value':[0.2,0.5,0.75], 'batch_equal':[True,False]
               }
    aug_args = makeGrid(aug_args)
    for i in aug_args:
        xaug = aug.random_FT_phase(**i)
print('   random FT phase OK: tested', N+len(aug_args), 'combinations of input arguments')


# In[15]:


print('Testing moving average...', end="", flush=True)
aug_args = { 'x': [x1,x2,x3,x4,x1np,x2np,x3np,x4np], 'order': [3, 5, 9] }
aug_args = makeGrid(aug_args)
for i in aug_args:
    xaug = aug.moving_avg(**i)
N = len(aug_args)
if device.type != 'cpu':
    aug_args = { 'x': [x1gpu, x2gpu, x3gpu, x4gpu], 'order': [3, 5, 9] }
    aug_args = makeGrid(aug_args)
    for i in aug_args:
        xaug = aug.moving_avg(**i)
print('   moving average OK: tested', N+len(aug_args), 'combinations of input arguments')


# In[16]:


print('Testing lowpass filter...', end="", flush=True)
aug_args = { 'x': [x1,x2,x3,x4,x1np,x2np,x3np,x4np], 'Fs':[128,256],
            'Wp':[30], 'Ws':[50], 'rp':[-20*np.log10(.95)], 'rs':[-20*np.log10(.15)], 
            'filter_type': ['butter', 'ellip', 'cheby1', 'cheby2']
           }
aug_args = makeGrid(aug_args)
for i in aug_args:
    xaug = aug.filter_lowpass(**i)
N = len(aug_args)
if device.type != 'cpu':
    aug_args = { 'x': [x1gpu, x2gpu, x3gpu, x4gpu],  'Fs':[128,256],
            'Wp':[30], 'Ws':[50], 'rp':[-20*np.log10(.95)], 'rs':[-20*np.log10(.15)], 
            'filter_type': ['butter', 'ellip', 'cheby1', 'cheby2']}
    aug_args = makeGrid(aug_args)
    for i in aug_args:
        xaug = aug.filter_lowpass(**i)
print('   lowpass filter OK: tested', N+len(aug_args), 'combinations of input arguments')


# In[17]:


print('Testing highpass filter...', end="", flush=True)
aug_args = { 'x': [x1,x2,x3,x4,x1np,x2np,x3np,x4np], 'Fs':[128,256],
            'Wp':[40], 'Ws':[20], 'rp':[-20*np.log10(.95)], 'rs':[-20*np.log10(.15)], 
            'filter_type': ['butter', 'ellip', 'cheby1', 'cheby2']
           }
aug_args = makeGrid(aug_args)
for i in aug_args:
    xaug = aug.filter_highpass(**i)
N = len(aug_args)
if device.type != 'cpu':
    aug_args = { 'x': [x1gpu, x2gpu, x3gpu, x4gpu],  'Fs':[128,256],
            'Wp':[40], 'Ws':[20], 'rp':[-20*np.log10(.95)], 'rs':[-20*np.log10(.15)], 
            'filter_type': ['butter', 'ellip', 'cheby1', 'cheby2']}
    aug_args = makeGrid(aug_args)
    for i in aug_args:
        xaug = aug.filter_highpass(**i)
print('   highpass filter OK: tested', N+len(aug_args), 'combinations of input arguments')


# In[18]:


print('Testing bandpass filter...', end="", flush=True)
aug_args = { 'x': [x1,x2,x3,x4,x1np,x2np,x3np,x4np], 'Fs':[128,256], 
            'eeg_band':[None,"delta", "theta", "alpha", "beta", 
                        "gamma", "gamma_low"],
            'filter_type': ['butter', 'ellip', 'cheby1', 'cheby2']
           }
aug_args = makeGrid(aug_args)
for i in aug_args:
    xaug = aug.filter_bandpass(**i)
N = len(aug_args)
if device.type != 'cpu':
    aug_args = { 'x': [x1gpu, x2gpu, x3gpu, x4gpu], 'Fs':[128,256], 
                'eeg_band':[None,"delta", "theta", "alpha", "beta", 
                            "gamma", "gamma_low"],
                'filter_type': ['butter', 'ellip', 'cheby1', 'cheby2']
               }
    aug_args = makeGrid(aug_args)
    for i in aug_args:
        xaug = aug.filter_bandpass(**i)
print('   bandpass filter OK: tested', N+len(aug_args), 'combinations of input arguments')


# In[19]:


print('Testing bandstop filter...', end="", flush=True)
aug_args = { 'x': [x1,x2,x3,x4,x1np,x2np,x3np,x4np], 'Fs':[128,256], 
            'eeg_band':[None,"delta", "theta", "alpha", "beta", 
                        "gamma", "gamma_low"],
            'filter_type': ['butter', 'ellip', 'cheby1', 'cheby2']
           }
aug_args = makeGrid(aug_args)
for i in aug_args:
    xaug = aug.filter_bandstop(**i)
N = len(aug_args)
if device.type != 'cpu':
    aug_args = { 'x': [x1gpu, x2gpu, x3gpu, x4gpu], 'Fs':[128,256], 
                'eeg_band':[None,"delta", "theta", "alpha", "beta", 
                            "gamma", "gamma_low"],
                'filter_type': ['butter', 'ellip', 'cheby1', 'cheby2']
               }
    aug_args = makeGrid(aug_args)
    for i in aug_args:
        xaug = aug.filter_bandstop(**i)
print('   bandstop filter OK: tested', N+len(aug_args), 'combinations of input arguments')


# In[20]:


print('Testing permute channels...', end="", flush=True)
channel_map = ['FP1', 'AF3', 'F1', 'F3', 'FC5', 'FC3', 'FC1', 'C1', 
               'C5', 'TP7', 'CP5', 'CP3', 'CP1', 'P7', 'PO7', 'POZ', 
               'PZ', 'FPZ', 'FP2', 'AFZ', 'FZ', 'F2', 'F4', 'F6', 
               'FT8', 'C4', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'PO8']
aug_args = { 'x': [x2,x3,x4,x2np,x3np,x4np], 'chan2shuf':[-1, 2, 5], 
            'mode':["random", "network"], 'chan_net':["DMN","FPN", ['DMN', 'FPN'],"all"],
            'batch_equal': [True,False], 'channel_map': [channel_map]
           }
aug_args = makeGrid(aug_args)
for i in aug_args:
    xaug = aug.permute_channels(**i)
N = len(aug_args)
if device.type != 'cpu':
    aug_args = { 'x': [x2gpu, x3gpu, x4gpu], 'chan2shuf':[-1, 2, 5], 
            'mode':["random", "network"], 'chan_net':["DMN","FPN", ['DMN', 'FPN'],"all"],
            'batch_equal': [True,False],'channel_map': [channel_map]
           }
    aug_args = makeGrid(aug_args)
    for i in aug_args:
        xaug = aug.permute_channels(**i)
print('   permute channels OK: tested', N+len(aug_args), 'combinations of input arguments')


# In[21]:


print('Testing permute signal...', end="", flush=True)
aug_args = { 'x': [x1,x2,x3,x4,x1np,x2np,x3np,x4np], 'segments':[10, 15, 20], 
            'seg_to_per':[-1,2,5,8],'batch_equal': [True,False]
           }
aug_args = makeGrid(aug_args)
for i in aug_args:
    xaug = aug.permutation_signal(**i)
N = len(aug_args)
if device.type != 'cpu':
    aug_args = { 'x': [x1gpu, x2gpu, x3gpu, x4gpu], 'segments':[10, 15, 20], 
            'seg_to_per':[-1,2,5,8],'batch_equal': [True,False]
           }
    aug_args = makeGrid(aug_args)
    for i in aug_args:
        xaug = aug.permutation_signal(**i)
print('   permute signal OK: tested', N+len(aug_args), 'combinations of input arguments')


# In[22]:


print('Testing warp signal (this may take some time)...', end="", flush=True)
aug_args = { 'x': [x1,x2,x3,x4,x1np,x2np,x3np,x4np], 'segments':[15], 
            'stretch_strength':[2,1.5],'squeeze_strength':[0.4,0.8],
            'batch_equal': [True,False]
           }
aug_args = makeGrid(aug_args)
for i in aug_args:
    xaug = aug.warp_signal(**i)
N = len(aug_args)
if device.type != 'cpu':
    aug_args = { 'x': [x1gpu, x2gpu, x3gpu, x4gpu], 'segments':[15], 
            'stretch_strength':[2,1.5],'squeeze_strength':[0.4,0.8],
            'batch_equal': [True,False]
           }
    aug_args = makeGrid(aug_args)
    for i in aug_args:
        xaug = aug.warp_signal(**i)
print('   warp signal OK: tested', N+len(aug_args), 'combinations of input arguments')


# In[23]:


print('Testing crop and resize...', end="", flush=True)
aug_args = { 'x': [x1,x2,x3,x4,x1np,x2np,x3np,x4np], 'segments':[15], 
            'N_cut':[1,5], 'batch_equal': [True,False]
           }
aug_args = makeGrid(aug_args)
for i in aug_args:
    xaug = aug.crop_and_resize(**i)
N = len(aug_args)
if device.type != 'cpu':
    aug_args = { 'x': [x1gpu, x2gpu, x3gpu, x4gpu], 'segments':[15], 
            'N_cut':[1,5], 'batch_equal': [True,False]
           }
    aug_args = makeGrid(aug_args)
    for i in aug_args:
        xaug = aug.crop_and_resize(**i)
print('   crop and resize OK: tested', N+len(aug_args), 'combinations of input arguments')


# In[24]:


print('Testing change reference...', end="", flush=True)
aug_args = { 'x': [x2,x3,x4,x2np,x3np,x4np], 'mode':['chan','avg'], 
            'reference':[None, 5], 'exclude_from_ref': [None, 9,[9,10]]
           }
aug_args = makeGrid(aug_args)
for i in aug_args:
    xaug = aug.change_ref(**i)
N = len(aug_args)
if device.type != 'cpu':
    aug_args = { 'x': [x2gpu, x3gpu, x4gpu], 'mode':['chan','avg'], 
            'reference':[None, 5], 'exclude_from_ref': [None, 9,[9,10]]
           }
    aug_args = makeGrid(aug_args)
    for i in aug_args:
        xaug = aug.change_ref(**i)
print('   change refeference OK: tested', N+len(aug_args), 'combinations of input arguments')


# In[25]:


print('Testing masking...', end="", flush=True)
aug_args = { 'x': [x1,x2,x3,x4,x1np,x2np,x3np,x4np], 'mask_number':[1,2,4], 
            'masked_ratio':[0.1,0.2,0.4], 'batch_equal': [True,False]
           }
aug_args = makeGrid(aug_args)
for i in aug_args:
    xaug = aug.masking(**i)
N = len(aug_args)
if device.type != 'cpu':
    aug_args = { 'x': [x1gpu,x2gpu, x3gpu, x4gpu], 'mask_number':[1,2,4], 
                'masked_ratio':[0.1,0.2,0.4], 'batch_equal': [True,False]
               }
    aug_args = makeGrid(aug_args)
    for i in aug_args:
        xaug = aug.masking(**i)
print('   masking OK: tested', N+len(aug_args), 'combinations of input arguments')


# In[26]:


print('Testing channel dropout...', end="", flush=True)
aug_args = { 'x': [x2,x3,x4,x2np,x3np,x4np], 'Nchan':[None,2,3],'batch_equal': [True,False]}
aug_args = makeGrid(aug_args)
for i in aug_args:
    xaug = aug.channel_dropout(**i)
N = len(aug_args)
if device.type != 'cpu':
    aug_args = { 'x': [x2gpu, x3gpu, x4gpu], 'Nchan':[None,2,3],'batch_equal': [True,False]}
    aug_args = makeGrid(aug_args)
    for i in aug_args:
        xaug = aug.channel_dropout(**i)
print('   channel dropout OK: tested', N+len(aug_args), 'combinations of input arguments')


# In[27]:


print('Testing eeg artifact...', end="", flush=True)
aug_args = { 'x': [x1,x2,x3,x4,x1np,x2np,x3np,x4np], 'Fs':[128], 
            'artifact':[None,'white','line','eye','muscle','drift','lost'],
            'amplitude':[None,1],'line_at_60Hz':[True,False],'lost_time':[0.5,None],
            'drift_slope':[None,0.2],'batch_equal': [True,False]}
aug_args = makeGrid(aug_args)
for i in aug_args:
    xaug = aug.add_eeg_artifact(**i)
N = len(aug_args)
if device.type != 'cpu':
    aug_args = { 'x': [x1gpu,x2gpu, x3gpu, x4gpu], 'Fs':[128], 
            'artifact':[None,'white','line','eye','muscle','drift','lost'],
            'amplitude':[None,1],'line_at_60Hz':[True,False],'lost_time':[0.5,None],
            'drift_slope':[None,0.2],'batch_equal': [True,False]}
    aug_args = makeGrid(aug_args)
    for i in aug_args:
        xaug = aug.add_eeg_artifact(**i)
print('   eeg artifact OK: tested', N+len(aug_args), 'combinations of input arguments')


# In[28]:


print('Testing augmentation composition by running introductory notebook...', end="", flush=True)
Fs = 128
BatchEEG = torch.zeros(16,32,1024) + torch.sin(torch.linspace(0, 8*np.pi,1024))
BatchEEGaug = aug.add_eeg_artifact(BatchEEG, Fs, 'eye' , amplitude=0.5, batch_equal=False)
Aug_eye = aug.StaticSingleAug(aug.add_eeg_artifact, 
                              {'Fs': Fs, 'artifact': 'eye', 'amplitude': 0.5, 'batch_equal': False}
                             )
BatchEEGaug = Aug_eye(BatchEEG)
Aug_eye = aug.StaticSingleAug(aug.add_eeg_artifact, 
                              [{'Fs': Fs, 'artifact': 'eye', 'amplitude': 0.5, 'batch_equal': False},
                               {'Fs': Fs, 'artifact': 'eye', 'amplitude': 1.0, 'batch_equal': False} #new set
                              ]
                             )
BatchEEGaug1 = Aug_eye(BatchEEG)
BatchEEGaug2 = Aug_eye(BatchEEG)
Aug_warp = aug.DynamicSingleAug(aug.warp_signal, 
                               discrete_arg = {'batch_equal': [True, False]}, #discrete args accepts single values if 
                                                                               #you want those to be static
                               range_arg= {'segments': [5,15], 'stretch_strength': [1.5,2.5],
                                           'squeeze_strength': [0.4,2/3]},
                               range_type={'segments': True, 'stretch_strength': False,
                                           'squeeze_strength': False} # true = int, false = float
                             )
BatchEEGaug1 = Aug_warp(BatchEEG)
BatchEEGaug2 = Aug_warp(BatchEEG)
Sequence1= aug.SequentialAug(Aug_eye, Aug_warp)
BatchEEGaug1 = Sequence1(BatchEEG)
BatchEEGaug2 = Sequence1(BatchEEG)
Sequence2= aug.RandomAug(Aug_eye, Aug_warp, p=[0.7, 0.3])
BatchEEGaug1 = Sequence2(BatchEEG)
BatchEEGaug2 = Sequence2(BatchEEG)

# DEFINE AUGMENTER
# FIRST RANDOM SELECTION: APPLY FLIP OR CHANGE REFERENCE OR NOTHING
AUG_flipv = aug.StaticSingleAug(aug.flip_vertical)
AUG_flipr = aug.StaticSingleAug(aug.flip_horizontal)
AUG_id = aug.StaticSingleAug(aug.identity)
Sequence1 = aug.RandomAug( AUG_id, AUG_flipv, AUG_id, p=[0.5, 0.25, 0.25])

# SECOND RANDOM SELECTION: ADD SOME NOISE
AUG_band = aug.DynamicSingleAug(aug.add_band_noise, 
                                 discrete_arg={'bandwidth': ["delta", "theta", "alpha", "beta", (30,49) ], 
                                               'samplerate': Fs,
                                               'noise_range': 0.1
                                              }
                                )
Aug_eye = aug.DynamicSingleAug(aug.add_eeg_artifact,
                               discrete_arg = {'Fs': Fs, 'artifact': 'eye', 'batch_equal': False},
                               range_arg= {'amplitude': [0.1,0.5]},
                               range_type={'amplitude': False}
                             )
Sequence2 = aug.RandomAug( AUG_band, Aug_eye)

# THIRD RANDOM SELECTION: CROP OR RANDOM PERMUTATION
AUG_crop = aug.DynamicSingleAug(aug.crop_and_resize,
                                discrete_arg={'batch_equal': False},
                                range_arg ={'N_cut': [1, 4], 'segments': [10,15]},
                                range_type ={'N_cut': True, 'segments': True})
Sequence3 = aug.RandomAug( AUG_crop, Aug_warp)

# FINAL AUGMENTER: SEQUENCE OF THE THREE RANDOM LISTS
Augmenter = aug.SequentialAug(Sequence1, Sequence2, Sequence3)
BatchEEGaug1 = Augmenter(BatchEEG)
BatchEEGaug2 = Augmenter(BatchEEG)

print('   augmentation composition OK')

