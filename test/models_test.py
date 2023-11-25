#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
sys.path.append(os.getcwd().split('/test')[0])
import itertools
import numpy as np
import torch

from selfeeg import models

def makeGrid(pars_dict):  
    keys=pars_dict.keys()
    combinations=itertools.product(*pars_dict.values())
    ds=[dict(zip(keys,cc)) for cc in combinations]
    return ds

device  = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
N       = 2
Chan    = 8
Samples = 2048
x   = torch.randn(N,Chan,Samples)
xl  = torch.randn(N,1,16,Samples)
xst = torch.randn(N,Samples,11,11)
xd  = torch.randn(N,128)

if device.type != 'cpu':
    x2  = torch.randn(N,Chan,Samples).to(device=device)
    xl2 = torch.randn(N,1,16,Samples).to(device=device) 
    xst = torch.randn(N,Samples,11,11).to(device=device)
    xd2 = torch.randn(N,128).to(device=device)


# In[20]:


print('\n---------------------')
print('TESTING MODELS MODULE')
if device.type != 'cpu':
    print('Found cuda device: testing nn.modules with both cpu and gpu (gpu load will be 3GB)')
else:
    print('Didn\'t found cuda device: testing nn.modules with only cpu')
print('---------------------')


# In[2]:


print('Testing Depthwise conv2d with max norm constraint...', end="", flush=True)
Depthwise_args = {'in_channels': [1],'depth_multiplier':[2,3,4], 'kernel_size': [(1,64),(5,1),(5,64)], 'stride':[1,2,3],
                 'dilation':[1,2], 'bias': [True, False], 'max_norm':[None, 2, 3], 'padding': ['valid']
                }
Depthwise_args = makeGrid(Depthwise_args)
for i in Depthwise_args:
    model = models.DepthwiseConv2d(**i)
    model.weight = torch.nn.Parameter(model.weight*10)
    model(xl)
    if i['max_norm'] is not None:
        norm= model.weight.norm(dim=2, keepdim=True).norm(dim=3,keepdim=True).squeeze()
        assert (norm>i['max_norm']).sum() == 0

if device.type != 'cpu':
    for i in Depthwise_args:
        model = models.DepthwiseConv2d(**i).to(device=device)
        model.weight = torch.nn.Parameter(model.weight*10)
        model(xl2)
        if i['max_norm'] is not None:
            norm= model.weight.norm(dim=2, keepdim=True).norm(dim=3,keepdim=True).squeeze()
            assert (norm>i['max_norm']).sum() == 0
print('   Depthwise conv2d OK: tested', len(Depthwise_args), ' combinations of input arguments') 


# In[3]:


print('Testing Separable conv2d with norm constraint...', end="", flush=True)
Separable_args = {'in_channels': [1],'out_channels': [5,16],
                  'depth_multiplier':[1,3], 'kernel_size': [(1,64),(5,1),(5,64)], 'stride':[1,2,3],
                  'dilation':[1,2], 'bias': [True, False], 'depth_max_norm':[None, 2, 3], 'padding': ['valid']
                 }
Separable_args = makeGrid(Separable_args)
for i in Separable_args:
    model = models.SeparableConv2d(**i)
    model(xl)

if device.type != 'cpu':
    for i in Separable_args:
        model = models.SeparableConv2d(**i).to(device=device)
        model(xl2)

print('   Separable conv2d OK: tested', len(Depthwise_args), 'combinations of input arguments') 


# In[4]:


print('Testing conv2d with max norm constraint...', end="", flush=True)
Conv_args = {'in_channels': [1],'out_channels':[5,16], 'kernel_size': [(1,64),(5,1),(5,64)], 
             'stride':[1,2,3], 'dilation':[1,2], 'bias': [True, False], 'max_norm':[None, 2, 3], 
             'padding': ['valid']
            }
Conv_args = makeGrid(Conv_args)
for i in Conv_args:
    model = models.ConstrainedConv2d(**i)
    model.weight = torch.nn.Parameter(model.weight*10)
    model(xl)
    if i['max_norm'] is not None:
        norm= model.weight.norm(dim=2, keepdim=True).norm(dim=3,keepdim=True).squeeze()
        assert (norm>i['max_norm']).sum() == 0

if device.type != 'cpu':
    for i in Conv_args:
        model = models.ConstrainedConv2d(**i).to(device=device)
        model.weight = torch.nn.Parameter(model.weight*10)
        model(xl2)
        if i['max_norm'] is not None:
            norm= model.weight.norm(dim=2, keepdim=True).norm(dim=3,keepdim=True).squeeze()
            assert (norm>i['max_norm']).sum() == 0
print('   Depthwise conv2d OK: tested', len(Depthwise_args), ' combinations of input arguments') 


# In[5]:


print('Testing Dense layer with max norm constraint...', end="", flush=True)
Dense_args = {'in_features': [128],'out_features':[32], 
              'bias': [True,False], 'max_norm':[None, 2, 3]}
Dense_args = makeGrid(Dense_args)
for i in Dense_args:
    model = models.ConstrainedDense(**i)
    model.weight = torch.nn.Parameter(model.weight*10)
    model(xd)
    if i['max_norm'] is not None:
        norm= model.weight.norm(dim=1, keepdim=True)
        assert (norm>i['max_norm']).sum() == 0

for i in Dense_args:
    model = models.ConstrainedDense(**i).to(device=device)
    model.weight = torch.nn.Parameter(model.weight*10)
    model(xd2)
    if i['max_norm'] is not None:
        norm= model.weight.norm(dim=1, keepdim=True)
        assert (norm>i['max_norm']).sum() == 0
print('   Dense layer OK: tested', len(Dense_args), ' combinations of input arguments') 


# In[6]:


print('Testing DeepConvNet...', end="", flush=True)
DCN_args = {'nb_classes': [2,4], 'Chans': [Chan], 
            'Samples': [Samples], 'kernLength':[10,20], 
            'F': [12,25], 'Pool': [3,4], 'stride': [3,4], 'max_norm': [2.0], 
            'batch_momentum': [0.9], 'ELUalpha': [1], 'dropRate': [0.5], 
            'max_dense_norm': [1.0], 'return_logits': [True,False]
           }
DCN_grid = makeGrid(DCN_args)
for i in DCN_grid:
    model = models.DeepConvNet(**i)
    model(x)
if device.type != 'cpu':
    for i in DCN_grid:
        model = models.DeepConvNet(**i).to(device=device)
        model(x2)
print('   DeepConvNet OK: tested ', len(DCN_grid), ' combinations of input arguments')


# In[7]:


print('Testing EEGInception...', end="", flush=True)
EEGin_args = {'nb_classes': [2,4], 'Chans': [Chan], 
            'Samples': [Samples], 'kernel_size':[32,128], 
            'F1': [4,16], 'D': [2,4], 'pool': [4, 8],
            'batch_momentum': [0.9], 'dropRate': [0.5], 
            'max_depth_norm': [1.0], 'return_logits': [True,False], 'bias':[True,False]
           }
EEGin_args = makeGrid(EEGin_args)
for i in EEGin_args:
    model = models.EEGInception(**i)
    model(x)

if device.type != 'cpu':
    for i in EEGin_args:
        model = models.EEGInception(**i).to(device=device)
        model(x2)
print('   EEGInception OK: tested', len(EEGin_args), ' combinations of input arguments')


# In[8]:


print('Testing EEGnet...', end="", flush=True)
EEGnet_args = {'nb_classes': [2,4], 'Chans': [Chan], 
               'Samples': [Samples], 'kernLength':[32,64,128], 
               'F1': [4,8,16], 'D': [2,4], 'F2':[8,16,32], 'pool1': [4, 8],
               'pool2':[8,16], 'separable_kernel':[16,32],
               'return_logits': [True,False]
              }
EEGnet_args = makeGrid(EEGnet_args)
for i in EEGnet_args:
    model = models.EEGNet(**i)
    model(x)

if device.type != 'cpu':
    for i in EEGnet_args:
        model = models.EEGNet(**i).to(device=device)
        model(x2)
print('   EEGnet OK: tested', len(EEGnet_args), ' combinations of input arguments')


# In[9]:


print('Testing EEGsym...', end="", flush=True)
EEGsym_args = {'nb_classes': [2,4],'Samples':[2048], 'Chans': [Chan], 
               'Fs': [64], 'scales_time':[(500,250,125),(250,183,95)], 
               'lateral_chans': [2,3], 'first_left': [True,False], 'F':[8,24],
               'pool':[2,3],
               'bias':[True,False],
               'return_logits': [True,False]
              }
EEGsym_args = makeGrid(EEGsym_args)
for i in EEGsym_args:
    model = models.EEGSym(**i)
    model(x)

if device.type != 'cpu':
    for i in EEGsym_args:
        model = models.EEGSym(**i).to(device=device)
        model(x2)
print('   EEGsym OK: tested', len(EEGsym_args), ' combinations of input arguments')


# In[10]:


print('Testing ResNet...', end="", flush=True)
EEGres_args = {'nb_classes': [2,4],'Samples':[2048], 'Chans': [Chan],
               'block':[models.BasicBlock1], 
               'Layers': [[1,1,1,1],[2,2,2,2],[1,2,4,3]], 'inplane':[8,16,32], 
               'kernLength': [7,13,15], 'addConnection': [True, False],
               'return_logits': [True,False]
              }
EEGres_args = makeGrid(EEGres_args)
for i in EEGres_args:
    model = models.ResNet1D(**i)
    model(x)

if device.type != 'cpu':
    for i in EEGres_args:
        model = models.ResNet1D(**i).to(device=device)
        model(x2)
print('   ResNet OK: tested', len(EEGres_args), ' combinations of input arguments')


# In[11]:


print('Testing ShallowNet...', end="", flush=True)
EEGsha_args = {'nb_classes': [2,4],'Samples':[2048], 'Chans': [Chan],
               'F': [20,40,80], 'K1':[25,12,50], 
               'Pool': [75,50,100],
               'return_logits': [True,False]
              }
EEGsha_args = makeGrid(EEGsha_args)
for i in EEGsha_args:
    model = models.ShallowNet(**i)
    model(x)

if device.type != 'cpu':
    for i in EEGsha_args:
        model = models.ShallowNet(**i).to(device=device)
        model(x2)
print('   ShallowNet OK: tested', len(EEGsha_args), ' combinations of input arguments')


# In[12]:


print('Testing StageRNet...', end="", flush=True)
EEGsta_args = {'nb_classes': [2,4],'Samples':[2048], 'Chans': [Chan],
               'F': [8,16,4], 'kernLength':[64,32,120], 
               'Pool': [16,30,8],
               'return_logits': [True,False]
              }
EEGsta_args = makeGrid(EEGsta_args)
for i in EEGsta_args:
    model = models.StagerNet(**i)
    model(x)

if device.type != 'cpu':
    for i in EEGsta_args:
        model = models.StagerNet(**i).to(device=device)
        model(x2)
print('   StageRNet OK: tested', len(EEGsha_args), ' combinations of input arguments')


# In[13]:


print('Testing STNet...', end="", flush=True)
EEGstn_args = {'nb_classes': [2,4],'Samples':[2048], 'grid_size': [5,9],
               'F': [256,512,64], 'kernlength':[5,7], 'dense_size':[1024,512],
               'return_logits': [True,False]
              }
EEGstn_args = makeGrid(EEGstn_args)
for i in EEGstn_args:
    model = models.STNet(**i)
    xst = torch.randn(N,Samples,i['grid_size'],i['grid_size'])
    model(xst)

if device.type != 'cpu':
    for i in EEGstn_args:
        model = models.STNet(**i).to(device=device)
        xst2 = torch.randn(N,Samples,i['grid_size'],i['grid_size']).to(device=device)
        model(xst2)
print('   STNet OK: tested', len(EEGstn_args), ' combinations of input arguments')


# In[14]:


print('Testing TinySleepNet...', end="", flush=True)
#nb_classes, Chans, Fs, F=128, kernlength=8, pool=8, 
#dropRate=0.5, batch_momentum=0.1, max_dense_norm=2.0, return_logits=True
EEGsleep_args = {'nb_classes': [2,4],'Chans':[Chan], 'Fs': [64], 'F':[128,64,32],
                 'kernlength':[8,16,30], 'pool': [16,5,8], 'hidden_lstm': [128,50],
                 'return_logits': [True,False]
                }
EEGsleep_args = makeGrid(EEGsleep_args)
for i in EEGsleep_args:
    model = models.TinySleepNet(**i)
    model(x)

if device.type != 'cpu':
    for i in EEGsleep_args:
        model = models.TinySleepNet(**i).to(device=device)
        model(x2)
print('   TinySleepNet OK: tested', len(EEGsleep_args), ' combinations of input arguments')

