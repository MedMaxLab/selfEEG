#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
sys.path.append(os.getcwd().split('/test')[0])
import itertools
import numpy as np
import torch

from selfeeg import losses

def makeGrid(pars_dict):  
    keys=pars_dict.keys()
    combinations=itertools.product(*pars_dict.values())
    ds=[dict(zip(keys,cc)) for cc in combinations]
    return ds

device  = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
N       = 64
Feat    = 128
x  = torch.randn(N,Feat)
y  = torch.randn(N,Feat)
p  = torch.randn(N,Feat)
z  = torch.randn(N,Feat)
u  = torch.randn(Feat, 1024)

if device.type != 'cpu':
    x2  = torch.randn(N,Feat).to(device=device)
    y2  = torch.randn(N,Feat).to(device=device)
    p2  = torch.randn(N,Feat).to(device=device)
    z2  = torch.randn(N,Feat).to(device=device)
    u2  = torch.randn(Feat, 1024).to(device=device)


# In[2]:


print('\n---------------------')
print('TESTING LOSSES MODULE')
if device.type != 'cpu':
    print('Found cuda device: testing nn.modules with both cpu and gpu')
else:
    print('Didn\'t found cuda device: testing nn.modules with only cpu')
print('---------------------')


# In[3]:


print('Testing Barlow Loss...', end="", flush=True)
Barlow_args = {'z1': [x],'z2': [y,None], 'lambda_coeff': [0.005, 0.05, 0.5, 1]}
Barlow_args = makeGrid(Barlow_args)
for i in Barlow_args:
    loss = losses.Barlow_loss(**i)

if device.type != 'cpu':
    Barlow_args = {'z1': [x2],'z2': [y2,None], 'lambda_coeff': [0.005, 0.05, 0.5, 1]}
    Barlow_args = makeGrid(Barlow_args)
    for i in Barlow_args:
        loss = losses.Barlow_loss(**i)
print('   Barlow Loss OK: tested', len(Barlow_args), 'combinations of input arguments')


# In[4]:


print('Testing BYOL Loss...', end="", flush=True)
BYOL_args = {'z1': [x],'z2': [y],'p1': [p],'p2': [z], 'projections_norm': [True,False]}
BYOL_args = makeGrid(BYOL_args)
for i in BYOL_args:
    loss = losses.BYOL_loss(**i)

if device.type != 'cpu':
    BYOL_args = {'z1': [x2],'z2': [y2],'p1': [p2],'p2': [z2], 'projections_norm': [True,False]}
    BYOL_args = makeGrid(BYOL_args)
    for i in BYOL_args:
        loss = losses.BYOL_loss(**i)
print('   BYOL Loss OK: tested', len(BYOL_args), 'combinations of input arguments')


# In[5]:


print('Testing SimCLR Loss...', end="", flush=True)
SimCLR_args = {'projections': [x],'temperature':[0.15, 0.5, 0.7], 'projections_norm': [True,False]}
SimCLR_args = makeGrid(SimCLR_args)
for i in SimCLR_args:
    loss = losses.SimCLR_loss(**i)

if device.type != 'cpu':
    SimCLR_args = {'projections': [x],'temperature':[0.15, 0.5, 0.7], 'projections_norm': [True,False]}
    SimCLR_args = makeGrid(SimCLR_args)
    for i in SimCLR_args:
        loss = losses.SimCLR_loss(**i)
print('   SimCLR Loss OK: tested', len(SimCLR_args), 'combinations of input arguments')


# In[6]:


print('Testing SimSiam Loss...', end="", flush=True)
Siam_args = {'z1': [x],'z2': [y],'p1': [p],'p2': [z], 'projections_norm': [True,False]}
Siam_args = makeGrid(Siam_args)
for i in Siam_args:
    loss = losses.SimSiam_loss(**i)

if device.type != 'cpu':
    Siam_args = {'z1': [x2],'z2': [y2],'p1': [p2],'p2': [z2], 'projections_norm': [True,False]}
    Siam_args = makeGrid(Siam_args)
    for i in Siam_args:
        loss = losses.SimSiam_loss(**i)
print('   BYOL Loss OK: tested', len(Siam_args), 'combinations of input arguments')


# In[7]:


print('Testing VICReg Loss...', end="", flush=True)
Vicreg_args = {'z1': [x],'z2': [y,None], 'Lambda': [25,10,50],'Mu': [25,5,50], 'Nu': [2,1,0.5]}
Vicreg_args = makeGrid(Vicreg_args)
for i in Vicreg_args:
    loss = losses.VICReg_loss(**i)

if device.type != 'cpu':
    Vicreg_args = {'z1': [x2],'z2': [y2,None], 'Lambda': [25,10,50],'Mu': [25,5,50], 'Nu': [2,1,0.5]}
    Vicreg_args = makeGrid(Vicreg_args)
    for i in Vicreg_args:
        loss = losses.VICReg_loss(**i)
print('   VICReg Loss OK: tested', len(Vicreg_args), 'combinations of input arguments')


# In[8]:


print('Testing MoCo Loss...', end="", flush=True)
Moco_args = {'q': [x],'k': [y], 'queue': [None, u],
             'projections_norm': [True, False], 'temperature': [0.15, 0.5, 0.9]}
Moco_args = makeGrid(Moco_args)
for i in Moco_args:
    loss = losses.Moco_loss(**i)

if device.type != 'cpu':
    Moco_args = {'q': [x],'k': [y], 'queue': [None, u],
                 'projections_norm': [True, False], 'temperature': [0.15, 0.5, 0.9]}
    Moco_args = makeGrid(Moco_args)
    for i in Moco_args:
        loss = losses.Moco_loss(**i)
print('   MoCo Loss OK: tested', len(Moco_args), 'combinations of input arguments')

