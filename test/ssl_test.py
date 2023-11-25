#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import pickle
import sys
sys.path.append(os.getcwd().split('/test')[0])
import itertools
import platform

# IMPORT CLASSICAL PACKAGES
import numpy as np
import pandas as pd

# IMPORT TORCH
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# IMPORT CUSTOM SELF-SUPERVISED LEARNING FOR EEG LIBRARY
from selfeeg import augmentation as aug
from selfeeg import dataloading as dl
from selfeeg import models as zoo
from selfeeg import ssl
from selfeeg import losses
from selfeeg import utils

def create_dataset(folder_name='Simulated_EEG',
                   Sample_range= [512, 1025],
                   Chans = 16,
                   return_labels = True,
                   seed=1234):
    N=1000
    if not(os.path.isdir(folder_name)):
        os.mkdir(folder_name)

    np.random.seed(seed=seed)
    classes = np.zeros(N)
    for i in range(N):
        Sample = np.random.randint(Sample_range[0],Sample_range[1])
        y = np.random.choice([0,1], p=[0.8,0.2])
        classes[i] = y
        x = 600
        while (np.max(x)>550 or np.min(x)<-550):
            if y == 1:
                stderr = np.sqrt(122.35423)
                F1 = np.random.normal(0.932649, 0.040448)
                F0 = np.random.normal(2.1159355, 2.3523977)
            else:
                stderr = np.sqrt(454.232666)
                F1 = np.random.normal(0.9619603, 0.0301687)
                F0 = np.random.normal(-0.1810323, 3.4712047)
            x = np.zeros((Chans,Sample))
            x[:,0] = np.random.normal( 0, stderr, Chans )  
            for k in range(1,Sample):
                x[:,k] = F0+ F1*x[:,k-1] + np.random.normal( 0, stderr, Chans )
                
        sample = {'data': x, 'label': y}
        A, B, C = (int(i//200)+1), (int( (i - 200*int(i//200)))//5+1), (i%5+1)
        file_name = 'Simulated_EEG/' + str(A) + '_' + str(B) + '_' + str(C) + '_1.pickle'
        with open(file_name, 'wb') as f:
            pickle.dump(sample, f)
    if return_labels:
        return classes

def loadEEG(path, return_label=False):
    with open(path, 'rb') as handle:
        EEG = pickle.load(handle)
    x = EEG['data']
    y = EEG['label']
    if return_label:
        return x, y
    else:
        return x

def transformEEG(EEG, value=64):
    EEG = EEG[:,:-value]
    return EEG

def makeGrid(pars_dict):  
    keys=pars_dict.keys()
    combinations=itertools.product(*pars_dict.values())
    ds=[dict(zip(keys,cc)) for cc in combinations]
    return ds

device  = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# In[2]:


print('\n---------------------------')
print('TESTING SSL MODULE')
if device.type != 'cpu':
    print('Found cuda device: testing ssl module on it')
else:
    print('Didn\'t found cuda device: testing ssl module on cpu')
print('---------------------------')


# In[3]:


# DEFINE FILE PATH, SAMPLING RATE, WINDOW LENGTH, OVERLAP PERCENTAGE, WORKERS AND BATCH SIZE
print('Defining dataloaders, augmenter and model')
eegpath = 'Simulated_EEG'
freq = 128
window = 2
overlap = 0.1
workers = 4
batchsize = 16
Chan = 16

classes = create_dataset()

# CALCULATE DATASET LENGTH
EEGlen = dl.GetEEGPartitionNumber(eegpath, freq, window, overlap, file_format='*.pickle', 
                                  load_function=loadEEG)

# SPLIT DATASET
EEGsplit= dl.GetEEGSplitTable(partition_table=EEGlen, val_ratio= 0.1, stratified=True, labels=classes,
                              test_data_id=[5], split_tolerance=0.001, perseverance=5000)

# DEFINE TRAINING DATALOADER
trainset = dl.EEGDataset(EEGlen, EEGsplit, [freq, window, overlap], load_function=loadEEG)
trainsampler = dl.EEGsampler(trainset, batchsize, workers)
trainloader = DataLoader(dataset = trainset, batch_size= batchsize, sampler=trainsampler, num_workers=workers)

# DEFINE VALIDATION DATALOADER
valset = dl.EEGDataset(EEGlen, EEGsplit, [freq, window, overlap], 'validation', load_function=loadEEG)
valloader = DataLoader(dataset = valset, batch_size= batchsize, shuffle=False)


# In[4]:


# DEFINE AUGMENTER
# First block: noise addition
AUG_band = aug.DynamicSingleAug(aug.add_band_noise, 
                                 discrete_arg={'bandwidth': ["delta", "theta", "alpha", "beta", (30,49) ], 
                                               'samplerate': freq,'noise_range': 0.5}
                               )
AUG_mask = aug.DynamicSingleAug(aug.masking, discrete_arg = {'mask_number': [1,2,3,4], 'masked_ratio': 0.25})
Block1 = aug.RandomAug( AUG_band, AUG_mask, p=[0.7, 0.3])

# second block: rescale
Block2 = lambda x: utils.scale_range_soft_clip(x, 500, 1.5, 'uV', True)

# FINAL AUGMENTER: SEQUENCE OF THE THREE RANDOM LISTS
Augmenter = aug.SequentialAug(Block1, Block2)


# In[5]:


# SSL model
emb_size= 16*((freq*window)//int(4*8))
head_size=[ emb_size, 128, 64]
predictor_size= [64, 64]


# In[6]:


print('Testing SimCLR (5 epochs, verbose True)...')

NNencoder= zoo.EEGNetEncoder(Chans=Chan, kernLength=65)
SelfMdl = ssl.SimCLR(encoder=NNencoder, projection_head=head_size).to(device=device)

# loss (fit method has a default loss based on the SSL algorithm
loss=losses.SimCLR_loss
loss_arg={'temperature': 0.5}

# earlystopper
earlystop = ssl.EarlyStopping(patience=2, min_delta=1e-05, record_best_weights=True)
# optimizer
optimizer = torch.optim.Adam(SelfMdl.parameters(), lr=1e-3)
# lr scheduler
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
loss_info = SelfMdl.fit(train_dataloader = trainloader, augmenter=Augmenter, epochs=5,
                        optimizer=optimizer, loss_func= loss, loss_args= loss_arg,
                        lr_scheduler= scheduler, EarlyStopper=earlystop,
                        validation_dataloader=valloader,
                        verbose=True, device= device, return_loss_info=True
                       )
print('   SimCLR OK')


# In[7]:


#Moco
print('Testing MoCo v2 (5 epochs, verbose False)...')

NNencoder= zoo.EEGNetEncoder(Chans=Chan, kernLength=65)
SelfMdl = ssl.MoCo(encoder=NNencoder, projection_head=head_size, bank_size=1024, m=0.9995).to(device=device)

# loss (fit method has a default loss based on the SSL algorithm
loss=losses.Moco_loss
loss_arg={'temperature': 0.5}

# earlystopper
earlystop = ssl.EarlyStopping(patience=6, min_delta=1e-05, record_best_weights=True)
# optimizer
optimizer = torch.optim.Adam(SelfMdl.parameters(), lr=1e-4)
# lr scheduler
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
loss_info = SelfMdl.fit(train_dataloader = trainloader, augmenter=Augmenter, epochs=5,
                        optimizer=optimizer, loss_func= loss, loss_args= loss_arg,
                        lr_scheduler= scheduler, EarlyStopper=earlystop,
                        validation_dataloader=valloader,
                        verbose=False, device= device, return_loss_info=True
                       )
print('   MoCo v2 OK')


# In[8]:


#Moco
print('Testing MoCo v3 (5 epochs, verbose False)...')

NNencoder= zoo.EEGNetEncoder(Chans=Chan, kernLength=65)
SelfMdl = ssl.MoCo(encoder=NNencoder, projection_head=head_size, predictor=predictor_size, m=0.9995).to(device=device)

# loss (fit method has a default loss based on the SSL algorithm
loss=losses.Moco_loss
loss_arg={'temperature': 0.5}

# earlystopper
earlystop = ssl.EarlyStopping(patience=6, min_delta=1e-05, record_best_weights=True)
# optimizer
optimizer = torch.optim.Adam(SelfMdl.parameters(), lr=1e-4)
# lr scheduler
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
loss_info = SelfMdl.fit(train_dataloader = trainloader, augmenter=Augmenter, epochs=5,
                        optimizer=optimizer, loss_func= loss, loss_args= loss_arg,
                        lr_scheduler= scheduler, EarlyStopper=earlystop,
                        validation_dataloader=valloader,
                        verbose=False, device= device, return_loss_info=True
                       )
print('   MoCo v3 OK')


# In[9]:


#Moco
print('Testing BYOL (5 epochs, verbose False)...')

NNencoder= zoo.EEGNetEncoder(Chans=Chan, kernLength=65)
SelfMdl = ssl.BYOL(encoder=NNencoder, projection_head=head_size, predictor=predictor_size, m=0.9995).to(device=device)

# loss (fit method has a default loss based on the SSL algorithm
loss=losses.BYOL_loss

# earlystopper
earlystop = ssl.EarlyStopping(patience=6, min_delta=1e-05, record_best_weights=True)
# optimizer
optimizer = torch.optim.Adam(SelfMdl.parameters(), lr=1e-4)
# lr scheduler
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
loss_info = SelfMdl.fit(train_dataloader = trainloader, augmenter=Augmenter, epochs=5,
                        optimizer=optimizer, loss_func= loss,
                        lr_scheduler= scheduler, EarlyStopper=earlystop,
                        validation_dataloader=valloader,
                        verbose=False, device= device, return_loss_info=True
                       )
print('   BYOL OK')


# In[10]:


#Moco
print('Testing SimSiam (5 epochs, verbose False)...')

NNencoder= zoo.EEGNetEncoder(Chans=Chan, kernLength=65)
SelfMdl = ssl.SimSiam(encoder=NNencoder, projection_head=head_size, predictor=predictor_size).to(device=device)

# loss (fit method has a default loss based on the SSL algorithm
loss=losses.SimSiam_loss

# earlystopper
earlystop = ssl.EarlyStopping(patience=6, min_delta=1e-05, record_best_weights=True)
# optimizer
optimizer = torch.optim.Adam(SelfMdl.parameters(), lr=1e-4)
# lr scheduler
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
loss_info = SelfMdl.fit(train_dataloader = trainloader, augmenter=Augmenter, epochs=5,
                        optimizer=optimizer, loss_func= loss,
                        lr_scheduler= scheduler, EarlyStopper=earlystop,
                        validation_dataloader=valloader,
                        verbose=False, device= device, return_loss_info=True
                       )
print('   SimSiam OK')


# In[11]:


print('Testing VICReg (5 epochs, verbose True)...')

NNencoder= zoo.EEGNetEncoder(Chans=Chan, kernLength=65)
SelfMdl = ssl.VICReg(encoder=NNencoder, projection_head=head_size).to(device=device)

# loss (fit method has a default loss based on the SSL algorithm
loss=losses.VICReg_loss

# earlystopper
earlystop = ssl.EarlyStopping(patience=2, min_delta=1e-05, record_best_weights=True)
# optimizer
optimizer = torch.optim.Adam(SelfMdl.parameters(), lr=1e-3)
# lr scheduler
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
loss_info = SelfMdl.fit(train_dataloader = trainloader, augmenter=Augmenter, epochs=5,
                        optimizer=optimizer, loss_func= loss,
                        lr_scheduler= scheduler, EarlyStopper=earlystop,
                        validation_dataloader=valloader,
                        verbose=False, device= device, return_loss_info=True
                       )
print('   VICReg OK')


# In[12]:


print('Testing Barlow_Twins (5 epochs, verbose True)...')

NNencoder= zoo.EEGNetEncoder(Chans=Chan, kernLength=65)
SelfMdl = ssl.Barlow_Twins(encoder=NNencoder, projection_head=head_size).to(device=device)

# loss (fit method has a default loss based on the SSL algorithm
loss=losses.Barlow_loss

# earlystopper
earlystop = ssl.EarlyStopping(patience=2, min_delta=1e-05, record_best_weights=True)
# optimizer
optimizer = torch.optim.Adam(SelfMdl.parameters(), lr=1e-3)
# lr scheduler
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
loss_info = SelfMdl.fit(train_dataloader = trainloader, augmenter=Augmenter, epochs=5,
                        optimizer=optimizer, loss_func= loss,
                        lr_scheduler= scheduler, EarlyStopper=earlystop,
                        validation_dataloader=valloader,
                        verbose=False, device= device, return_loss_info=True
                       )
print('   Barlow_Twins OK')


# In[13]:


print('testing fine-tuning phase (10 epochs, verbose True)...')

# Extract only the samples for fine-tuning
filesFT= EEGsplit.loc[EEGsplit['split_set']==2, 'file_name'].values
EEGlenFT= EEGlen.loc[EEGlen['file_name'].isin(filesFT)].reset_index().drop(columns=['index'])
labels = classes[ EEGsplit[EEGsplit['split_set']==2].index.tolist()]

# split the fine-tuning data in train-test-validation
EEGsplitFT = dl.GetEEGSplitTable(partition_table=EEGlenFT, test_ratio = 0.2, val_ratio= 0.1, val_ratio_on_all_data=False,
                                 stratified=True, labels=labels, split_tolerance=0.001, perseverance=10000)

# TRAINING DATALOADER
trainsetFT = dl.EEGDataset(EEGlenFT, EEGsplitFT, [freq, window, overlap], 'train', 
                           supervised=True, label_on_load=True, 
                           load_function=loadEEG, optional_load_fun_args=[True])
trainsamplerFT = dl.EEGsampler(trainsetFT, batchsize, workers)
trainloaderFT = DataLoader(dataset = trainsetFT, batch_size= batchsize, sampler=trainsamplerFT, num_workers=workers)

# VALIDATION DATALOADER
valsetFT = dl.EEGDataset(EEGlenFT, EEGsplitFT, [freq, window, overlap], 'validation', 
                         supervised=True, label_on_load=True, 
                         load_function=loadEEG, optional_load_fun_args=[True])
valloaderFT = DataLoader(dataset = valsetFT, batch_size= batchsize, num_workers=workers, shuffle=False)

#TEST DATALOADER
testsetFT = dl.EEGDataset(EEGlenFT, EEGsplitFT, [freq, window, overlap], 'test', 
                          supervised=True, label_on_load=True, 
                          load_function=loadEEG, optional_load_fun_args=[True])
testloaderFT = DataLoader(dataset = testsetFT, batch_size= batchsize, shuffle=False)

FinalMdl = zoo.EEGNet(nb_classes = 2, Chans = Chan, Samples = int(freq*window), kernLength = 65)

# Transfer the pretrained backbone and move the final model to the right device
SelfMdl.train() 
SelfMdl.to(device='cpu') 
FinalMdl.encoder = SelfMdl.get_encoder()
FinalMdl.train()
FinalMdl.to(device=device)

# DEFINE LOSS
def loss_fineTuning(yhat, ytrue):
    ytrue = ytrue + 0.
    yhat = torch.squeeze(yhat)
    return F.binary_cross_entropy_with_logits(yhat, ytrue, pos_weight = torch.tensor([2.5]).to(device=device) )

# DEFINE EARLYSTOPPER
earlystopFT = ssl.EarlyStopping(patience=10, min_delta=1e-03, record_best_weights=True)

# DEFINE OPTIMIZER 
optimizerFT = torch.optim.Adam(FinalMdl.parameters(), lr=1e-3)
schedulerFT = torch.optim.lr_scheduler.ExponentialLR(optimizerFT, gamma=0.97)

finetuning_loss=ssl.fine_tune(model                 = FinalMdl,
                              train_dataloader      = trainloaderFT,
                              epochs                = 10,
                              optimizer             = optimizerFT,
                              loss_func             = loss_fineTuning, 
                              lr_scheduler          = schedulerFT,
                              EarlyStopper          = earlystopFT,
                              validation_dataloader = valloaderFT,
                              verbose               = True,
                              device                = device,
                              return_loss_info      = True
                             )

print('   fine-tuning OK')


# In[ ]:





# In[ ]:





# In[ ]:




