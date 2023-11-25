import os
import random
import pickle
import sys
sys.path.append(os.getcwd().split('/test')[0])
import itertools
import numpy as np
from torch.utils.data import DataLoader
from selfeeg import dataloading as dl
import platform

def create_dataset(folder_name='Simulated_EEG',
                   data_len=[1024,4097], 
                   is3D=False, 
                   seed=1234):
    N=1000
    if not(os.path.isdir(folder_name)):
        os.mkdir(folder_name)

    np.random.seed(seed=seed)
    for i in range(N):
        if is3D:
            x = np.random.randn(2,2,np.random.randint(data_len[0],data_len[1]))
        else:
            x = np.random.randn(2,np.random.randint(data_len[0],data_len[1]))
        y = np.random.randint(1,5)
        sample = {'data': x, 'label': y}
        A, B, C = (int(i//200)+1), (int( (i - 200*int(i//200)))//5+1), (i%5+1)
        file_name = 'Simulated_EEG/' + str(A) + '_' + str(B) + '_' + str(C) + '_1.pickle'
        with open(file_name, 'wb') as f:
            pickle.dump(sample, f)

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


# In[2]:


print('\n---------------------------')
print('TESTING DATALOADING MODULE')
print('the process will generetate two additional folders')
print('---------------------------')


# In[3]:


# Define partition spec
if not(os.path.isdir('tmpsave')):
        os.mkdir('tmpsave')
eegpath  = 'Simulated_EEG'
freq     = 128        # sampling frequency in [Hz]
overlap  = 0.15       # overlap between partitions
window   = 2          # window length in [seconds]

create_dataset()
print('Testing GetEEGPartitionNumber...', end="", flush=True)
input_grid = {'EEGpath': [eegpath], 'freq': [freq], 'window': [2], 'overlap':[0,0.15],
               'includePartial':[True,False],'file_format':['*.pickle'],'load_function':[loadEEG],
              'optional_load_fun_args':[[False]],'transform_function':[transformEEG],
              'optional_transform_fun_args':[[32]],'keep_zero_sample':[True,False],'save':[True,False],
              'save_path':['tmpsave/results1.csv'],'verbose':[False]}
input_grid = makeGrid(input_grid)
for i in input_grid:
    EEGlen = dl.GetEEGPartitionNumber(**i)
print('   GetEEGPartitionNumber OK: tested', len(input_grid), 'combination of input arguments')


# In[4]:


print('Testing GetEEGSplitTable...', end="", flush=True)
EEGlen = dl.GetEEGPartitionNumber(eegpath, freq, window, overlap, file_format='*.pickle', 
                                  load_function=loadEEG, optional_load_fun_args=[False],
                                  transform_function=transformEEG )
Labels = np.zeros(EEGlen.shape[0], dtype=int)
for i in range(EEGlen.shape[0]):
    _ , Labels[i] = loadEEG(EEGlen.iloc[i]['full_path'], True)

input_grid = {'partition_table': [EEGlen], 'test_ratio': [0, 0.2], 'val_ratio': [0, 0.2], 
              'test_split_mode':[0,1,2], 'val_split_mode':[1,2],
              'exclude_data_id':[None,{x:[13,23] for x in range(1,6)}],
              'test_data_id':[None,{x:[14,22] for x in range(1,6)}, 4 ],
              'val_data_id':[None,{x:[15,21] for x in range(1,6)},[3] ],
              'val_ratio_on_all_data': [True,False], 'stratified':[False,True], 'labels':[Labels],
              'dataset_id_extractor':[lambda x: int(x.split('_')[0])],
              'subject_id_extractor':[None, lambda x: int(x.split('_')[0])],
              'save':[True], 'split_tolerance':[0.005], 'perseverance':[5000],
              'save_path':['tmpsave/results1.csv']
             }
input_grid = makeGrid(input_grid)
for i in input_grid:
    EEGsplit = dl.GetEEGSplitTable(**i)
print('   GetEEGSplitTable OK: tested', len(input_grid), 'combination of input arguments')


# In[5]:


print('Testing GetEEGSplitTableKfold...', end="", flush=True)
EEGlen = dl.GetEEGPartitionNumber(eegpath, freq, window, overlap, file_format='*.pickle', 
                                  load_function=loadEEG, optional_load_fun_args=[False],
                                  transform_function=transformEEG )
input_grid = {'partition_table': [EEGlen], 'test_ratio': [0, 0.2], 'kfold': [5,10], 
              'test_split_mode':[1,2], 'val_split_mode':[1,2],
              'exclude_data_id':[None,{x:[13,23] for x in range(1,6)}],
              'test_data_id':[None,{x:[14,22] for x in range(1,6)}, 4 ],
              'stratified':[False,True], 'labels':[Labels],
              'save':[True], 'split_tolerance':[0.005], 'perseverance':[5000],
              'save_path':['tmpsave/results1.csv']
             }
input_grid = makeGrid(input_grid)
for i in input_grid:
    EEGsplit = dl.GetEEGSplitTableKfold(**i)
print('   GetEEGSplitTableKfold OK: tested', len(input_grid), 'combination of input arguments')


# In[6]:


print('Testing EEGDataset on both unsupervised and supervised mode...', end="", flush=True)
EEGlen = dl.GetEEGPartitionNumber(eegpath, freq, window, overlap, file_format='*.pickle', 
                                  load_function=loadEEG, optional_load_fun_args=[False],
                                  transform_function=transformEEG )
EEGsplit = dl.GetEEGSplitTable(EEGlen, 
                               test_ratio=0.1, val_ratio=0.1,
                               test_split_mode='file', val_split_mode= 'file',
                               stratified=True, labels=Labels,
                               perseverance=5000, split_tolerance=0.005
                              )
dataset_pretrain = dl.EEGDataset(EEGlen, EEGsplit, 
                                 [freq, window, overlap], # split parameters must be given as list
                                 mode = 'train', #default, select all samples in the train set
                                 load_function = loadEEG, 
                                 transform_function=transformEEG
                                )
sample_1 = dataset_pretrain.__getitem__(0)
dataset_finetune = dl.EEGDataset(EEGlen, EEGsplit, 
                                 [freq, window, overlap], # split parameters must be given as list
                                 mode = 'train', #default, select all samples in the train set
                                 supervised = True, # !IMPORTANT!
                                 load_function = loadEEG,
                                 optional_load_fun_args= [True], #tells loadEEG to return a label
                                 transform_function=transformEEG,
                                 label_on_load=True, #default, 
                                )
sample_2, label_2 = dataset_finetune.__getitem__(0)
print('   EEGDataset OK')


# In[7]:


print('Testing Sampler on both mode...', end="", flush=True)
sampler_linear = dl.EEGsampler(dataset_pretrain, Mode=0)
sampler_custom = dl.EEGsampler(dataset_pretrain, 16, 4)
print('   EEGDataset OK')


# In[8]:


print('Testing Dataloader batch construction...', end="", flush=True)
Final_Dataloader = DataLoader( dataset = dataset_pretrain, 
                               batch_size= 16, 
                               sampler=sampler_custom, 
                               num_workers=4)
for X in Final_Dataloader:
    pass
print('   Dataloader OK')

