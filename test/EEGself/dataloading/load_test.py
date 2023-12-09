import unittest
import os
import platform
import random
import pickle
import sys
import itertools
import numpy as np
import torch
from torch.utils.data import DataLoader
from selfeeg import dataloading as dl

class TestDataloading(unittest.TestCase):

    def create_dataset(self, folder_name='Simulated_EEG', data_len=[1024,4097], seed=1234):
        N=1000
        if not(os.path.isdir(folder_name)):
            os.mkdir(folder_name)
        np.random.seed(seed=seed)
        for i in range(N):
            x = np.random.randn(2,np.random.randint(data_len[0],data_len[1]))
            y = np.random.randint(1,5)
            sample = {'data': x, 'label': y}
            A, B, C = (int(i//200)+1), (int( (i - 200*int(i//200)))//5+1), (i%5+1)
            file_name = 'Simulated_EEG/' + str(A) + '_' + str(B) + '_' + str(C) + '_1.pickle'
            with open(file_name, 'wb') as f:
                pickle.dump(sample, f)

    def loadEEG(self, path, return_label=False):
        with open(path, 'rb') as handle:
            EEG = pickle.load(handle)
        x = EEG['data']
        y = EEG['label']
        if return_label:
            return x, y
        else:
            return x
    
    def transformEEG(self, EEG, value=64):
        EEG = EEG[:,:-value]
        return EEG
    
    def makeGrid(self, pars_dict):  
        keys=pars_dict.keys()
        combinations=itertools.product(*pars_dict.values())
        ds=[dict(zip(keys,cc)) for cc in combinations]
        return ds

    @classmethod
    def setUpClass(cls):
        print('\n---------------------------')
        print('TESTING DATALOADING MODULE')
        print('the process will generetate two additional folders')
        print('---------------------------')
        if not(os.path.isdir('tmpsave')):
                os.mkdir('tmpsave')
        cls.eegpath  = 'Simulated_EEG'
        cls.create_dataset(cls)
        cls.freq     = 128        # sampling frequency in [Hz]
        cls.overlap  = 0.15       # overlap between partitions
        cls.window   = 2          # window length in [seconds]

    
    def setUp(self):
        self.seed=1234
        random.seed(self.seed)
        np.random.seed(self.seed)
                

    def test_GetEEGPartitionNumber(self):
        # checks = Table with length 1000 and results in specific setting is correct
        print('Testing GetEEGPartitionNumber...', end="", flush=True)
        input_grid = {'EEGpath': [self.eegpath], 'freq': [self.freq], 
                      'window': [2], 'overlap':[0,0.15],
                      'includePartial':[True,False],'file_format':['*.pickle'],
                      'load_function':[self.loadEEG],
                      'optional_load_fun_args':[[False]],
                      'transform_function':[None,self.transformEEG],
                      'optional_transform_fun_args':[None,[32]],
                      'keep_zero_sample':[True,False],'save':[True,False],
                      'save_path':['tmpsave/results1.csv'],'verbose':[False]}
        input_grid = self.makeGrid(input_grid)
        for i in input_grid:
            EEGlen = dl.GetEEGPartitionNumber(**i)
            self.assertEqual( EEGlen.shape[0], 1000)
        #EEGlen = dl.GetEEGPartitionNumber(self.eegpath, self.freq, self.window, 0,
        #                                  file_format='*.pickle', load_function=self.loadEEG, 
        #                                  optional_load_fun_args=[False], includePartial=False)
        #self.assertEqual(EEGlen['N_samples'].sum(),9448)                               
        print('   GetEEGPartitionNumber OK: tested', 
              len(input_grid), 'combination of input arguments')

    
    
    def test_GetEEGSplitTable(self):
        # checks: table has length 1000, ratio = 0 means empty set, ids when given are splitted
        # corretly,
        print('Testing GetEEGSplitTable (this may take some time)...', end="", flush=True)
        EEGlen = dl.GetEEGPartitionNumber(self.eegpath, self.freq, self.window, self.overlap,
                                          file_format='*.pickle', load_function=self.loadEEG, 
                                          optional_load_fun_args=[False],
                                          transform_function=self.transformEEG )
        Labels = np.zeros(EEGlen.shape[0], dtype=int)
        for i in range(EEGlen.shape[0]):
            _ , Labels[i] = self.loadEEG(EEGlen.iloc[i]['full_path'], True)
        val_dict_id= [ 25,  26,  27,  28,  29,  60,  61,  62,  63,  64, 225, 226, 227,
            228, 229, 260, 261, 262, 263, 264, 425, 426, 427, 428, 429, 460,
            461, 462, 463, 464, 625, 626, 627, 628, 629, 660, 661, 662, 663,
            664, 825, 826, 827, 828, 829, 860, 861, 862, 863, 864]
        test_dict_id=[ 20,  21,  22,  23,  24,  65,  66,  67,  68,  69, 220, 221, 222,
            223, 224, 265, 266, 267, 268, 269, 420, 421, 422, 423, 424, 465,
            466, 467, 468, 469, 620, 621, 622, 623, 624, 665, 666, 667, 668,
            669, 820, 821, 822, 823, 824, 865, 866, 867, 868, 869]
        excl_dict_id = [ 15,  16,  17,  18,  19,  70,  71,  72,  73,  74, 215, 216, 217,
            218, 219, 270, 271, 272, 273, 274, 415, 416, 417, 418, 419, 470,
            471, 472, 473, 474, 615, 616, 617, 618, 619, 670, 671, 672, 673,
            674, 815, 816, 817, 818, 819, 870, 871, 872, 873, 874]
        input_grid = {'partition_table': [EEGlen], 'test_ratio': [0, 0.2], 'val_ratio': [0, 0.2], 
                      'test_split_mode':[0,1,2], 'val_split_mode':[1,2],
                      'exclude_data_id':[None,{x:[13,23] for x in range(1,6)}],
                      'test_data_id':[None,{x:[14,22] for x in range(1,6)}, 4 ],
                      'val_data_id':[None,{x:[15,21] for x in range(1,6)},[3] ],
                      'val_ratio_on_all_data': [True,False], 
                      'stratified':[False,True], 'labels':[Labels],
                      'dataset_id_extractor':[lambda x: int(x.split('_')[0])],
                      'subject_id_extractor':[None],
                      'save':[True], 'split_tolerance':[0.001], 'perseverance':[10000],
                      'save_path':['tmpsave/results1.csv'],
                      'seed':[self.seed]}
        input_grid = self.makeGrid(input_grid)
        for n, i in enumerate(input_grid):

            if n==500:
                print('proceding...', end="", flush=True)
            elif n==1000:
                print('a little more...', end="", flush=True)
            elif n==1500:
                print('almost done...', end="", flush=True)
            
            EEGsplit = dl.GetEEGSplitTable(**i)
            total_list=EEGsplit[EEGsplit['split_set']!=-1].index.tolist()
            tot = EEGlen.iloc[total_list]['N_samples'].sum()
            self.assertEqual(EEGsplit.shape[0],1000)
            
            if isinstance(i['exclude_data_id'],dict):
                check = EEGsplit.iloc[ excl_dict_id ]['split_set'].unique()
                self.assertTrue(len(check)==1)
                self.assertTrue(check[0]==-1)
            
            if i['test_ratio'] ==0:
                if isinstance(i['test_data_id'],dict):
                    check = EEGsplit.iloc[ test_dict_id ]['split_set'].unique()
                    self.assertTrue(len(check)==1)
                    self.assertTrue(check[0]==2)
                elif isinstance(i['test_data_id'],int):
                    if not(isinstance(i['exclude_data_id'],dict)):
                        cond = (EEGsplit['file_name'].str[0]=='4').values
                        check = EEGsplit[ cond ]['split_set'].unique()
                        self.assertTrue(len(check)==1)
                        self.assertTrue(check[0]==2)
                else:
                    self.assertEqual(EEGlen['N_samples'][EEGsplit['split_set']==2].sum(),0)
            else:
                if i['test_data_id'] is None:
                    ratio = abs(0.2-EEGlen['N_samples'][EEGsplit['split_set']==2].sum()/tot)
                    self.assertTrue(ratio<1e-2)
                    if not(i['stratified']) and i['test_split_mode']==0:
                        EEGsplit['dataid']= EEGsplit['file_name'].str[0]
                        group = EEGsplit.groupby(['dataid', 'split_set'])
                        lst=list(group.split_set.groups.keys())
                        result = [t for t in lst if t[1] == 2]
                        self.assertEqual(len(result),1)
                        check= 200 if i['exclude_data_id'] is None else 190
                        self.assertEqual(group.get_group(result[0]).shape[0],check)
            
            if i['val_ratio']==0:
                if isinstance(i['val_data_id'],dict):
                    if isinstance(i['test_data_id'],dict):
                        check = EEGsplit.iloc[ val_dict_id ]['split_set'].unique()
                        self.assertTrue(len(check)==1)
                        self.assertTrue(check[0]==1)
                #elif isinstance(i['val_data_id'],list):
                #    if not(isinstance(i['test_data_id'],dict) or 
                #           isinstance(i['exclude_data_id'],dict)):
                #        cond = (EEGsplit['file_name'].str[0]=='3').values
                #        check = EEGsplit[ cond ]['split_set'].unique()
                #        self.assertTrue(len(check)==1)
                #        self.assertTrue(check[0]==1)
                elif i['val_data_id'] is None:
                    self.assertEqual(EEGlen['N_samples'][EEGsplit['split_set']==1].sum(),0)
            else:
                if i['val_data_id'] is None:
                    thresh = 0.2 
                    if not(i['val_ratio_on_all_data']):
                        test_list = EEGsplit[EEGsplit['split_set']==2].index.tolist()
                        test_ratio = EEGlen.iloc[test_list]['N_samples'].sum()/tot
                        thresh = 0.2*(1-test_ratio)
                    ratio = abs(thresh-EEGlen['N_samples'][EEGsplit['split_set']==1].sum()/tot)
                    self.assertTrue(ratio<1e-2)
        
        EEGsplit = dl.GetEEGSplitTable(EEGlen, 0.2, 0.2, 2, 2, val_ratio_on_all_data=True,
                                       stratified=True, labels=Labels,split_tolerance=0.001,
                                       perseverance=10000, seed=1234)
        ratio = dl.check_split(EEGlen,EEGsplit,Labels,True,False)['class_ratio']
        self.assertTrue(np.abs(ratio-ratio.mean(0)).max() < 1e-3 )
        print('   GetEEGSplitTable OK: tested', len(input_grid), 'combination of input arguments')

    
    
    def test_GetEEGSplitTableKfold(self):
        # check: since this function is based on multiple calls of the previous one, 
        # we have already verified the quality of the single splits, so checks will be done on 
        # the size of the table and if each file is placed only ones in validation set, excluding
        # those placed in test or excluded
        print('Testing GetEEGSplitTableKfold...', end="", flush=True)
        EEGlen = dl.GetEEGPartitionNumber(self.eegpath, self.freq, self.window, self.overlap,
                                          file_format='*.pickle', load_function=self.loadEEG, 
                                          optional_load_fun_args=[False],
                                          transform_function=self.transformEEG )
        Labels = np.zeros(EEGlen.shape[0], dtype=int)
        for i in range(EEGlen.shape[0]):
            _ , Labels[i] = self.loadEEG(EEGlen.iloc[i]['full_path'], True)
        input_grid = {'partition_table': [EEGlen], 'test_ratio': [0, 0.2], 'kfold': [5,10], 
                      'test_split_mode':[1,2], 'val_split_mode':[1,2],
                      'exclude_data_id':[None,{x:[13,23] for x in range(1,6)}],
                      'test_data_id':[None,{x:[14,22] for x in range(1,6)}, 4 ],
                      'stratified':[False,True], 'labels':[Labels],
                      'save':[True], 'split_tolerance':[0.005], 'perseverance':[5000],
                      'save_path':['tmpsave/results1.csv']
                     }
        input_grid = self.makeGrid(input_grid)
        for i in input_grid:
            EEGsplit = dl.GetEEGSplitTableKfold(**i)
            self.assertEqual(EEGsplit.shape[0],1000)
            self.assertEqual(EEGsplit.shape[1],i['kfold']+1)
            sums = set(EEGsplit.sum(axis=1, numeric_only=True).unique().tolist())
            self.assertTrue(sums.issubset(set([-1*i['kfold'],1,2*i['kfold']])) )
            
        print('   GetEEGSplitTableKfold OK: tested', 
              len(input_grid), 'combination of input arguments')

    
    
    def test_EEGDataset(self):
        #checks: extraction is performed correctly
        print('Testing EEGDataset on both unsupervised and supervised mode...',
              end="", flush=True)
        EEGlen = dl.GetEEGPartitionNumber(self.eegpath, self.freq, self.window, self.overlap,
                                          file_format='*.pickle', load_function=self.loadEEG, 
                                          optional_load_fun_args=[False],
                                          transform_function=self.transformEEG )
        Labels = np.zeros(EEGlen.shape[0], dtype=int)
        for i in range(EEGlen.shape[0]):
            _ , Labels[i] = self.loadEEG(EEGlen.iloc[i]['full_path'], True)
        EEGsplit = dl.GetEEGSplitTable(EEGlen, 
                                       test_ratio=0.1, val_ratio=0.1,
                                       test_split_mode='file', val_split_mode= 'file',
                                       #stratified=False, labels=Labels,
                                       perseverance=5000, split_tolerance=0.005
                                      )
        dataset_pretrain = dl.EEGDataset(EEGlen, EEGsplit, 
                                         [self.freq, self.window, self.overlap],
                                         mode = 'train',
                                         load_function = self.loadEEG, 
                                         transform_function=self.transformEEG
                                        )
        sample_1 = dataset_pretrain.__getitem__(0)
        self.assertTrue(isinstance(sample_1,torch.Tensor))
        self.assertEqual(sample_1.shape[-1],256)
        dataset_finetune = dl.EEGDataset(EEGlen, EEGsplit, 
                                         [self.freq, self.window, self.overlap],
                                         mode = 'train',
                                         supervised = True,
                                         load_function = self.loadEEG,
                                         optional_load_fun_args= [True],
                                         transform_function=self.transformEEG,
                                         label_on_load=True
                                        )
        sample_2, label_2 = dataset_finetune.__getitem__(0)
        self.assertTrue(isinstance(sample_1,torch.Tensor))
        self.assertEqual(sample_2.shape[-1],256)
        self.assertTrue(isinstance(label_2,int))
        print('   EEGDataset OK')

    
    
    def test_EEGsamples(self):
        print('Testing Sampler on both mode...', end="", flush=True)
        EEGlen = dl.GetEEGPartitionNumber(self.eegpath, self.freq, self.window, self.overlap,
                                          file_format='*.pickle', load_function=self.loadEEG, 
                                          optional_load_fun_args=[False],save=True, 
                                          save_path='tmpsave/results1.csv')
        Labels = np.zeros(EEGlen.shape[0])#, dtype=in)
        for i in range(EEGlen.shape[0]):
            EEG , Labels[i] = self.loadEEG(EEGlen['full_path'][i], True)
        Labels= Labels.astype(int)
        EEGsplit = dl.GetEEGSplitTable(EEGlen, 
                                       test_ratio=0.1, val_ratio=0.1,
                                       test_split_mode='file', val_split_mode= 'file',
                                       stratified=True, labels=Labels,
                                       perseverance=5000, split_tolerance=0.005
                                      )
        dataset_pretrain = dl.EEGDataset(EEGlen, EEGsplit, 
                                         [self.freq, self.window, self.overlap],
                                         mode = 'train',
                                         load_function = self.loadEEG, 
                                         transform_function=self.transformEEG
                                        )
        sampler_linear = dl.EEGsampler(dataset_pretrain, Mode=0)
        sampler_custom = dl.EEGsampler(dataset_pretrain, 16, 4)
        print('   EEGDataset OK')


    @classmethod
    def tearDownClass(cls):
        print('removing generated residual directories (Simulated_EEG, tmpsave)')
        try: 
            if platform.system() == 'Windows':
                os.system('rmdir /Q /S Simulated_EEG')
                os.system('rmdir /Q /S tmpsave')
            else:
                os.system('rm -r Simulated_EEG')
                os.system('rm -r tmpsave')
        except:
            print('Failed to delete \"Simulated_EEG\" and \"tmpsave\" folders.'
                  ' Please don\'t hate me and do it manually')

