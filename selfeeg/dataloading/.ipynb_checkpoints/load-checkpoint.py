import os
import glob
import time
import math
import random
import copy
import warnings
from typing import Union, Sequence
import numpy as np
import pandas as pd
import tqdm
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, Sampler
from ..utils.utils import subarray_closest_sum, get_subarray_closest_sum

__all__ = ['GetEEGPartitionNumber', 
           'GetEEGSplitTable', 'GetEEGSplitTableKfold',
           'EEGDataset', 'EEGsampler']

# TO DO: ADD parallel loop in get EEG Partition Number

def GetEEGPartitionNumber(EEGpath: str, 
                          freq: int or float=250, 
                          window: int or float=2, 
                          overlap: int or float=0.10,
                          includePartial: bool=True,
                          file_format: str or list[str]='*',
                          load_function: 'function'=None,
                          optional_load_fun_args: list or dict=None,
                          transform_function: 'function'=None,
                          optional_transform_fun_args: list or dict=None,
                          keep_zero_sample: bool=True,
                          save: bool=False,
                          save_path: str=None,
                          verbose: bool=False
                         ) -> pd.DataFrame :
    """
    GetEEGPartitionNumber(EEGpath, freq, window, overlap=0)
    Find the number of unique partitions from EEG signals stored inside a given directory.
    Some default parameters are designed to work with the 'auto-BIDS' library. For more info,
    See: Link
    
    Return a Pandas DataFrame with the exact number of samples which can be extracted from each 
    EEG file in the EEGpath directory.
    
    Parameters
    ----------
    EEGpath : string 
        Directory with all EEG files. If the last element of the string is not "/", the character will
        be added automatically
    freq : int or float, optional 
        EEG sampling rate. Must be the same for all EEG files. 
        Default: 250
    window : int or float, optional
        Time window length in seconds.
        Default: 2
    overlap : float, optional 
        Same EEG recording overlap percentage. Must be in the interval [0,1)
        Default: 0.10
    includePartial : bool, optional
        Count also final EEG partitions at least half of the time windows filled with EEG recording
        Default: True
    file_format : str or list[str], optional
        A string used to detect a set of specific EEG files inside the give EEGpath. It is directly put after 
        EEGpath during call of the glob.glob() method. Therefore, it can contain shell-style wildcards (see glob.glob()
        help for more info). This parameter might be helpful if you have other files other than the EEGs in your 
        directory.
        Alternatively, you can provide a list of strings to cast multiple glob.glob() searches. This might be useful if
        you want to combine multiple identification criteria (e.g. specific file extensions, specific file names, etc.)
        Default: '*'
    load_function : 'function', optional
        A custom EEG file loading function. It will be used instead of the default: 
        loadmat(ii, simplify_cells=True)['DATA_STRUCT']['data'] (for files preprocessed with the AUTO-BIDS library)
        The function must take only one required argument, which is the full path to the EEG file 
        (e.g. the function will be called in this way: load_function(fullpath, optional_arguments) )
        Default: None
    optional_load_fun_args: list or dict, optional
        Optional arguments to give to the custom loading function. Can be a list or a dict.
        Default: None
    transform_function : 'function', optional
        A custom transformation to be applied after the EEG is loaded. Might be useful if there are portion of 
        the signal to cut. 
        The function must take only one required argument, which is the EEG file to transform
        (e.g. the function will be called in this way: transform_function(EEG, optional_arguments) )
        Default: None
    optional_transform_fun_args: list or dict, optional
        Optional arguments to give to the EEG transformation function. Can be a list or a dict.
        Default: None
    keep_zero_sample : bool, optional
        Whether to preserve Dataframe rows where the number of samples to extract is equal to zero.
    save : bool, optional
        Save the resulted DataFrame as a .csv file 
    save_path: str, optional
        A custom path to be used instead of the current working directory. It's the string given to the 
        pandas.DataFrame.to_csv() method.
        Default: None
    verbose: bool, optional
        whether to print or not some information during function excecution. Useful to keep track
                
    Returns
    -------
    lenEEG : DataFrame
        Three columns Pandas DataFrame. 
        The first column has the full path to the EEG file, the second the file name, 
        the third its number of partitions.
    
    Note
    ----
    freq*window must give an integer with the number of samples
    
    """
    # Check Inputs
    if (overlap<0) or (overlap>=1):
        raise ValueError("overlap must be a number in the interval [0,1)")
    if freq<=0:
        raise ValueError("the EEG sampling rate cannot be negative")
    if window<=0:
        raise ValueError("the time window cannot be negative")
    if (freq*window) != int(freq*window):
        raise ValueError("freq*window must give an integer number ")
        
    # Extract all files from directory
    if isinstance(file_format, str):
        if EEGpath[-1]=='/':
            EEGfiles=glob.glob(EEGpath + file_format)
        else:
            EEGfiles=glob.glob(EEGpath + '/' + file_format )
    else:
        try:
            if EEGpath[-1]=='/':
                EEGfiles = [glob.glob(eegpath + i) for i in file_format]
            else:
                EEGfiles = [glob.glob(eegpath + '/' + i) for i in file_format]
            EEGfiles = [item for sublist in EEGfiles for item in sublist]
        except:
            print('file_format must be a string or an iterable (e.g. list) of strings')
            return None

    if len(EEGfiles)==0:
        print('didn\'t found any with the given format')
        return None
        
                
               
    EEGfiles=sorted(EEGfiles)
    NumFiles= len(EEGfiles)

    # Create Table
    EEGlen=[]
    WindSample=freq*window
    overlapInt=round(WindSample*overlap)

    with tqdm.tqdm(total=len(EEGfiles), disable=not(verbose), 
                   desc='extracting EEG samples', unit=' batch') as pbar:
        for i, ii in enumerate(EEGfiles):
            
            if verbose:
                pbar.update()
            
            # load file, if custom function is provided use it to load data according 
            # to possible optional arguments
            if load_function is None:         
                EEG=loadmat(ii, simplify_cells=True)['DATA_STRUCT']['data']
            else:
                if isinstance(optional_load_fun_args, list):
                    EEG = load_function(ii, *optional_load_fun_args)
                elif isinstance(optional_load_fun_args, dict):
                    EEG = load_function(ii, **optional_load_fun_args)
                else:
                    EEG = load_function(ii)
    
            # transform loaded file if custom function is provided
            # and call function according to possible optional arguments
            if transform_function is not None:
                if isinstance(optional_transform_fun_args, list):
                    EEG = transform_function(EEG, *optional_transform_fun_args)
                elif isinstance(optional_transform_fun_args, dict):
                    EEG = transform_function(EEG, **optional_transform_fun_args)
                else:
                    EEG= transform_function(EEG)
    
            M=len(EEG.shape)
            if overlap==0:
                N_Partial=EEG.shape[-1]/(WindSample)
            else:
                L=EEG.shape[-1]
                N=(L-overlapInt)//(WindSample-overlapInt)
                #R=L-WindSample*N+overlapInt*(N-1)
                #N_Partial=N+(R+overlapInt)/WindSample
                R=(overlapInt-WindSample)*N
                N_Partial=N+(L+R)/WindSample
            
            if includePartial:
                N_EEG=round(N_Partial) if N_Partial>=1 else 0
            else:
                N_EEG=N
    
            # check for extra dimension (file with multiple trials)
            if M>2:
                warnings.warn(('Loaded a file with multiple EEGs ('+ str(M)+'-D array).'
                               ' Found number of sample will be multiplied by the size of each '
                               'extra dimension. Note that this may create problems to the '
                               '__getitem()__ method in the custom EEGDataset class'), Warning)
                N_EEG*= np.prod(EEG.shape[0:-2])
                
            EEGlen.append([ii, ii.split('/')[-1],N_EEG])
    
    del EEG
    EEGlen=pd.DataFrame(EEGlen,columns=['full_path','file_name','N_samples'])
    
    if not(keep_zero_sample):
        EEGlen = EEGlen.drop(EEGlen[EEGlen.N_samples == 0].index).reset_index()
        EEGlen = EEGlen.drop(columns= 'index')

    try:
        if save:
            if save_path is not None:
                EEGlen.to_csv(save_path)
            else:
                condition=True
                cnt=-1
                while condition:
                    cnt +=1
                    if cnt==0:
                        filename='EEGPartitionNumber.csv'
                        condition=os.path.isfile(filename)
                    else:
                        filename='EEGPartitionNumber_'+str(cnt)+'.csv'
                        condition=os.path.isfile(filename)
                EEGlen.to_csv(filename)
    except:
        print('failed to save file. Function output will be returned but not saved.')

    if verbose:
        w,o,s,d = 'window', 'overlap', 'sampling rate', 'dataset length'
        NN= EEGlen['N_samples'].sum()
        print('\nConcluded extraction of repository length with the following specific: \n')
        print(f'{w:15} ==> {window:5d} s')
        print(f'{o:15} ==> {overlap*100:5.0f} %')
        print(f'{s:15} ==> {freq:5d} Hz')
        print('-----------------------------')
        print(f'{d:15} ==> {NN:8d}')
                
    return EEGlen


def GetEEGSplitTable(partition_table: pd.DataFrame,
                     test_ratio: float= None,
                     val_ratio: float= None,
                     test_split_mode: str or int =2,
                     val_split_mode: str or int= 2,
                     exclude_data_id: list or dict =None,
                     test_data_id: list or dict=None,
                     val_data_id: list or dict =None,
                     val_ratio_on_all_data: bool=True,
                     stratified: bool=False,
                     labels: 'array like'=None,
                     dataset_id_extractor: 'function'=None,
                     subject_id_extractor: 'function'=None,
                     split_tolerance=0.01,
                     perseverance=1000,
                     save: bool=False,
                     save_path: str=None
                    ) -> pd.DataFrame:

    """   
    GetEEGSplitTable create a table defining the files to use for train, validation and test of the models
    
    Return a Pandas DataFrame defining which file should be included in the training, validation or test set
    
    Split is done in the following way:
    Dataset --> Train / Test 
                Train --> Train / Validation
    If specific ID are given, the split is done using them ignoring split ratio, otherwise split is done randomly
    using the given ratio. Keep in mind that Test or Validation can be empty, if for example you want to split 
    the dataset only in two subsets.
    
    Parameters
    ----------
    partition_table: pd.Dataframe, optional
        A two columns dataframe where:
            1-the first column has name 'file_name' and contain all the file names
            2-the second column has name 'N_samples' and has the number of samples which can be extracted from the file
        This table can be automatically created with a custom setting with the provided function GetEEGPartitionNumber()
        Default: None
    test_ratio: float, optional
        The percentage of data with respect to the whole number of samples (partitions) of the dataset to be included 
        in the test set. Must be a number in [0,1]. 0 means that the test split is skipped if test_data_id isn't given
        Default: None
    val_ratio: float, optional
        The percentage of data with respect to the whole number of samples (partitions) of the dataset or the remaining 
        ones after test split (see val_ratio_on_all_data argument) to be included in the validation set. 
        Must be a number in [0,1]. 0 means that the validation split is skipped if val_data_id isn't given
        Default: None
    test_split_mode: int or str, optional
        The type of split to perform in the step not_excluded_data --> train / test sets. It can be one of the following:
            1) any of [0, 'd', 'set', 'dataset'] = split will be performed using dataset ids, i.e. all files of the same
                dataset will be put in the same split set
            2) any of [1, 's', 'subj', 'subject'] = split will be performed using subjects ids, i.e. all files of the same
                subjects will be put in the same split set
            3) any of [2, 'file', 'record'] = split will be performed looking at single files
        Default: 2
    val_split_mode: int or str, optional
        The type of split to perform in the step train --> train / validation sets. Input allowed are the same as in
        test_split_mode.
        Default: 2
    exclude_data_id  : list or dict, optional 
        Dataset ID to be excluded. It can be:
            1) a list with all dataset ids to exclude
            2) a dictionary where keys are the dataset ids and values its relative subject ids. If a key is store with 
                None as a value, then all the files from that dataset will be included
        Note1: To work, the function must be able to identify the dataset or subject IDs from the file name in order to check
            if it is in the list/dict. Custom extraction function can be given as arguments; however, if nothing is given, 
            the function will try to extract IDs considering that file names are in the format a_b_c_d.extension 
            (the output of the AUTO-BIDS library), where "a" is an integer with the dataset ID and "b" an integer 
            with the subject ID. If this fail, all files will be considered from the same datasets (id=0), and each file from a 
            different subject (id from 0 to N-1).
        Note2: if the input argument is not a list or a dict, it will be automatically converted to a list
        Default: None
    test_data_id: list or dict, optional 
        Same as exclude_data_id but for the test split
        Defaul: None
    val_data_id: list or dict, optional  
        Same as exclude_data_id but for validation split
        Default: None
    val_ratio_on_all_data: bool, optional
        Whether to calculate the validation split ratio only on the training set size (False) or on the entire considered
        dataset (True), i.e. the size of all files except for the ones excluded
        files not excluded  placed in the test set.
        Default: True
    stratified: bool, optional
        Whether to apply stratification to the split or not. Might be used for fine-tuning split (the only phase where labels
        are involved). Stratification will preserve, if possible, the label's ratio on the training/validation/test sets.
        Works only when each file has an unique label, which must be given in input.
        Default: False
    labels: list, array like, optional
        A list or array like objects with the label of each file listed in the partition table. 
        Must be given if stratification is set to True
        Indeces of labels must match row indeces in the partition table, i.e. label1 --> row1, label2 --> row2.
        Default: None
    dataset_id_extractor: function, optional
        A custom function to be used to extract the dataset ID from file the file name. It must accept only one argument,
        which is the file name
        Default: None
    subject_id_extractor: function, optional
        A custom function to be used to extract the subject ID from file the file name. It must accept only one argument,
        which is the file name
        Default: None
    split_tolerance: float, optional
        Argument for get_subarray_closest_sum function. Set the maximum accepted tolerance between the given split ratio
        and the one got with the obtained subset. Must be a number in [0,1]
        Default: 0.01
    perseverance: int, optional
        Argument for get_subarray_closest_sum function. Set the maximum number of tries before stop searching for a split 
        whose ratio is in the range [target_ratio - tolerance, target_ratio + tolerance]
        Default: 1000
    save : bool, optional
        Whether to save the resulted DataFrame as a .csv file or not
        Default: False
    save_path: str, optional
        A custom path to be used instead of the current working directory. It's the string given to the 
        pandas.DataFrame.to_csv() method.
        Default: None
                 
    Returns
    -------
    EEGSplit : DataFrame
        Two columns Pandas DataFrame. The first column has the EEG file name, 
        the second define the file usage with 0-1, Train-Test
            
    """ 
    
    # VARIOUS CHECKS ON INPUTS
    # check given ratios
    if test_ratio != None:
        if (test_ratio<0) or (test_ratio>=1):
            raise ValueError('test_ratio must be in [0,1)')
    if val_ratio != None:
        if (val_ratio<0) or (val_ratio>=1):
            raise ValueError('val_ratio must be in [0,1)')
    if (test_ratio != None) and (val_ratio != None):
        if val_ratio_on_all_data and ((val_ratio+test_ratio)>=1):
            raise ValueError('if val_ratio_on_all_data is set to true, val_ratio+test_ratio must be in [0,1) ')
     
    # check if given data ids are list or dict
    if exclude_data_id!=None:
        if not( isinstance(exclude_data_id,list) or isinstance(exclude_data_id,dict) ):
            exclude_data_id=[exclude_data_id]
    if test_data_id!=None:
        if not( isinstance(test_data_id,list) or isinstance(test_data_id,dict) ):
            test_data_id=[test_data_id]
    if val_data_id!=None:
        if not( isinstance(val_data_id,list) or isinstance(val_data_id,dict) ):
            val_data_id=[val_data_id]
        
    # align split modes to integer
    if isinstance(val_split_mode,str):
        val_split_mode = val_split_mode.lower()
    if isinstance(test_split_mode,str):
        test_split_mode = test_split_mode.lower()
    
    if val_split_mode in [1, 's', 'subj', 'subject']:
        val_split_mode= 1
    elif val_split_mode in [0, 'd', 'set', 'dataset']:
        val_split_mode= 0
    elif val_split_mode in [2, 'file', 'record']:
        val_split_mode= 2
    else:
        raise ValueError('validation split mode not supported')
    
    if test_split_mode in [1, 's', 'subj', 'subject']:
        test_split_mode= 1
    elif test_split_mode in [0, 'd', 'set', 'dataset']:
        test_split_mode= 0
    elif test_split_mode in [2, 'file', 'record']:
        test_split_mode= 2
    else:
        raise ValueError('test split mode not supported')

    # check if stratification must be applied 
    if stratified:
        if (test_ratio==None) and (val_ratio==None):
            print('STRATIFICATION can be applied only if at least one split ratio is given.')
        else:
            N_classes = np.unique(labels)
            classSplit = [None]*len(N_classes)
            # Call the split for each class 
            for i, n in enumerate(N_classes):
                #classIdx= np.where(labels==n)[0]
                classIdx = [ index_i for index_i, label_i in enumerate(labels) if label_i==n]
                subClassTable= partition_table.iloc[classIdx]
                classSplit[i] = GetEEGSplitTable(partition_table=subClassTable,  
                                                 test_ratio=test_ratio, 
                                                 val_ratio = val_ratio,
                                                 test_split_mode=test_split_mode,
                                                 val_split_mode=val_split_mode,
                                                 exclude_data_id= exclude_data_id,  
                                                 test_data_id =test_data_id,  
                                                 val_data_id = val_data_id,
                                                 val_ratio_on_all_data=val_ratio_on_all_data,
                                                 stratified= False,
                                                 labels = None,
                                                 dataset_id_extractor = dataset_id_extractor,
                                                 subject_id_extractor = subject_id_extractor,
                                                 split_tolerance=split_tolerance,
                                                 perseverance=perseverance,
                                                 save= False,
                                                )
            
            # merge subclass tables and check for mysterious duplicates
            EEGsplit= pd.concat(classSplit, axis=0, ignore_index=True)
            try:
                EEGsplit.drop(columns='index') #useless but to be sure
            except:
                pass
            EEGsplit= EEGsplit.drop_duplicates(ignore_index=True)
            EEGsplit = EEGsplit.sort_values(by='file_name')

    else:

        ex_id_list=isinstance(exclude_data_id,list)
        test_id_list=isinstance(test_data_id,list)
        val_id_list=isinstance(val_data_id,list)
        
        # COPY PARTITION TABLE AND ADD DATASET AND SUBJECT IDS COLUMNS
        if isinstance(partition_table, pd.DataFrame):
            partition2 = partition_table.copy()
            # NOTE: keep the list is faster for access to the data compared to iloc
            # extract dataset id 
            if dataset_id_extractor !=None:
                dataset_ID = [ dataset_id_extractor(x) for x in partition2['file_name'] ]
            else:
                try:
                    dataset_ID = [ int(x.split('_')[0]) for x in partition2['file_name'] ]
                except:
                    dataset_ID = [ 0 for _ in range(len(partition2['file_name'])) ]
            partition2['dataset_ID']=dataset_ID
            # extract subject id 
            if subject_id_extractor !=None:
                subj_ID = [ subject_id_extractor(x) for x in partition2['file_name'] ]
            else:
                try:
                    subj_ID = [ int(x.split('_')[1]) for x in partition2['file_name'] ]
                except:
                    subj_ID = [ x for x in range(len(partition2['file_name'])) ]
            partition2['subj_ID']=subj_ID
            EEGfiles= partition_table['file_name'].values.tolist()
        
        # it's faster to update a list than a table
        EEGsplit=[ [filename, 0] for filename in EEGfiles]
        
        # PRE SPLIT:  DATASET  -->  DATASET WITH ONLY CONSIDERED DATA
        if exclude_data_id!=None:
            for ii in range(len(EEGfiles)):
                DatasetID = dataset_ID[ii]
                if ex_id_list:
                    if DatasetID in exclude_data_id:
                        EEGsplit[ii][1]=-1
                else:
                    SubjID = subj_ID[ii]
                    if DatasetID in exclude_data_id.keys():
                        if (exclude_data_id[DatasetID] is None) or (SubjID in exclude_data_id[DatasetID]):
                            EEGsplit[ii][1]=-1
        
        # calculate the sum of all remaining samples after data exclusion
        # it will be used in train test split (when test ratio is given)
        # or in train validation split (if validation_on_all_data is set to true)
        idx_val= [i for i in range(len(EEGsplit)) if EEGsplit[i][1]!=-1]
        arr=partition2.iloc[idx_val]['N_samples']
        alldatasum=sum(arr)
        
        # FIRST SPLIT:  DATASET  -->  TRAIN/TEST
        if test_data_id!=None: 
            # if test_data_id is given, ignore test ratio and use given IDs
            for ii in range(len(EEGfiles)):
                if EEGsplit[ii][1] != -1:
                    DatasetID= dataset_ID[ii]
                    if test_id_list:
                        if DatasetID in test_data_id:
                            EEGsplit[ii][1]=2
                    else:
                        SubjID=subj_ID[ii]
                        if DatasetID in test_data_id.keys():
                            if (test_data_id[DatasetID] is None) or (SubjID in test_data_id[DatasetID]):
                                EEGsplit[ii][1]=2
        elif test_ratio>0:
            # split data according to test ratio and test_split_on_subj
            # group data according to test_split_mode
            partition3 = partition2.iloc[idx_val]
            if test_split_mode==1:
                group1 = partition3.groupby( ['dataset_ID','subj_ID'] )['N_samples'].sum().reset_index(name='N_samples')
            elif test_split_mode==0:
                group1 = partition3.groupby( ['dataset_ID'] )['N_samples'].sum().reset_index(name='N_samples')
            else:
                group1 = partition3

            # get split subarray
            arr=group1['N_samples'].values.tolist()
            target=test_ratio*alldatasum
            final_idx, subarray = get_subarray_closest_sum(arr, target, split_tolerance, perseverance)
            final_idx.sort()

            # update split list according to returned subarray
            if test_split_mode==2:
                fileName=group1.iloc[final_idx]['file_name'].values.tolist()
                cntmax = len(fileName)
                cnt=0
                for ii in range(len(EEGfiles)):
                    if cnt==cntmax:
                        break
                    if EEGsplit[ii][0]==fileName[cnt]:
                        cnt +=1
                        EEGsplit[ii][1] = 2
            else: 
                data_test_ID = set(group1['dataset_ID'].iloc[final_idx].values.tolist())
                if test_split_mode==1:
                    subj_test_ID = {key: [] for key in data_test_ID}
                    for i in final_idx:
                        subj_test_ID[group1['dataset_ID'].iloc[i]].append( group1['subj_ID'].iloc[i] )
                        
                for ii in range(len(EEGfiles)):
                    if EEGsplit[ii][1] != -1:
                        DatasetID= dataset_ID[ii]
                        if DatasetID in data_test_ID: 
                            if test_split_mode==1:
                                subjID=subj_ID[ii]
                                if subjID in subj_test_ID[DatasetID]:
                                    EEGsplit[ii][1]=2
                                else:
                                    EEGsplit[ii][1]=0
                            else:
                                EEGsplit[ii][1]=2
                        else:
                            EEGsplit[ii][1]=0
    
                        
        # SECOND SPLIT:  TRAIN  -->  TRAIN/VALIDATION
        if val_data_id!=None:
            for ii in range(len(EEGfiles)):
                if EEGsplit[ii][1] == 0:
                    DatasetID= dataset_ID[ii]
                    if val_id_list:
                        if DatasetID in val_data_id:
                            EEGsplit[ii][1]=1
                    else:
                        SubjID=subj_ID[ii]
                        if DatasetID in val_data_id.keys():
                            if (val_data_id[DatasetID] is None) or (SubjID in val_data_id[DatasetID]):
                                EEGsplit[ii][1]=1
        elif val_ratio>0:
            # split data according to test ratio and test_split_on_subj
            idx_val= [i for i in range(len(EEGsplit)) if EEGsplit[i][1]==0]
            partition3 = partition2.iloc[idx_val]
            if val_split_mode == 1:
                group2 = partition3.groupby( ['dataset_ID','subj_ID'] )['N_samples'].sum().reset_index(name='N_samples')
            elif val_split_mode == 0:
                group2 = partition3.groupby( ['dataset_ID'] )['N_samples'].sum().reset_index(name='N_samples')
            else:
                group2 = partition3
            
            arr=group2['N_samples'].values.tolist()
            target=val_ratio*alldatasum if val_ratio_on_all_data else val_ratio*sum(arr)
            final_idx, subarray = get_subarray_closest_sum(arr, target, split_tolerance, perseverance)
            final_idx.sort()
            
            if val_split_mode==2:
                fileName=group2.iloc[final_idx]['file_name'].values.tolist()
                cntmax = len(fileName)
                cnt=0
                for ii in range(len(EEGfiles)):
                    if cnt==cntmax:
                        break
                    if EEGsplit[ii][0]==fileName[cnt]:
                        cnt +=1
                        EEGsplit[ii][1] = 1
            else:
                data_val_ID = set(group2['dataset_ID'].iloc[final_idx].values.tolist())
                if val_split_mode == 1:
                    subj_val_ID = {key: [] for key in data_val_ID}
                    for i in final_idx:
                        subj_val_ID[group2['dataset_ID'].iloc[i]].append( group2['subj_ID'].iloc[i] )
    
                for ii in range(len(EEGfiles)):
                    if EEGsplit[ii][1] == 0:
                        DatasetID= dataset_ID[ii]
                        if DatasetID in data_val_ID: 
                            if val_split_mode==1:
                                subjID= subj_ID[ii]
                                if subjID in subj_val_ID[DatasetID]:
                                    EEGsplit[ii][1]=1
                                else:
                                    EEGsplit[ii][1]=0
                            else:
                                EEGsplit[ii][1]=1
                        else:
                            EEGsplit[ii][1]=0
            
        EEGsplit=pd.DataFrame(EEGsplit,columns=['file_name','split_set'])
    
    try:
        if save:
            if save_path is not None:
                EEGsplit.to_csv(save_path)
            else:
                condition=True
                cnt=-1
                while condition:
                    cnt +=1
                    if cnt==0:
                        filename='EEGTrainTestSplit.csv'
                        condition=os.path.isfile(filename)
                    else:
                        filename='EEGTrainTestSplit_'+str(cnt)+'.csv'
                        condition=os.path.isfile(filename)
                EEGsplit.to_csv(filename)
    except:
        print('failed to save file. Function output will be returned but not saved.')      
     
    return EEGsplit


def GetEEGSplitTableKfold(partition_table: pd.DataFrame,
                          kfold: int = 10,
                          test_ratio: float= None,
                          test_split_mode: str or int =2,
                          val_split_mode: str or int= 2,
                          exclude_data_id: list or dict =None,
                          test_data_id: list or dict=None,
                          stratified: bool=False,
                          labels: 'array like'=None,
                          dataset_id_extractor: 'function'=None,
                          subject_id_extractor: 'function'=None,
                          split_tolerance=0.01,
                          perseverance=1000,
                          save: bool=False,
                          save_path: str=None
                         ):
    """   
    GetEEGSplitTableKfold create a table with multiple splits for cross-validation.
    
    Return a Pandas DataFrame defining multiple training-validation for cross validation applications.
    Test split, if calculated, is kept equal in every CV split.
    
    Split is done in the following way:
    Dataset --> Train / Test (optional)
                Train --> Train / Validation  * Fold Number
    Test split is optional and can be done with the same modalities described in GetEEGSplitTable function, i.e.
    by giving specific ID or by giving a split ratio. 
    CV's train/validation split can't be done in this way, since this does not guarantee the preservation of 
    the split ratio.
    
    Parameters
    ----------
    partition_table: pd.Dataframe, optional
        A two columns dataframe where:
            1-the first column has name 'file_name' and contain all the file names
            2-the second column has name 'N_samples' and has the number of samples which can be extracted from the file
        This table can be automatically created with a custom setting with the provided function GetEEGPartitionNumber()
        Default: None
    Kfold: int, optional
        The number of folds to extract. Must be a number higher or equal than 2.
        Default: 10
    test_ratio: float, optional
        The percentage of data with respect to the whole number of samples (partitions) of the dataset to be included 
        in the test set. Must be a number in [0,1]. 0 means that the test split is skipped if test_data_id isn't given
        Default: None
    test_split_mode: int or str, optional
        The type of split to perform in the step not_excluded_data --> train / test sets. It can be one of the following:
            1) any of [0, 'd', 'set', 'dataset'] = split will be performed using dataset ids, i.e. all files of the same
                dataset will be put in the same split set
            2) any of [1, 's', 'subj', 'subject'] = split will be performed using subjects ids, i.e. all files of the same
                subjects will be put in the same split set
            3) any of [2, 'file', 'record'] = split will be performed looking at single files
        Default: 2
    val_split_mode: int or str, optional
        The type of split to perform in the step train --> train / validation sets. Input allowed are the same as in
        test_split_mode.
        Default: 2
    exclude_data_id  : list or dict, optional 
        Dataset ID to be excluded. It can be:
            1) a list with all dataset ids to exclude
            2) a dictionary where keys are the dataset ids and values its relative subject ids. If a key is store with 
                None as a value, then all the files from that dataset will be included
        Note1: To work, the function must be able to identify the dataset or subject IDs from the file name in order to check
            if it is in the list/dict. Custom extraction function can be given as arguments; however, if nothing is given, 
            the function will try to extract IDs considering that file names are in the format a_b_c_d.extension 
            (the output of the AUTO-BIDS library), where "a" is an integer with the dataset ID and "b" an integer 
            with the subject ID. If this fail, all files will be considered from the same datasets (id=0), and each file from a 
            different subject (id from 0 to N-1).
        Note2: if the input argument is not a list or a dict, it will be automatically converted to a list
        Default: None
    test_data_id: list or dict, optional 
        Same as exclude_data_id but for the test split
        Defaul: None
    stratified: bool, optional
        Whether to apply stratification to the split or not. Might be used for fine-tuning split (the only phase where labels
        are involved). Stratification will preserve, if possible, the label's ratio on the training/validation/test sets.
        Works only when each file has an unique label, which must be given in input.
        Default: False
    labels: list, array like, optional
        A list or array like objects with the label of each file listed in the partition table. 
        Must be given if stratification is set to True
        Indeces of labels must match row indeces in the partition table, i.e. label1 --> row1, label2 --> row2.
        Default: None
    dataset_id_extractor: function, optional
        A custom function to be used to extract the dataset ID from file the file name. It must accept only one argument,
        which is the file name
        Default: None
    subject_id_extractor: function, optional
        A custom function to be used to extract the subject ID from file the file name. It must accept only one argument,
        which is the file name
        Default: None
    split_tolerance: float, optional
        Argument for get_subarray_closest_sum function. Set the maximum accepted tolerance between the given split ratio
        and the one got with the obtained subset. Must be a number in [0,1]
        Default: 0.01
    perseverance: int, optional
        Argument for get_subarray_closest_sum function. Set the maximum number of tries before stop searching for a split 
        whose ratio is in the range [target_ratio - tolerance, target_ratio + tolerance]
        Default: 1000
    save : bool, optional
        Whether to save the resulted DataFrame as a .csv file or not
        Default: False
    save_path: str, optional
        A custom path to be used instead of the current working directory. It's the string given to the 
        pandas.DataFrame.to_csv() method.
        Default: None
                 
    Returns
    -------
    EEGSplit : DataFrame
        Two columns Pandas DataFrame. The first column has the EEG file name, 
        the second define the file usage with 0-1, Train-Test
            
    """ 
    if kfold<2:
        raise ValueError('kfold must be greater than or equal to 2. '
                         'If you don\'t need multiple splits use the GetEEGSplitTable function'
                        )
    kfold=int(kfold)
    if (test_ratio is None) and (test_data_id is None):
        test_ratio = 0.
        
    # FIRST STEP: Create test set or exclude data if necessary
    # the result of this function call will be an initialization of the split table
    # if no data need to be excluded or placed in a test set, the split_set column will 
    # simply have all zeros.
    EEGsplit = GetEEGSplitTable(partition_table=partition_table,
                                test_ratio= test_ratio,
                                test_split_mode=test_split_mode,
                                val_split_mode= val_split_mode,
                                exclude_data_id=exclude_data_id,
                                test_data_id=test_data_id,
                                val_data_id=[],
                                stratified= stratified,
                                labels = labels,
                                dataset_id_extractor = dataset_id_extractor,
                                subject_id_extractor = subject_id_extractor,
                                split_tolerance=split_tolerance,
                                perseverance=perseverance
                               )
    
    # Find index of elements in train set
    EEGsplit = EEGsplit.assign(**{x: EEGsplit.iloc[:, 1] for x in ['split_'+str(i+1) for i in range(kfold)]})
    idxSplit= EEGsplit.index[(EEGsplit['split_set'] != 0)]
    idxAll= np.arange(EEGsplit.shape[0])
    idx2assign = np.setdiff1d(idxAll, idxSplit)
    for i in range(kfold-1):
        EEGsplit.iloc[idx2assign,i+2]= GetEEGSplitTable(partition_table=partition_table.iloc[idx2assign],
                                                      val_ratio= 1/(kfold-i),
                                                      val_split_mode= val_split_mode,
                                                      exclude_data_id=[],
                                                      test_data_id=[],
                                                      stratified= stratified,
                                                      labels = labels[idx2assign] if labels is not None else labels,
                                                      dataset_id_extractor = dataset_id_extractor,
                                                      subject_id_extractor = subject_id_extractor,
                                                      split_tolerance=split_tolerance,
                                                      perseverance=perseverance
                                                    )['split_set']
        idxSplit= EEGsplit.index[(EEGsplit['split_'+str(i+1)] ==1)]
        idx2assign = np.setdiff1d(idx2assign, idxSplit)

    # assign last fold and delete useless initial split column
    EEGsplit.iloc[idx2assign,-1]=1
    EEGsplit.drop(columns='split_set', inplace=True)

    try:
        if save:
            if save_path is not None:
                EEGsplit.to_csv(save_path)
            else:
                condition=True
                cnt=-1
                while condition:
                    cnt +=1
                    if cnt==0:
                        filename='EEGTrainTestSplitKfold.csv'
                        condition=os.path.isfile(filename)
                    else:
                        filename='EEGTrainTestSplitKfold_'+str(cnt)+'.csv'
                        condition=os.path.isfile(filename)
                EEGsplit.to_csv(filename)
    except:
        print('failed to save file. Function output will be returned but not saved.')  
    
    return EEGsplit



class EEGDataset(Dataset):
    """
    Parameters
    ----------
    EEGlen : DataFrame
        DataFrame with the number of partition per EEG record. Must be the output of GetEEGPartitionNumber()
    EEGsplit : DataFrame 
        DataFrame with the train/test split info. Must be the output of GetEEGSplitTable()
    EEGpartition_spec : list
        3-element list with the input gave to GetEEGPartitionNumber() in 
        [sampling_rate, window_length, overlap_percentage] format.
    mode: string, optional
        if the dataset is intended for train, test or validation. It accept only 'train','test','validation'
        strings.
        Default: 'train'
    supervised: bool, optional
        Whether the getItem method must return a label or not
        Default: False
    load_function : 'function', optional
        A pointer to a custom EEG file loading function. It will be used instead of the default: 
        loadmat(ii, simplify_cells=True)['DATA_STRUCT']['data']
        The function:
            1) must take only one required argument, which is the full path to the EEG file 
            2) can output one or two arguments where the first must be the EEG file and the second its (if there is one)
               label
        (e.g. the function will be called in this way: load_function(fullpath, optional_arguments) )
        Note: the assumed number of output is based on the parameter label_on_load. So if the function will return only the 
              EEG remember to set label_on_load on False
        Default: None
    transform_function : 'function', optional
        A pointer to a custom transformation to be applied after the EEG is loaded. Might be useful if there are portion of 
        the signal to cut. 
        The function must take only one required argument, which is the EEG file to transform
        (e.g. the function will be called in this way: transform_function(EEG, optional_arguments) )
        Default: None
    label_function : 'function', optional
        A pointer to a custom function for the label extraction. Might be useful for the fine-tuning phase. 
        Considering that an EEG file can have single or multiple labels the function will be called with
        2 required arguments:
            1) full path to the EEG file 
            2) list with all indeces necessary to identify the extracted partition (if EEG is a 2-D array 
                the list will have only the starting and ending indeces of the slice of the last axis, if the
                EEG is N-D the list will also add all the other indeces from the first to the second to last axis)
        e.g. the function will be called in this way: 
                      label_function(full_path, [*first_axis_idx, start, end], optional arguments)
        NOTE: it is strongly suggested to save EEG labels in a separate file in order to avoid loading every time
              the entire EEG file which is the purpose of this entire module implementation
        Default: None
    optional_load_fun_args: list or dict, optional
        Optional arguments to give to the custom loading function. Can be a list or a dict.
        Default: None
    optional_transform_fun_args: list or dict, optional
        Optional arguments to give to the EEG transformation function. Can be a list or a dict.
        Default: None
    optional_label_fun_args: list or dict, optional
        Optional arguments to give to the EEG transformation function. Can be a list or a dict.
        Default: None
    label_on_load: bool, optional
        Whether the custom loading function will also load a label associated to the eeg file 
        Default: True
    label_key: str or list of str, optional
        A single or set of dictionary keys given as list of strings to use to get access to the label. Might be useful if
        the loading function will return a dictionary of labels associated to the file, for example when you have a set of 
        patient info but you want to use only a specific one (as in the AUTO-BIDS library)
        Default: None
            
    """ 
    def __init__(self, 
                 EEGlen: pd.DataFrame, 
                 EEGsplit: pd.DataFrame, 
                 EEGpartition_spec: list,
                 mode: str='train',
                 supervised: bool=False,
                 load_function: 'function'=None,
                 transform_function: 'function'=None,
                 label_function: "function"=None,
                 optional_load_fun_args: list or dict=None,
                 optional_transform_fun_args: list or dict=None,
                 optional_label_fun_args: list or dict=None,
                 label_on_load: bool=False,
                 label_key: list=None,
                 default_dtype=torch.float32
                ):               
        # Instantiate parent class
        super().__init__()
        
        # Store all Input arguments
        self.default_dtype = default_dtype
        self.EEGsplit = EEGsplit
        self.EEGlen = EEGlen
        self.mode=mode
        self.supervised = supervised    
        
        self.load_function = load_function
        self.optional_load_fun_args = optional_load_fun_args
        self.transform_function = transform_function
        self.optional_transform_fun_args = optional_transform_fun_args
        self.label_function = label_function
        self.optional_label_fun_args = optional_label_fun_args
        
        self.label_on_load = label_on_load
        self.given_label_keys= None
        self.curr_key = None
        if label_key is not None:
            self.given_label_keys = label_key if isinstance(label_key,list) else [label_key]
            self.curr_key = self.given_label_keys[0] if len(self.given_label_keys)==1 else None
        
        # Check if the dataset is for train test or validation
        # and extract relative file names
        if mode.lower()=='train':
            FileNames = EEGsplit.loc[EEGsplit['split_set']==0,'file_name'].values
        elif mode.lower()=='validation':
            FileNames = EEGsplit.loc[EEGsplit['split_set']==1,'file_name'].values
        else:
            FileNames = EEGsplit.loc[EEGsplit['split_set']==2,'file_name'].values
        
        # initialize attributes for __len__ and __getItem__
        self.EEGlenTrain = EEGlen.loc[EEGlen['file_name'].isin(FileNames)].reset_index()
        self.EEGlenTrain = self.EEGlenTrain.drop(columns='index')
        self.DatasetSize = self.EEGlenTrain['N_samples'].sum()
        
        # initialize other attributes for __getItem__
        self.freq = EEGpartition_spec[0]
        self.window = EEGpartition_spec[1]
        self.overlap = EEGpartition_spec[2]
        self.Nsample= EEGpartition_spec[0]*EEGpartition_spec[1]
        self.EEGcumlen = np.cumsum(self.EEGlenTrain['N_samples'].values)
        
        
        # Set Current EEG loaded attributes (speed up getItem method)
        # Keep in mind that multiple workers use copy of the dataset
        # saving a copy of the current loaded EEG file can use lots of memory if EEGs are pretty large
        self.currEEG = None
        self.dimEEG = 0
        self.dimEEGprod = None
        self.file_path=None
        self.minIdx = -1
        self.maxIdx = -1
        self.label_info=None
        self.label_info_keys=None
        
    
    def __len__(self):
        return self.DatasetSize
    
    def __getitem__(self,index):

        # Check if a new EEG file must be loaded. If so, a new EEG file is loaded,
        # transformed (if necessary) and all loading attributes are updated according to the new file
        if ((index<self.minIdx) or (index>self.maxIdx)):
            # Get full path to new file to load 
            nameIdx=np.searchsorted(self.EEGcumlen, index, side='right')
            self.file_path= self.EEGlenTrain.iloc[nameIdx].full_path

            # load file according to given setting (custom load or not)
            if self.load_function is not None:
                if isinstance(self.optional_load_fun_args, list):
                    EEG = self.load_function(self.file_path, *self.optional_load_fun_args)
                elif isinstance(self.optional_load_fun_args, dict):
                    EEG = self.load_function(self.file_path, **self.optional_load_fun_args)
                else:
                    EEG = self.load_function(self.file_path)
                if (self.label_on_load):
                    self.currEEG=EEG[0]
                    if self.supervised:
                        self.label_info = EEG[1]
                        if self.given_label_keys is not None:
                            self.label_info_keys=self.label_info.keys()
                            if (self.given_label_keys is not None) and (len(self.given_label_keys)>1):
                                self.curr_key = list(set(self.label_info_keys).intersection(self.given_label_keys))[0]
                            self.label = self.label_info[self.curr_key]
                        else:
                            self.label = EEG[1]
                else:
                    self.currEEG = EEG
            else:
                # load things considering files coming from the auto-BIDS library
                EEG=loadmat(self.file_path, simplify_cells=True)
                self.currEEG=EEG['DATA_STRUCT']['data']
                if (self.supervised) and (self.label_on_load):
                    self.label_info=EEG['DATA_STRUCT']['subj_info']
                    self.label_info_keys=self.label_info.keys()
                    if (self.given_label_keys is not None) and (len(self.given_label_keys)>1):
                        self.curr_key = list(set(self.label_info_keys).intersection(self.given_label_keys))[0]
                    self.label = self.label_info[self.curr_key]

            # transform data if transformation function is given
            if self.transform_function is not None:
                if isinstance(self.optional_transform_fun_args, list):
                    self.currEEG = self.transform_function(self.currEEG, *self.optional_transform_fun_args)
                elif isinstance(self.optional_transform_fun_args, dict):
                    self.currEEG = self.transform_function(self.currEEG, **self.optional_transform_fun_args)
                else:
                    self.currEEG = self.transform_function(self.currEEG)

            # convert loaded eeg to torch tensor of specific dtype
            if isinstance(self.currEEG, np.ndarray):
                self.currEEG=torch.from_numpy(self.currEEG)
            if self.currEEG.dtype != self.default_dtype:
                self.currEEG= self.currEEG.to(dtype=self.default_dtype)

            # store dimensionality of EEG files (some datasets stored 3D tensors)
            # This might be helpful for partition selection of multiple EEG in a single file
            self.dimEEG = len(self.currEEG.shape)
            if self.dimEEG>2:
                self.dimEEGprod = (self.EEGlenTrain.iloc[nameIdx].N_samples)/np.cumprod(self.currEEG.shape[:-2])
                self.dimEEGprod = self.dimEEGprod.astype(int)

            # change minimum and maximum index according to new loaded file
            self.minIdx=0 if nameIdx==0 else self.EEGcumlen[nameIdx-1]
            self.maxIdx=self.EEGcumlen[nameIdx]-1

        
        # Calculate start and end of the partition
        # Manage the multidimensional EEG 
        # NOTE: using the if add lines but avoid making useless operation in case of 2D tensors
        partition=index-self.minIdx
        first_dims_idx=[0]*(self.dimEEG-2)
        if self.dimEEG>2:
            cumidx=0
            for i in range(self.dimEEG-2):
                first_dims_idx[i]=(partition-cumidx)//self.dimEEGprod[i]
                cumidx += first_dims_idx[i]*self.dimEEGprod[i]
            start=(self.Nsample-round(self.Nsample*self.overlap))*(partition - cumidx)
            end=start+self.Nsample
            if end>self.currEEG.shape[-1]: # in case of partial ending samples
                sample=self.currEEG[*first_dims_idx,:,-self.Nsample:]
            else:
                sample=self.currEEG[*first_dims_idx,:,start:end]    
        else:
            start=(self.Nsample-round(self.Nsample*self.overlap))*(partition)
            end=start+self.Nsample
            if end>self.currEEG.shape[-1]: # in case of partial ending samples
                sample=self.currEEG[...,-self.Nsample:]
            else:
                sample=self.currEEG[...,start:end]  

        # extract label if training is supervised (fine-tuning purposes)
        if self.supervised:
            if self.label_on_load:
                label = self.label
            else:
                if isinstance(self.optional_label_fun_args, list):
                    label = self.label_function(self.file_path, [*first_dims_idx, start, end], *self.optional_label_fun_args)
                elif isinstance(self.optional_label_fun_args, dict):
                    label = self.label_function(self.file_path, [*first_dims_idx, start, end], **self.optional_label_fun_args)
                else:
                    label = self.label_function(self.file_path, [*first_dims_idx, start, end])
            return sample, label
        else: 
            return sample


class EEGsampler(Sampler):
    """
    data_source: EEGDataset
        The instance of the EEGdataset provided in this module
    BatchSize: int, optional
        The size of the batch size used during training. It will be used to create the custom iterator (not linear)
        Default: 1
    Workers: Int, optional
        The number of workers used by the dataloader. It will be used to create the custom iterator (not linear)
    Mode: int, optional
        The mode to be used to create the dataloader. it can be 0 or 1, where:
            1) 0 = the iterator is a simple linear
            2) 1 = the indeces are first shuffled at the inter-file level, then at the intra-file level; ultimately
                   all indeces are rearranged based on the batch size and the number of workers in order to reduce
                   the number of times a new EEG is loaded. The iterator can be seen as a good compromise between 
                   batch heterogeneity and batch creation speed
    """
    
    
    def __init__(self, 
                 data_source: Dataset, 
                 BatchSize: int=1, 
                 Workers: int=0,
                 Mode: int=1,
                 Keep_only_ratio: float=1
                ):
        self.data_source = data_source
        self.SubjectSamples = np.insert(data_source.EEGcumlen, 0, 0)
        self.Nsubject=len(self.SubjectSamples)
        self.BatchSize=BatchSize
        self.Workers=Workers if Workers>0 else 1
        if Mode not in [0,1]:
            raise ValueError('supported modes are 0 (linear sampler) and 1 (custom randomization)')
        else:
            self.Mode= Mode
        if Keep_only_ratio>1 or Keep_only_ratio<=0:
            raise ValueError('Keep_only_ratio must be in (0,1]')
        else:
            self.Keep_only_ratio = Keep_only_ratio
        self.shrink_data=True if Keep_only_ratio<1 else False
        
    
    
    def __len__(self):
        return len(self.data_source)
    
    def __iter__(self):
        """
        Return an iterator where subject are passed sequentially for each worker but the samples of each subjects are shuffled
        """
        iterator=[]
        Nseed=random.randint(0,9999999)

        if self.Mode==0:
            return iter(range(len(self.data_source)))
        
        # -------------------------------------------------------------
        # 1st - create a list of shuffled subjects
        # -------------------------------------------------------------
        SubjList=[i for i in range(self.Nsubject-1)]
        random.seed(Nseed)
        random.shuffle(SubjList)
        
        # -------------------------------------------------------------
        # 2nd - shuffle partitions of the same subject for each subject 
        # -------------------------------------------------------------
        for ii in SubjList:
            random.seed(Nseed)
            idx = list(range(self.SubjectSamples[ii],self.SubjectSamples[ii+1]))
            random.shuffle(idx)
            if self.shrink_data:
                iterator += idx[0: int(len(idx)*self.Keep_only_ratio)]
            else:
                iterator += idx
         
        # ------------------------------------------------------------
        # 3rd - Arrange index According to batch and number of workers
        # ------------------------------------------------------------
        batch=self.BatchSize
        worker=self.Workers
        Ntot=len(iterator)

        Nbatch=math.ceil(Ntot/batch)
        Nrow, Ncol = batch*math.ceil(Nbatch/worker) , worker
        Npad= Nrow*Ncol-Ntot

        # Matrix Initialization
        b=np.zeros((Nrow,Ncol),order='C', dtype=int)

        # Assign index to first block of the matrix (Rows until the last batch)
        b[0:-batch,:].flat=iterator[:((Nrow-batch)*Ncol)]

        # Assign -1 to the bottom left part of the matrix
        block2=(Ncol-int(Npad/batch))
        b[-batch:,block2:]=-1

        # Assign the remaining -1
        block3=(Npad-(Ncol-block2)*batch)
        if block3!=0:
            b[Nrow-block3:,block2-1]=-1


        # Complete index matrix with the remaining index to insert
        iterator=iterator[((Nrow-batch)*Ncol):]
        for i in range(batch):
            if len(iterator)==0:
                break
            Nel=(b.shape[1]-np.count_nonzero(b[-batch+i]))
            b[-batch+i,:Nel] = iterator[:Nel]
            iterator = iterator[Nel:]
            
        # Convert matrix to list by scrolling elements according to batchsize and workers
        c=[None]*(Nrow*Ncol)
        cnt=0
        Rstart=-batch
        Rend=0
        for ii in range(int(Nrow/batch)):
            Rstart += batch
            Rend += batch
            for jj in range(Ncol):   
                c[cnt:(cnt+batch)]=b[Rstart:Rend, jj].tolist()
                cnt += batch
        
        # Remove -1 if there are 
        if Npad==0:
            iterator=c
        else:
            iterator=c[:-Npad]
        return iter(iterator)


