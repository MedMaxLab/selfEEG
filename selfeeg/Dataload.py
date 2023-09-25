import os
import glob
import time
import math
import random
import copy
import sys
from typing import Union, Sequence
import numpy as np
import pandas as pd
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, get_worker_info

__all__ = ['GetEEGPartitionNumber', 'get_subarray_closest_sum', 'GetEEGSplitTable',
           'EEGDataset', 'EEGsampler']


def GetEEGPartitionNumber(EEGpath: str, 
                          freq: int or float=250, 
                          window: int or float=2, 
                          overlap: int or float=0.10,
                          includePartial: bool=True, 
                          save: bool=False,
                          verbose: bool=False
                         ) -> pd.DataFrame :
    """
    GetEEGPartitionNumber(EEGpath, freq, window, overlap=0)
    Find the number of unique partitions from EEG signals stored with the AZ's library.
    
    Return a Pandas DataFrame with the exact number of samples which can be extracted from each 
    EEG file in the EEGpath directory
    
    Parameters
    ----------
    EEGpath : string 
        Directory with all EEG files in .mat extension and file names in the format 'a_b_c_d', where
        a is the dataset ID, b the subject ID, c the session ID, d the ith file of the session. To get
        files preprocessed in this way you can use the matlab library provided in the same repository
        as the one of this framework, which is able to preprocess different datasets using EEGlab
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
    save : bool, optional
        Save the resulted DataFrame as a .csv file 
                
    Returns
    -------
    lenEEG : DataFrame
        Two columns Pandas DataFrame. 
        The first column has the EEG file name, the second its number of partitions.
    
    NOTE: freq*window must give an integer with the number of samples
    
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
        
    # Extract all .mat files from directory
    if EEGpath[-1]=='/':
        EEGfiles=glob.glob(EEGpath + '*_*_*_*.mat')
    else:
        EEGfiles=glob.glob(EEGpath + '/*_*_*_*.mat')
    
    if len(EEGfiles)==0:
        print('didn\'t found any .mat files')
        return None
    
    EEGfiles=sorted(EEGfiles)
    NumFiles= len(EEGfiles)

    # Create Table
    EEGlen=[]
    WindSample=freq*window
    overlapInt=round(WindSample*overlap)

    if verbose:
        print('Extracting Number of Samples per file')
    for i, ii in enumerate(EEGfiles):
        
        if verbose:
            # PROGRESS BAR PRINT
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" % ('='*int(20*i/NumFiles), 100*i/NumFiles))
            sys.stdout.flush()
        # load file
        EEG=loadmat(ii, simplify_cells=True)['DATA_STRUCT']['data']
        
        if overlap==0:
            N_Partial=EEG.shape[1]/(WindSample)
        else:
            L=EEG.shape[1]
            N=(L-overlapInt)//(WindSample-overlapInt)
            #R=L-WindSample*N+overlapInt*(N-1)
            #N_Partial=N+(R+overlapInt)/WindSample
            R=(overlapInt-WindSample)*N
            N_Partial=N+(L+R)/WindSample
        
        if includePartial:
            N_EEG=round(N_Partial)
        else:
            N_EEG=N
        EEGlen.append([ii.split('/')[-1],N_EEG])
    
    del EEG
    EEGlen=pd.DataFrame(EEGlen,columns=['file_name','N_samples'])
    
    if save:
        EEGlen.to_csv('EEGPartitionNumber.csv')
    
    print('Concluded')
    return EEGlen


def subarray_closest_sum(arr, n, k):
    """
    subarrat_closest_sum return a subarray whose element sum is closest to k.
    
    This function is taken from geeksforgeeks at the following link:
    https://www.geeksforgeeks.org/subarray-whose-sum-is-closest-to-k/
    
    It is important to note that this function return a subarray and not a subset of the array.
    A subset is a collection of elements in the array taken from any index, a subarray here is 
    a slice of the array (arr[start:end]). If you are looking for a subset with closest sum, which is more
    accurate but more computationally and memory demanding, search for another function.
    
    Arguments
    ---------
    arr: list
        the array to search
    n: int
        the length of the array
    k: float
        the target value
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
                            ):
    """
    get_subarray_closest_sum find the subarray of arr whose values sum is 
    closer to a target up to a specified tolerance (if possible) and return the index of the 
    selected values in the original array.
    
    To find the subarray, get_subarray_closest_sum calls multiple times subarray_closest_sum, 
    until the subarray has the sum within [target*(1-tolerance), target*(1+tolerance).
    At each try the array is shuffled in order to get a different solution. Keep in mind that 
    the solution is not always the optimal, but is the first which satisfy the requirements 
    given.
    
     Arguments
    ---------
    arr: list
        the array to search
    target: float
        the target sum
    tolerance: float, optional
        the tolerance to apply to the sum in percentage. It must be a value between 0 and 1.
        Default: 0.01
    return_subarray: bool, optional
        whether to also return the subarray or not
        Default: True
    perseverance: int, optional
        The maximum number of tries before stopping searching the subarray with closest sum.
        Default: 1000
    
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


def check_input_split_table(EEGpath, test_ratio, val_ratio, exclude_data_id, test_data_id, val_data_id,
                            val_ratio_on_all_data, partition_table, dataset_info
                           ):
    """
    function called by GetEEGSplitTable to check if inputs given are ok
    """
    # Check EEG path
    if EEGpath==None and not(isinstance(partition_table, pd.DataFrame)):
        msgErr= 'Cannot get file names. EEGpath or partition_table with file name on first column must be given'
        raise ValueError(msgErr)
    
    if EEGpath != None:
        if not(isinstance(EEGpath,str)):
            raise TypeError("Argument EEGpath must be a string")
    
    # Check test val ratio
    if test_ratio != None:
        if (test_ratio<0) or (test_ratio>=1):
            raise ValueError('test_ratio must be in [0,1)')
    if val_ratio != None:
        if (val_ratio<0) or (val_ratio>=1):
            raise ValueError('val_ratio must be in [0,1)')
    if (test_ratio != None) and (val_ratio != None):
        if val_ratio_on_all_data and ((val_ratio+test_ratio)>=1):
            raise ValueError('if val_ratio_on_all_data is set to true, val_ratio+test_ratio must be in [0,1) ')
     
    # check on exclude_data_id
    if exclude_data_id!=None:
        if not( isinstance(exclude_data_id,list) or isinstance(exclude_data_id,dict) ):
            exclude_data_id=[exclude_data_id]
        if isinstance(exclude_data_id,list):
            if (all([isinstance(x,str) for x in exclude_data_id])):
                if not(isinstance(partition_table, pd.DataFrame)):
                    msgErr= 'if given as a list of str, a pandas table with first column the dataset ID '
                    msgErr += 'and second column the dataset name must be given'
                    raise ValueError(msgErr)
                else:
                    exclude_data_id = dataset_info.loc[dataset_info[dataset_info.columns[1]].isin(exclude_data_id), 
                                                   dataset_info.columns[0]].values.tolist()
            elif not(all([isinstance(x,int) for x in exclude_data_id])):
                raise TypeError("if given as a list, exclude_data_id must have only int or str values")
        else:
            cond1 = not(all([isinstance(x,int) for x in exclude_data_id.keys() ]))
            flat_list = [item for sublist in exclude_data_id.values() for item in sublist]
            cond2 = not(all([isinstance(x,int) for x in flat_list]))
            if cond1 or cond2:
                raise TypeError("if given as a dict, val_data_id must have only integer keys and values") 
    
    # check on test_data_id
    if test_data_id!=None:
        if not( isinstance(test_data_id,list) or isinstance(test_data_id,dict) ):
            test_data_id=[test_data_id]
        if isinstance(test_data_id,list):
            if (all([isinstance(x,str) for x in test_data_id])):
                if not(isinstance(partition_table, pd.DataFrame)):
                    msgErr= 'if given as a list of str, a pandas table with first column the dataset ID '
                    msgErr += 'and second column the dataset name must be given'
                    raise ValueError(msgErr)
                else:
                    test_data_id = dataset_info.loc[dataset_info[dataset_info.columns[1]].isin(test_data_id), 
                                                   dataset_info.columns[0]].values.tolist()
            elif not(all([isinstance(x,int) for x in test_data_id])):
                raise TypeError("if given as a list, test_data_id must have only integer values")
        else:
            cond1 = not(all([isinstance(x,int) for x in test_data_id.keys() ]))
            flat_list = [item for sublist in test_data_id.values() for item in sublist]
            cond2 = not(all([isinstance(x,int) for x in flat_list]))
            if cond1 or cond2:
                raise TypeError("if given as a dict, test_data_id must have only integer keys and values")

    # check on val_data_id
    if val_data_id!=None:
        if not( isinstance(val_data_id,list) or isinstance(val_data_id,dict) ):
            val_data_id=[val_data_id]
        if isinstance(val_data_id,list):
            if (all([isinstance(x,str) for x in val_data_id])):
                if not(isinstance(partition_table, pd.DataFrame)):
                        msgErr= 'if given as a list of str, a pandas table with first column the dataset ID '
                        msgErr += 'and second column the dataset name must be given'
                        raise ValueError(msgErr)
                else:
                    val_data_id = dataset_info.loc[dataset_info[dataset_info.columns[1]].isin(val_data_id), 
                                                       dataset_info.columns[0]].values.tolist()
            elif not(all([isinstance(x,int) for x in val_data_id])):
                raise TypeError("if given as a list, val_data_id must have only integer values")
        else:
            cond1 = not(all([isinstance(x,int) for x in val_data_id.keys() ]))
            flat_list = [item for sublist in val_data_id.values() for item in sublist]
            cond2 = not(all([isinstance(x,int) for x in flat_list]))
            if cond1 or cond2:
                raise TypeError("if given as a dict, val_data_id must have only integer keys and values") 
    
    return exclude_data_id, test_data_id , val_data_id


def GetEEGSplitTable(partition_table: pd.DataFrame=None,
                     test_ratio: float= None,
                     val_ratio: float= None,
                     exclude_data_id: Union[int,Sequence[int or str], dict]=None,
                     test_data_id: Union[int,Sequence[int or str], dict]=None,
                     val_data_id: Union[int,Sequence[int or str], dict]=None,
                     val_ratio_on_all_data: bool=True,
                     test_split_mode: str or int =0,
                     val_split_mode: str or int= 1,
                     split_tolerance=0.01,
                     perseverance=1000,
                     dataset_info = None,
                     EEGpath: str=None,
                     save: bool=False
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
    exclude_data_id  : int or list[int or str] or dict[int, list[int]], optional 
        Dataset ID to be excluded. The ID must be the one used to process EEG with the AZ's library. So, given a file
        name in the format a_b_c_d.extension, a is the dataset ID and b is the subject ID. the IDs can be given in the
        following formats:
            1) a single int identifying a specific dataset ID
            2) a list of integers with the dataset ID
            3) a list of strings with the dataset ID
            4) a dictionary where keys are integers defining the dataset ID and values are list of integers defining
               the subject IDs for that dataset. Strings as dictionary keys are not supported
        Default: None
    test_data_id: int or list[int or str] or dict[int, list[int]], optional 
        Same as exclude_data_id but for the test split
        Defaul: None
    val_data_id: int or list[int or str] or dict[int, list[int]], optional 
        Same as exclude_data_id but for validation split
        Default: None
    split_tolerance: float, optional
        Argument for get_subarray_closest_sum function. Set the maximum accepted tolerance between the given split ratio
        and the one achieved with the obtained subset. Must be a number in [0,1]
        Default: 0.01
    perseverance: int, optional
        Argument for get_subarray_closest_sum function. Set the maximum number of tries before stop searching for a split 
        whose ratio is in the range [target_ratio - tolerance, target_ratio + tolerance]
        Default: 1000
    dataset_info = pd.Dataframe, optional
        Dataframe with having in the first column the ID of the Dataset, and the second column the name of the Dataset as 
        a string. Must be given if test_data_id, val_data_id or exclude_data_id are given as a list of string with the name
        of the datasets.
        Default: None
    EEGpath : string, optional 
        Directory with all EEG files. Can be given in place of partition table if all split are made by id. It's highly 
        suggested to run GetEEGPartitionNumber() and give the partition_table though. 
        Default: None
    save : bool, optional
        Whether to save the resulted DataFrame as a .csv file or not
        Default: False
                 
    Returns
    -------
    EEGSplit : DataFrame
        Two columns Pandas DataFrame. The first column has the EEG file name, 
        the second define the file usage with 0-1, Train-Test
            
    """ 
    
    # Various check on inputs
    exclude_data_id, test_data_id, val_data_id = check_input_split_table(EEGpath, test_ratio, val_ratio, 
                                                              exclude_data_id, test_data_id, val_data_id, 
                                                              val_ratio_on_all_data, partition_table, dataset_info)
    ex_id_list=isinstance(exclude_data_id,list)
    test_id_list=isinstance(test_data_id,list)
    val_id_list=isinstance(val_data_id,list)
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
        
    
    # copy and modify table
    if isinstance(partition_table, pd.DataFrame):
        partition2 = partition_table.copy()
        partition2['dataset_ID'] = [ x.split('_')[0] for x in partition2['file_name'] ]
        partition2['subj_ID']    = [ x.split('_')[1] for x in partition2['file_name'] ]
        EEGfiles= partition_table['file_name'].values.tolist()
    else:
        if EEGpath[-1]=='/':
            EEGfiles=glob.glob(EEGpath + '*_*_*_*.mat')
        else:
            EEGfiles=glob.glob(EEGpath + '/*_*_*_*.mat')
        EEGfiles = [i.split('/')[-1] for i in EEGfiles]

        if len(EEGfiles)==0:
            print('didn\'t found any .mat files')
            return
    
    EEGsplit=[ [filename, 0] for filename in EEGfiles]
    
    # PRE SPLIT:  DATASET  -->  DATASET WITH ONLY CONSIDERED DATA
    if exclude_data_id!=None: # if test_data_id is given, ignore test ratio
        for ii in range(len(EEGfiles)):
            FileID= int(EEGsplit[ii][0].split('_')[0])
            if ex_id_list:
                if FileID in exclude_data_id:
                    EEGsplit[ii][1]=-1
            else:
                SubjID=int(EEGsplit[ii][0].split('_')[1])
                if FileID in exclude_data_id.keys():
                    if SubjID in exclude_data_id[FileID]:
                        EEGsplit[ii][1]=-1
    
    
    
    # FIRST SPLIT:  DATASET  -->  TRAIN/TEST
    if test_data_id!=None: # if test_data_id is given, ignore test ratio
        for ii in range(len(EEGfiles)):
            if EEGsplit[ii][1] != -1:
                FileID= int(EEGsplit[ii][0].split('_')[0])
                if test_id_list:
                    if FileID in test_data_id:
                        EEGsplit[ii][1]=2
                else:
                    SubjID=int(EEGsplit[ii][0].split('_')[1])
                    if FileID in test_data_id.keys():
                        if SubjID in test_data_id[FileID]:
                            EEGsplit[ii][1]=2
    elif test_ratio>0:
        # split data according to test ratio and test_split_on_subj
        idx_val= [i for i in range(len(EEGsplit)) if EEGsplit[i][1]!=-1]
        partition3 = partition2.iloc[idx_val]
        if test_split_mode==1:
            sample1 = partition3.groupby( ['dataset_ID','subj_ID'] )['N_samples'].sum().reset_index(name='N_samples')
        elif test_split_mode==0:
            sample1 = partition3.groupby( ['dataset_ID'] )['N_samples'].sum().reset_index(name='N_samples')
        else:
            sample1 = partition3

        arr=sample1['N_samples'].values.tolist()
        alldatasum=sum(arr)
        target=test_ratio*alldatasum
        final_idx, subarray = get_subarray_closest_sum(arr, target, tolerance=split_tolerance, 
                                                       perseverance=perseverance)
        final_idx.sort()

        if test_split_mode==2:
            for ii in final_idx:
                EEGsplit[ii][1] = 2
        else: 
            data_test_ID = set(sample1['dataset_ID'].iloc[final_idx].values.tolist())
            if test_split_mode==1:
                subj_test_ID = {key: [] for key in data_test_ID}
                for i in final_idx:
                    subj_test_ID[sample1['dataset_ID'].iloc[i]].append( sample1['subj_ID'].iloc[i] )
            
            for ii in range(len(EEGfiles)):
                if EEGsplit[ii][1] != -1:
                    FileID= EEGsplit[ii][0].split('_')[0]
                    if FileID in data_test_ID: 
                        if test_split_mode==1:
                            subjID=EEGsplit[ii][0].split('_')[1]
                            if subjID in subj_test_ID[FileID]:
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
                    FileID= int(EEGsplit[ii][0].split('_')[0])
                    if val_id_list:
                        if FileID in val_data_id:
                            EEGsplit[ii][1]=1
                    else:
                        SubjID=int(EEGsplit[ii][0].split('_')[1])
                        if FileID in val_data_id.keys():
                            if SubjID in val_data_id[FileID]:
                                EEGsplit[ii][1]=1
    elif val_ratio>0:
        # split data according to test ratio and test_split_on_subj
        idx_val= [i for i in range(len(EEGsplit)) if EEGsplit[i][1]==0]
        partition3 = partition2.iloc[idx_val]
        if val_split_mode == 1:
            sample1 = partition3.groupby( ['dataset_ID','subj_ID'] )['N_samples'].sum().reset_index(name='N_samples')
        elif val_split_mode == 0:
            sample1 = partition3.groupby( ['dataset_ID'] )['N_samples'].sum().reset_index(name='N_samples')
        else:
            sample1 = partition3
        
        arr=sample1['N_samples'].values.tolist()
        target=val_ratio*alldatasum if val_ratio_on_all_data else val_ratio*sum(arr)
        final_idx, subarray = get_subarray_closest_sum(arr, target, tolerance=split_tolerance, perseverance=perseverance)
        final_idx.sort()
        
        if val_split_mode==2:
            fileName=sample1.iloc[final_idx]['file_name'].values.tolist()
            for ii in range(len(EEGfiles)):
                if EEGsplit[ii][0] in fileName:
                    EEGsplit[ii][1] = 1
        else:
            data_val_ID = set(sample1['dataset_ID'].iloc[final_idx].values.tolist())
            if val_split_mode == 1:
                subj_val_ID = {key: [] for key in data_val_ID}
                for i in final_idx:
                    subj_val_ID[sample1['dataset_ID'].iloc[i]].append( sample1['subj_ID'].iloc[i] )

            for ii in range(len(EEGfiles)):
                if EEGsplit[ii][1] == 0:
                    FileID= EEGsplit[ii][0].split('_')[0]
                    if FileID in data_val_ID: 
                        if val_split_mode==1:
                            subjID=EEGsplit[ii][0].split('_')[1]
                            if subjID in subj_val_ID[FileID]:
                                EEGsplit[ii][1]=1
                            else:
                                EEGsplit[ii][1]=0
                        else:
                            EEGsplit[ii][1]=1
                    else:
                        EEGsplit[ii][1]=0
        
    EEGsplit=pd.DataFrame(EEGsplit,columns=['file_name','split_set'])
    
    if save:
        EEGsplit.to_csv('EEGTrainTestSplit.csv')        
     
    return EEGsplit


class EEGDataset(Dataset):
    def __init__(self, 
                 EEGpath: str, 
                 EEGlen: pd.DataFrame, 
                 EEGsplit: pd.DataFrame, 
                 EEGpartition_spec: list,
                 mode: str='train',
                 supervised: bool=False,
                 label_on_subj_info: bool=True,
                 subj_info_key: list=None,
                 get_label_fun: "function"=None,
                 optional_fun_args: list or dict=None,
                 default_dtype=torch.float32
                ):
        """
        Parameters
        ----------
        EEGpath : string 
            Directory with all the EEG files.
        EEGlen : DataFrame
            DataFrame with the number of partition per EEG record. Must be the output of GetEEGPartitionNumber()
        EEGsplit : DataFrame 
            DataFrame with the train/test split info. Must be the output of GetEEGSplitTable()
        EEGpartition_spec : list
            3-element list with the input gave to GetEEGPartitionNumber() in [freq, window, overlap] format.
        mode: string, optional
            if the dataset is intended for train, test or validation. It accept only 'train','test','validation'
            strings.
            Default: 'train'
        supervised: bool, optional
            Whether the getItem method must return a label or not
            Default: False
        label_on_subj_info: bool, optional
            Whether the label is in the subj_info attribute or not
            Default: True
        subj_info_key: str or list of str, optional
            A single or set of dictionary keys given as strings to use to get access to the subject info to use as 
            label.
            Default: None
        get_label_fun: function, optional
            A pointer to a custom function to use to get the label of the ith sample. This function must accept
            as first four arguments, given automatically in the __getitem__ method:
            1) file name
            2) file path
            3) start index of the sample
            4) end index of the sample
            Since the method must be generic, it is expected that at least some of these four arguments will be 
            necessary to get the correct label. The function can accept other optional argument, but keep in mind 
            this setting when designing the custom one to give here.
            Default: None
        optional_fun_args: list or dict, optional
            Other optional arguments to give to the get_label_fun function. List or dict are both accepted
            Default: None
        """
        
        if supervised:
            if subj_info_key==None and get_label_fun==None:
                raise ValueError('if supervised dataset, subj_info_key or get_label_fun must be given')
        
        # Instantiate parent class
        super().__init__()

        self.default_dtype = default_dtype
        # Set EEGpath and get all EEG files
        if EEGpath[-1]=='/':
            self.EEGpath = EEGpath
        else: 
            self.EEGpath = EEGpath +'/'
        
        # Save Input arguments
        self.EEGsplit = EEGsplit
        self.EEGlen = EEGlen
        self.mode=mode
        self.supervised = supervised
        self.label_on_subj_info = label_on_subj_info
        self.subj_info_key = subj_info_key if isinstance(subj_info_key,list) else [subj_info_key]
        self.get_label_fun = get_label_fun
        self.optional_fun_args = optional_fun_args
        
        # Check if the dataset is for train test or validation
        if mode.lower()=='train':
            FileNames = EEGsplit.loc[EEGsplit['split_set']==0,'file_name'].values
        elif mode.lower()=='validation':
            FileNames = EEGsplit.loc[EEGsplit['split_set']==1,'file_name'].values
        else:
            FileNames = EEGsplit.loc[EEGsplit['split_set']==2,'file_name'].values
        
        # initialize attributes for __len__
        self.EEGlenTrain = EEGlen.loc[EEGlen['file_name'].isin(FileNames)]
        self.DatasetSize = self.EEGlenTrain['N_samples'].sum()
        
        # initialize attributes for __getItem__
        self.freq = EEGpartition_spec[0]
        self.window = EEGpartition_spec[1]
        self.overlap = EEGpartition_spec[2]
        self.Nsample= EEGpartition_spec[0]*EEGpartition_spec[1]
        self.EEGcumlen = np.cumsum(self.EEGlenTrain['N_samples'].values)
        
        
        # Set Current EEG passed (speed up getItem method)
        # Keep in mind that multiple workers use copy of the dataset
        # saving a copy of the current passed EEG file will saturate the memory 
        # if combined with the rest of the memory used by the system and that used to store the batches
        # it can saturate fast
        self.currEEG = None
        self.subj_info=None
        self.subj_info_keys=None
        self.file_path=None
        self.minIdx = -1
        self.maxIdx = -1
        
    
    def __len__(self):
        return self.DatasetSize
    
    def __getitem__(self,index):
        nameIdx=np.searchsorted(self.EEGcumlen, index, side='right')
        if ((index<self.minIdx) or (index>self.maxIdx)):
            file_name= self.EEGlenTrain.iloc[nameIdx].file_name
            EEG=loadmat(self.EEGpath + file_name, 
                                 simplify_cells=True)
            self.currEEG=EEG['DATA_STRUCT']['data']
            if isinstance(self.currEEG, np.ndarray):
                self.currEEG=torch.from_numpy(self.currEEG)
            if self.currEEG.dtype != self.default_dtype:
                self.currEEG= self.currEEG.to(dtype=self.default_dtype)
            
            self.minIdx=0 if nameIdx==0 else self.EEGcumlen[nameIdx-1]
            self.maxIdx=self.EEGcumlen[nameIdx]-1
            if self.supervised:
                self.subj_info=EEG['DATA_STRUCT']['subj_info']
                self.subj_info_keys=self.subj_info.keys()
                self.file_path=EEG['DATA_STRUCT']['filepath']
                self.curr_key = list(set(self.subj_info_keys).intersection(self.subj_info_key))[0]
                self.label = self.subj_info[self.curr_key]

        
        
        partition=index-self.minIdx
        start=(self.Nsample-round(self.Nsample*self.overlap))*(partition)
        end=start+self.Nsample
        #deal with partial end sample
        if end>self.currEEG.shape[1]:
            sample=self.currEEG[:,-self.Nsample:]
        else:
            sample=self.currEEG[:,start:end]
        
        if isinstance(sample, np.ndarray):
            sample= torch.from_numpy(sample)
        
        if self.supervised:
            if self.label_on_subj_info:
                label = self.label
            else:
                if isinsance(optional_fun_args, list):
                    label = self.get_label_fun(file_name, self.file_path, start, end, *self.optional_fun_args)
                elif isinsance(optional_fun_args, dict):
                    label = self.get_label_fun(file_name, self.file_path, start, end, **self.optional_fun_args)
                else:
                    label = self.get_label_fun(file_name, self.file_path, start, end)
            return sample, label
        else: 
            return sample
     
    def shuffleFiles(self):
        self.EEGlenTrain=self.EEGlenTrain.sample(frac=1).reset_index(drop=True)
        self.EEGcumlen = np.cumsum(self.EEGlenTrain['N_samples'].values)
        

class EEGsampler(Sampler):
    
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
        Return an iterator where subject are passed sequentially but the samples of each subjects are shuffled
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

    
# ## Function to create a table with the number of partitions per EEG record
#    # ALTERNATIVE WITH WHILE CYCLE    
#    EEGlen=[]
#    for ii in EEGfiles:
#        # load file
#        EEG=loadmat(ii)['DATA_STRUCT']['data'][0][0]
#        
#        # Calculate number of total partitions
#        if overlap==0:
#            N_EEG=EEG.shape[1]/(freq*window)
#        else:
#            Nsample=EEG.shape[1]
#            WindSample=freq*window
#            overlapInt=round(WindSample*overlap)
#            N_EEG=0
#            currS=0
#            InPartition=True
#            while InPartition:
#                currS += WindSample
#                if currS>Nsample:
#                    if (currS-Nsample)/WindSample <0.5:
#                        N_EEG=N_EEG+(1-(currS-Nsample)/WindSample)
#                    InPartition=False
#                else:    
#                    N_EEG=N_EEG+1
#                    if currS==Nsample:
#                        InPartition=False
#                    else:
#                        currS -= overlapInt
#                
#        # round base on how partial partition is handled
#        if includePartial:
#            N_EEG=round(N_EEG)
#        else:
#            N_EEG=math.floor(N_EEG)
#        EEGlen.append([ii.split('/')[-1],N_EEG])