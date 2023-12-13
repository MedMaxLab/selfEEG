import os
import sys
import glob
import time
import math
import random
import copy
import warnings
from typing import Union, Sequence, Optional
import numpy as np
import pandas as pd
import tqdm
from scipy.io import loadmat
import torch
from numpy.typing import ArrayLike
from torch.utils.data import Dataset, Sampler
from ..utils.utils import subarray_closest_sum, get_subarray_closest_sum


__all__ = ['GetEEGPartitionNumber', 
           'GetEEGSplitTable', 'GetEEGSplitTableKfold',
           'EEGDataset', 'EEGsampler']

def GetEEGPartitionNumber(EEGpath: str, 
                          freq: int or float=250, 
                          window: int or float=2, 
                          overlap: float=0.10,
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
    ``GetEEGPartitionNumber`` finds the number of unique partitions from the EEG signals 
    stored inside a given input directory. Some default parameters are designed to work with 
    the 'BIDSAlign' library. For more info, see [bids]_.
    To further check how to use this function see the introductory notebook provided 
    in the documentation.
    
    
    Parameters
    ----------
    EEGpath : str 
        Directory with all EEG files. If the last element of the string is not 
        "/", the character will be added automatically.
    freq : int or float, optional 
        EEG sampling rate. Must be the same for all EEG files. 
        
        Default = 250
    window : int or float, optional
        The window length given in seconds.
        
        Default = 2
    overlap : float, optional 
        Same EEG contiguous partitions overlap in percentage. Must be in the interval [0,1).
        
        Default = 0.1
    includePartial : bool, optional
        Whether to also count the final EEG portions which could potentially cover 
        at least half of the time windows. In this case the overlap between the last
        included partition and the previous one will increase in order to fill the incomplete
        partition with real recorded values. Note that this apply only if at 
        least half of such partition will include new values.
        
        Default = True
    file_format : str or list[str], optional
        A string used to detect a set of specific EEG files inside the given EEGpath. 
        It is directly put after ``EEGpath`` during call of the glob.glob() method. 
        Therefore, it can contain shell-style wildcards (see glob.glob() help for more info). 
        This parameter might be helpful if you have other files other than the EEGs in your 
        directory. Alternatively, you can provide a list of strings to cast multiple 
        glob.glob() searches. This might be useful if you want to combine multiple 
        identification criteria (e.g. specific file extensions, specific file names, etc.)
        
        Default = '*'
    load_function : 'function', optional
        A custom EEG file loading function. It will be used instead of the default: 
        
        ``loadmat(ii, simplify_cells=True)['DATA_STRUCT']['data']`` 
        
        which is the default output format for files preprocessed with the BIDSalign library.
        The function must take only one required argument, which is the full path to the 
        EEG file (e.g. the function will be called in this way: 
        load_function(fullpath, optional_arguments) ).
        
        Default = None
    optional_load_fun_args: list or dict, optional
        Optional arguments to give to the custom loading function. Can be a list or a dict.
        
        Default = None
    transform_function : 'function', optional
        A custom transformation to be applied after the EEG is loaded. 
        Might be useful if there are portions of the signal to cut (usually the initial or the
        final). The function must take only one required argument, which is the loaded EEG 
        file to transform (e.g. the function will be called in this way: 
        transform_function(EEG, optional_arguments) ).
        
        Default = None
    optional_transform_fun_args: list or dict, optional
        Optional arguments to give to the EEG transformation function. Can be a list or a dict.
        
        Default = None
    keep_zero_sample : bool, optional
        Whether to preserve Dataframe's rows with calculated zero number of samples or not.

        Default = True
    save : bool, optional
        Whether to save the resulting DataFrame as a .csv file.

        Default = False
    save_path: str, optional
        A custom path to be used instead of the current working directory. 
        It is the string given to the ``pandas.DataFrame.to_csv()`` method.
        Note that if save is True and no save_path is given, the file will be saved in the
        current working directory as `EEGPartitionNumber.csv` or 
        `EEGPartitionNumber_k.csv` with k integer used to avoid overwriting a file.
        
        Default = None
    verbose: bool, optional
        whether to print some information during function excecution. 
        Useful to keep track of the calculation process. Might be useful for large
        datasets.
                
    Returns
    -------
    lenEEG : DataFrame
        Three columns Pandas DataFrame. 
        The first column has the full path to the EEG file, the second the file name, 
        the third its number of partitions.
    
    Note
    ----
    freq*window must give an integer with the number of samples.

    Note
    ----
    This function can handle array with more than 2 dimensions. In this case a warning will
    be generated and calculation will be performed in this way. First the length of the last 
    dimension will be used to calculate the number of partitions, then the number will be 
    multiplied by the product of the shape of all dimensions from the first to the second to
    last (the last two dimensions are supposed to be the Channel and Sample dimension of a
    single EEG file).

    Example
    -------
    >>> import pickle
    >>> import pandas as pd
    >>> import selfeeg.dataloading as dl
    >>> import selfeeg.utils
    >>> utils.create_dataset()
    >>> def loadEEG(path):
    ...     with open(path, 'rb') as handle:
    ...         EEG = pickle.load(handle)
    ...     x = EEG['data']
    ...     return x
    >>> EEGlen = dl.GetEEGPartitionNumber('Simulated_EEG',freq=128, window=2, 
    ...                                   overlap=0.3, load_function=loadEEG )
    >>> EEGlen.head()

    References
    ----------
    .. [bids] https://github.com/MedMaxLab/BIDSAlign
    
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
        if EEGpath[-1]==os.sep:
            EEGfiles=glob.glob(EEGpath + file_format)
        else:
            EEGfiles=glob.glob(EEGpath + os.sep + file_format )
    else:
        try:
            if EEGpath[-1]==os.sep:
                EEGfiles = [glob.glob(eegpath + i) for i in file_format]
            else:
                EEGfiles = [glob.glob(eegpath + os.sep + i) for i in file_format]
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
                   desc='extracting EEG samples', unit=' files',file=sys.stdout) as pbar:
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

            # calculate number of samples based on the 
            # overlap and includepartial arguments
            M=len(EEG.shape)
            if overlap==0:
                if includePartial:
                    N_Partial=EEG.shape[-1]/(WindSample)
                else:
                    N = EEG.shape[-1]//(WindSample)
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
                               ' Found number of samples will be multiplied by the size of each '
                               'extra dimension. Note that this may create problems to the '
                               '__getitem()__ method in the custom EEGDataset class'), Warning)
                N_EEG*= np.prod(EEG.shape[0:-2])
                
            EEGlen.append([ii, ii.split(os.sep)[-1],N_EEG])
    
    del EEG
    
    # create dataframe and check if 0 length files must be kept
    EEGlen=pd.DataFrame(EEGlen,columns=['full_path','file_name','N_samples'])
    if not(keep_zero_sample):
        EEGlen = EEGlen.drop(EEGlen[EEGlen.N_samples == 0].index).reset_index()
        EEGlen = EEGlen.drop(columns= 'index')

    # save block
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

    # generate summary to print 
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
                     test_ratio: float= 0.2,
                     val_ratio: float= 0.2,
                     test_split_mode: str or int =2,
                     val_split_mode: str or int= 2,
                     exclude_data_id: list or dict =None,
                     test_data_id: list or dict=None,
                     val_data_id: list or dict =None,
                     val_ratio_on_all_data: bool=True,
                     stratified: bool=False,
                     labels: ArrayLike=None,
                     dataset_id_extractor: 'function'=None,
                     subject_id_extractor: 'function'=None,
                     split_tolerance=0.01,
                     perseverance=1000,
                     save: bool=False,
                     save_path: str=None,
                     seed: int= None
                    ) -> pd.DataFrame:

    """   
    ``GetEEGSplitTable`` creates a table defining the files to use for train, 
    validation and test of the models.
    
    Split is done in the following way:

        1. Dataset is split in Train and Test sets
        2. Train set is split in Train and Validation sets
    
    If specific IDs are given, the split is done using them ignoring any split ratio, 
    otherwise split is done randomly using the given ratio. Note that Test or Validation 
    sets can be empty, if for example you want to split the dataset only in two subsets.
    To further understand how to use this function see the introductory notebook provided 
    in the documentation.
    
    Parameters
    ----------
    partition_table: pd.Dataframe
        A two columns dataframe where:
            
            1. the first column has name 'file_name' and contain all the file names
            2. the second column has name 'N_samples' and has the number of samples 
               which can be extracted from the file
        
        This table can be automatically created with a custom setting with the provided 
        function ``GetEEGPartitionNumber()`` .
    test_ratio: float, optional
        The percentage of data with respect to the whole number of samples (partitions) 
        of the dataset to be included in the test set. Must be a number in [0,1]. 
        0 means that the test split is skipped if test_data_id is not given.
        
        Default = 0.2
    val_ratio: float, optional
        The percentage of data with respect to the whole number of samples (partitions) 
        of the dataset or the remaining ones after test split 
        (see val_ratio_on_all_data argument) to be included in the validation set. 
        Must be a number in [0,1]. 0 means that the validation split is skipped 
        if val_data_id is not given. 
        
        Default = 0.2
    test_split_mode: int or str, optional
        The type of split to perform in the step train test split. It can be one of the
        following:
        
            1. any of [0, 'd', 'set', 'dataset']: split will be performed using dataset 
               IDs, i.e. all files of the same dataset will be put in the same split set
            2. any of [1, 's', 'subj', 'subject']: split will be performed using subjects IDs, 
               i.e. all files of the same subjects will be put in the same split set
            3. any of [2, 'file', 'record']: split will be performed looking at single files
            
        Default = 2
    val_split_mode: int or str, optional
        The type of split to perform in the step train to train - validation split. 
        Inputs allowed are the same as in test_split_mode.
        
        Default = 2
    exclude_data_id  : list or dict, optional 
        Dataset ID to be excluded. It can be given in the following formats:
        
            1. a list with all dataset IDs to exclude
            2. a dictionary where keys are the dataset IDs and values its relative subject IDs. 
               If a key has an empty value, then all the files with that dataset ID
               will be included
        
        Note that to work, the function must be able to identify the dataset or subject IDs 
        from the file name in order to check if they are in the given list or dict. 
        Custom extraction functions can be given as arguments; however, if nothing is given, 
        the function will try to extract IDs considering that file names are in the format 
        a_b_c_d.extension (the typical output of the BIDSAlign library), 
        where "a" is an integer with the dataset ID and "b" an integer with the subject ID. 
        If this fails, all files will be considered from the same datasets (id=0), 
        and each file from a different subject (id from 0 to N-1).
        
        Also note that if the input argument is not a list or a dict, it will be 
        automatically converted to a list. No checks about what is converted to a list will
        be performed.
        
        Default = None
    test_data_id: list or dict, optional 
        Same as exclude_data_id but for the test split.
        
        Defaul = None
    val_data_id: list or dict, optional  
        Same as exclude_data_id but for validation split.
        
        Default = None
    val_ratio_on_all_data: bool, optional
        Whether to calculate the validation split size only on the training 
        set size (False) or on the entire "considered" dataset (True), i.e., the size 
        of all files except ones included in `exclude_data_id`.
        
        Default = True
    stratified: bool, optional
        Whether to apply stratification to the split or not. Might be used for fine-tuning 
        split (the typical phase where labels are involved). Stratification will preserve, 
        if possible, the label's ratio on the training, validation, and test sets.
        Works only when each file has an unique label, which must be given in input.
        
        Default = False
    labels: list or ArrayLike, optional
        A list or 1d ArrayLike objects with the label of each file listed 
        in the partition table. Must be given if stratification is set to True.
        Indeces of labels must match row indeces in the partition table, i.e. 
        label1 -> row1, label2 -> row2, etc.
        
        Default = None
    dataset_id_extractor: function, optional
        A custom function to be used to extract the dataset ID from file the file name. 
        It must accept only one argument, which is the file name (not the file path, 
        only the file name).
        
        Default = None
    subject_id_extractor: function, optional
        A custom function to be used to extract the subject ID from the file name. 
        It must accept only one argument, which is the file name (not the file path, 
        only the file name).
        
        Default = None
    split_tolerance: float, optional
        Argument for ``get_subarray_closest_sum`` function. 
        Set the maximum accepted tolerance between the given split ratio
        and the one obtained with the resulting subset. Must be a number in [0,1].
        
        Default = 0.01
    perseverance: int, optional
        Argument for ``get_subarray_closest_sum`` function. Set the maximum number 
        of tries before stop searching for a split whose ratio is in the range 
        [target_ratio - tolerance, target_ratio + tolerance].
        
        Default = 1000
    save : bool, optional
        Whether to save the resulting DataFrame as a .csv file or not.
        
        Default = False
    save_path: str, optional
        A custom path to be used instead of the current working directory. 
        It is the string given to the ``pandas.DataFrame.to_csv()`` method.
        
        Default = None
    seed: int, optional
        An integer defining the seed to use. Set it to reproduce split results.

        Default = None
                 
    Returns
    -------
    EEGSplit : DataFrame
        Two columns Pandas DataFrame. The first column has the EEG file name, 
        the second defines the split. 
        The split will assign the following labels to a file:

           1. -1 : the file is excluded 
           2. 0  : the file is included in the training set
           3. 1  : the file is included in the validation set
           4. 2  : the file is included in the test set

    Example
    -------
    >>> import pickle
    >>> import pandas as pd
    >>> import selfeeg.dataloading as dl
    >>> import selfeeg.utils
    >>> labels = utils.create_dataset()
    >>> def loadEEG(path):
    ...     with open(path, 'rb') as handle:
    ...         EEG = pickle.load(handle)
    ...     x = EEG['data']
    ...     return x
    >>>  EEGlen = dl.GetEEGPartitionNumber('Simulated_EEG',freq=128, window=2, 
    ...                                    overlap=0.3, load_function=loadEEG )
    >>>  EEGsplit = dl.GetEEGSplitTable(EEGlen, seed=1234) #default 60/20/20 split
    >>>  dl.check_split(EEGlen,EEGsplit) #will return 60/20/20
            
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
            raise ValueError('if val_ratio_on_all_data is set to true,'
                             ' val_ratio+test_ratio must be in [0,1) ')
     
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

    if seed is not None:
        random.seed(seed)

    # check if stratification must be applied
    # in case stratification must be performed, the function will be called
    # multiple times using the same ratio but only files having the same label
    # single results will be then concatenated and sorted to preserve index positioning
    if stratified:
        if (test_ratio==None) and (val_ratio==None):
            print('STRATIFICATION can be applied only if at least one split ratio is given.')
        else:
            N_classes = np.unique(labels)
            classSplit = [None]*len(N_classes)
            # Call the split for each class 
            for i, n in enumerate(N_classes):
                classIdx = [ index_i for index_i, label_i in enumerate(labels) if label_i==n]
                subClassTable = partition_table.iloc[classIdx]
                classSplit[i] = GetEEGSplitTable(partition_table = subClassTable,  
                                                 test_ratio = test_ratio, 
                                                 val_ratio = val_ratio,
                                                 test_split_mode = test_split_mode,
                                                 val_split_mode = val_split_mode,
                                                 exclude_data_id = exclude_data_id,  
                                                 test_data_id = test_data_id,  
                                                 val_data_id = val_data_id,
                                                 val_ratio_on_all_data = val_ratio_on_all_data,
                                                 stratified = False,
                                                 labels = None,
                                                 dataset_id_extractor = dataset_id_extractor,
                                                 subject_id_extractor = subject_id_extractor,
                                                 split_tolerance = split_tolerance,
                                                 perseverance = perseverance,
                                                 save = False,
                                                )
            
            # merge subclass tables and check for mysterious duplicates
            EEGsplit = pd.concat(classSplit, axis=0, ignore_index=True)
            try:
                EEGsplit.drop(columns='index') #useless but to be sure
            except:
                pass
            EEGsplit= EEGsplit.drop_duplicates(ignore_index=True)
            EEGsplit = EEGsplit.sort_values(by='file_name').reset_index().drop(columns='index')

    else:

        # boolean to check that ids are given as list or dict
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
        
        # It is faster to update a list than a table
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
            final_idx, subarray = get_subarray_closest_sum(arr, target, 
                                                           split_tolerance, perseverance)
            final_idx.sort()

            # update split list according to returned subarray
            # and test split mode
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
        # the flow is basically the same as in the first split aside for some minor modifications
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
            final_idx, subarray = get_subarray_closest_sum(arr, target, 
                                                           split_tolerance, perseverance)
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

    # save block
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
                          test_ratio: float= 0.2,
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
                          save_path: str=None,
                          seed: int = None
                         ) -> pd.DataFrame:
    """   
    ``GetEEGSplitTableKfold`` creates a table with multiple splits for cross-validation.
    Test split, if calculated, is kept equal in every CV split.
    Split is done in the following way:

        1. dataset is split in Train and Test sets
        2. train set is split in Train and Validation sets
    
    Test split is optional and can be done with the same modalities described 
    in the ``GetEEGSplitTable`` function, i.e. by giving specific ID or by giving a split ratio. 
    CV's train/validation split cannot be done in this way, since this does not 
    guarantee the preservation of the split ratio, which is the core of cross validation.
    
    Parameters
    ----------
    partition_table: pd.Dataframe
        A two columns dataframe where:
            
            1. the first column has name 'file_name' and contain all the file names
            2. the second column has name 'N_samples' and has the number of samples 
               which can be extracted from the file
        
        This table can be automatically created with a custom setting with the provided 
        function ``GetEEGPartitionNumber()``.
    Kfold: int, optional
        The number of folds to extract. Must be a number higher or equal than 2. 
        
        Default = 10
    test_ratio: float, optional
        The percentage of data with respect to the whole number of samples (partitions) 
        of the dataset to be included in the test set. Must be a number in [0,1]. 
        0 means that the test split is skipped if test_data_id is not given.
        
        Default = 0.2
    test_split_mode: int or str, optional
        The type of split to perform in the step train test split. It can be one of the
        following:
        
            1. any of [0, 'd', 'set', 'dataset']: split will be performed using dataset 
               IDs, i.e. all files of the same dataset will be put in the same split set
            2. any of [1, 's', 'subj', 'subject']: split will be performed using subjects IDs, 
               i.e. all files of the same subjects will be put in the same split set
            3. any of [2, 'file', 'record']: split will be performed looking at single files
            
        Default = 2
    val_split_mode: int or str, optional
        The type of split to perform in the step train to train - validation split. 
        Input allowed are the same as in test_split_mode.
        
        Default = 2
    exclude_data_id  : list or dict, optional 
        Dataset ID to be excluded. It can be given in the following formats:
        
            1. a list with all dataset IDs to exclude
            2. a dictionary where keys are the dataset IDs and values its relative subject IDs. 
               If a key has an empty value, then all the files with that dataset ID
               will be included
        
        Note that to work, the function must be able to identify the dataset or subject IDs 
        from the file name in order to check if they are in the given list or dict. 
        Custom extraction functions can be given as arguments; however, if nothing is given, 
        the function will try to extract IDs considering that file names are in the format 
        a_b_c_d.extension (the output of the BIDSalign library), 
        where "a" is an integer with the dataset ID and "b" an integer with the subject ID. 
        If this fail, all files will be considered from the same datasets (id=0), 
        and each file from a different subject (id from 0 to N-1).
        
        Also note that if the input argument is not a list or a dict, it will be 
        automatically converted to a list. No checks about what is converted to a list will
        be performed.
        
        Default = None
    test_data_id: list or dict, optional 
        Same as exclude_data_id but for the test split.
        
        Defaul = None
    stratified: bool, optional
        Whether to apply stratification to the split or not. Might be used for fine-tuning 
        split (the typical phase where labels are involved). Stratification will preserve, 
        if possible, the label's ratio on the training, validation, and test sets.
        Works only when each file has an unique label, which must be given in input.
        
        Default = False
    labels: list or ArrayLike, optional
        A list or 1d ArrayLike objects with the label of each file listed 
        in the partition table. Must be given if stratification is set to True
        Indeces of labels must match row indeces in the partition table, i.e. 
        label1 -> row1, label2 -> row2, etc.
        
        Default = None
    dataset_id_extractor: function, optional
        A custom function to be used to extract the dataset ID from the file name. 
        It must accept only one argument, which is the file name (not the full path, 
        only the file name).
        
        Default = None
    subject_id_extractor: function, optional
        A custom function to be used to extract the subject ID from the file name. 
        It must accept only one argument, which is the file name (not the full path, 
        only the file name).
        
        Default = None
    split_tolerance: float, optional
        Argument for ``get_subarray_closest_sum`` function. 
        Set the maximum accepted tolerance between the given split ratio
        and the one got with the obtained subset. Must be a number in [0,1]
        
        Default = 0.01
    perseverance: int, optional
        Argument for ``get_subarray_closest_sum`` function. Set the maximum number 
        of tries before stop searching for a split whose ratio is in the range 
        [target_ratio - tolerance, target_ratio + tolerance]
        
        Default = 1000
    save : bool, optional
        Whether to save the resulted DataFrame as a .csv file or not.
        
        Default = False
    save_path: str, optional
        A custom path to be used instead of the current working directory. 
        It is the string given to the ``pandas.DataFrame.to_csv()`` method.
        
        Default = None
    seed: int, optional
        An integer defining the seed to use. Set it to reproduce split results.

        Default = None
                 
    Returns
    -------
    EEGSplitKfold : pd.DataFrame
       Pandas DataFrame where the first column has the EEG file names, while the
       others will have the assigned split for each CV split. Each split is included 
       in a column with the name "split_k" with k from 1 to the given Kfold argument.
       Each split will assign the following labels to a file:

           1. -1 : the file is excluded 
           2. 0  : the file is included in the training set
           3. 1  : the file is included in the validation set
           4. 2  : the file is included in the test set

    See Also
    --------
    getsplit : extract a specific split from the output dataframe.
    
    Warnings
    --------
    Some configurations may produce strange results. For example, if you want to do a 10 fold
    CV with a subject based split, but your dataset has only 5 subjects, the function will
    not throw an error, but some splits won't have a validation split.

    Example
    -------
    >>> import pickle
    >>> import pandas as pd
    >>> import selfeeg.dataloading as dl
    >>> import selfeeg.utils
    >>> labels = utils.create_dataset()
    >>> def loadEEG(path):
    ...     with open(path, 'rb') as handle:
    ...         EEG = pickle.load(handle)
    ...     x = EEG['data']
    ...     return x
    >>>  EEGlen = dl.GetEEGPartitionNumber('Simulated_EEG',freq=128, window=2, 
    ...                                    overlap=0.3, load_function=loadEEG )
    >>>  EEGsplit = dl.GetEEGSplitTableKfold(EEGlen, seed=1234) #default 60/20 train/test split
    >>>  dl.check_split(EEGlen,dl.getsplit(EEGsplit,1)) # will return 0.72/0.08/0.2 ratios
            
    """ 
    if kfold<2:
        raise ValueError('kfold must be greater than or equal to 2. '
                         'If you don\'t need multiple splits use the GetEEGSplitTable function'
                        )
    kfold=int(kfold)
    if (test_ratio is None) and (test_data_id is None):
        test_ratio = 0.

    if seed is not None:
        random.seed(seed)
        
    # FIRST STEP: Create test set or exclude data if necessary
    # the result of this function call will be an initialization of the split table
    # if no data need to be excluded or placed in a test set, the split_set column will 
    # simply have all zeros.
    EEGsplit = GetEEGSplitTable(partition_table=partition_table,
                                test_ratio= test_ratio,
                                val_ratio= 0.0,
                                test_split_mode=test_split_mode,
                                val_split_mode= val_split_mode,
                                exclude_data_id=exclude_data_id,
                                test_data_id=test_data_id,
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
    # to perform CV it is necessary to perform multiple train/validation split. Each time the data already 
    # included in the test or any validation set will be excluded and the val_ratio is scaled according to 
    # the remaining portions of the data
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
        # update list of files not assigned to any validation set
        idxSplit= EEGsplit.index[(EEGsplit['split_'+str(i+1)] ==1)]
        idx2assign = np.setdiff1d(idx2assign, idxSplit)

    # assign last fold and delete useless initial split column
    EEGsplit.iloc[idx2assign,-1]=1
    EEGsplit.drop(columns='split_set', inplace=True)

    # save block
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

def getsplit(split_table: pd.DataFrame, split: int) -> pd.DataFrame:
    """
    ``getsplit`` extracts a split from the output of the ``GetEEGSplitTableKfold`` .
    It also changes column names in order to make them equals to the output DataFrame
    of the ``GetEEGSplitTable`` function. 

    Parameters
    ----------
    split_table: pd.DataFrame
        The table with all the Cross Validation Splits. It is the output of the 
        ``GetEEGSplitTableKfold`` function. Such table has a first column named "file_name",
        where the EEG file names are placed, and other sets of columns named "split_k",
        where the k-th is placed.
    split: int
        An integer indicating the specific split to extract. Note that the output of the 
        ``GetEEGSplitTableKfold`` function has split starting from 1, i.e. "split_0" doesn't 
        exist.

    Returns
    -------
    new_table: pd.DataFrame
        A 2 columns DataFrame with same format as GetEEGSplitTable, i.e. first column with file
        names and second their split ID.

    Example
    -------
    >>> import pickle
    >>> import selfeeg.dataloading as dl
    >>> import selfeeg.utils
    >>> labels = utils.create_dataset()
    >>> def loadEEG(path):
    ...     with open(path, 'rb') as handle:
    ...         EEG = pickle.load(handle)
    ...     x = EEG['data']
    ...     return x
    >>>  EEGlen = dl.GetEEGPartitionNumber('Simulated_EEG',freq=128, window=2, 
    ...                                    overlap=0.3, load_function=loadEEG )
    >>>  EEGsplit = dl.GetEEGSplitTableKfold(EEGlen) #default 60/20 train/test split
    >>>  EEGsplit1 = dl.getsplit(EEGsplit,1) #will extract first CV split
    >>>  EEGsplit1.head()

    """
    split_str = 'split_'+str(int(split))
    new_table = split_table.loc[:,('file_name',split_str)]
    new_table.rename(columns={"file_name": "file_name", split_str: "split_set"}, inplace=True)
    return new_table

def check_split(EEGlen: pd.DataFrame, 
                EEGsplit: pd.DataFrame, 
                Labels=None, 
                return_ratio=False,
                verbose = True,
               ) -> Optional[dict]:
    """
    ``check_split`` calculate and print split ratios to check if the split 
    has been performed correctly.

    Parameters
    ----------
    EEGlen: pd.DataFrame
        The output of the ``GetEEGPartitionNumber`` function.
    EEGsplit: pd.DataFrame
        The output of the ``GetEEGSplitTable`` function. If you have used the 
        ``GetEEGSplitTableKfold`` function, make sure to get a specific split by
        calling the ``getsplit`` function.
    Labels: ArrayLike, optional
        A list or 1d array like objects with the label of each file listed in the 
        partition table. It is the same object given to the called split function.

        Default = None
    return_ratio: bool, otional
        Whether to return the calculated ratio in a dictionary or simply print them.

        Default = False
    verbose: bool, optional
        Wheter to generate a summary print of the calculated ratios or not.

        Default = True

    Returns
    -------
    ratios: dict, optional
        A dictionary with the calculated ratios. If labels were given, a numpy array

    Example
    -------
    >>> import pickle
    >>> import selfeeg.dataloading as dl
    >>> import selfeeg.utils
    >>> labels = utils.create_dataset()
    >>> def loadEEG(path):
    ...     with open(path, 'rb') as handle:
    ...         EEG = pickle.load(handle)
    ...     x = EEG['data']
    ...     return x
    >>>  EEGlen = dl.GetEEGPartitionNumber('Simulated_EEG',freq=128, window=2, 
    ...                                    overlap=0.3, load_function=loadEEG )
    >>>  EEGsplit = dl.GetEEGSplitTable(EEGlen) #default 60/20/20 ratio
    >>>  ratios = dl.check_split(EEGlen, EEGsplit, return_ratio=True) # will return 0.6/0.2/0.2
    >>>  print(ratios['train_ratio'], ratios['val_ratio'], ratios['test_ratio'])

    """
    # Check split ratio
    # simply the ratio between the sum of all samples with a specific label set 
    # and the sum of all samples with label different from -1
    total_list=EEGsplit[EEGsplit['split_set']!=-1].index.tolist()
    total = EEGlen.iloc[total_list]['N_samples'].sum()
    train_list = EEGsplit[EEGsplit['split_set']==0].index.tolist()
    train_ratio = EEGlen.iloc[train_list]['N_samples'].sum()/total
    val_list = EEGsplit[EEGsplit['split_set']==1].index.tolist()
    val_ratio = EEGlen.iloc[val_list]['N_samples'].sum()/total
    test_list = EEGsplit[EEGsplit['split_set']==2].index.tolist()
    test_ratio = EEGlen.iloc[test_list]['N_samples'].sum()/total

    if verbose:
        print(f"\ntrain ratio:      {train_ratio:.2f}")
        print(f"validation ratio: {val_ratio:.2f}")
        print(f"test ratio:       {test_ratio:.2f}")
    ratios = {'train_ratio': train_ratio, 'val_ratio': val_ratio,
              'test_ratio': test_ratio, 'class_ratio': None}
    
    # Check class ratio
    # similar to the previous one but the ratios are calculated with respect to 
    # the subset sizes (train test validation sets)
    if Labels is not None:
        Labels = np.array(Labels)
        if len(Labels.shape)!=1:
            raise ValueError('Labels must be a 1d array or a list')
        lab_unique = np.unique(Labels)
        EEGlen2 = EEGlen.copy() # copy to avoid strange behaviours in the given dataframes
        EEGlen2['split_set']=EEGsplit['split_set']
        EEGlen2['Labels'] = Labels
        tottrain = EEGlen2.iloc[train_list]['N_samples'].sum()
        totval = EEGlen2.iloc[val_list]['N_samples'].sum()
        tottest = EEGlen2.iloc[test_list]['N_samples'].sum()
        class_ratio = np.zeros((3,len(lab_unique)))
        # iterate through train/validation/test sets
        for i in range(3):
            # iterate through each label
            for k in range(len(lab_unique)):
                if i==0:
                    train_k = EEGlen2.loc[ ((EEGlen2['split_set']==0) & 
                                           (EEGlen2['Labels']==lab_unique[k])),'N_samples'].sum()
                    class_ratio[i,k] = train_k/tottrain
                elif i==1:
                    val_k = EEGlen2.loc[ ((EEGlen2['split_set']==1) &
                                           (EEGlen2['Labels']==lab_unique[k])),'N_samples'].sum()
                    class_ratio[i,k] = val_k/totval
                else:
                    test_k = EEGlen2.loc[ ((EEGlen2['split_set']==2) & 
                                           (EEGlen2['Labels']==lab_unique[k])),'N_samples'].sum()
                    class_ratio[i,k] = test_k/tottest
        # print results
        if verbose:
            print(f"\ntrain labels ratio:", 
                  *[f"{lab_unique[k]}={class_ratio[0,k]:.3f}, " for k in range(len(lab_unique))])
            print(f"val   labels ratio:", 
                  *[f"{lab_unique[k]}={class_ratio[1,k]:.3f}, " for k in range(len(lab_unique))])
            print(f"test  labels ratio:", 
                  *[f"{lab_unique[k]}={class_ratio[2,k]:.3f}, " for k in range(len(lab_unique))])
            print('')
        ratios['class_ratio']=class_ratio

    # return calculated ratios if necessary
    if return_ratio:
        return ratios
    else:
        return None


class EEGDataset(Dataset):
    """
    ``EEGDataset`` is a custom pytorch.Dataset class designed to manage different
    loading configuration, both for pretraining and fine tuning.
    Its main functionalities reside in the ability to accepts different ways to 
    load, transform and extract optional labels from the data without preallocate
    the entire dataset, which is especially useful in SSL experiments, where multiple 
    and large datasets are used.
    To further check how to use this class see the introductory notebook provided 
    in the documentation.
    
    Parameters
    ----------
    EEGlen : DataFrame
        DataFrame with the number of partition per EEG record. 
        Must be the output of the ``GetEEGPartitionNumber()`` function.
    EEGsplit : DataFrame 
        DataFrame with the train/test split info. Must be the output of the
        ``GetEEGSplitTable()`` or a split extracted from the ``GetEEGSplitTableKfold`` function
        output with the ``getsplit`` function.
    EEGpartition_spec : list
        3-element list with the input gave to ``GetEEGPartitionNumber()`` in 
        [sampling_rate, window_length, overlap_percentage] format.
    mode: string, optional
        If the dataset is intended for train, test or validation. 
        It accept only the following strings: 'train','test','validation'.
        
        Default = 'train'
    supervised: bool, optional
        Whether the class ``__getItem__()`` method must return a label or not. Must be
        set to True during fine-tuning.
        
        Default = False
    load_function : 'function', optional
        A custom EEG file loading function. It will be used instead of the default: 
        
        ``loadmat(ii, simplify_cells=True)['DATA_STRUCT']['data']`` 
        
        which is the default output format for files preprocessed with the BIDSalign library.
        The function must take only one required argument, which is the full path to the 
        EEG file (e.g. the function will be called in this way: 
        load_function(fullpath, optional_arguments) )

        The function can output one or two arguments where the first must be the EEG file 
        and the second (if there is one) is its label.
        Note that the assumed number of outputs is based on the parameter label_on_load. 
        So if the function will return only the EEG remember to set label_on_load on False.
        Note also that this function must load the EEGs in the same way as during 
        ``GetEEGPartitionNumber`` call.
        
        Default = None
    transform_function : 'function', optional
        A custom transformation to be applied after the EEG is loaded. 
        Might be useful if there are portions of the signal to cut (usually the initial or the
        final). The function must take only one required argument, which is the loaded EEG 
        file to transform (e.g. the function will be called in this way: 
        transform_function(EEG, optional_arguments) ).
        Note that this function must transform the EEGs in the same way as during 
        ``GetEEGPartitionNumber`` call.
        
        Default = None
    label_function : 'function', optional
        A custom transformation to be applied for the label extraction.
        Might be useful for the fine-tuning phase. 
        Considering that an EEG file can have single or multiple labels the function 
        will be called with 2 required arguments:
            
            1. full path to the EEG file 
            2. list with all indeces necessary to identify the extracted partition 
               (if EEG is a 2-D array the list will have only the starting and ending 
               indeces of the slice of the last axis, if the EEG is N-D the list will 
               also add all the other indeces from the first to the second to last axis)
               
        e.g. the function will be called in this way: 
        
        ``label_function(full_path, [*first_axis_idx, start, end], optional arguments)``
        
        It is strongly suggested to save EEG labels in a separate file in order to avoid 
        loading every time the entire EEG file which is the purpose of this entire 
        module implementation.
        
        Default = None
    optional_load_fun_args: list or dict, optional
        Optional arguments to give to the custom loading function. Can be a list or a dict.
        
        Default = None
    optional_transform_fun_args: list or dict, optional
        Optional arguments to give to the EEG transformation function. Can be a list or a dict.
        
        Default = None
    optional_label_fun_args: list or dict, optional
        Optional arguments to give to the EEG transformation function. Can be a list or a dict.
        
        Default = None
    label_on_load: bool, optional
        Whether the custom loading function will also load a label associated to the eeg file.
        
        Default = True
    label_key: str or list of str, optional
        A single or set of dictionary keys given as list of strings, used to access 
        a specific label if multiple were loaded. Might be useful if the loading function
        will return a dictionary of labels associated to the file, 
        for example when you have a set of patient info but you want to use only a 
        specific one.
        
        Default = None
    default_dtype: torch.dtype
        The dtype to use when converting loaded EEG to torch tensors. It is suggested
        to change the default float32 only if there are specific requirements since 
        float32 are faster on GPU devices.

    Example
    -------
    >>> import pickle
    >>> import selfeeg.dataloading as dl
    >>> import selfeeg.utils
    >>> labels = utils.create_dataset()
    >>> def loadEEG(path):
    ...     with open(path, 'rb') as handle:
    ...         EEG = pickle.load(handle)
    ...     x = EEG['data']
    ...     return x
    >>>  EEGlen = dl.GetEEGPartitionNumber('Simulated_EEG',freq=128, window=2, 
    ...                                    overlap=0.3, load_function=loadEEG )
    >>>  EEGsplit = dl.GetEEGSplitTable(EEGlen, seed=1234) #default 60/20/20 ratio
    >>>  TrainSet = dl.EEGDataset(EEGlen,EEGsplit,[128,2,0.3],load_function=loadEEG)
    >>>  print(len(TrainSet))
    >>>  print(TrainSet.__getitem__(10).shape) # will return torch.Size([8, 256])
    >>>  print(TrainSet.file_path) # will return 'Simulated_EEG/1_11_3_1.pickle'


    This image summarizes how to set up the main arguments of the EEGDataset class:

    .. image:: ../../Images/DatasetClassScheme.jpeg
        :align: center
            
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
        # saving a copy of the current loaded EEG file can use lots of memory 
        # if EEGs are pretty large
        self.currEEG = None
        self.dimEEG = 0
        self.dimEEGprod = None
        self.file_path=None
        self.minIdx = -1
        self.maxIdx = -1
        self.label_info=None
        self.label_info_keys=None
        
    
    def __len__(self):
        """
        :meta private:
        
        """
        return self.DatasetSize
    
    def __getitem__(self,index):
        """
        :meta private:
        
        """

        # Check if a new EEG file must be loaded. If so, a new EEG file is loaded,
        # transformed (if necessary) and all loading attributes are
        # updated according to the new file
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

            # store dimensionality of EEG files (some datasets stored 3D tensors, unfortunately)
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
    ``EEGsampler`` is a custom pytorch Sampler designed to efficiently reduce the file 
    loading operations when combined with the ``EEGDataset`` class. To do that, it exploits the 
    parallelization properties of the pytorch Dataloader and the buffer of EEGDataset.
    To further check how the custom iterator is created see image reported below and check 
    the introductory notebook provided in the documentation.

    Parameters
    ----------
    data_source: EEGDataset
        The instance of the ``EEGdataset`` class provided in this module.
    BatchSize: int, optional
        The batch size used during training. It will be used to create the 
        custom iterator (not linear).
        
        Default = 1
    Workers: Int, optional
        The number of workers used by the Dataloader. Must be the same as the argument
        workers in the Dataloader classs. It will be used to create 
        the custom iterator (not linear).

        Default = 0
    Mode: int, optional
        The mode to be used to create the iterator. It can be 0 or 1, where:
            
            - 0 = the iterator is a simple linear iterator (range(0,len(dataset))
            - 1 = the indeces are first shuffled at the inter-file level, 
              then at the intra-file level; ultimately all indeces are rearranged based on the 
              batch size and the number of workers in order to reduce the number of times a new 
              EEG is loaded. The iterator can be seen as a good compromise between 
              batch heterogeneity and batch creation speed

        Default = 1
    Keep_only_ratio: float, optional
        Whether to preserve only a given ratio of samples for each files in the given EEGdataset.
        It can be used to reduce the training time of each epoch while being sure to feed at
        least a portion of each EEG file in your dataset. If not given, all samples of the
        given dataset will be used. Note that the sample indices will be chosen after the 
        intra-file level shuffle so to avoid selecting the same initial portions of the EEG
        record.

        Default = 1

    Example
    -------
    >>> import pickle
    >>> import random
    >>> import selfeeg.dataloading as dl
    >>> import selfeeg.utils
    >>> labels = utils.create_dataset()
    >>> def loadEEG(path):
    ...     with open(path, 'rb') as handle:
    ...         EEG = pickle.load(handle)
    ...     x = EEG['data']
    ...     return x
    >>> random.seed(1234)
    >>> EEGlen = dl.GetEEGPartitionNumber('Simulated_EEG',freq=128, window=2, 
    ...                                   overlap=0.3, load_function=loadEEG )
    >>> EEGsplit = dl.GetEEGSplitTable(EEGlen, seed=1234) #default 60/20/20 ratio
    >>> TrainSet = dl.EEGDataset(EEGlen,EEGsplit,[128,2,0.3],load_function=loadEEG)
    >>> smplr = EEGsampler(TrainSet, 16, 8)
    >>> print([i for i in a][:8]) # will return [599, 1661, 1354, 1942, 1907, 495, 489, 1013]


    This image summarizes how the custom sampler iterator is created:

    .. image:: ../../Images/sampler_example.png
        :align: center
    
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
            raise ValueError('supported modes are 0 (linear sampler) '
                             'and 1 (custom randomization)')
        else:
            self.Mode= Mode
        if Keep_only_ratio>1 or Keep_only_ratio<=0:
            raise ValueError('Keep_only_ratio must be in (0,1]')
        else:
            self.Keep_only_ratio = Keep_only_ratio
        self.shrink_data=True if Keep_only_ratio<1 else False
        
    
    
    def __len__(self):
        """
        :meta private:
        
        """
        return len(self.data_source)
    
    def __iter__(self):
        """
        Return an iterator where subject are passed sequentially for 
        each worker but the samples of each subjects are shuffled.
        
        :meta private:
        
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


