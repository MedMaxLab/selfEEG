import os
import math
import copy
import datetime
import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..losses import losses as Loss
from ..dataloading import EEGsampler
from typing import Any, Callable, Iterable, TypeVar, Generic, Sequence, List, Optional, Union

__all__ = ['EarlyStopping', 'SSL_Base', 'SimCLR', 'SimSiam', 'MoCo', 'BYOL', 'Barlow_Twins', 'VICReg']

def Default_augmentation(x):
    """
    simple default augmentation used when non data augmenter is given in SSL 
    fit methods. It's just a programming choice to avoid putting the augmenter 
    as non optional parameter. No justification 
    for the choice of random flip + random noise. 
    Just that it can be written in few lines of code.
    
    :meta private:
    
    """
    if not(isinstance(x, torch.Tensor)):
        x=torch.Tensor(x)
    x = x*(random.getrandbits(1))
    std = torch.std(x)
    noise = std * torch.randn(*x.shape, device=x.device)
    x_noise = x + noise 
    return x_noise

def evaluateLoss( loss_fun: 'function', 
                  arguments, 
                  loss_arg: Union[list, dict]=None
                ) -> 'loss_fun output':
    """
    ``evaluateLoss`` evaluate a custom loss function using `arguments` 
    as required arguments and loss_arg as optional ones. It is simply
    the ``SSL_Base's evaluate_loss`` method exported as a function.

    Parameters
    ----------
    loss_fun: function
        the custom loss function. It can be any loss function which 
        accepts as input the model's prediction (or predictions) and
        true labels as required argument and loss_args as optional arguments.
        Note that for the ``fine_tune`` method the number of required
        arguments must be 2, i.e. the model's prediction and true labels.
    arguments: torch.Tensor or list[torch.Tensors]
        the required arguments. Based on the way this function is used 
        in a training pipeline it can be a single or multiple tensors.
    loss_arg: Union[list, dict], optional
        The optional arguments to pass to the function. it can be a list
        or a dict

        Default = None

    Returns
    -------
    loss: 'loss_fun output'
        the output of the given loss function. It is expected to be 
        a torch.Tensor
    
    """
    if isinstance(arguments, list):
        if loss_arg==None or loss_arg==[]:
            loss=loss_fun( *arguments )
        elif isinstance(loss_arg, dict):
            loss=loss_fun(*arguments,**loss_arg)
        else:
            loss=loss_fun(*arguments,*loss_arg)
    else:
        if loss_arg==None or loss_arg==[]:
            loss=loss_fun(arguments)
        elif isinstance(loss_arg, dict):
            loss=loss_fun(arguments,**loss_arg)
        else:
            loss=loss_fun(arguments,*loss_arg)
    return loss

def fine_tune(model: nn.Module,
              train_dataloader: torch.utils.data.DataLoader,
              epochs=1,
              optimizer=None,
              augmenter=None,
              loss_func: 'function'= None, 
              loss_args: list or dict=[],
              label_encoder: 'function' = None,
              lr_scheduler=None,
              EarlyStopper=None,
              validation_dataloader: torch.utils.data.DataLoader=None,
              verbose=True,
              device: str or torch.device=None,
              return_loss_info: bool=False
             ) -> Optional[dict]:
    """
    ``fine_tune`` is a custom fit function designed to perform 
    fine tuning on a given model with the given dataloader.

    Parameters
    ----------
    model: nn.Module
        the pytorch model to fine tune. Must be a nn.Module.
    train_dataloader: Dataloader
        the pytorch Dataloader used to get the training batches. It
        must return a batch as a tuple (X, Y), with X the feature tensor
        and Y the label tensor
    epochs: int, optional
        The number of training epochs. Must be an integer bigger than 0.

        Default = 1
    optimizer: torch Optimizer, optional
        The optimizer used for weight's update. It can be any optimizer
        provided in the torch.optim module. If not given Adam with default
        parameters will be instantiated.

        Default = torch.optim.Adam
    augmenter: function, optional
        Any function (or callable object) used to perform data augmentation
        on the batch. It is highly suggested to resort to the augmentation
        module, which implements different data augmentation function and 
        classes to combine them. Note that data augmentation is not performed
        on the validation set, since its goal is to increase the size of the 
        training set and to get more different samples.

        Default = None
    loss_func: function
        the custom loss function. It can be any loss function which 
        accepts as input the model's prediction and the true labels 
        as required arguments and loss_args as optional arguments.
    loss_args: Union[list, dict], optional
        The optional arguments to pass to the function. it can be a list
        or a dict.

        Default = None
    label_encoder: function, optional
        A custom function used to encode the returned Dataloaders true labels.
        If None, the Dataloader's true label is used directly.

        Default = None
    lr_scheduler: torch Scheduler
        A pytorch learning rate scheduler used to update the learning rate 
        during the fine-tuning.

        Default = None
    EarlyStopper: EarlyStopping, optional
        An instance of the provided EarlyStopping class.

        Default = None
    validation_dataloader: Dataloader, optional
        the pytorch Dataloader used to get the validation batches. It
        must return a batch as a tuple (X, Y), with X the feature tensor
        and Y the label tensor. If not given, no validation loss will be 
        calculated

        Default = None
    verbose: bool, optional
        Whether to print a progression bar or not.

        Default = None
    device: torch.device or str, optional
        The devide to use for fine-tuning. If given as a string it will
        be converted in a torch.device instance. If not given, 'cpu' device
        will be used.
        
        Device = None
    return_loss_info: bool, optional
        Whether to return the calculated training validation losses at 
        each epoch. 

        Default = False

    Returns
    -------
    loss_info: dict, optional
        A dictionary with keys the epoch number (as integer) and values
        a two element list with the average epoch's training and validation
        loss.
        
    Note
    ----
    If an EarlyStopping instance is given with monitoring loss set to 
    validation loss, but no validation dataloader is given, monitoring
    loss will be automatically set to training loss.
    
    """
    
    if device is None:
        device=torch.device('cpu')
    else:
        if isinstance(device, str):
            device=torch.device(device.lower())
        elif isinstance(device, torch.device):
            pass
        else:
            raise ValueError('device must be a string or a torch.device instance')
    model.to(device=device)
    
    if not( isinstance(train_dataloader, torch.utils.data.DataLoader)):
        raise ValueError('Current implementation accept only training data'
                         ' as a pytorch DataLoader')
    if not(isinstance(epochs, int)):
        epochs= int(epochs)
    if epochs<1:
        raise ValueError('epochs must be bigger than 1')
    if optimizer is None:
        optimizer=torch.optim.Adam(self.parameters())
    if loss_func is None:
        raise ValueError('loss function not given') 
    if not( isinstance(loss_args,list) or isinstance(loss_args,dict)):
        raise ValueError('loss_args must be a list or a dict with '
                         'all optional arguments of the loss function')
    
    perform_validation=False
    if validation_dataloader!=None:
        if not( isinstance(validation_dataloader, torch.utils.data.DataLoader)):
            raise ValueError('Current implementation accept only'
                             ' validation data as a pytorch DataLoader')
        else:
            perform_validation=True
    if EarlyStopper is not None:
        if EarlyStopper.monitored=='validation' and not(perform_validation):
            print('Early stopper monitoring is set to validation loss'
                  ', but not validation data are given. '
                  'Internally changing monitoring to training loss')
            EarlyStopper.monitored = 'train'

    loss_info={i: [None, None] for i in range(epochs)}
    N_train = len(train_dataloader)
    N_val = 0 if validation_dataloader is None else len(validation_dataloader)
    for epoch in range(epochs):
        print(f'epoch [{epoch+1:6>}/{epochs:6>}]')
        
        train_loss=0
        val_loss=0
        train_loss_tot=0
        val_loss_tot=0
        
        if not(model.training):
            model.train()
        with tqdm.tqdm(total=N_train+N_val, ncols=100, 
                       bar_format='{desc}{percentage:3.0f}%|{bar:15}| {n_fmt}/{total_fmt}'
                       ' [{rate_fmt}{postfix}]',
                       disable=not(verbose), unit=' Batch') as pbar:
            for batch_idx, (X, Ytrue) in enumerate(train_dataloader):
                
                optimizer.zero_grad()
                if X.device.type!=device.type:
                    X = X.to(device=device)
                if augmenter is not None:
                    X = augmenter(X)
                if label_encoder is not None:
                    Ytrue= label_encoder(Ytrue)
                if Ytrue.device.type!=device.type:
                    Ytrue = Ytrue.to(device=device)                
                
                Yhat = model(X)
                train_loss = evaluateLoss( loss_func, [Yhat , Ytrue], loss_args )
    
                train_loss.backward()
                optimizer.step()
                train_loss_tot += train_loss.item()
                # verbose print
                if verbose:
                    pbar.set_description(f" train {batch_idx+1:8<}/{len(train_dataloader):8>}")
                    pbar.set_postfix_str(f"train_loss={train_loss_tot/(batch_idx+1):.5f}, val_loss={val_loss_tot:.5f}")
                    pbar.update()        
            train_loss_tot /= (batch_idx+1)
    
            if lr_scheduler!=None:
                lr_scheduler.step()
            
            # Perform validation if validation dataloader were given
            if perform_validation:
                model.eval()
                with torch.no_grad():
                    val_loss=0
                    for batch_idx, (X, Ytrue) in enumerate(validation_dataloader):
                        
                        if X.device.type!=device.type:
                            X = X.to(device=device)
                        if label_encoder != None:
                            Ytrue= label_encoder(Ytrue)
                        if Ytrue.device.type!=device.type:
                            Ytrue = Ytrue.to(device=device) 
                                    
                        Yhat = model(X)
                        val_loss = evaluateLoss( loss_func, [Yhat, Ytrue], loss_args )
                        val_loss_tot += val_loss.item()
                        if verbose:
                            pbar.set_description(f"   val {batch_idx+1:8<}/{len(validation_dataloader):8>}")
                            pbar.set_postfix_str(f"train_loss={train_loss_tot:.5f}, val_loss={val_loss_tot/(batch_idx+1):.5f}")
                            pbar.update()
                    
                    val_loss_tot /= (batch_idx+1)
                    
                    
        # Deal with earlystopper if given
        if EarlyStopper!=None:
            updated_mdl=False
            curr_monitored = val_loss_tot if EarlyStopper.monitored=='validation' else train_loss_tot
            EarlyStopper.early_stop(curr_monitored)
            if EarlyStopper.record_best_weights:
                if EarlyStopper.best_loss==curr_monitored:
                    EarlyStopper.rec_best_weights(model)
                    updated_mdl=True
            if EarlyStopper():
                print('no improvement after {} epochs. Training stopped.'.format(
                    EarlyStopper.patience))
                if EarlyStopper.record_best_weights and not(updated_mdl):
                    EarlyStopper.restore_best_weights(model)
                if return_loss_info:
                    return loss_info
                else:
                    return
        
        if return_loss_info:
            loss_info[epoch]=[train_loss_tot, val_loss_tot]        
    if return_loss_info:
        return loss_info


class EarlyStopping:
    """
    Simple implementation of an early stopping class for pytorch.
    It can monitor the validation or the training loss (no other metrics
    are currently supported).
    
    Some arguments are similar to Keras EarlyStopping class [early]_ .
    If you want to use other implemented functionalities take a look at 
    PyTorch Ignite [ign]_ .

    Parameters
    ----------
    patience: int, optional
        The number of epochs to wait before stopping the training. Can
        be any positive integer.

        Default = 5
    min_delta: float, optional
        The minimum difference between the current best loss and the 
        calculated one to consider as an improvement.

        Default = 1e-9
    improvement: str, optional
        Whether to consider an increase or decrease in the best loss
        as an improvement. Accepted strings are:

            - ['d','dec','decrease'] for decrease
            - ['i','inc','increase'] for increase

        Default = "decrease"
    monitored: str, optional
        Whether to monitor the training or validation loss. This 
        attribute is used in the ``fine_tuning`` function or 
        others class ``fit`` methods to check which calculated loss 
        must be given. Accepted values are "increase" or "decrease"

        Default = "decrease"
    record_best_weights: bool, optional
        Whether to record the best weights after every new best loss
        is reached or not. It will be used to restore such weights
        if the training is stopped.

        Default = True

    References
    ----------
    .. [early] https://keras.io/api/callbacks/early_stopping/ 
    .. [ign] https://pytorch.org/ignite/
    
    """
    def __init__(self, 
                 patience: int=5, 
                 min_delta: float=1e-9,
                 improvement: str='decrease',
                 monitored: str='validation',
                 record_best_weights: bool=True,
                ):
        
        if patience < 0:
            self.patience = 0
        else:
            self.patience = int(patience)
        
        if isinstance(monitored, str) and (monitored in ['validation', 'train']):
            self.monitored = monitored.lower()
        else:
            raise ValueError('supported monitoring modes are train or validation')
                            
        if min_delta<0:
            msgErr='min_delta must be >= 0. '
            msgErr += 'Use improvement to set if decrease or increase in the loss must be considered'
            raise ValueError(msgErr)
        else:
            self.min_delta = min_delta
        
        if improvement.lower() not in ['d','i','dec','inc','decrease','increase']:
            msgErr= 'got ' + str(improvement) + ' as improvement argument. Accepted arguments are '
            msgErr += 'd, dec or decrease for decrease; i, inc or increase for increase'
            raise ValueError(msgErr)
        else: 
            if improvement.lower() in ['d','dec','decrease']:
                self.improvement= 'decrease'
            else:
                self.improvement= 'increase'
            
        self.record_best_weights = record_best_weights
        self.best_loss = 1e12 if improvement.lower()=='decrease' else -1*1e12
        
        self.best_model=None
        self.counter = 0
        self.earlystop = False

    def __call__(self):
        """
        :meta private:
        """
        return self.earlystop
    
    def early_stop(self, loss, count_add=1):
        """
        Method used to updated the counter and the best loss

        Parameters
        ----------
        loss: float
            The calculated loss
        count_add: int, optional
            The number to add to the counter. It can be useful if early stopping
            checks will not be performed after each epoch
        
        """
        
        # The function can be compressed with a big if. 
        #This expansion is faster and better understandable
        if self.improvement=='decrease':
            #Check if current loss is better than recorded best loss
            if loss < (self.best_loss - self.min_delta):
                self.best_loss = loss
                self.counter = 0  # During train if self.counter==0 record_best_weights will be called
            else:
                self.counter += count_add
                if self.counter >= self.patience:
                    self.earlystop=True
                     
        elif self.improvement=='increase': # mirror with increase option
            if loss > (self.best_loss + self.min_delta):
                self.best_loss = loss
                self.counter = 0    
            else:
                self.counter += count_add
                if self.counter >= self.patience:
                    self.earlystop=True
            
                    
    def rec_best_weights(self, model):
        """
        record model's best weights. The copy of the model is stored
        in the cpu device

        Parameters
        ----------
        model: nn.Module
            The model to record
        
        """
        self.best_model = copy.deepcopy(model).to(device='cpu').state_dict()
    
    def restore_best_weights(self, model):
        """
        restore model's best weights.

        Parameters
        ----------
        model: nn.Module
            The model to restore

        Warnings
        --------
        The model is moved to the cpu device before restoring weights.
        Remember to move again to the desired device if cpu is not 
        the selected one
        
        """
        model.to(device='cpu')
        model.load_state_dict(self.best_model)

    def reset_counter(self):
        """
        method to reset the counter and early stopping flag. 
        It might be useful if you want to further train 
        your model after the first training is stopped.
        
        """
        self.counter=0
        self.earlystop= False
        



class SSL_Base(nn.Module):
    """
    Baseline Self-Supervised Learning nn.Module. It is used as parent class 
    by the other implemented SSL methods.

    Parameters
    ----------
    encoder: nn.Module
        The encoder part of the module. It is the one you wish to pretrain and
        transfer to the new model
    
    """
    
    def __init__(self, encoder: nn.Module):
        super(SSL_Base, self).__init__()
        self.encoder = encoder
        
    def forward(self,x):
        """
        :meta private:
        
        """
        pass
    
        
    def evaluate_loss(self, 
                      loss_fun: 'function', 
                      arguments, 
                      loss_arg: Union[list, dict]=None
                     ) -> 'loss_fun output':
        """
        ``evaluate_loss`` evaluate a custom loss function using `arguments` 
        as required arguments and loss_arg as optional ones.
    
        Parameters
        ----------
        loss_fun: function
            the custom loss function. It can be any loss function which 
            accepts as input the model's prediction (or predictions) 
            as required argument and loss_args as optional arguments.
            Note that the number of required arguments can change based on the
            specific pretraining method used
        arguments: torch.Tensor or list[torch.Tensors]
            the required arguments. Based on the way this function is used 
            in a training pipeline it can be a single or multiple tensors.
        loss_arg: Union[list, dict], optional
            The optional arguments to pass to the function. it can be a list
            or a dict
    
            Default = None
    
        Returns
        -------
        loss: 'loss_fun output'
            the output of the given loss function. It is expected to be 
            a torch.Tensor
        
        """
        if isinstance(arguments, list):
            if loss_arg==None or loss_arg==[]:
                loss=loss_fun(*arguments)
            elif isinstance(loss_arg, dict):
                loss=loss_fun(*arguments,**loss_arg)
            else:
                loss=loss_fun(*arguments,*loss_arg)
        else:
            if loss_arg==None or loss_arg==[]:
                loss=loss_fun(arguments)
            elif isinstance(loss_arg, dict):
                loss=loss_fun(arguments,**loss_arg)
            else:
                loss=loss_fun(arguments,*loss_arg)
        return loss
        
    def get_encoder(self, device='cpu'):
        """
        return a copy of the encoder on the selected device.

        Parameters
        ----------
        device: torch.device or str, optional
            the pytorch device

            Default = 'cpu'
        
        """
        enc= copy.deepcopy(self.encoder).to(device=device)
        return enc
    
    def save_encoder(self, path: str=None):
        """
        a method for saving the pretrained encoder.

        Parameters
        ----------
        path: str, optional
            the saving path. it will be given to the ``torch.save()`` 
            method. If None is given, the encoder will be saved in a created
            SSL_encoders subdirectory. The name will contain the pretraining
            method used (e.g. SimCLR, MoCo etc) and the current time.

            Default = None
            
        """
        if path is None:
            path=os.getcwd()
            if os.path.isdir(path + '/SSL_encoders'):
                path += '/SSL_encoders'
            else:
                os.mkdir(path + '/SSL_encoders')
                path += '/SSL_encoders'
            
            if isinstance(self, SimSiam):
                sslName= '_SimSiam'
            elif isinstance(self, VICReg):
                sslName= '_VICreg'
            elif isinstance(self, Barlow_Twins):
                sslName= '_BarTw'
            elif isinstance(self, BYOL):
                sslName= '_BYOL'
            elif isinstance(self, MoCo):
                sslName= '_MoCo'
            elif isinstance(self, SimCLR):
                sslName= '_SimCLR'
            else:
                sslName= '_Custom'
            save_path= path+'/encoder_' + datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")+sslName+ '.pt'
        torch.save(self.encoder.state_dict(), save_path)
        



class SimCLR(SSL_Base):
    """
    Implementation of the SimCLR SSL method. To check
    how SimCLR works, read the following paper [NTXent1]_ .
    Official repository at [simgit1]_ .

    Parameters
    ----------
    encoder: nn.Module
        The encoder part of the module. It is the one you wish to pretrain and
        transfer to the new model
    projection_head: Union[list[int], nn.Module]
        The projection head to use. It can be an nn.Module or a list of ints.
        In case a list is given, a nn.Sequential module with Dense, BatchNorm
        and Relu will be automtically created. The list will be used to set 
        input and output dimension of each Dense Layer. For instance, if 
        [64, 128, 64] is given, two hidden layers will be created. The first
        with input 64 and output 128, the second with input 128 and output 64.

    Note
    ----
    BatchNorm is not applied to the last output layer due to findings in the
    most recent SSL works (see BYOL and SimSiam)

    Warnings
    --------
    This class will not check the compatibility of the encoder's output and
    the projection head's input. Make sure that they have the same size. 

    References
    ----------
    .. [NTXent1] Chen et al. A Simple Framework for Contrastive Learning of Visual
      Representations. (2020). https://doi.org/10.48550/arXiv.2002.05709
    .. [simgit1] To check the original tensorflow implementation visit the following repository:
      https://github.com/google-research/simclr (look at the function add_contrastive_loss 
      in objective.py)
      
    """
    
    def __init__(self, 
                 encoder: nn.Module, 
                 projection_head: Union[list[int], nn.Module],
                ):
        
        super(SimCLR,self).__init__(encoder)
        self.encoder = encoder
        
        if isinstance(projection_head, list):
            if len(projection_head)<2:
                raise ValueError('got a list with only one element')
            else:
                if all(isinstance(i,int) for i in projection_head):
                    DenseList=[]
                    for i in range(len(projection_head)-1):
                        DenseList.append(nn.Linear(projection_head[i],projection_head[i+1]))
                        # Batch Norm Not applied on output due to BYOL and SimSiam 
                        # choices, since those two are more recent SSL implementations
                        DenseList.append(nn.BatchNorm1d(num_features=projection_head[i+1]))
                        if i<(len(projection_head)-2):
                            DenseList.append(nn.ReLU())
                    self.projection_head= nn.Sequential(*DenseList)
                else:
                    raise ValueError('got a list with non integer values')
        else:    
            self.projection_head = projection_head
        
        
    def forward(self,x):
        """
        :meta private:
        
        """
        x   = self.encoder(x)
        emb = self.projection_head(x)
        return emb
    
    
    def fit(self,
            train_dataloader,
            epochs=1,
            optimizer=None,
            augmenter=None,
            loss_func: 'function'= None, 
            loss_args: list or dict=[],
            lr_scheduler=None,
            EarlyStopper=None,
            validation_dataloader=None,
            verbose=True,
            device: str or torch.device=None,
            cat_augmentations: bool=False,
            return_loss_info: bool=False
           ):
        """
        ``fit`` is a custom fit function designed to perform 
        pretraining on a given model with the given dataloader.
    
        Parameters
        ----------
        train_dataloader: Dataloader
            the pytorch Dataloader used to get the training batches. It
            must return a batch as a tuple (X, Y), with X the feature tensor
            and Y the label tensor
        epochs: int, optional
            The number of training epochs. Must be an integer bigger than 0.
    
            Default = 1
        optimizer: torch Optimizer, optional
            The optimizer used for weight's update. It can be any optimizer
            provided in the torch.optim module. If not given Adam with default
            parameters will be instantiated.
    
            Default = torch.optim.Adam
        augmenter: function, optional
            Any function (or callable object) used to perform data augmentation
            on the batch. It is highly suggested to resort to the augmentation
            module, which implements different data augmentation functions and 
            classes to combine them. If none is given a default augmentation with 
            random vertical flip + random noise is applied.
            Note that in this case data augmentation 
            is also performed on the validation set, since it's part of the 
            SSL algorithm.
    
            Default = None
        loss_func: function, optional
            the custom loss function. It can be any loss function which 
            accepts as input the model's prediction and the true labels 
            as required arguments and loss_args as optional arguments.
            If not given SimCLR loss will be automatically used.

            Default = None
        loss_args: Union[list, dict], optional
            The optional arguments to pass to the function. it can be a list
            or a dict.
    
            Default = None
        label_encoder: function, optional
            A custom function used to encode the returned Dataloaders true labels.
            If None, the Dataloader's true label is used directly.
    
            Default = None
        lr_scheduler: torch Scheduler
            A pytorch learning rate scheduler used to update the learning rate 
            during the fine-tuning.
    
            Default = None
        EarlyStopper: EarlyStopping, optional
            An instance of the provided EarlyStopping class.
    
            Default = None
        validation_dataloader: Dataloader, optional
            the pytorch Dataloader used to get the validation batches. It
            must return a batch as a tuple (X, Y), with X the feature tensor
            and Y the label tensor. If not given, no validation loss will be 
            calculated
    
            Default = None
        verbose: bool, optional
            Whether to print a progression bar or not.
    
            Default = None
        device: torch.device or str, optional
            The devide to use for fine-tuning. If given as a string it will
            be converted in a torch.device instance. If not given, 'cpu' device
            will be used.
            
            Device = None
        cat_augmentations: bool, optional
            Whether to calculate the loss on the cat version of the two
            projection's or not. It might affect some statistical layers.

            Default = False
        return_loss_info: bool, optional
            Whether to return the calculated training validation losses at 
            each epoch. 
    
            Default = False
    
        Returns
        -------
        loss_info: dict, optional
            A dictionary with keys the epoch number (as integer) and values
            a two element list with the average epoch's training and validation
            loss.
            
        Note
        ----
        If an EarlyStopping instance is given with monitoring loss set to 
        validation loss, but no validation dataloader is given, monitoring
        loss will be automatically set to training loss.
        
        """
        # Various check on input parameters. If some arguments weren't given
        if device is None:
            # If device is None cannot assume if the model is on gpu and so if to send the batch
            # on other device, so cpu will be used. If model is sent to another device set the 
            # device attribute with a proper string or torch device
            device=torch.device('cpu')
        else:
            if isinstance(device, str):
                device=torch.device(device.lower())
            elif isinstance(device, torch.device):
                pass
            else:
                raise ValueError('device must be a string or a torch.device instance')
        self.to(device=device)
        
        if not( isinstance(train_dataloader, torch.utils.data.DataLoader)):
            raise ValueError('Current implementation accept only training data'
                             ' as a pytorch DataLoader')
        if not(isinstance(epochs, int)):
            epochs= int(epochs)
        if epochs<1:
            raise ValueError('epochs must be bigger than 1')
        if optimizer is None:
            optimizer=torch.optim.Adam(self.parameters())
        if augmenter is None:
            print('augmenter not given. Using a basic one with with flip + random noise')
            augmenter=Default_augmentation
        if loss_func is None:
            print('Use base SimCLR loss')
            loss_func=Loss.SimCLR_loss   
        if not( isinstance(loss_args,list) or isinstance(loss_args,dict) or loss_args==None):
            raise ValueError('loss_args must be a list or a dict with all'
                             ' optional arguments of the loss function')
        perform_validation=False
        if validation_dataloader!=None:
            if not( isinstance(validation_dataloader, torch.utils.data.DataLoader)):
                raise ValueError('Current implementation accept only '
                                 'training data as a pytorch DataLoader')
            else:
                perform_validation=True
        
        if EarlyStopper is not None:
            if EarlyStopper.monitored=='validation' and not(perform_validation):
                print('Early stopper monitoring is set to validation loss'
                      ', but not validation data are given. '
                      'Internally changing monitoring to training loss')
                EarlyStopper.monitored = 'train'
    
        loss_info={i: [None, None] for i in range(epochs)}
        N_train = len(train_dataloader)
        if isinstance(train_dataloader.sampler, EEGsampler):
            if train_dataloader.sampler.Keep_only_ratio != 1:
                if train_dataloader.drop_last:
                    N_train = math.floor(sum(1 for _ in train_dataloader.sampler.__iter__())/(train_dataloader.batch_size))
                else:
                    N_train = math.ceil(sum(1 for _ in train_dataloader.sampler.__iter__())/(train_dataloader.batch_size))
                
        N_val = 0 if validation_dataloader is None else len(validation_dataloader)
        if isinstance(validation_dataloader.sampler, EEGsampler):
            if validation_dataloader.sampler.Keep_only_ratio != 1:
                if validation_dataloader.drop_last:
                    N_val = math.floor(sum(1 for _ in validation_dataloader.sampler.__iter__())/(validation_dataloader.batch_size))
                else:
                    N_val = math.ceil(sum(1 for _ in validation_dataloader.sampler.__iter__())/(validation_dataloader.batch_size))
        for epoch in range(epochs):
            print(f'epoch [{epoch+1:6>}/{epochs:6>}]')
        
            train_loss=0
            val_loss=0
            train_loss_tot=0
            val_loss_tot=0
            
            if not(self.training):
                self.train()

            with tqdm.tqdm(total=N_train+N_val, ncols=100, 
                       bar_format='{desc}{percentage:3.0f}%|{bar:15}| {n_fmt}/{total_fmt}'
                       ' [{rate_fmt}{postfix}]',
                       disable=not(verbose), unit=' Batch') as pbar:
                for batch_idx, X in enumerate(train_dataloader):
                    
                    optimizer.zero_grad()
    
                    if X.device.type!=device.type:
                        X = X.to(device=device)
    
                    data_aug1 = augmenter(X)
                    data_aug2 = augmenter(X)
                    
                    if cat_augmentations:
                        data_aug = torch.cat((data_aug1, data_aug2))
                        z = self(data_aug)
                        train_loss = self.evaluate_loss(loss_func, z, loss_args )
                    else:
                        z1 = self(data_aug1)
                        z2 = self(data_aug2)
                        train_loss = self.evaluate_loss(loss_func, torch.cat((z1,z2)), loss_args )
    
                    train_loss.backward()
                    optimizer.step()
                    train_loss_tot += train_loss.item()
                    # verbose print
                    if verbose:
                        pbar.set_description(f" train {batch_idx+1:8<}/{N_train:8>}")
                        pbar.set_postfix_str(f"train_loss={train_loss_tot/(batch_idx+1):.5f}, val_loss={val_loss_tot:.5f}")
                        pbar.update()        
                train_loss_tot /= (batch_idx+1)
    
                if lr_scheduler!=None:
                    lr_scheduler.step()
                
                # Perform validation if validation dataloader were given
                if perform_validation:
                    self.eval()
                    with torch.no_grad():
                        val_loss=0
                        val_loss_tot=0
                        for batch_idx, X in enumerate(validation_dataloader):
                            
                            if X.device.type!=device.type:
                                X = X.to(device=device)
    
                            data_aug = torch.cat((augmenter(X), augmenter(X)), dim=0 )
                            z = self(data_aug)
                            val_loss = self.evaluate_loss(loss_func, z, loss_args )
                            val_loss_tot += val_loss
                            if verbose:
                                pbar.set_description(f"   val {batch_idx+1:8<}/{N_val:8>}")
                                pbar.set_postfix_str(f"train_loss={train_loss_tot:.5f}, val_loss={val_loss_tot/(batch_idx+1):.5f}")
                                pbar.update()
                        
                        val_loss_tot /= (batch_idx+1)
                    
            # Deal with earlystopper if given
            if EarlyStopper!=None:
                updated_mdl=False
                curr_monitored = val_loss_tot if EarlyStopper.monitored=='validation' else train_loss_tot
                EarlyStopper.early_stop(curr_monitored)
                if EarlyStopper.record_best_weights:
                    if EarlyStopper.best_loss==curr_monitored:
                        EarlyStopper.rec_best_weights(self)
                        updated_mdl=True
                if EarlyStopper():
                    print('no improvement after {} epochs. Training stopped.'.format(
                        EarlyStopper.patience))
                    if EarlyStopper.record_best_weights and not(updated_mdl):
                        EarlyStopper.restore_best_weights(self)
                    if return_loss_info:
                        return loss_info
                    else:
                        return
            
            if return_loss_info:
                loss_info[epoch]=[train_loss_tot, val_loss_tot]        
        if return_loss_info:
            return loss_info
                    

    
    def test(self, 
             test_dataloader,
             augmenter=None,
             loss_func: 'function'= None, 
             loss_args: list or dict=[],
             verbose: bool=True,
             device: str=None
            ):
        """
        a method to evaluate the loss on a test dataloader.
        Parameters are the same as described in the fit method, aside for 
        those related to model training which are removed.
        
        It is rare to evaluate the pretraing loss function on a test set.
        Nevertheless this function provides a way to do that. 
        An example of usage could be to assess the quality of the learned 
        features on the fine-tuning dataset.
        
        """
        if device==None:
            # If device is None cannot assume if the model is on gpu and so if to send the batch
            # on other device, so cpu will be used. If model is sent to another device set the 
            # device attribute with a proper string or torch device
            device=torch.device('cpu')
        else:
            if isinstance(device, str):
                device=torch.device(device.lower())
            elif isinstance(device, torch.device):
                pass
            else:
                raise ValueError('device must be a string or a torch.device instance')
        self.to(device=device)
        if not( isinstance(test_dataloader, torch.utils.data.DataLoader)):
            raise ValueError('Current implementation accept only test data'
                             ' as a pytorch DataLoader')
        if augmenter==None:
            print('augmenter not given. Using a basic one with with flip + random noise')
            augmenter=Default_augmentation
        if loss_func==None:
            loss_func=Loss.SimCLR_loss   
        if not( isinstance(loss_args,list) or isinstance(loss_args,dict) or loss_args==None):
            raise ValueError('loss_args must be a list or a dict with all optional'
                             ' arguments of the loss function')
        
        self.eval()
        N_test = len(test_dataloader)
        if isinstance(test_dataloader.sampler, EEGsampler):
            if test_dataloader.sampler.Keep_only_ratio != 1:
                if test_dataloader.drop_last:
                    N_test = math.floor(sum(1 for _ in test_dataloader.sampler.__iter__())/(test_dataloader.batch_size))
                else:
                    N_test = math.ceil(sum(1 for _ in test_dataloader.sampler.__iter__())/(test_dataloader.batch_size))
        with torch.no_grad():
            test_loss=0
            test_loss_tot=0
            with tqdm.tqdm(total=N_test, ncols=100, 
                           bar_format='{desc}{percentage:3.0f}%|{bar:15}| {n_fmt}/{total_fmt}'
                           ' [{rate_fmt}{postfix}]',
                           disable=not(verbose), unit=' Batch') as pbar:
                for batch_idx, X in enumerate(test_dataloader):
                    
                    if X.device.type!=device.type:
                        X = X.to(device=device)
    
                    # two forward may be slower but uses less memory
                    data_aug1 = augmenter(X)
                    z1 = self(data_aug1)
                    data_aug2 = augmenter(X)
                    z2 = self(data_aug2)
                    test_loss = self.evaluate_loss(loss_func, torch.cat((z1,z2)), loss_args )
                    test_loss_tot += test_loss   
                    # verbose print
                    if verbose:
                        pbar.set_description(f"   test {batch_idx+1:8<}/{N_test:8>}")
                        pbar.set_postfix_str(f"test_loss={test_loss_tot/(batch_idx+1):.5f}")
                        pbar.update()
                              
                test_loss_tot /= (batch_idx+1)
        return test_loss_tot
    
    
    

class SimSiam(SSL_Base):
    """
    Implementation of the SimSiam SSL method. To check
    how SimSIam works, read the following paper [simsiam1]_ .
    Official repo at [siamgit1]_ .

    Parameters
    ----------
    encoder: nn.Module
        The encoder part of the module. It is the one you wish to pretrain and
        transfer to the new model
    projection_head: Union[list[int], nn.Module]
        The projection head to use. It can be an nn.Module or a list of ints.
        In case a list is given, a nn.Sequential module with Dense, BatchNorm
        and Relu will be automtically created. The list will be used to set 
        input and output dimension of each Dense Layer. For instance, if 
        [64, 128, 64] is given, two hidden layers will be created. The first
        with input 64 and output 128, the second with input 128 and output 64.
    predictor: Union[list[int], nn.Module]
        The predictor to put after the projection head. Accepted arguments
        are the same as for the projection_head.

    Warnings
    --------
    This class will not check the compatibility of the encoder's output and
    the projection head's input (as well as between the projection head and the 
    predictor). Make sure that they have the same size. 
        
    References
    ----------
    .. [siamgit1] Original github repo: https://github.com/facebookresearch/simsiam
    .. [simsiam1] Original paper: Chen & He. Exploring Simple Siamese Representation Learning.
      https://arxiv.org/abs/2011.10566
      
    """
    
    def __init__(self, 
                 encoder: nn.Module, 
                 projection_head: Union[list[int], nn.Module],
                 predictor: Union[list[int], nn.Module],
                ):
        
        super(SimSiam,self).__init__(encoder)
        self.encoder = encoder
        if isinstance(projection_head, list):
            if len(projection_head)<2:
                raise ValueError('got a list with only one element')
            else:
                if all(isinstance(i,int) for i in projection_head):
                    DenseList=[]
                    for i in range(len(projection_head)-1):
                        DenseList.append(nn.Linear(projection_head[i],projection_head[i+1]))
                        DenseList.append(nn.BatchNorm1d(num_features=projection_head[i+1]))
                        if i<(len(projection_head)-2):
                            DenseList.append(nn.ReLU())
                    self.projection_head= nn.Sequential(*DenseList)
                else:
                    raise ValueError('got a list with non integer values')
        else:    
            self.projection_head = projection_head
        
        if isinstance(predictor, list):
            if len(predictor)<2:
                raise ValueError('got a list with only one element')
            else:
                if all(isinstance(i,int) for i in predictor):
                    DenseList=[]
                    for i in range(len(predictor)-1):
                        DenseList.append(nn.Linear(predictor[i],predictor[i+1]))
                        if i<(len(predictor)-2):
                            DenseList.append(nn.BatchNorm1d(num_features=predictor[i+1]))
                            DenseList.append(nn.ReLU())
                    self.predictor= nn.Sequential(*DenseList)
                else:
                    raise ValueError('got a list with non integer values')
        else:    
            self.predictor = predictor
        
        
    def forward(self,x):
        """
        :meta private:
        
        """
        x   = self.encoder(x)
        x   = self.projection_head(x)
        emb = self.predictor(x)
        return emb
    
    
    def fit(self,
            train_dataloader,
            epochs=1,
            optimizer=None,
            augmenter=None,
            loss_func: 'function'= None, 
            loss_args: list or dict=[],
            lr_scheduler=None,
            EarlyStopper=None,
            validation_dataloader=None,
            verbose: bool=True,
            device: str=None,
            return_loss_info: bool=False
           ):
        """
        ``fit`` is a custom fit function designed to perform 
        pretraining on a given model with the given dataloader.
    
        Parameters
        ----------
        train_dataloader: Dataloader
            the pytorch Dataloader used to get the training batches. It
            must return a batch as a tuple (X, Y), with X the feature tensor
            and Y the label tensor
        epochs: int, optional
            The number of training epochs. Must be an integer bigger than 0.
    
            Default = 1
        optimizer: torch Optimizer, optional
            The optimizer used for weight's update. It can be any optimizer
            provided in the torch.optim module. If not given Adam with default
            parameters will be instantiated.
    
            Default = torch.optim.Adam
        augmenter: function, optional
            Any function (or callable object) used to perform data augmentation
            on the batch. It is highly suggested to resort to the augmentation
            module, which implements different data augmentation functions and 
            classes to combine them. If none is given a default augmentation with 
            random vertical flip + random noise is applied.
            Note that in this case data augmentation 
            is also performed on the validation set, since it's part of the 
            SSL algorithm.
    
            Default = None
        loss_func: function, optional
            the custom loss function. It can be any loss function which 
            accepts as input the model's prediction and the true labels 
            as required arguments and loss_args as optional arguments.
            If not given SimSiam loss will be automatically used.

            Default = None
        loss_args: Union[list, dict], optional
            The optional arguments to pass to the function. it can be a list
            or a dict.
    
            Default = None
        label_encoder: function, optional
            A custom function used to encode the returned Dataloaders true labels.
            If None, the Dataloader's true label is used directly.
    
            Default = None
        lr_scheduler: torch Scheduler
            A pytorch learning rate scheduler used to update the learning rate 
            during the fine-tuning.
    
            Default = None
        EarlyStopper: EarlyStopping, optional
            An instance of the provided EarlyStopping class.
    
            Default = None
        validation_dataloader: Dataloader, optional
            the pytorch Dataloader used to get the validation batches. It
            must return a batch as a tuple (X, Y), with X the feature tensor
            and Y the label tensor. If not given, no validation loss will be 
            calculated
    
            Default = None
        verbose: bool, optional
            Whether to print a progression bar or not.
    
            Default = None
        device: torch.device or str, optional
            The devide to use for fine-tuning. If given as a string it will
            be converted in a torch.device instance. If not given, 'cpu' device
            will be used.
            
            Device = None
        return_loss_info: bool, optional
            Whether to return the calculated training validation losses at 
            each epoch. 
    
            Default = False
    
        Returns
        -------
        loss_info: dict, optional
            A dictionary with keys the epoch number (as integer) and values
            a two element list with the average epoch's training and validation
            loss.
            
        Note
        ----
        If an EarlyStopping instance is given with monitoring loss set to 
        validation loss, but no validation dataloader is given, monitoring
        loss will be automatically set to training loss.
        
        """
        
        # Various check on input parameters. If some arguments weren't given
        if device==None:
            # If device is None cannot assume if the model is on gpu and so if to send the batch
            # on other device, so cpu will be used. If model is sent to another device set the 
            # device attribute with a proper string or torch device
            device=torch.device('cpu')
        else:
            if isinstance(device, str):
                device=torch.device(device.lower())
            elif isinstance(device, torch.device):
                pass
            else:
                raise ValueError('device must be a string or a torch.device instance')
        self.to(device=device)

        if not( isinstance(train_dataloader, torch.utils.data.DataLoader)):
            raise ValueError('Current implementation accept only training data as a pytorch DataLoader')
        if not(isinstance(epochs, int)):
            epochs= int(epochs)
        if epochs<1:
            raise ValueError('epochs must be bigger than 1')
        if optimizer==None:
            optimizer=torch.optim.Adam(self.parameters())
        if augmenter==None:
            print('augmenter not given. Using a basic one with with flip + random noise')
            augmenter=Default_augmentation
        if loss_func==None:
            loss_func=Loss.SimSiam_loss   
        if not( isinstance(loss_args,list) or isinstance(loss_args,dict) or loss_args==None):
            raise ValueError('loss_args must be a list or a dict with all'
                             ' optional arguments of the loss function')
        perform_validation=False
        if validation_dataloader!=None:
            if not( isinstance(validation_dataloader, torch.utils.data.DataLoader)):
                raise ValueError('Current implementation accept only validation data as'
                                 ' a pytorch DataLoader')
            else:
                perform_validation=True

        if EarlyStopper is not None:
            if EarlyStopper.monitored=='validation' and not(perform_validation):
                print('Early stopper monitoring is set to validation loss'
                      ', but not validation data are given. '
                      'Internally changing monitoring to training loss')
                EarlyStopper.monitored = 'train'
    
        loss_info={i: [None, None] for i in range(epochs)}
        N_train = len(train_dataloader)
        if isinstance(train_dataloader.sampler, EEGsampler):
            if train_dataloader.sampler.Keep_only_ratio != 1:
                if train_dataloader.drop_last:
                    N_train = math.floor(sum(1 for _ in train_dataloader.sampler.__iter__())/(train_dataloader.batch_size))
                else:
                    N_train = math.ceil(sum(1 for _ in train_dataloader.sampler.__iter__())/(train_dataloader.batch_size))
                
        N_val = 0 if validation_dataloader is None else len(validation_dataloader)
        if isinstance(validation_dataloader.sampler, EEGsampler):
            if validation_dataloader.sampler.Keep_only_ratio != 1:
                if validation_dataloader.drop_last:
                    N_val = math.floor(sum(1 for _ in validation_dataloader.sampler.__iter__())/(validation_dataloader.batch_size))
                else:
                    N_val = math.ceil(sum(1 for _ in validation_dataloader.sampler.__iter__())/(validation_dataloader.batch_size))
        for epoch in range(epochs):
            print(f'epoch [{epoch+1:6>}/{epochs:6>}]')
        
            train_loss=0
            val_loss=0
            train_loss_tot=0
            val_loss_tot=0
            
            if not(self.training):
                self.train()

            with tqdm.tqdm(total=N_train+N_val, ncols=100, 
                       bar_format='{desc}{percentage:3.0f}%|{bar:15}| {n_fmt}/{total_fmt}'
                       ' [{rate_fmt}{postfix}]',
                       disable=not(verbose), unit=' Batch') as pbar:
                for batch_idx, X in enumerate(train_dataloader):
    
                    optimizer.zero_grad()
                    if X.device.type!=device.type:
                        X = X.to(device=device)
                        
                    data_aug1 = augmenter(X)
                    data_aug2 = augmenter(X)
                    
                    p1 = self(data_aug1)
                    p2 = self(data_aug2)
                    
                    with torch.no_grad():
                        z1 = self.encoder(data_aug1)
                        z1 = self.projection_head(z1).detach()
                        z2 = self.encoder(data_aug2)
                        z2 = self.projection_head(z2).detach()
                    
                    train_loss = self.evaluate_loss(loss_func, [p1, z1, p2, z2] , loss_args )
                    train_loss.backward()
                    optimizer.step()
                    train_loss_tot += train_loss.item()
                    # verbose print
                    if verbose:
                        pbar.set_description(f" train {batch_idx+1:8<}/{N_train:8>}")
                        pbar.set_postfix_str(f"train_loss={train_loss_tot/(batch_idx+1):.5f}, val_loss={val_loss_tot:.5f}")
                        pbar.update()        
                train_loss_tot /= (batch_idx+1)
    
                if lr_scheduler!=None:
                    lr_scheduler.step()
                
                # Perform validation if validation dataloader were given
                if perform_validation:
                    self.eval()
                    with torch.no_grad():
                        val_loss=0
                        val_loss_tot=0
                        for batch_idx, X in enumerate(validation_dataloader):
                            if X.device.type!=device.type:
                                X = X.to(device=device)
                            
                            data_aug1 = augmenter(X)
                            data_aug2 = augmenter(X)
                            
                            p1 = self(data_aug1)
                            z1 = self.encoder(data_aug1)
                            z1 = self.projection_head(z1).detach()
    
                            p2 = self(data_aug2)
                            z2 = self.encoder(data_aug2)
                            z2 = self.projection_head(z2).detach()
                            val_loss = self.evaluate_loss( loss_func, [p1, z1, p2, z2], loss_args )
                            val_loss_tot += val_loss
                            if verbose:
                                pbar.set_description(f"   val {batch_idx+1:8<}/{N_val:8>}")
                                pbar.set_postfix_str(f"train_loss={train_loss_tot:.5f}, val_loss={val_loss_tot/(batch_idx+1):.5f}")
                                pbar.update()
                        val_loss_tot /= (batch_idx+1)
                    
            # Deal with earlystopper if given
            if EarlyStopper!=None:
                updated_mdl=False
                curr_monitored = val_loss_tot if EarlyStopper.monitored=='validation' else train_loss_tot
                EarlyStopper.early_stop(curr_monitored)
                if EarlyStopper.record_best_weights:
                    if EarlyStopper.best_loss==curr_monitored:
                        EarlyStopper.rec_best_weights(self)
                        updated_mdl=True
                if EarlyStopper():
                    print('no improvement after {} epochs. Training stopped.'.format(
                        EarlyStopper.patience))
                    if EarlyStopper.record_best_weights and not(updated_mdl):
                        EarlyStopper.restore_best_weights(self)
                    if return_loss_info:
                        return loss_info
                    else:
                        return
            
            if return_loss_info:
                loss_info[epoch]=[train_loss_tot, val_loss_tot]        
        if return_loss_info:
            return loss_info
                    

    
    def test(self, 
             test_dataloader,
             augmenter=None,
             loss_func: 'function'= None, 
             loss_args: list or dict=[],
             verbose: bool=True,
             device: str=None
            ):
        """
        a method to evaluate the loss on a test dataloader.
        Parameters are the same as described in the fit method, aside for 
        those related to model training which are removed.
        
        It is rare to evaluate the pretraing loss function on a test set.
        Nevertheless this function provides a way to do that. 
        An example of usage could be to assess the quality of the learned 
        features on the fine-tuning dataset.
        
        """
        if device==None:
            # If device is None cannot assume if the model is on gpu and so if to send the batch
            # on other device, so cpu will be used. If model is sent to another device set the 
            # device attribute with a proper string or torch device
            device=torch.device('cpu')
        else:
            if isinstance(device, str):
                device=torch.device(device.lower())
            elif isinstance(device, torch.device):
                pass
            else:
                raise ValueError('device must be a string or a torch.device instance')
        self.to(device=device)
        if not( isinstance(test_dataloader, torch.utils.data.DataLoader)):
            raise ValueError('Current implementation accept only test data'
                             ' as a pytorch DataLoader')
        if augmenter==None:
            print('augmenter not given. Using a basic one with with flip + random noise')
            augmenter=Default_augmentation
        if loss_func==None:
            loss_func=Loss.SimSiam_loss   
        if not( isinstance(loss_args,list) or isinstance(loss_args,dict) or loss_args==None):
            raise ValueError('loss_args must be a list or a dict with all '
                             'optional arguments of the loss function')
        
        N_test = len(test_dataloader)
        if isinstance(test_dataloader.sampler, EEGsampler):
            if test_dataloader.sampler.Keep_only_ratio != 1:
                if test_dataloader.drop_last:
                    N_test = math.floor(sum(1 for _ in test_dataloader.sampler.__iter__())/(test_dataloader.batch_size))
                else:
                    N_test = math.ceil(sum(1 for _ in test_dataloader.sampler.__iter__())/(test_dataloader.batch_size))
        self.eval()
        with torch.no_grad():
            test_loss=0
            test_loss_tot=0
            with tqdm.tqdm(total=N_test, ncols=100, 
                           bar_format='{desc}{percentage:3.0f}%|{bar:15}| {n_fmt}/{total_fmt}'
                           ' [{rate_fmt}{postfix}]',
                           disable=not(verbose), unit=' Batch') as pbar:
                for batch_idx, X in enumerate(test_dataloader):
                    if X.device.type!=device.type:
                        X = X.to(device=device)
    
                    # two forward may be slower but uses less memory
                    data_aug1 = augmenter(X)
                    p1 = self(data_aug1)
                    z1 = self.encoder(data_aug1)
                    z1 = self.projection_head(z1).detach()
                    
                    data_aug2 = augmenter(X)
                    p2 = self(data_aug2)
                    z2 = self.encoder(data_aug2)
                    z2 = self.projection_head(z2).detach()
                    test_loss = self.evaluate_loss(loss_func, [p1, z1, p2, z2], loss_args )
                    test_loss_tot += test_loss     
                    # verbose print
                    if verbose:
                        pbar.set_description(f"   test {batch_idx+1:8<}/{N_test:8>}")
                        pbar.set_postfix_str(f"test_loss={test_loss_tot/(batch_idx+1):.5f}")
                        pbar.update()
                            
                test_loss_tot /= (batch_idx+1)
        return test_loss_tot
    
    
    
    
class MoCo(SSL_Base):
    """
    Implementation of the MoCo SSL method. To check
    how MoCo works, read the following paper [moco21]_ [moco31]_ .

    Parameters
    ----------
    encoder: nn.Module
        The encoder part of the module. It is the one you wish to pretrain and
        transfer to the new model
    projection_head: Union[list[int], nn.Module]
        The projection head to use. It can be an nn.Module or a list of ints.
        In case a list is given, a nn.Sequential module with Dense, BatchNorm
        and Relu will be automtically created. The list will be used to set 
        input and output dimension of each Dense Layer. For instance, if 
        [64, 128, 64] is given, two hidden layers will be created. The first
        with input 64 and output 128, the second with input 128 and output 64.
    predictor: Union[list[int], nn.Module], optional
        The predictor to put after the projection head. Accepted arguments
        are the same as for the projection_head. It can be left to None since
        MoCo v2 doesn't use it (MoCo v3 use it although).

        Default = None
    feat_size: int, optional
        The size of the feature vector (encoder's output last dim shape).
        It will be used to initialize the queue (for MoCo v2). If not given
        the last element of the projection_head list is used. Of course, it 
        must be given if a custom projection head is used.

        Default = -1
    bank_size: int, optional
        The size of the queue, i.e. the number of projection to keep memory.
        If not given, fit will trigger the calculation of the MoCo v3 loss.

        Default = 0
    m: float, optional
        The value of the momentum coefficient. Suggested values are in the 
        range [0.995, 0.999].

        Default = 0.995

    Warnings
    --------
    This class will not check the compatibility of the encoder's output and
    the projection head's input (as well as between the projection head and the 
    predictor). Make sure that they have the same size. 
        
    References
    ----------
    .. [moco21] K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick, 
      “Momentum contrast for unsupervised visual representation learning,” in Proceedings of 
      the IEEE/CVF conference on computer vision and pattern recognition, pp. 9729–9738, 2020.
    .. [moco31] X. Chen, H. Fan, R. Girshick, and K. He, “Improved base- lines with momentum 
      contrastive learning,” arXiv preprint arXiv:2003.04297, 2020.
      
    """
    
    def __init__(self, 
                 encoder: nn.Module, 
                 projection_head: Union[list[int], nn.Module],
                 predictor: Union[list[int], nn.Module]=None,
                 feat_size: int=-1,
                 bank_size: int=0,
                 m: float=0.995
                ):
        
        super(MoCo,self).__init__(encoder)

        self.bank_size = bank_size
        self.m = m
        
        self.encoder = encoder
        self.momentum_encoder = copy.deepcopy(encoder)
        
        if isinstance(projection_head, list):
            if len(projection_head)<2:
                raise ValueError('got a list with only one element')
            else:
                if all(isinstance(i,int) for i in projection_head):
                    DenseList=[]
                    for i in range(len(projection_head)-1):
                        DenseList.append(nn.Linear(projection_head[i],projection_head[i+1]))
                        DenseList.append(nn.BatchNorm1d(num_features=projection_head[i+1]))
                        if i<(len(projection_head)-2):
                            DenseList.append(nn.ReLU())
                    self.projection_head= nn.Sequential(*DenseList)
                    self.momentum_projection_head= nn.Sequential(*DenseList)
                    
                else:
                    raise ValueError('got a list with non integer values')
        else:    
            self.projection_head = projection_head
            self.momentum_projection_head= copy.deepcopy(projection_head)

        if predictor!=None:
            if isinstance(predictor, list):
                if len(predictor)<2:
                    raise ValueError('got a list with only one element')
                else:
                    if all(isinstance(i,int) for i in predictor):
                        DenseList=[]
                        for i in range(len(predictor)-1):
                            DenseList.append(nn.Linear(predictor[i],predictor[i+1]))
                            DenseList.append(nn.BatchNorm1d(num_features=predictor[i+1]))
                            if i<(len(predictor)-2):
                                DenseList.append(nn.ReLU())
                        self.predictor= nn.Sequential(*DenseList)
                    else:
                        raise ValueError('got a list with non integer values')
            else:    
                self.predictor = predictor

        if self.predictor==None and bank_size<=0:
            msgWarning= 'You are trying to initialize MoCo with only the projection head and no memory bank. '
            msgWarning += ' Training will follow MoCo v3 setup for loss calculation during training,'
            msgWarning += ' but it\'s suggested to set up an 2-hidden layer MLP predictor as in the original paper'
            print(msgWarning)

        if self.bank_size>0:
            # need to set feature vector size
            if feat_size>0:
                self.feat_size=feat_size
            elif isinstance(projection_head,list):
                self.feat_size=projection_head[-1]
            else:
                msgErr= 'feature size cannot be extracted from a nn.Module.'
                msgErr += ' Please provide the feature size, otherwise memory bank cannot be initialized'
                raise ValueError(msgErr)
            # create the queue
            self.register_buffer( "queue", torch.randn(self.feat_size, self.bank_size) )
            self.queue = nn.functional.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        for param_base, param_mom in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            param_mom.requires_grad = False
        for param_base, param_mom in zip(self.projection_head.parameters(), self.momentum_projection_head.parameters()):
            param_mom.requires_grad = False

    
    @torch.no_grad()
    def _update_momentum_encoder(self):
        """
        :meta private:
        
        """
        for param_b, param_m in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * self.m + param_b.data * (1. - self.m)
        for param_b, param_m in zip(self.projection_head.parameters(), self.momentum_projection_head.parameters()):
            param_m.data = param_m.data * self.m + param_b.data * (1. - self.m)

    @torch.no_grad()
    def _update_queue(self, keys):
        """
        :meta private:
        
        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if batch_size > self.bank_size:
            raise ValueError('cannot add a batch bigger than bank size')

        if (ptr+batch_size)>self.bank_size:
            diff1 = self.bank_size-ptr
            diff2 = batch_size-diff1
            self.queue[:, ptr:]= keys[:diff1].T
            self.queue[:, :diff2]= keys[diff1:].T
            self.queue_ptr[0]=diff2
        else:
            self.queue[:, ptr:ptr+batch_size]=keys.T
            self.queue_ptr +=batch_size

    
    def forward(self, x):
        """
        :meta private:
        
        """
        x   = self.encoder(x)
        emb   = self.projection_head(x)
        if self.predictor!=None:
            emb = self.predictor(emb)
        return emb

    
    def fit(self,
            train_dataloader,
            epochs=1,
            optimizer=None,
            augmenter=None,
            loss_func: 'function'= None, 
            loss_args: list or dict=[],
            lr_scheduler=None,
            EarlyStopper=None,
            validation_dataloader=None,
            verbose:bool=True,
            device: str=None,
            return_loss_info: bool=False
           ):
        """
        ``fit`` is a custom fit function designed to perform 
        pretraining on a given model with the given dataloader.
    
        Parameters
        ----------
        train_dataloader: Dataloader
            the pytorch Dataloader used to get the training batches. It
            must return a batch as a tuple (X, Y), with X the feature tensor
            and Y the label tensor
        epochs: int, optional
            The number of training epochs. Must be an integer bigger than 0.
    
            Default = 1
        optimizer: torch Optimizer, optional
            The optimizer used for weight's update. It can be any optimizer
            provided in the torch.optim module. If not given Adam with default
            parameters will be instantiated.
    
            Default = torch.optim.Adam
        augmenter: function, optional
            Any function (or callable object) used to perform data augmentation
            on the batch. It is highly suggested to resort to the augmentation
            module, which implements different data augmentation functions and 
            classes to combine them. If none is given a default augmentation with 
            random vertical flip + random noise is applied.
            Note that in this case data augmentation 
            is also performed on the validation set, since it's part of the 
            SSL algorithm.
    
            Default = None
        loss_func: function, optional
            the custom loss function. It can be any loss function which 
            accepts as input the model's prediction and the true labels 
            as required arguments and loss_args as optional arguments.
            If not given MoCo loss will be automatically chosen.

            Default = None
        loss_args: Union[list, dict], optional
            The optional arguments to pass to the function. it can be a list
            or a dict.
    
            Default = None
        label_encoder: function, optional
            A custom function used to encode the returned Dataloaders true labels.
            If None, the Dataloader's true label is used directly.
    
            Default = None
        lr_scheduler: torch Scheduler
            A pytorch learning rate scheduler used to update the learning rate 
            during the fine-tuning.
    
            Default = None
        EarlyStopper: EarlyStopping, optional
            An instance of the provided EarlyStopping class.
    
            Default = None
        validation_dataloader: Dataloader, optional
            the pytorch Dataloader used to get the validation batches. It
            must return a batch as a tuple (X, Y), with X the feature tensor
            and Y the label tensor. If not given, no validation loss will be 
            calculated
    
            Default = None
        verbose: bool, optional
            Whether to print a progression bar or not.
    
            Default = None
        device: torch.device or str, optional
            The devide to use for fine-tuning. If given as a string it will
            be converted in a torch.device instance. If not given, 'cpu' device
            will be used.
            
            Device = None
        return_loss_info: bool, optional
            Whether to return the calculated training validation losses at 
            each epoch. 
    
            Default = False
    
        Returns
        -------
        loss_info: dict, optional
            A dictionary with keys the epoch number (as integer) and values
            a two element list with the average epoch's training and validation
            loss.
            
        Note
        ----
        If an EarlyStopping instance is given with monitoring loss set to 
        validation loss, but no validation dataloader is given, monitoring
        loss will be automatically set to training loss.
        
        """
        
        # Various check on input parameters. If some arguments weren't given
        if device==None:
            # If device is None cannot assume if the model is on gpu and so if to send the batch
            # on other device, so cpu will be used. If model is sent to another device set the 
            # device attribute with a proper string or torch device
            device=torch.device('cpu')
        else:
            if isinstance(device, str):
                device=torch.device(device.lower())
            elif isinstance(device, torch.device):
                pass
            else:
                raise ValueError('device must be a string or a torch.device instance')
        self.to(device=device)

        if not( isinstance(train_dataloader, torch.utils.data.DataLoader)):
            raise ValueError('Current implementation accept only training data'
                             ' as a pytorch DataLoader')
        if not(isinstance(epochs, int)):
            epochs= int(epochs)
        if epochs<1:
            raise ValueError('epochs must be bigger than 1')
        if optimizer==None:
            optimizer=torch.optim.Adam(self.parameters())
        if augmenter==None:
            print('augmenter not given. Using a basic one with with flip + random noise')
            augmenter=Default_augmentation
        if loss_func==None:
            loss_func=Loss.MoCo_loss   
        if not( isinstance(loss_args,list) or isinstance(loss_args,dict) or loss_args==None):
            raise ValueError('loss_args must be a list or a dict with all'
                             ' optional arguments of the loss function')
        perform_validation=False
        if validation_dataloader!=None:
            if not( isinstance(validation_dataloader, torch.utils.data.DataLoader)):
                raise ValueError('Current implementation accept only validation data as'
                                 ' a pytorch DataLoader')
            else:
                perform_validation=True

        if EarlyStopper is not None:
            if EarlyStopper.monitored=='validation' and not(perform_validation):
                print('Early stopper monitoring is set to validation loss'
                      ', but not validation data are given. '
                      'Internally changing monitoring to training loss')
                EarlyStopper.monitored = 'train'
    
        loss_info={i: [None, None] for i in range(epochs)}
        N_train = len(train_dataloader)
        if isinstance(train_dataloader.sampler, EEGsampler):
            if train_dataloader.sampler.Keep_only_ratio != 1:
                if train_dataloader.drop_last:
                    N_train = math.floor(sum(1 for _ in train_dataloader.sampler.__iter__())/(train_dataloader.batch_size))
                else:
                    N_train = math.ceil(sum(1 for _ in train_dataloader.sampler.__iter__())/(train_dataloader.batch_size))
                
        N_val = 0 if validation_dataloader is None else len(validation_dataloader)
        if isinstance(validation_dataloader.sampler, EEGsampler):
            if validation_dataloader.sampler.Keep_only_ratio != 1:
                if validation_dataloader.drop_last:
                    N_val = math.floor(sum(1 for _ in validation_dataloader.sampler.__iter__())/(validation_dataloader.batch_size))
                else:
                    N_val = math.ceil(sum(1 for _ in validation_dataloader.sampler.__iter__())/(validation_dataloader.batch_size))
        for epoch in range(epochs):
            print(f'epoch [{epoch+1:6>}/{epochs:6>}]')
        
            train_loss=0
            val_loss=0
            train_loss_tot=0
            val_loss_tot=0
            
            if not(self.training):
                self.train()

            with tqdm.tqdm(total=N_train+N_val, ncols=100, 
                       bar_format='{desc}{percentage:3.0f}%|{bar:15}| {n_fmt}/{total_fmt}'
                       ' [{rate_fmt}{postfix}]',
                       disable=not(verbose), unit=' Batch') as pbar:
                for batch_idx, X in enumerate(train_dataloader):
    
                    if X.device.type!=device.type:
                        X = X.to(device=device)
    
                    optimizer.zero_grad()
                    self._update_momentum_encoder()
                    
                    data_aug1 = augmenter(X)
                    data_aug2 = augmenter(X)
                    
                    if self.bank_size>0:
                        # follow moco v2 setup
                        q = self(data_aug1) # queries
                        with torch.no_grad():
                            k = self.momentum_encoder(data_aug2)
                            k = self.momentum_projection_head(k)
                            k = k.detach() # keys
                        self._update_queue(k)
                        train_loss = self.evaluate_loss(loss_func, [q, k, self.queue] , loss_args )
                    else:
                        # if no memory bank, follow moco v3 setup
                        q1 = self(data_aug1)
                        q2 = self(data_aug2)
                        with torch.no_grad():
                            k1 = self.momentum_encoder(data_aug1)
                            k1 = self.momentum_projection_head(k1)
                            k1 = k1.detach() # keys
                            k2 = self.momentum_encoder(data_aug2)
                            k2 = self.momentum_projection_head(k2)
                            k2 = k2.detach() # keys
                        train_loss1 = self.evaluate_loss(loss_func, [q1, k2] , loss_args )
                        train_loss2 = self.evaluate_loss(loss_func, [q2, k1] , loss_args )
                        train_loss= train_loss1 + train_loss2
                    
                    train_loss.backward()
                    optimizer.step()
                    train_loss_tot += train_loss.item()
                    # verbose print
                    if verbose:
                        pbar.set_description(f" train {batch_idx+1:8<}/{N_train:8>}")
                        pbar.set_postfix_str(f"train_loss={train_loss_tot/(batch_idx+1):.5f}, val_loss={val_loss_tot:.5f}")
                        pbar.update()        
                train_loss_tot /= (batch_idx+1)
    
                if lr_scheduler!=None:
                    lr_scheduler.step()
                
                # Perform validation if validation dataloader were given
                # Note that validation in moco can be misleading if there's a memory_bank
                # since calculated keys cannot be added
                if perform_validation:
                    self.eval()
                    with torch.no_grad():
                        val_loss=0
                        val_loss_tot=0
                        for batch_idx, X in enumerate(validation_dataloader):
                            if X.device.type!=device.type:
                                X = X.to(device=device)
                            
                            data_aug1 = augmenter(X)
                            data_aug2 = augmenter(X)
    
                            if self.bank_size>0:
                                # follow moco v2 setup
                                q = self(data_aug1) # queries
                                k = self.momentum_encoder(data_aug2)
                                k = self.momentum_projection_head(k)
                                k = k.detach() # keys
                                val_loss = self.evaluate_loss(loss_func, [q, k, self.queue] , loss_args )
                            else:
                                # if no memory bank, follow moco v3 setup
                                q1 = self(data_aug1)
                                q2 = self(data_aug2)
                                k1 = self.momentum_encoder(data_aug1)
                                k1 = self.momentum_projection_head(k1)
                                k1 = k1.detach() # keys
                                k2 = self.momentum_encoder(data_aug2)
                                k2 = self.momentum_projection_head(k2)
                                k2 = k2.detach() # keys
                                val_loss1 = self.evaluate_loss(loss_func, [q1, k2] , loss_args )
                                val_loss2 = self.evaluate_loss(loss_func, [q2, k1] , loss_args )
                                val_loss = val_loss1 + val_loss2
                            val_loss_tot += val_loss
                            if verbose:
                                pbar.set_description(f"   val {batch_idx+1:8<}/{N_val:8>}")
                                pbar.set_postfix_str(f"train_loss={train_loss_tot:.5f}, val_loss={val_loss_tot/(batch_idx+1):.5f}")
                                pbar.update()
                        val_loss_tot /= (batch_idx+1)
                    
            # Deal with earlystopper if given
            if EarlyStopper!=None:
                updated_mdl=False
                curr_monitored = val_loss_tot if EarlyStopper.monitored=='validation' else train_loss_tot
                EarlyStopper.early_stop(curr_monitored)
                if EarlyStopper.record_best_weights:
                    if EarlyStopper.best_loss==curr_monitored:
                        EarlyStopper.rec_best_weights(self)
                        updated_mdl=True
                if EarlyStopper():
                    print('no improvement after {} epochs. Training stopped.'.format(
                        EarlyStopper.patience))
                    if EarlyStopper.record_best_weights and not(updated_mdl):
                        EarlyStopper.restore_best_weights(self)
                    if return_loss_info:
                        return loss_info
                    else:
                        return
            
            if return_loss_info:
                loss_info[epoch]=[train_loss_tot, val_loss_tot]        
        if return_loss_info:
            return loss_info
                    

    
    def test(self, 
             test_dataloader,
             augmenter=None,
             loss_func: 'function'= None, 
             loss_args: list or dict=[],
             verbose: bool=True,
             device: str=None
            ):
        """
        a method to evaluate the loss on a test dataloader.
        Parameters are the same as described in the fit method, aside for 
        those related to model training which are removed.
        
        It is rare to evaluate the pretraing loss function on a test set.
        Nevertheless this function provides a way to do that. 
        An example of usage could be to assess the quality of the learned 
        features on the fine-tuning dataset.
        
        """
        
        if device==None:
            # If device is None cannot assume if the model is on gpu and so if to send the batch
            # on other device, so cpu will be used. If model is sent to another device set the 
            # device attribute with a proper string or torch device
            device=torch.device('cpu')
        else:
            if isinstance(device, str):
                device=torch.device(device.lower())
            elif isinstance(device, torch.device):
                pass
            else:
                raise ValueError('device must be a string or a torch.device instance')
        self.to(device=device)
        if not( isinstance(test_dataloader, torch.utils.data.DataLoader)):
            raise ValueError('Current implementation accept only test'
                             ' data as a pytorch DataLoader')
        if augmenter==None:
            print('augmenter not given. Using a basic one with with flip + random noise')
            augmenter=Default_augmentation
        if loss_func==None:
            los_func=Loss.Moco_loss   
        if not( isinstance(loss_args,list) or isinstance(loss_args,dict) or loss_args==None):
            raise ValueError('loss_args must be a list or a dict with all'
                             ' optional arguments of the loss function')
        
        N_test = len(test_dataloader)
        if isinstance(test_dataloader.sampler, EEGsampler):
            if test_dataloader.sampler.Keep_only_ratio != 1:
                if test_dataloader.drop_last:
                    N_test = math.floor(sum(1 for _ in test_dataloader.sampler.__iter__())/(test_dataloader.batch_size))
                else:
                    N_test = math.ceil(sum(1 for _ in test_dataloader.sampler.__iter__())/(test_dataloader.batch_size))
        elf.eval()
        with torch.no_grad():
            test_loss=0
            test_loss_tot=0
            with tqdm.tqdm(total=N_test, ncols=100, 
                           bar_format='{desc}{percentage:3.0f}%|{bar:15}| {n_fmt}/{total_fmt}'
                           ' [{rate_fmt}{postfix}]',
                           disable=not(verbose), unit=' Batch') as pbar:
                for batch_idx, X in enumerate(test_dataloader):
                    if X.device.type!=device.type:
                        X = X.to(device=device)
    
                    data_aug1 = augmenter(X)
                    data_aug2 = augmenter(X)
                    
                    if self.bank_size>0:
                        # follow moco v2 setup
                        q = self(data_aug1) # queries
                        k = self.momentum_encoder(data_aug2)
                        k = self.momentum_projection_head(k)
                        k = k.detach() # keys
                        test_loss = self.evaluate_loss(loss_func, [q, k, self.queue] , loss_args )
                    else:
                        # if no memory bank, follow moco v3 setup
                        q1 = self(data_aug1)
                        q2 = self(data_aug2)
                        k1 = self.momentum_encoder(data_aug1)
                        k1 = self.momentum_projection_head(k1)
                        k1 = k1.detach() # keys
                        k2 = self.momentum_encoder(data_aug2)
                        k2 = self.momentum_projection_head(k2)
                        k2 = k2.detach() # keys
                        test_loss1 = self.evaluate_loss(loss_func, [q1, k2] , loss_args )
                        test_loss2 = self.evaluate_loss(loss_func, [q2, k1] , loss_args )
                        test_loss = test_loss1 + test_loss2 
                    test_loss_tot += test_loss
                    # verbose print
                    if verbose:
                        pbar.set_description(f"   test {batch_idx+1:8<}/{N_test:8>}")
                        pbar.set_postfix_str(f"test_loss={test_loss_tot/(batch_idx+1):.5f}")
                        pbar.update()
                            
                test_loss_tot /= (batch_idx+1)
        return test_loss_tot




class BYOL(SSL_Base):
    """
    Implementation of the BYOL SSL method. To check
    how BYOL works, read the following paper [BYOL1]_ .

    Parameters
    ----------
    encoder: nn.Module
        The encoder part of the module. It is the one you wish to pretrain and
        transfer to the new model
    projection_head: Union[list[int], nn.Module]
        The projection head to use. It can be an nn.Module or a list of ints.
        In case a list is given, a nn.Sequential module with Dense, BatchNorm
        and Relu will be automtically created. The list will be used to set 
        input and output dimension of each Dense Layer. For instance, if 
        [64, 128, 64] is given, two hidden layers will be created. The first
        with input 64 and output 128, the second with input 128 and output 64.
    predictor: Union[list[int], nn.Module]
        The predictor to put after the projection head. Accepted arguments
        are the same as for the projection_head.
    m: float, optional
        The value of the momentum coefficient. Suggested values are in the 
        range [0.995, 0.999].

        Default = 0.995

    Warnings
    --------
    This class will not check the compatibility of the encoder's output and
    the projection head's input (as well as between the projection head and the 
    predictor). Make sure that they have the same size. 
        
    References
    ----------
    .. [BYOL1] J.-B. Grill, F. Strub, F. Altché, C. Tallec, P. Richemond, E. Buchatskaya, 
      C. Doersch, B. Avila Pires, Z. Guo, M. Gheshlaghi Azar, et al., “Bootstrap your own 
      latent- a new approach to self-supervised learning,” Advances in neural information
      processing systems, vol. 33, pp. 21271– 21284, 2020.
      
    """
    def __init__(self, 
                 encoder: nn.Module, 
                 projection_head: Union[list[int], nn.Module],
                 predictor: Union[list[int], nn.Module]=None,
                 m: float=0.99
                ):
        
        super(BYOL,self).__init__(encoder)
        
        self.m = m
        self.encoder = encoder
        self.momentum_encoder = copy.deepcopy(encoder)
        
        if isinstance(projection_head, list):
            if len(projection_head)<2:
                raise ValueError('got a list with only one element')
            else:
                if all(isinstance(i,int) for i in projection_head):
                    DenseList=[]
                    for i in range(len(projection_head)-1):
                        DenseList.append(nn.Linear(projection_head[i],projection_head[i+1]))
                        if i<(len(projection_head)-2):
                            DenseList.append(nn.BatchNorm1d(num_features=projection_head[i+1]))
                            DenseList.append(nn.ReLU())
                    self.projection_head= nn.Sequential(*DenseList)
                    self.momentum_projection_head= nn.Sequential(*DenseList)
                    
                else:
                    raise ValueError('got a list with non integer values')
        else:    
            self.projection_head = projection_head
            self.momentum_projection_head= copy.deepcopy(projection_head)

        if isinstance(predictor, list):
            if len(predictor)<2:
                raise ValueError('got a list with only one element')
            else:
                if all(isinstance(i,int) for i in predictor):
                    DenseList=[]
                    for i in range(len(predictor)-1):
                        DenseList.append(nn.Linear(predictor[i],predictor[i+1]))
                        if i<(len(predictor)-2):
                            DenseList.append(nn.BatchNorm1d(num_features=predictor[i+1]))
                            DenseList.append(nn.ReLU())
                    self.predictor= nn.Sequential(*DenseList)
                else:
                    raise ValueError('got a list with non integer values')
        else:    
            self.predictor = predictor

        for param_base, param_mom in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            param_mom.requires_grad = False
        for param_base, param_mom in zip(self.projection_head.parameters(), self.momentum_projection_head.parameters()):
            param_mom.requires_grad = False

    
    @torch.no_grad()
    def _update_momentum_encoder(self):
        """
        :meta private:
        
        """
        for param_b, param_m in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * self.m + param_b.data * (1. - self.m)
        for param_b, param_m in zip(self.projection_head.parameters(), self.momentum_projection_head.parameters()):
            param_m.data = param_m.data * self.m + param_b.data * (1. - self.m)
    
    def forward(self, x):
        """
        :meta private:
        
        """
        x   = self.encoder(x)
        x   = self.projection_head(x)
        emb = self.predictor(x)
        return emb

    
    def fit(self,
            train_dataloader,
            epochs=1,
            optimizer=None,
            augmenter=None,
            loss_func: 'function'= None, 
            loss_args: list or dict=[],
            lr_scheduler=None,
            EarlyStopper=None,
            validation_dataloader=None,
            verbose:bool =True,
            device: str=None,
            return_loss_info: bool=False
           ):
        """
        ``fit`` is a custom fit function designed to perform 
        pretraining on a given model with the given dataloader.
    
        Parameters
        ----------
        train_dataloader: Dataloader
            the pytorch Dataloader used to get the training batches. It
            must return a batch as a tuple (X, Y), with X the feature tensor
            and Y the label tensor
        epochs: int, optional
            The number of training epochs. Must be an integer bigger than 0.
    
            Default = 1
        optimizer: torch Optimizer, optional
            The optimizer used for weight's update. It can be any optimizer
            provided in the torch.optim module. If not given Adam with default
            parameters will be instantiated.
    
            Default = torch.optim.Adam
        augmenter: function, optional
            Any function (or callable object) used to perform data augmentation
            on the batch. It is highly suggested to resort to the augmentation
            module, which implements different data augmentation functions and 
            classes to combine them. If none is given a default augmentation with 
            random vertical flip + random noise is applied.
            Note that in this case data augmentation 
            is also performed on the validation set, since it's part of the 
            SSL algorithm.
    
            Default = None
        loss_func: function, optional
            the custom loss function. It can be any loss function which 
            accepts as input the model's prediction and the true labels 
            as required arguments and loss_args as optional arguments.
            If not given BYOL loss will be automatically chosen.

            Default = None
        loss_args: Union[list, dict], optional
            The optional arguments to pass to the function. it can be a list
            or a dict.
    
            Default = None
        label_encoder: function, optional
            A custom function used to encode the returned Dataloaders true labels.
            If None, the Dataloader's true label is used directly.
    
            Default = None
        lr_scheduler: torch Scheduler
            A pytorch learning rate scheduler used to update the learning rate 
            during the fine-tuning.
    
            Default = None
        EarlyStopper: EarlyStopping, optional
            An instance of the provided EarlyStopping class.
    
            Default = None
        validation_dataloader: Dataloader, optional
            the pytorch Dataloader used to get the validation batches. It
            must return a batch as a tuple (X, Y), with X the feature tensor
            and Y the label tensor. If not given, no validation loss will be 
            calculated
    
            Default = None
        verbose: bool, optional
            Whether to print a progression bar or not.
    
            Default = None
        device: torch.device or str, optional
            The devide to use for fine-tuning. If given as a string it will
            be converted in a torch.device instance. If not given, 'cpu' device
            will be used.
            
            Device = None
        return_loss_info: bool, optional
            Whether to return the calculated training validation losses at 
            each epoch. 
    
            Default = False
    
        Returns
        -------
        loss_info: dict, optional
            A dictionary with keys the epoch number (as integer) and values
            a two element list with the average epoch's training and validation
            loss.
            
        Note
        ----
        If an EarlyStopping instance is given with monitoring loss set to 
        validation loss, but no validation dataloader is given, monitoring
        loss will be automatically set to training loss.
        
        """
        
        # Various check on input parameters. If some arguments weren't given
        if device==None:
            # If device is None cannot assume if the model is on gpu and so if to send the batch
            # on other device, so cpu will be used. If model is sent to another device set the 
            # device attribute with a proper string or torch device
            device=torch.device('cpu')
        else:
            if isinstance(device, str):
                device=torch.device(device.lower())
            elif isinstance(device, torch.device):
                pass
            else:
                raise ValueError('device must be a string or a torch.device instance')
        self.to(device=device)

        if not( isinstance(train_dataloader, torch.utils.data.DataLoader)):
            raise ValueError('Current implementation accept only'
                             ' training data as a pytorch DataLoader')
        if not(isinstance(epochs, int)):
            epochs= int(epochs)
        if epochs<1:
            raise ValueError('epochs must be bigger than 1')
        if optimizer==None:
            optimizer=torch.optim.Adam(self.parameters())
        if augmenter==None:
            print('augmenter not given. Using a basic one with with flip + random noise')
            augmenter=Default_augmentation
        if loss_func==None:
            loss_func=Loss.BYOL_loss   
        if not( isinstance(loss_args,list) or isinstance(loss_args,dict) or loss_args==None):
            raise ValueError('loss_args must be a list or a dict with all'
                             ' optional arguments of the loss function')
        perform_validation=False
        if validation_dataloader!=None:
            if not( isinstance(validation_dataloader, torch.utils.data.DataLoader)):
                raise ValueError('Current implementation accept only validation data as'
                                 ' a pytorch DataLoader')
            else:
                perform_validation=True

        if EarlyStopper is not None:
            if EarlyStopper.monitored=='validation' and not(perform_validation):
                print('Early stopper monitoring is set to validation loss'
                      ', but not validation data are given. '
                      'Internally changing monitoring to training loss')
                EarlyStopper.monitored = 'train'
    
        loss_info={i: [None, None] for i in range(epochs)}
        N_train = len(train_dataloader)
        if isinstance(train_dataloader.sampler, EEGsampler):
            if train_dataloader.sampler.Keep_only_ratio != 1:
                if train_dataloader.drop_last:
                    N_train = math.floor(sum(1 for _ in train_dataloader.sampler.__iter__())/(train_dataloader.batch_size))
                else:
                    N_train = math.ceil(sum(1 for _ in train_dataloader.sampler.__iter__())/(train_dataloader.batch_size))
                
        N_val = 0 if validation_dataloader is None else len(validation_dataloader)
        if isinstance(validation_dataloader.sampler, EEGsampler):
            if validation_dataloader.sampler.Keep_only_ratio != 1:
                if validation_dataloader.drop_last:
                    N_val = math.floor(sum(1 for _ in validation_dataloader.sampler.__iter__())/(validation_dataloader.batch_size))
                else:
                    N_val = math.ceil(sum(1 for _ in validation_dataloader.sampler.__iter__())/(validation_dataloader.batch_size))
        for epoch in range(epochs):
            print(f'epoch [{epoch+1:6>}/{epochs:6>}]')
        
            train_loss=0
            val_loss=0
            train_loss_tot=0
            val_loss_tot=0
            if not(self.training):
                self.train()
            with tqdm.tqdm(total=N_train+N_val, ncols=100, 
                       bar_format='{desc}{percentage:3.0f}%|{bar:15}| {n_fmt}/{total_fmt}'
                       ' [{rate_fmt}{postfix}]',
                       disable=not(verbose), unit=' Batch') as pbar:
                for batch_idx, X in enumerate(train_dataloader):
    
                    if X.device.type!=device.type:
                        X = X.to(device=device)
    
                    optimizer.zero_grad()
                    self._update_momentum_encoder()
                    
                    data_aug1 = augmenter(X)
                    data_aug2 = augmenter(X)
                    
                    p1 = self(data_aug1)
                    p2 = self(data_aug2)
                    with torch.no_grad():
                        z1 = self.momentum_encoder(data_aug1)
                        z1 = self.momentum_projection_head(z1)
                        z1 = z1.detach() # keys
                        z2 = self.momentum_encoder(data_aug2)
                        z2 = self.momentum_projection_head(z2)
                        z2 = z2.detach() # keys
                    train_loss = self.evaluate_loss(loss_func, [p1,z1,p2,z2] , loss_args )
                    
                    train_loss.backward()
                    optimizer.step()
                    train_loss_tot += train_loss.item()
                    # verbose print
                    if verbose:
                        pbar.set_description(f" train {batch_idx+1:8<}/{N_train:8>}")
                        pbar.set_postfix_str(f"train_loss={train_loss_tot/(batch_idx+1):.5f}, val_loss={val_loss_tot:.5f}")
                        pbar.update()        
                train_loss_tot /= (batch_idx+1)
    
                if lr_scheduler!=None:
                    lr_scheduler.step()
                
                # Perform validation if validation dataloader were given
                # Note that validation in moco can be misleading if there's a memory_bank
                # since calculated keys cannot be added
                if perform_validation:
                    self.eval()
                    with torch.no_grad():
                        for batch_idx, X in enumerate(validation_dataloader):
                            if X.device.type!=device.type:
                                X = X.to(device=device)
                            
                            data_aug1 = augmenter(X)
                            data_aug2 = augmenter(X)
                            p1 = self(data_aug1)
                            p2 = self(data_aug2)
                            z1 = self.momentum_encoder(data_aug1)
                            z1 = self.momentum_projection_head(z1)
                            z1 = z1.detach() # keys
                            z2 = self.momentum_encoder(data_aug2)
                            z2 = self.momentum_projection_head(z2)
                            z2 = z2.detach() # keys
                            val_loss = self.evaluate_loss(loss_func, [p1,z1,p2,z2] , loss_args )
            
                            val_loss_tot += val_loss
                            if verbose:
                                pbar.set_description(f"   val {batch_idx+1:8<}/{N_val:8>}")
                                pbar.set_postfix_str(f"train_loss={train_loss_tot:.5f}, val_loss={val_loss_tot/(batch_idx+1):.5f}")
                                pbar.update()
                        val_loss_tot /= (batch_idx+1)
                    
            # Deal with earlystopper if given
            if EarlyStopper!=None:
                updated_mdl=False
                curr_monitored = val_loss_tot if EarlyStopper.monitored=='validation' else train_loss_tot
                EarlyStopper.early_stop(curr_monitored)
                if EarlyStopper.record_best_weights:
                    if EarlyStopper.best_loss==curr_monitored:
                        EarlyStopper.rec_best_weights(self)
                        updated_mdl=True
                if EarlyStopper():
                    print('no improvement after {} epochs. Training stopped.'.format(
                        EarlyStopper.patience))
                    if EarlyStopper.record_best_weights and not(updated_mdl):
                        EarlyStopper.restore_best_weights(self)
                    if return_loss_info:
                        return loss_info
                    else:
                        return
            
            if return_loss_info:
                loss_info[epoch]=[train_loss_tot, val_loss_tot]        
        if return_loss_info:
            return loss_info
                    

    
    def test(self, 
             test_dataloader,
             augmenter=None,
             loss_func: 'function'= None, 
             loss_args: list or dict=[],
             verbose: bool=True,
             device: str=None
            ):
        """
        a method to evaluate the loss on a test dataloader.
        Parameters are the same as in the fit method, apart for the 
        ones specific for the training which are removed.
        
        It is rare to evaluate the pretraing loss function on a test set.
        Nevertheless this function provides a way to do that. 
        An example of usage could be to assess the quality of the learned 
        features on the fine-tuning dataset.
        
        """
        
        if device==None:
            # If device is None cannot assume if the model is on gpu and so if to send the batch
            # on other device, so cpu will be used. If model is sent to another device set the 
            # device attribute with a proper string or torch device
            device=torch.device('cpu')
        else:
            if isinstance(device, str):
                device=torch.device(device.lower())
            elif isinstance(device, torch.device):
                pass
            else:
                raise ValueError('device must be a string or a torch.device instance')
        self.to(device=device)
        if not( isinstance(test_dataloader, torch.utils.data.DataLoader)):
            raise ValueError('Current implementation accept only training data as a pytorch DataLoader')
        if augmenter==None:
            print('augmenter not given. Using a basic one with with flip + random noise')
            augmenter=Default_augmentation
        if loss_func==None:
            loss_func=Loss.BYOL_loss   
        if not( isinstance(loss_args,list) or isinstance(loss_args,dict) or loss_args==None):
            raise ValueError('loss_args must be a list or a dict with all optional arguments of the loss function')
        
        N_test = len(test_dataloader)
        if isinstance(test_dataloader.sampler, EEGsampler):
            if test_dataloader.sampler.Keep_only_ratio != 1:
                if test_dataloader.drop_last:
                    N_test = math.floor(sum(1 for _ in test_dataloader.sampler.__iter__())/(test_dataloader.batch_size))
                else:
                    N_test = math.ceil(sum(1 for _ in test_dataloader.sampler.__iter__())/(test_dataloader.batch_size))
        self.eval()
        with torch.no_grad():
            test_loss=0
            test_loss_tot=0
            with tqdm.tqdm(total=N_test, ncols=100, 
                           bar_format='{desc}{percentage:3.0f}%|{bar:15}| {n_fmt}/{total_fmt}'
                           ' [{rate_fmt}{postfix}]',
                           disable=not(verbose), unit=' Batch') as pbar:
                for batch_idx, X in enumerate(test_dataloader):
                    if X.device.type!=device.type:
                        X = X.to(device=device)
    
                    data_aug1 = augmenter(X)
                    data_aug2 = augmenter(X)
                    
                    p1 = self(data_aug1)
                    p2 = self(data_aug2)
                    z1 = self.momentum_encoder(data_aug1)
                    z1 = self.momentum_projection_head(z1)
                    z1 = z1.detach() # keys
                    z2 = self.momentum_encoder(data_aug2)
                    z2 = self.momentum_projection_head(z2)
                    z2 = z2.detach() # keys
                    test_loss = self.evaluate_loss(loss_func, [p1,z1,p2,z2] , loss_args )
                    test_loss_tot += test_loss
                    # verbose print
                    if verbose:
                        pbar.set_description(f"   test {batch_idx+1:8<}/{N_test:8>}")
                        pbar.set_postfix_str(f"test_loss={test_loss_tot/(batch_idx+1):.5f}")
                        pbar.update()
                            
                test_loss_tot /= (batch_idx+1)
        return test_loss_tot



class Barlow_Twins(SimCLR):
    """
    Implementation of the Barlow twins SSL method. To check
    how Barlow Twins works, read the following paper [barlow]_ .

    Parameters
    ----------
    encoder: nn.Module
        The encoder part of the module. It is the one you wish to pretrain and
        transfer to the new model
    projection_head: Union[list[int], nn.Module]
        The projection head to use. It can be an nn.Module or a list of ints.
        In case a list is given, a nn.Sequential module with Dense, BatchNorm
        and Relu will be automtically created. The list will be used to set 
        input and output dimension of each Dense Layer. For instance, if 
        [64, 128, 64] is given, two hidden layers will be created. The first
        with input 64 and output 128, the second with input 128 and output 64.

    Warnings
    --------
    This class will not check the compatibility of the encoder's output and
    the projection head's input (as well as between the projection head and the 
    predictor). Make sure that they have the same size. 
        
    References
    ----------
    .. [barlow] J. Zbontar, L. Jing, I. Misra, Y. LeCun, and S. Deny, 
      “Barlow twins: Self-supervised learning via redundancy re- duction,” in International 
      Conference on Machine Learning, pp. 12310–12320, PMLR, 2021.
      
    """
    def __init__(self, 
                 encoder: nn.Module, 
                 projection_head: Union[list[int], nn.Module],
                ):
        super(Barlow_Twins, self).__init__(encoder, projection_head)

    def fit(self,
            train_dataloader,
            epochs=1,
            optimizer=None,
            augmenter=None,
            loss_func: 'function'= None, 
            loss_args: list or dict=[],
            lr_scheduler=None,
            EarlyStopper=None,
            validation_dataloader=None,
            verbose: bool=True,
            device: str=None,
            cat_augmentations: bool=False,
            return_loss_info: bool=False
           ):
        """
        ``fit`` is a custom fit function designed to perform 
        pretraining on a given model with the given dataloader.
    
        Parameters
        ----------
        train_dataloader: Dataloader
            the pytorch Dataloader used to get the training batches. It
            must return a batch as a tuple (X, Y), with X the feature tensor
            and Y the label tensor
        epochs: int, optional
            The number of training epochs. Must be an integer bigger than 0.
    
            Default = 1
        optimizer: torch Optimizer, optional
            The optimizer used for weight's update. It can be any optimizer
            provided in the torch.optim module. If not given Adam with default
            parameters will be instantiated.
    
            Default = torch.optim.Adam
        augmenter: function, optional
            Any function (or callable object) used to perform data augmentation
            on the batch. It is highly suggested to resort to the augmentation
            module, which implements different data augmentation functions and 
            classes to combine them. If none is given a default augmentation with 
            random vertical flip + random noise is applied.
            Note that in this case data augmentation 
            is also performed on the validation set, since it's part of the 
            SSL algorithm.
    
            Default = None
        loss_func: function, optional
            the custom loss function. It can be any loss function which 
            accepts as input the model's prediction and the true labels 
            as required arguments and loss_args as optional arguments.
            If not given Barlow loss will be automatically used.

            Default = None
        loss_args: Union[list, dict], optional
            The optional arguments to pass to the function. it can be a list
            or a dict.
    
            Default = None
        label_encoder: function, optional
            A custom function used to encode the returned Dataloaders true labels.
            If None, the Dataloader's true label is used directly.
    
            Default = None
        lr_scheduler: torch Scheduler
            A pytorch learning rate scheduler used to update the learning rate 
            during the fine-tuning.
    
            Default = None
        EarlyStopper: EarlyStopping, optional
            An instance of the provided EarlyStopping class.
    
            Default = None
        validation_dataloader: Dataloader, optional
            the pytorch Dataloader used to get the validation batches. It
            must return a batch as a tuple (X, Y), with X the feature tensor
            and Y the label tensor. If not given, no validation loss will be 
            calculated
    
            Default = None
        verbose: bool, optional
            Whether to print a progression bar or not.
    
            Default = None
        device: torch.device or str, optional
            The devide to use for fine-tuning. If given as a string it will
            be converted in a torch.device instance. If not given, 'cpu' device
            will be used.
            
            Device = None
        cat_augmentations: bool, optional
            Whether to calculate the loss on the cat version of the two
            projection's or not. It might affect some statistical layers.

            Default = False
        return_loss_info: bool, optional
            Whether to return the calculated training validation losses at 
            each epoch. 
    
            Default = False
    
        Returns
        -------
        loss_info: dict, optional
            A dictionary with keys the epoch number (as integer) and values
            a two element list with the average epoch's training and validation
            loss.
            
        Note
        ----
        If an EarlyStopping instance is given with monitoring loss set to 
        validation loss, but no validation dataloader is given, monitoring
        loss will be automatically set to training loss.
        
        """
        
        if loss_func==None:
            loss_func=Loss.Barlow_loss
            loss_args=[]
        super().fit(train_dataloader, epochs, optimizer, augmenter, loss_func, loss_args, 
                    lr_scheduler, EarlyStopper, validation_dataloader, verbose, device, 
                    cat_augmentations, return_loss_info
                   ) 

    def test(self, 
             test_dataloader,
             augmenter=None,
             loss_func: 'function'= None, 
             loss_args: list or dict=[],
             verbose: bool=True,
             device: str=None
            ):
        
        if loss_func==None:
            loss_func=Loss.Barlow_loss
            loss_args=[]
        super().test(test_dataloader, augmenter, loss_func, loss_args,  verbose, device) 



class VICReg(SimCLR):
    """
    Implementation of the VICReg SSL method. To check
    how VICReg works, read the following paper [VIC1]_ .

    Parameters
    ----------
    encoder: nn.Module
        The encoder part of the module. It is the one you wish to pretrain and
        transfer to the new model
    projection_head: Union[list[int], nn.Module]
        The projection head to use. It can be an nn.Module or a list of ints.
        In case a list is given, a nn.Sequential module with Dense, BatchNorm
        and Relu will be automtically created. The list will be used to set 
        input and output dimension of each Dense Layer. For instance, if 
        [64, 128, 64] is given, two hidden layers will be created. The first
        with input 64 and output 128, the second with input 128 and output 64.
    predictor: Union[list[int], nn.Module]
        The predictor to put after the projection head. Accepted arguments
        are the same as for the projection_head.

    Warnings
    --------
    This class will not check the compatibility of the encoder's output and
    the projection head's input (as well as between the projection head and the 
    predictor). Make sure that they have the same size. 
        
    References
    ----------
    .. [VIC1] A. Bardes, J. Ponce, and Y. LeCun, “Vicreg: Variance- invariance-covariance
      regularization for self-supervised learning,” arXiv preprint arXiv:2105.04906, 2021.
      
    """
    def __init__(self, 
                 encoder: nn.Module, 
                 projection_head: Union[list[int], nn.Module],
                ):
        super(VICReg, self).__init__(encoder, projection_head)

    def fit(self,
            train_dataloader,
            epochs=1,
            optimizer=None,
            augmenter=None,
            loss_func: 'function'= None, 
            loss_args: list or dict=[],
            lr_scheduler=None,
            EarlyStopper=None,
            validation_dataloader=None,
            verbose: bool=True,
            device: str=None,
            cat_augmentations: bool=False,
            return_loss_info: bool=False
           ):
        """
        ``fit`` is a custom fit function designed to perform 
        pretraining on a given model with the given dataloader.
    
        Parameters
        ----------
        train_dataloader: Dataloader
            the pytorch Dataloader used to get the training batches. It
            must return a batch as a tuple (X, Y), with X the feature tensor
            and Y the label tensor
        epochs: int, optional
            The number of training epochs. Must be an integer bigger than 0.
    
            Default = 1
        optimizer: torch Optimizer, optional
            The optimizer used for weight's update. It can be any optimizer
            provided in the torch.optim module. If not given Adam with default
            parameters will be instantiated.
    
            Default = torch.optim.Adam
        augmenter: function, optional
            Any function (or callable object) used to perform data augmentation
            on the batch. It is highly suggested to resort to the augmentation
            module, which implements different data augmentation functions and 
            classes to combine them. If none is given a default augmentation with 
            random vertical flip + random noise is applied.
            Note that in this case data augmentation 
            is also performed on the validation set, since it's part of the 
            SSL algorithm.
    
            Default = None
        loss_func: function, optional
            the custom loss function. It can be any loss function which 
            accepts as input the model's prediction and the true labels 
            as required arguments and loss_args as optional arguments.
            If not given VICReg loss will be automatically used.

            Default = None
        loss_args: Union[list, dict], optional
            The optional arguments to pass to the function. it can be a list
            or a dict.
    
            Default = None
        label_encoder: function, optional
            A custom function used to encode the returned Dataloaders true labels.
            If None, the Dataloader's true label is used directly.
    
            Default = None
        lr_scheduler: torch Scheduler
            A pytorch learning rate scheduler used to update the learning rate 
            during the fine-tuning.
    
            Default = None
        EarlyStopper: EarlyStopping, optional
            An instance of the provided EarlyStopping class.
    
            Default = None
        validation_dataloader: Dataloader, optional
            the pytorch Dataloader used to get the validation batches. It
            must return a batch as a tuple (X, Y), with X the feature tensor
            and Y the label tensor. If not given, no validation loss will be 
            calculated
    
            Default = None
        verbose: bool, optional
            Whether to print a progression bar or not.
    
            Default = None
        device: torch.device or str, optional
            The devide to use for fine-tuning. If given as a string it will
            be converted in a torch.device instance. If not given, 'cpu' device
            will be used.
            
            Device = None
        cat_augmentations: bool, optional
            Whether to calculate the loss on the cat version of the two
            projection's or not. It might affect some statistical layers.

            Default = False
        return_loss_info: bool, optional
            Whether to return the calculated training validation losses at 
            each epoch. 
    
            Default = False
    
        Returns
        -------
        loss_info: dict, optional
            A dictionary with keys the epoch number (as integer) and values
            a two element list with the average epoch's training and validation
            loss.
            
        Note
        ----
        If an EarlyStopping instance is given with monitoring loss set to 
        validation loss, but no validation dataloader is given, monitoring
        loss will be automatically set to training loss.
        
        """
        
        if loss_func==None:
            loss_func=Loss.VICReg_loss
            loss_args=[]
        super().fit(train_dataloader, epochs, optimizer, augmenter, loss_func, loss_args, 
                    lr_scheduler, EarlyStopper, validation_dataloader, verbose, device,
                    cat_augmentations, return_loss_info
                   ) 

    def test(self, 
             test_dataloader,
             augmenter=None,
             loss_func: 'function'= None, 
             loss_args: list or dict=[],
             verbose: bool=True,
             device: str=None
            ):
        
        if loss_func==None:
            loss_func=Loss.VICReg_loss
            loss_args=[]
        super().test(test_dataloader, augmenter, loss_func, loss_args,  verbose, device) 

