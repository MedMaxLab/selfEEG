import os
import copy
import datetime
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import losses as Loss
from typing import Any, Callable, Iterable, TypeVar, Generic, Sequence, List, Optional, Union

__all__ = ['EarlyStopping', 'SSL_Base', 'SimCLR', 'SimSiam', 'MoCo', 'BYOL', 'Barlow_Twins', 'VICReg']

def Default_augmentation(x):
    """
    simple default augmentation used when non data augmenter is given in SSL fit methods. It's just
    a programming choice to avoid putting the augmenter as non optional parameter. No justification 
    for the choice of flip + random noise. Just that it can be written in few lines of code.
    """
    if not(isinstance(x, torch.Tensor)):
        x=torch.Tensor(x)
    x = x*(-1)
    std = torch.std(x)
    noise = std * torch.randn(*x.shape, device=x.device)
    x_noise = x + noise 
    return x_noise

def evaluateLoss( loss_fun: 'loss function', 
                  arguments, 
                  loss_arg: Union[list, dict]=None):
    """
    evaluate current batch loss
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

def fine_tune(model,
              train_dataloader,
              epochs=1,
              optimizer=None,
              augmenter=None,
              loss_func: 'function'= None, 
              loss_args: list or dict=[],
              label_encoder: 'function' = None,
              lr_scheduler=None,
              EarlyStopper=None,
              validation_dataloader=None,
              verbose=True,
              device: str or torch.device=None,
              return_loss_info: bool=False
             ):
    
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
                        val_loss_tot += val_loss
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
    Some arguments are similar to Keras EarlyStopping ones ( https://keras.io/api/callbacks/early_stopping/ ).
    If you want to use other implemented functionalities take a look at 
    PyTorch Ignite (https://pytorch.org/ignite/)
    """
    def __init__(self, 
                 patience: int=5, 
                 min_delta: float=1e-9,
                 improvement: str='decrease',
                 monitored: str='validation',
                 record_best_weights: bool=True,
                 start_from_epoch: int=0,
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
        self.start_from_epoch = start_from_epoch
        self.best_loss = 1e12 if improvement.lower()=='decrease' else -1*1e12
        
        self.best_model=None
        self.counter = 0
        self.earlystop = False

    def __call__(self):
        return self.earlystop
    
    def early_stop(self, loss, count_add=1):
        
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
        self.best_model = copy.deepcopy(model).to(device='cpu').state_dict()
    
    def restore_best_weights(self, model):
        model.to(device='cpu')
        model.load_state_dict(self.best_model)

    def reset_counter(self):
        self.counter=0
        self.earlystop= False
        



class SSL_Base(nn.Module):
    """
    Contrastive SSL is the basic class for every implemented contrastive learning alghoritm
    """
    
    def __init__(self, encoder: nn.Module):
        super(SSL_Base, self).__init__()
        self.encoder = encoder
        
    def forward(self,x):
        pass
    
        
    def evaluate_loss(self, 
                      loss_fun: 'loss function', 
                      arguments, 
                      loss_arg: Union[list, dict]=None):
        """
        evaluate current batch loss
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
        """
        enc= copy.deepcopy(self.encoder).to(device=device)
        return enc
    
    def save_encoder(self, path: str=None):
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
        N_val = 0 if validation_dataloader is None else len(validation_dataloader)
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
                        pbar.set_description(f" train {batch_idx+1:8<}/{len(train_dataloader):8>}")
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
        with torch.no_grad():
            test_loss=0
            test_loss_tot=0
            with tqdm.tqdm(total=len(test_dataloader), ncols=100, 
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
                        pbar.set_description(f"   test {batch_idx+1:8<}/{len(test_dataloader):8>}")
                        pbar.set_postfix_str(f"test_loss={test_loss_tot/(batch_idx+1):.5f}")
                        pbar.update()
                            
    
                test_loss_tot /= (batch_idx+1)
        return test_loss_tot
    
    
    

class SimSiam(SSL_Base):
    
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
            verbose=0,
            device: str=None,
            validation_freq: int=1,
            return_loss_info: bool=False
           ):
        
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
            raise ValueError('loss_args must be a list or a dict with all optional arguments of the loss function')
        if verbose<0:
            verbose=0
        else:
            freq_list=[9e9, 10, 5, 3, 2, 1]
            verbose_freq= freq_list[min(int(verbose),5)]
        if validation_freq<1:
            raise ValueError('validation_freq must be >=1')
        perform_validation=False
        if validation_dataloader!=None:
            if not( isinstance(validation_dataloader, torch.utils.data.DataLoader)):
                raise ValueError('Current implementation accept only training data as a pytorch DataLoader')
            else:
                perform_validation=True

        loss_info={i: [None, None] for i in range(epochs)}
        for epoch in range(epochs):
            train_losses=[]
            val_losses=[]
            
            if not(self.training):
                self.train()
            
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
                
                # verbose print
                if verbose>0:
                    if batch_idx % verbose_freq == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch+1, (batch_idx+1) * train_dataloader.batch_size, len(train_dataloader.dataset),
                            100. * (batch_idx+1) / len(train_dataloader), train_loss.item()))
                train_losses.append(train_loss.item())

            if lr_scheduler!=None:
                lr_scheduler.step()
            
            # Perform validation if validation dataloader were given
            if perform_validation and ((epoch % validation_freq) == 0):
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
                    val_loss_tot /= len(validation_dataloader)
                    print(' -------------------------------------------------------------')
                    print(' ----------- VALIDATION LOSS AT EPOCH {}: {:.6f} ------------'.format(
                        epoch+1, val_loss_tot.item()))
                    print(' -------------------------------------------------------------')
                    val_losses.append(val_loss_tot.item())
                    
                    # Deal with earlystopper if given
                    if EarlyStopper!=None:
                        updated_mdl=False
                        EarlyStopper.early_stop(val_loss_tot)
                        if EarlyStopper.record_best_weights:
                            if EarlyStopper.best_loss==val_loss_tot:
                                EarlyStopper.rec_best_weights(self)
                                updated_mdl=True
                        if EarlyStopper():
                            print('no improvement after {} epochs. Training stopped.'.format(
                                EarlyStopper.patience*validation_freq))
                            if EarlyStopper.record_best_weights and not(updated_mdl):
                                EarlyStopper.restore_best_weights(self)
                            if return_loss_info:
                                return loss_info
                            else:
                                return
            
            if return_loss_info:
                loss_info[epoch]=[train_losses, val_losses]       
        if return_loss_info:
            return loss_info
                    

    
    def test(self, 
             test_dataloader,
             augmenter=None,
             loss_func: 'function'= None, 
             loss_args: list or dict=[],
             verbose: int=0,
             device: str=None
            ):
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
            loss_func=Loss.SimSiam_loss   
        if not( isinstance(loss_args,list) or isinstance(loss_args,dict) or loss_args==None):
            raise ValueError('loss_args must be a list or a dict with all optional arguments of the loss function')
        if verbose<0:
            verbose=0
        else:
            freq_list=[1e9, 10, 5, 3, 2, 1]
            verbose_freq= freq_list[min(int(verbose),5)]
        
        self.eval()
        with torch.no_grad():
            test_loss=0
            test_loss_tot=0
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
                if verbose>0:
                    if batch_idx % verbose_freq == 0:
                        print('Test loss : [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            (batch_idx+1) * test_dataloader.batch_size, len(test_dataloader.dataset),
                            100. * (batch_idx+1) / len(test_dataloader), test_loss.item()))
                        

            test_loss_tot /= len(test_dataloader)
            if verbose>0:
                print(' -------------------------------------------------------------')
                print(' -----------    TEST LOSS: {:.6f}    ------------'.format(test_loss_tot.item()))
                print(' -------------------------------------------------------------')
        return test_loss_tot
    
    
    
    
class MoCo(SSL_Base):
    """
    Implementation of the third version of MoCo. This class also make the possibility to use 
    the memory bank as in previous MoCo versions.
    """
    
    def __init__(self, 
                 encoder: nn.Module, 
                 projection_head: Union[list[int], nn.Module],
                 predictor: Union[list[int], nn.Module]=None,
                 feat_size: int=-1,
                 bank_size: int=0,
                 m: float=0.99
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
        for param_b, param_m in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * self.m + param_b.data * (1. - self.m)
        for param_b, param_m in zip(self.projection_head.parameters(), self.momentum_projection_head.parameters()):
            param_m.data = param_m.data * self.m + param_b.data * (1. - self.m)

    @torch.no_grad()
    def _update_queue(self, keys):
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
            verbose=0,
            device: str=None,
            validation_freq: int=1,
            return_loss_info: bool=False
           ):
        
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
            loss_func=Loss.MoCo_loss   
        if not( isinstance(loss_args,list) or isinstance(loss_args,dict) or loss_args==None):
            raise ValueError('loss_args must be a list or a dict with all optional arguments of the loss function')
        if verbose<0:
            verbose=0
        else:
            freq_list=[9e9, 10, 5, 3, 2, 1]
            verbose_freq= freq_list[min(int(verbose),5)]
        if validation_freq<1:
            raise ValueError('validation_freq must be >=1')
        perform_validation=False
        if validation_dataloader!=None:
            if not( isinstance(validation_dataloader, torch.utils.data.DataLoader)):
                raise ValueError('Current implementation accept only training data as a pytorch DataLoader')
            else:
                perform_validation=True

        loss_info={i: [None, None] for i in range(epochs)}
        for epoch in range(epochs):
            train_losses=[]
            val_losses=[]
            
            if not(self.training):
                self.train()
            
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
                    train_loss = self.evaluate_loss(loss_func, [q, k] , loss_args )
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

                # verbose print
                if verbose>0:
                    if batch_idx % verbose_freq == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch+1, (batch_idx+1) * train_dataloader.batch_size, len(train_dataloader.dataset),
                            100. * (batch_idx+1) / len(train_dataloader), train_loss.item()))
                train_losses.append(train_loss.item())

            if lr_scheduler!=None:
                lr_scheduler.step()
            
            # Perform validation if validation dataloader were given
            # Note that validation in moco can be misleading if there's a memory_bank
            # since calculated keys cannot be added
            if perform_validation and ((epoch % validation_freq) == 0):
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
                            val_loss = self.evaluate_loss(loss_func, [q, k] , loss_args )
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

                    val_loss_tot /= len(validation_dataloader)
                    print(' -------------------------------------------------------------')
                    print(' ----------- VALIDATION LOSS AT EPOCH {}: {:.6f} ------------'.format(
                        epoch+1, val_loss_tot.item()))
                    print(' -------------------------------------------------------------')
                    val_losses.append(val_loss_tot.item())
                    
                    # Deal with earlystopper if given
                    if EarlyStopper!=None:
                        updated_mdl=False
                        EarlyStopper.early_stop(val_loss_tot)
                        if EarlyStopper.record_best_weights:
                            if EarlyStopper.best_loss==val_loss_tot:
                                EarlyStopper.rec_best_weights(self)
                                updated_mdl=True
                        if EarlyStopper():
                            print('no improvement after {} epochs. Training stopped.'.format(
                                EarlyStopper.patience*validation_freq))
                            if EarlyStopper.record_best_weights and not(updated_mdl):
                                EarlyStopper.restore_best_weights(self)
                            if return_loss_info:
                                return loss_info
                            else:
                                return
            
            if return_loss_info:
                loss_info[epoch]=[train_losses, val_losses]
        
        if return_loss_info:
            return loss_info
                    

    
    def test(self, 
             test_dataloader,
             augmenter=None,
             loss_func: 'function'= None, 
             loss_args: list or dict=[],
             verbose: int=0,
             device: str=None
            ):
        
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
            los_func=Loss.Moco_loss   
        if not( isinstance(loss_args,list) or isinstance(loss_args,dict) or loss_args==None):
            raise ValueError('loss_args must be a list or a dict with all optional arguments of the loss function')
        if verbose<0:
            verbose=0
        else:
            freq_list=[1e9, 10, 5, 3, 2, 1]
            verbose_freq= freq_list[min(int(verbose),5)]
        
        self.eval()
        with torch.no_grad():
            test_loss=0
            test_loss_tot=0
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
                    test_loss = self.evaluate_loss(loss_func, [q, k] , loss_args )
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
                if verbose>0:
                    if batch_idx % verbose_freq == 0:
                        print('Test loss : [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            (batch_idx+1) * test_dataloader.batch_size, len(test_dataloader.dataset),
                            100. * (batch_idx+1) / len(test_dataloader), test_loss.item()))
                        

            test_loss_tot /= len(test_dataloader)
            if verbose>0:
                print(' -------------------------------------------------------------')
                print(' -----------    TEST LOSS: {:.6f}    ------------'.format(test_loss_tot.item()))
                print(' -------------------------------------------------------------')
        return test_loss_tot




class BYOL(SSL_Base):
    """
    Implementation of BYOL.
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
        for param_b, param_m in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * self.m + param_b.data * (1. - self.m)
        for param_b, param_m in zip(self.projection_head.parameters(), self.momentum_projection_head.parameters()):
            param_m.data = param_m.data * self.m + param_b.data * (1. - self.m)
    
    def forward(self, x):
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
            verbose=0,
            device: str=None,
            validation_freq: int=1,
            return_loss_info: bool=False
           ):
        
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
            loss_func=Loss.BYOL_loss   
        if not( isinstance(loss_args,list) or isinstance(loss_args,dict) or loss_args==None):
            raise ValueError('loss_args must be a list or a dict with all optional arguments of the loss function')
        if verbose<0:
            verbose=0
        else:
            freq_list=[9e9, 10, 5, 3, 2, 1]
            verbose_freq= freq_list[min(int(verbose),5)]
        if validation_freq<1:
            raise ValueError('validation_freq must be >=1')
        perform_validation=False
        if validation_dataloader!=None:
            if not( isinstance(validation_dataloader, torch.utils.data.DataLoader)):
                raise ValueError('Current implementation accept only training data as a pytorch DataLoader')
            else:
                perform_validation=True

        loss_info={i: [None, None] for i in range(epochs)}
        for epoch in range(epochs):
            train_losses=[]
            val_losses=[]
            
            if not(self.training):
                self.train()
            
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

                # verbose print
                if verbose>0:
                    if batch_idx % verbose_freq == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch+1, (batch_idx+1) * train_dataloader.batch_size, len(train_dataloader.dataset),
                            100. * (batch_idx+1) / len(train_dataloader), train_loss.item()))
                train_losses.append(train_loss.item())

            if lr_scheduler!=None:
                lr_scheduler.step()
            
            # Perform validation if validation dataloader were given
            # Note that validation in moco can be misleading if there's a memory_bank
            # since calculated keys cannot be added
            if perform_validation and ((epoch % validation_freq) == 0):
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
                        p2 = self(data_aug2)
                        z1 = self.momentum_encoder(data_aug1)
                        z1 = self.momentum_projection_head(z1)
                        z1 = z1.detach() # keys
                        z2 = self.momentum_encoder(data_aug2)
                        z2 = self.momentum_projection_head(z2)
                        z2 = z2.detach() # keys
                        val_loss = self.evaluate_loss(loss_func, [p1,z1,p2,z2] , loss_args )
        
                        val_loss_tot += val_loss

                    val_loss_tot /= len(validation_dataloader)
                    print(' -------------------------------------------------------------')
                    print(' ----------- VALIDATION LOSS AT EPOCH {}: {:.6f} ------------'.format(
                        epoch+1, val_loss_tot.item()))
                    print(' -------------------------------------------------------------')
                    val_losses.append(val_loss_tot.item())
                    
                    # Deal with earlystopper if given
                    if EarlyStopper!=None:
                        updated_mdl=False
                        EarlyStopper.early_stop(val_loss_tot)
                        if EarlyStopper.record_best_weights:
                            if EarlyStopper.best_loss==val_loss_tot:
                                EarlyStopper.rec_best_weights(self)
                                updated_mdl=True
                        if EarlyStopper():
                            print('no improvement after {} epochs. Training stopped.'.format(
                                EarlyStopper.patience*validation_freq))
                            if EarlyStopper.record_best_weights and not(updated_mdl):
                                EarlyStopper.restore_best_weights(self)
                            if return_loss_info:
                                return loss_info
                            else:
                                return
            
            if return_loss_info:
                loss_info[epoch]=[train_losses, val_losses]
        
        if return_loss_info:
            return loss_info
                    

    
    def test(self, 
             test_dataloader,
             augmenter=None,
             loss_func: 'function'= None, 
             loss_args: list or dict=[],
             verbose: int=0,
             device: str=None
            ):
        
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
        if verbose<0:
            verbose=0
        else:
            freq_list=[1e9, 10, 5, 3, 2, 1]
            verbose_freq= freq_list[min(int(verbose),5)]
        
        self.eval()
        with torch.no_grad():
            test_loss=0
            test_loss_tot=0
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
                if verbose>0:
                    if batch_idx % verbose_freq == 0:
                        print('Test loss : [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            (batch_idx+1) * test_dataloader.batch_size, len(test_dataloader.dataset),
                            100. * (batch_idx+1) / len(test_dataloader), test_loss.item()))
                        

            test_loss_tot /= len(test_dataloader)
            if verbose>0:
                print(' -------------------------------------------------------------')
                print(' -----------    TEST LOSS: {:.6f}    ------------'.format(test_loss_tot.item()))
                print(' -------------------------------------------------------------')
        return test_loss_tot



class Barlow_Twins(SimCLR):
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
            verbose=0,
            device: str=None,
            validation_freq: int=1,
            cat_augmentations: bool=False,
            return_loss_info: bool=False
           ):
        
        if loss_func==None:
            loss_func=Loss.Barlow_loss
            loss_args=[]
        super().fit(train_dataloader, epochs, optimizer, augmenter, loss_func, loss_args, 
                    lr_scheduler, EarlyStopper, validation_dataloader, verbose, device,
                    validation_freq, cat_augmentations, return_loss_info
                   ) 

    def test(self, 
             test_dataloader,
             augmenter=None,
             loss_func: 'function'= None, 
             loss_args: list or dict=[],
             verbose: int=0,
             device: str=None
            ):
        
        if loss_func==None:
            loss_func=Loss.Barlow_loss
            loss_args=[]
        super().test(test_dataloader, augmenter, loss_func, loss_args,  verbose, device) 



class VICReg(SimCLR):
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
            verbose=0,
            device: str=None,
            validation_freq: int=1,
            cat_augmentations: bool=False,
            return_loss_info: bool=False
           ):
        
        if loss_func==None:
            loss_func=Loss.VICReg_loss
            loss_args=[]
        super().fit(train_dataloader, epochs, optimizer, augmenter, loss_func, loss_args, 
                    lr_scheduler, EarlyStopper, validation_dataloader, verbose, device,
                    validation_freq, cat_augmentations, return_loss_info
                   ) 

    def test(self, 
             test_dataloader,
             augmenter=None,
             loss_func: 'function'= None, 
             loss_args: list or dict=[],
             verbose: int=0,
             device: str=None
            ):
        
        if loss_func==None:
            loss_func=Loss.VICReg_loss
            loss_args=[]
        super().test(test_dataloader, augmenter, loss_func, loss_args,  verbose, device) 

