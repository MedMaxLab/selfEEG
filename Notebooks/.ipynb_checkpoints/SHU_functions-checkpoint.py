import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score, 
                              precision_score, recall_score, roc_auc_score,
                             classification_report)

# IMPORT TORCH
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from selfeeg import dataloading as dl



def count_classes(dataloader, 
                  nclasses: int=2, 
                  class_name: list[str]=None
                 ):
    class_count=[0]*nclasses
    N = range(nclasses)
    if class_name==None:
        class_name=[i for i in N]
    for X , y in dataloader:
        for i in N:
            try: 
                class_count[i]= class_count[i] + sum(1 for x in y if x in class_name[i])
            except:
                class_count[i]= class_count[i] + sum(1 for x in y if x == class_name[i])
    return class_count

def plot_loss_progression(loss_info):
    # PLOT LOSS PER EPOCH
    train_loss= []
    val_loss= []
    if loss_info==[]:
        print('No available loss_info')
    else:
        for i in list(loss_info.keys()):
            if loss_info[i][0]==None:
                loss_info.pop(i,None)
        for i in range(len(loss_info)):
            train_loss.append( loss_info[i][0])
            val_loss.append( loss_info[i][1])
        
        epochs=[i for i in range(1, len(loss_info)+1)]
        
        plt.plot(epochs, train_loss, 'o-')
        plt.plot(epochs, val_loss, 'o-')
        plt.title('Loss Progression')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['Training loss', 'validation loss'])
        plt.show()


def evaluate_model(Mdl=None, 
                   loader_to_evaluate=None,
                   device= 'cpu',
                   Class_thresh=0.5,
                   ytrue = None,
                   ypred = None,
                   ws_box = False,
                   cs_box = False,
                   subj_box = False
                  ):

    nb_classes=2
    if (ytrue==None) and (ypred==None):
        Mdl.eval()
        Mdl.to(device=device)
        
        ytrue=torch.zeros(len(loader_to_evaluate.dataset))
        ypred=torch.zeros_like(ytrue)
        
        cnt=0
        for i, (X, Y) in enumerate(loader_to_evaluate):
            X=X.to(device=device)
            ytrue[cnt:cnt+X.shape[0]]= Y 
            
            # GET PREDICTIONS
            with torch.no_grad():
                yhat = torch.sigmoid(Mdl(X)).to(device='cpu')
                ypred[cnt:cnt+X.shape[0]] = torch.squeeze(yhat) 
            cnt += X.shape[0]

    Class_thresh=0.5
    index1  = ['Left', 'Right']
    labels1 = [0,1]
    # MAKE CONFUSION MATRIX
    ypred2 = ypred > Class_thresh
    ytrue2=ytrue.flatten()
    ypred2= ypred2.flatten()
    
    ConfMat = confusion_matrix(ytrue2, ypred2, labels=labels1).T
    ConfMat_df = pd.DataFrame(ConfMat, index = index1, columns = index1)
    ConfMat2=ConfMat
    Acc_mat = confusion_matrix(ytrue2, ypred2, labels=labels1, normalize='true').T
    Acc_mat_df = pd.DataFrame(Acc_mat, index = index1, columns = index1)

    # PLOT CONFUSION MATRIX
    vmin = np.min(ConfMat)
    vmax = np.max(ConfMat)
    off_diag_mask = np.eye(*ConfMat.shape, dtype=bool)
    
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    sns.heatmap(ConfMat_df, vmin= 0, vmax=vmax, mask=~off_diag_mask, 
                annot=True, cmap='Blues', linewidths=1)
    sns.heatmap(ConfMat_df, annot=True, mask=off_diag_mask, cmap='OrRd', 
                vmin=vmin, vmax=vmax, cbar_kws=dict(ticks=[]), linewidths=1)
    plt.xlabel('true labels', fontsize=15)
    plt.ylabel('predicted labels', fontsize=15)
    plt.title('Accuracy Absolute Values', fontsize=15)
    
    plt.subplot(1,2,2)
    sns.heatmap(Acc_mat_df, vmin= 0, vmax=1, mask=~off_diag_mask, 
                annot=True, cmap='Blues', linewidths=1)
    sns.heatmap(Acc_mat_df, annot=True, mask=off_diag_mask, cmap='OrRd', 
                vmin=0, vmax=1, cbar_kws=dict(ticks=[]), linewidths=1)
    plt.xlabel('true labels', fontsize=15)
    plt.ylabel('predicted labels', fontsize=15)
    plt.title('Accuracy normalized on True Values', fontsize=15)
    plt.show()

    F1_mat = f1_score(ytrue2, ypred2, average=None, zero_division=0.0)
    prec_mat = precision_score(ytrue2, ypred2, average=None, zero_division=0.0)
    rec_mat = recall_score(ytrue2, ypred2, average=None, zero_division=0.0)
    
    Accuracy = accuracy_score(ytrue2, ypred2)
    
    F1_micro = f1_score(ytrue2, ypred2, average='micro', zero_division=0.0)
    F1_macro = f1_score(ytrue2, ypred2, average='macro', zero_division=0.0)
    F1_macro_weighted = f1_score(ytrue2, ypred2, average='weighted',zero_division=0.0)
    
    precision_micro = precision_score(ytrue2, ypred2, average='micro',zero_division=0.0)
    precision_macro = precision_score(ytrue2, ypred2, average='macro',zero_division=0.0)
    precision_macro_weighted = precision_score(ytrue2, ypred2, 
                                               average='weighted',zero_division=0.0)
    
    recall_micro = recall_score(ytrue2, ypred2, average='micro',zero_division=0.0)
    recall_macro = recall_score(ytrue2, ypred2, average='macro',zero_division=0.0)
    recall_macro_weighted = recall_score(ytrue2, ypred2, average='weighted',zero_division=0.0)
    
    if nb_classes<=2:
        AUC_micro = roc_auc_score(ytrue2, ypred2, average='micro')
        AUC_macro = roc_auc_score(ytrue2, ypred2, average='macro')
        AUC_macro_weighted = roc_auc_score(ytrue2, ypred2, average='weighted')
        F1_bin = f1_score(ytrue2, ypred2, zero_division=0.0)
        precision_bin = precision_score(ytrue2, ypred2, zero_division=0.0)
        recall_bin = recall_score(ytrue2, ypred2, zero_division=0.0)

    print(' |-----------------------------------------|')
    print(' |              OTHER METRICS              |')
    print(' |-----------------------------------------|')
    print(' |Accuracy score:                 %.3f    |' % Accuracy) 
    print(' |-----------------------------------------|')
    print(' |Precision binary score:         %.3f    |' %precision_bin ) if nb_classes<=2 else None
    print(' |Precision micro score:          %.3f    |' %precision_micro )
    print(' |Precision macro score:          %.3f    |' %precision_macro )
    print(' |Precision macro weighted score: %.3f    |' %precision_macro_weighted )
    print(' |-----------------------------------------|')
    print(' |Recall binary score:            %.3f    |' %recall_bin ) if nb_classes<=2 else None
    print(' |Recall micro score:             %.3f    |' %recall_micro )
    print(' |Recall macro score:             %.3f    |' %recall_macro )
    print(' |Recall macro weighted score:    %.3f    |' %recall_macro_weighted )
    print(' |-----------------------------------------|')
    print(' |F1 binary score:                %.3f    |' %F1_bin ) if nb_classes<=2 else None
    print(' |F1 micro score:                 %.3f    |' %F1_micro )
    print(' |F1 macro score:                 %.3f    |' %F1_macro )
    print(' |F1 macro weighted score:        %.3f    |' %F1_macro_weighted )
    print(' |-----------------------------------------|')
    if nb_classes<=2:
        print(' |AUC micro score:                %.3f    |' %AUC_micro )
        print(' |AUC macro score:                %.3f    |' %AUC_macro )
        print(' |AUC macro weighted score:       %.3f    |' %AUC_macro_weighted )
        print(' |-----------------------------------------|')

    print('\n\n                    classification report')
    print(classification_report(ytrue2, ypred2, target_names= index1))

    ws_acc= []
    cs_acc= [] 
    cs_cleaned = []
    subj_acc = [] 
    subj_cleaned = []
    
    if ws_box:
        ws_acc = []
        curr_subj = int(loader_to_evaluate.dataset.EEGlenTrain['file_name'][0].split('_')[1])
        curr_sess =  int(loader_to_evaluate.dataset.EEGlenTrain['file_name'][0].split('_')[2])
        curr_lab = []
        curr_pred = []
        for i in range(len(loader_to_evaluate.dataset.EEGlenTrain)):
            subj = int(loader_to_evaluate.dataset.EEGlenTrain['file_name'][i].split('_')[1])
            sess = int(loader_to_evaluate.dataset.EEGlenTrain['file_name'][i].split('_')[2])
            if subj==curr_subj and sess==curr_sess:
                curr_lab.append(ytrue2[i])
                curr_pred.append(ypred2[i])
            else:
                curr_subj=subj
                curr_sess=sess
                ws_acc.append(accuracy_score(curr_lab, curr_pred))
                curr_lab = []
                curr_pred = []
        ws_acc = np.array(ws_acc)
        plt.figure(figsize=(4,6))
        sns.set(style='whitegrid')
        sns.boxplot(ws_acc, width=0.3, color='lightblue', 
                    medianprops=dict(color="red", alpha=0.7))
        plt.yticks([i/10 for i in range(1,11)]) 
        plt.title("Whithin session accuracy", fontsize=16)
        plt.xlabel('')  # remove unuseful xlabel ('Variable')
        plt.ylabel("Accuracy", fontsize=14)
        plt.show()

    if cs_box:
        cs_acc = np.zeros((25,5)) -1
        for k in range(5):
            curr_subj = int(loader_to_evaluate.dataset.EEGlenTrain['file_name'][0].split('_')[1])
            curr_subj = -1
            curr_sess =  k+1
            curr_lab = []
            curr_pred = []
            for i in range(len(loader_to_evaluate.dataset.EEGlenTrain)):
                subj = int(loader_to_evaluate.dataset.EEGlenTrain['file_name'][i].split('_')[1])
                sess = int(loader_to_evaluate.dataset.EEGlenTrain['file_name'][i].split('_')[2])
                if sess==curr_sess:
                    if subj==curr_subj:
                        curr_lab.append(ytrue2[i])
                        curr_pred.append(ypred2[i])
                    else:
                        if curr_lab!=[]:
                            cs_acc[curr_subj-1,k] =accuracy_score(curr_lab, curr_pred)
                        curr_subj=subj
                        curr_lab = []
                        curr_pred = []
            if curr_lab!=[]:
                cs_acc[curr_subj-1,k] =accuracy_score(curr_lab, curr_pred)
        cs_cleaned = [ cs_acc[cs_acc[:,i]>0,i] for i in range(5)]
        plt.figure(figsize=(5,6))
        sns.set(style='whitegrid')
        sns.boxplot(cs_cleaned, width=0.3, color='lightblue', 
                    medianprops=dict(color="red", alpha=0.7))
        plt.title("Cross session accuracy", fontsize=16)
        plt.xticks([0, 1, 2, 3, 4], ['s1', 's2', 's3', 's4','s5'])
        plt.yticks([i/10 for i in range(1,11)]) 
        plt.xlabel('session', fontsize=14)  # remove unuseful xlabel ('Variable')
        plt.ylabel("Accuracy", fontsize=14)
        plt.show()
    
    
    if cs_box or ws_box or subj_box:
        boxes = {'within_session': ws_acc, 
                 'cross_session': (cs_acc, cs_cleaned),
                 'cross_subj': (subj_acc, subj_cleaned)
                }
        return ConfMat, ytrue, ypred , boxes
    else:
        return ConfMat, ytrue, ypred