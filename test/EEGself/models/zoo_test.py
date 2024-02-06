import unittest

import os
import sys
import itertools
import numpy as np
import torch
from selfeeg import models


class TestModels(unittest.TestCase):

    def makeGrid(self, pars_dict):
        keys=pars_dict.keys()
        combinations=itertools.product(*pars_dict.values())
        ds=[dict(zip(keys,cc)) for cc in combinations]
        return ds

    @classmethod
    def setUpClass(cls):
        cls.device= torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        if cls.device.type == 'cpu':
            cls.device  = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print('\n---------------------')
        print('TESTING MODELS MODULE')
        if cls.device.type != 'cpu':
            print('Found gpu device: testing module with both cpu and gpu')
        else:
            print('Didn\'t found cuda device: testing module with only cpu')
        print('---------------------')
        cls.N       = 2
        cls.Chan    = 8
        cls.Samples = 2048
        cls.x   = torch.randn(cls.N, cls.Chan, cls.Samples)
        cls.xl  = torch.randn(cls.N, 1, 16, cls.Samples)
        cls.xd  = torch.randn(cls.N,128)
        if cls.device.type != 'cpu':
            cls.x2  = torch.randn(cls.N,cls.Chan,cls.Samples).to(device=cls.device)
            cls.xl2 = torch.randn(cls.N,1,16,cls.Samples).to(device=cls.device)
            cls.xd2 = torch.randn(cls.N,128).to(device=cls.device)

    def setUp(self):
        torch.manual_seed(1234)

    def test_DepthwiseConv2d(self):
        print('Testing Depthwise conv2d with max norm constraint...', end="", flush=True)
        Depthwise_args = {'in_channels': [1],'depth_multiplier':[2,3,4],
                          'kernel_size': [(1,64),(5,1),(5,64)], 'stride':[1,2,3],
                         'dilation':[1,2], 'bias': [True, False],
                          'max_norm':[None, 2, 3], 'padding': ['valid']
                        }
        Depthwise_args = self.makeGrid(Depthwise_args)
        for i in Depthwise_args:
            model = models.DepthwiseConv2d(**i)
            model.weight = torch.nn.Parameter(model.weight*10)
            out = model(self.xl)
            if i['max_norm'] is not None:
                norm= model.weight.norm(dim=2, keepdim=True).norm(dim=3,keepdim=True).squeeze()
                self.assertEqual((norm>i['max_norm']).sum(), 0)
            self.assertEqual(torch.isnan(out).sum(),0)
            self.assertEqual(out.shape[1],i['depth_multiplier'])

        if self.device.type != 'cpu':
            for i in Depthwise_args:
                model = models.DepthwiseConv2d(**i).to(device=self.device)
                model.weight = torch.nn.Parameter(model.weight*10)
                out = model(self.xl2)
                if i['max_norm'] is not None:
                    norm= model.weight.norm(dim=2, keepdim=True).norm(dim=3,keepdim=True).squeeze()
                    self.assertEqual((norm>i['max_norm']).sum(), 0)
                self.assertEqual(torch.isnan(out).sum(),0)
                self.assertEqual(out.shape[1],i['depth_multiplier'])
        print('   Depthwise conv2d OK: tested', len(Depthwise_args), ' combinations of input arguments')


    def test_SeparableConv2d(self):
        print('Testing Separable conv2d with norm constraint...', end="", flush=True)
        Separable_args = {'in_channels': [1],'out_channels': [5,16], 'depth_multiplier':[1,3],
                          'kernel_size': [(1,64),(5,1),(5,64)], 'stride':[1,2,3],
                          'dilation':[1,2], 'bias': [True, False], 'depth_max_norm':[None, 2, 3],
                          'padding': ['valid']
                         }
        Separable_args = self.makeGrid(Separable_args)
        for i in Separable_args:
            model = models.SeparableConv2d(**i)
            out = model(self.xl)
            self.assertEqual(torch.isnan(out).sum(),0)
            self.assertEqual(out.shape[1],i['out_channels'])

        if self.device.type != 'cpu':
            for i in Separable_args:
                model = models.SeparableConv2d(**i).to(device=self.device)
                out = model(self.xl2)
                self.assertEqual(torch.isnan(out).sum(),0)
                self.assertEqual(out.shape[1],i['out_channels'])
        print('   Separable conv2d OK: tested', len(Separable_args), 'combinations of input arguments')


    def test_ConstrainedConv2d(self):
        print('Testing conv2d with max norm constraint...', end="", flush=True)
        Conv_args = {'in_channels': [1],'out_channels':[5,16], 'kernel_size': [(1,64),(5,1),(5,64)],
                     'stride':[1,2,3], 'dilation':[1,2], 'bias': [True, False], 'max_norm':[None, 2, 3],
                     'padding': ['valid']
                    }
        Conv_args = self.makeGrid(Conv_args)
        for i in Conv_args:
            model = models.ConstrainedConv2d(**i)
            model.weight = torch.nn.Parameter(model.weight*10)
            out = model(self.xl)
            if i['max_norm'] is not None:
                norm = model.weight.norm(dim=2, keepdim=True).norm(dim=3,keepdim=True).squeeze()
                self.assertEqual((norm>i['max_norm']).sum(), 0)
            self.assertEqual(torch.isnan(out).sum(),0)

        if self.device.type != 'cpu':
            for i in Conv_args:
                model = models.ConstrainedConv2d(**i).to(device=self.device)
                model.weight = torch.nn.Parameter(model.weight*10)
                out = model(self.xl2)
                if i['max_norm'] is not None:
                    norm = model.weight.norm(dim=2, keepdim=True).norm(dim=3,keepdim=True).squeeze()
                    self.assertEqual((norm>i['max_norm']).sum(), 0)
                self.assertEqual(torch.isnan(out).sum(),0)
        print('   Constrained conv2d OK: tested', len(Conv_args), ' combinations of input arguments')

    def test_ConstrainedDense(self):
        print('Testing Dense layer with max norm constraint...', end="", flush=True)
        Dense_args = {'in_features': [128],'out_features':[32],
                      'bias': [True,False], 'max_norm':[None, 2, 3]}
        Dense_args = self.makeGrid(Dense_args)
        for i in Dense_args:
            model = models.ConstrainedDense(**i)
            model.weight = torch.nn.Parameter(model.weight*10)
            out = model(self.xd)
            if i['max_norm'] is not None:
                norm= model.weight.norm(dim=1, keepdim=True)
                self.assertEqual((norm>i['max_norm']).sum(), 0)
            self.assertEqual(torch.isnan(out).sum(),0)
            self.assertEqual(out.shape[1], 32)

        if self.device.type != 'cpu':
            for i in Dense_args:
                model = models.ConstrainedDense(**i).to(device=self.device)
                model.weight = torch.nn.Parameter(model.weight*10)
                out = model(self.xd2)
                if i['max_norm'] is not None:
                    norm= model.weight.norm(dim=1, keepdim=True)
                    self.assertEqual((norm>i['max_norm']).sum(), 0)
                self.assertEqual(torch.isnan(out).sum(),0)
                self.assertEqual(out.shape[1], 32)
        print('   Dense layer OK: tested', len(Dense_args), ' combinations of input arguments')


    def test_DeepConvNet(self):
        print('Testing DeepConvNet...', end="", flush=True)
        DCN_args = {'nb_classes': [2,4], 'Chans': [self.Chan],
                    'Samples': [self.Samples], 'kernLength':[10,20],
                    'F': [12,25], 'Pool': [3,4], 'stride': [3,4], 'max_norm': [2.0],
                    'batch_momentum': [0.9], 'ELUalpha': [1], 'dropRate': [0.5],
                    'max_dense_norm': [1.0], 'return_logits': [True,False]
                   }
        DCN_grid = self.makeGrid(DCN_args)
        for i in DCN_grid:
            model = models.DeepConvNet(**i)
            out = model(self.x)
            self.assertEqual(torch.isnan(out).sum(),0)
            self.assertEqual(out.shape[1], i['nb_classes'] if i['nb_classes']>2 else 1)
            if not(i['return_logits']):
                self.assertLessEqual(out.max(), 1)
                self.assertGreaterEqual(out.min(), 0)
        if self.device.type != 'cpu':
            for i in DCN_grid:
                model = models.DeepConvNet(**i).to(device=self.device)
                out = model(self.x2)
                self.assertEqual(torch.isnan(out).sum(),0)
                self.assertEqual(out.shape[1], i['nb_classes'] if i['nb_classes']>2 else 1)
                if not(i['return_logits']):
                    self.assertLessEqual(out.max(), 1)
                    self.assertGreaterEqual(out.min(), 0)
        print('   DeepConvNet OK: tested ', len(DCN_grid), ' combinations of input arguments')

    def test_EEGInception(self):
        print('Testing EEGInception...', end="", flush=True)
        EEGin_args = {'nb_classes': [2,4], 'Chans': [self.Chan],
                    'Samples': [self.Samples], 'kernel_size':[32,128],
                    'F1': [4,16], 'D': [2,4], 'pool': [4, 8],
                    'batch_momentum': [0.9], 'dropRate': [0.5],
                    'max_depth_norm': [1.0], 'return_logits': [True,False], 'bias':[True,False]
                   }
        EEGin_args = self.makeGrid(EEGin_args)
        for i in EEGin_args:
            model = models.EEGInception(**i)
            out = model(self.x)
            self.assertEqual(torch.isnan(out).sum(),0)
            self.assertEqual(out.shape[1], i['nb_classes'] if i['nb_classes']>2 else 1)
            if not(i['return_logits']):
                self.assertLessEqual(out.max(), 1)
                self.assertGreaterEqual(out.min(), 0)

        if self.device.type != 'cpu':
            for i in EEGin_args:
                model = models.EEGInception(**i).to(device=self.device)
                out = model(self.x2)
                self.assertEqual(torch.isnan(out).sum(),0)
                self.assertEqual(out.shape[1], i['nb_classes'] if i['nb_classes']>2 else 1)
                if not(i['return_logits']):
                    self.assertLessEqual(out.max(), 1)
                    self.assertGreaterEqual(out.min(), 0)
        print('   EEGInception OK: tested', len(EEGin_args), ' combinations of input arguments')


# In[8]:

    def test_EEGNet(self):
        print('Testing EEGnet...', end="", flush=True)
        EEGnet_args = {'nb_classes': [2,4], 'Chans': [self.Chan],
                       'Samples': [self.Samples], 'kernLength':[32,64,128],
                       'F1': [4,8,16], 'D': [2,4], 'F2':[8,16,32], 'pool1': [4, 8],
                       'pool2':[8,16], 'separable_kernel':[16,32],
                       'return_logits': [True,False]
                      }
        EEGnet_args = self.makeGrid(EEGnet_args)
        for i in EEGnet_args:
            model = models.EEGNet(**i)
            out = model(self.x)
            self.assertEqual(torch.isnan(out).sum(),0)
            self.assertEqual(out.shape[1], i['nb_classes'] if i['nb_classes']>2 else 1)
            if not(i['return_logits']):
                self.assertLessEqual(out.max(), 1)
                self.assertGreaterEqual(out.min(), 0)

        if self.device.type != 'cpu':
            for i in EEGnet_args:
                model = models.EEGNet(**i).to(device=self.device)
                out = model(self.x2)
                self.assertEqual(torch.isnan(out).sum(),0)
                self.assertEqual(out.shape[1], i['nb_classes'] if i['nb_classes']>2 else 1)
                if not(i['return_logits']):
                    self.assertLessEqual(out.max(), 1)
                    self.assertGreaterEqual(out.min(), 0)
        print('   EEGnet OK: tested', len(EEGnet_args), ' combinations of input arguments')


# In[9]:

    def test_EEGSym(self):
        print('Testing EEGsym...', end="", flush=True)
        EEGsym_args = {'nb_classes': [2,4],'Samples':[2048], 'Chans': [self.Chan],
                       'Fs': [64], 'scales_time':[(500,250,125),(250,183,95)],
                       'lateral_chans': [2,3], 'first_left': [True,False], 'F':[8,24],
                       'pool':[2,3],
                       'bias':[True,False],
                       'return_logits': [True,False]
                      }
        EEGsym_args = self.makeGrid(EEGsym_args)
        for i in EEGsym_args:
            model = models.EEGSym(**i)
            out = model(self.x)
            self.assertEqual(torch.isnan(out).sum(),0)
            self.assertEqual(out.shape[1], i['nb_classes'] if i['nb_classes']>2 else 1)
            if not(i['return_logits']):
                self.assertLessEqual(out.max(), 1)
                self.assertGreaterEqual(out.min(), 0)

        if self.device.type != 'cpu':
            for i in EEGsym_args:
                model = models.EEGSym(**i).to(device=self.device)
                out = model(self.x2)
                self.assertEqual(torch.isnan(out).sum(),0)
                self.assertEqual(out.shape[1], i['nb_classes'] if i['nb_classes']>2 else 1)
                if not(i['return_logits']):
                    self.assertLessEqual(out.max(), 1)
                    self.assertGreaterEqual(out.min(), 0)
        print('   EEGsym OK: tested', len(EEGsym_args), ' combinations of input arguments')


# In[10]:

    def test_EEGSym(self):
        print('Testing ResNet...', end="", flush=True)
        EEGres_args = {'nb_classes': [2,4],'Samples':[2048], 'Chans': [self.Chan],
                       'block':[models.BasicBlock1],
                       'Layers': [[1,1,1,1],[2,2,2,2],[1,2,4,3]], 'inplane':[8,16,32],
                       'kernLength': [7,13,15], 'addConnection': [True, False],
                       'return_logits': [True,False]
                      }
        EEGres_args = self.makeGrid(EEGres_args)
        for i in EEGres_args:
            model = models.ResNet1D(**i)
            out = model(self.x)
            self.assertEqual(torch.isnan(out).sum(),0)
            self.assertEqual(out.shape[1], i['nb_classes'] if i['nb_classes']>2 else 1)
            if not(i['return_logits']):
                self.assertLessEqual(out.max(), 1)
                self.assertGreaterEqual(out.min(), 0)

        if self.device.type != 'cpu':
            for i in EEGres_args:
                model = models.ResNet1D(**i).to(device=self.device)
                out = model(self.x2)
                self.assertEqual(torch.isnan(out).sum(),0)
                self.assertEqual(out.shape[1], i['nb_classes'] if i['nb_classes']>2 else 1)
                if not(i['return_logits']):
                    self.assertLessEqual(out.max(), 1)
                    self.assertGreaterEqual(out.min(), 0)
        print('   ResNet OK: tested', len(EEGres_args), ' combinations of input arguments')


# In[11]:

    def test_ShallowNet(self):
        print('Testing ShallowNet...', end="", flush=True)
        EEGsha_args = {'nb_classes': [2,4],'Samples':[2048], 'Chans': [self.Chan],
                       'F': [20,40,80], 'K1':[25,12,50],
                       'Pool': [75,50,100],
                       'return_logits': [True,False]
                      }
        EEGsha_args = self.makeGrid(EEGsha_args)
        for i in EEGsha_args:
            model = models.ShallowNet(**i)
            out = model(self.x)
            self.assertEqual(torch.isnan(out).sum(),0)
            self.assertEqual(out.shape[1], i['nb_classes'] if i['nb_classes']>2 else 1)
            if not(i['return_logits']):
                self.assertLessEqual(out.max(), 1)
                self.assertGreaterEqual(out.min(), 0)

        if self.device.type != 'cpu':
            for i in EEGsha_args:
                model = models.ShallowNet(**i).to(device=self.device)
                out = model(self.x2)
                self.assertEqual(torch.isnan(out).sum(),0)
                self.assertEqual(out.shape[1], i['nb_classes'] if i['nb_classes']>2 else 1)
                if not(i['return_logits']):
                    self.assertLessEqual(out.max(), 1)
                    self.assertGreaterEqual(out.min(), 0)
        print('   ShallowNet OK: tested', len(EEGsha_args), ' combinations of input arguments')




    def test_StagerNet(self):
        print('Testing StageRNet...', end="", flush=True)
        EEGsta_args = {'nb_classes': [2,4],'Samples':[2048], 'Chans': [self.Chan],
                       'F': [8,16,4], 'kernLength':[64,32,120],
                       'Pool': [16,30,8],
                       'return_logits': [True,False]
                      }
        EEGsta_args = self.makeGrid(EEGsta_args)
        for i in EEGsta_args:
            model = models.StagerNet(**i)
            out = model(self.x)
            self.assertEqual(torch.isnan(out).sum(),0)
            self.assertEqual(out.shape[1], i['nb_classes'] if i['nb_classes']>2 else 1)
            if not(i['return_logits']):
                self.assertLessEqual(out.max(), 1)
                self.assertGreaterEqual(out.min(), 0)

        if self.device.type != 'cpu':
            for i in EEGsta_args:
                model = models.StagerNet(**i).to(device=self.device)
                out = model(self.x2)
                self.assertEqual(torch.isnan(out).sum(),0)
                self.assertEqual(out.shape[1], i['nb_classes'] if i['nb_classes']>2 else 1)
                if not(i['return_logits']):
                    self.assertLessEqual(out.max(), 1)
                    self.assertGreaterEqual(out.min(), 0)
        print('   StageRNet OK: tested', len(EEGsta_args), ' combinations of input arguments')



    def test_STNet(self):
        print('Testing STNet...', end="", flush=True)
        EEGstn_args = {'nb_classes': [2,4],'Samples':[2048], 'grid_size': [5,9],
                       'F': [256,512,64], 'kernlength':[5,7], 'dense_size':[1024,512],
                       'return_logits': [True,False]
                      }
        EEGstn_args = self.makeGrid(EEGstn_args)
        for i in EEGstn_args:
            model = models.STNet(**i)
            xst = torch.randn(self.N,i['Samples'],i['grid_size'],i['grid_size'])
            out = model(xst)
            self.assertEqual(torch.isnan(out).sum(),0)
            self.assertEqual(out.shape[1], i['nb_classes'] if i['nb_classes']>2 else 1)
            if not(i['return_logits']):
                self.assertLessEqual(out.max(), 1)
                self.assertGreaterEqual(out.min(), 0)

        if self.device.type != 'cpu':
            for i in EEGstn_args:
                model = models.STNet(**i).to(device=self.device)
                xst2 = torch.randn(self.N,i['Samples'],i['grid_size'],i['grid_size']).to(device=self.device)
                out = model(xst2)
                self.assertEqual(torch.isnan(out).sum(),0)
                self.assertEqual(out.shape[1], i['nb_classes'] if i['nb_classes']>2 else 1)
                if not(i['return_logits']):
                    self.assertLessEqual(out.max(), 1)
                    self.assertGreaterEqual(out.min(), 0)
        print('   STNet OK: tested', len(EEGstn_args), ' combinations of input arguments')


    def test_TinySleepNet(self):
        print('Testing TinySleepNet...', end="", flush=True)
        #nb_classes, Chans, Fs, F=128, kernlength=8, pool=8,
        #dropRate=0.5, batch_momentum=0.1, max_dense_norm=2.0, return_logits=True
        EEGsleep_args = {'nb_classes': [2,4],'Chans':[self.Chan], 'Fs': [64], 'F':[128,64,32],
                         'kernlength':[8,16,30], 'pool': [16,5,8], 'hidden_lstm': [128,50],
                         'return_logits': [True,False]
                        }
        EEGsleep_args = self.makeGrid(EEGsleep_args)
        for i in EEGsleep_args:
            model = models.TinySleepNet(**i)
            out = model(self.x)
            self.assertEqual(torch.isnan(out).sum(),0)
            self.assertEqual(out.shape[1], i['nb_classes'] if i['nb_classes']>2 else 1)
            if not(i['return_logits']):
                self.assertLessEqual(out.max(), 1)
                self.assertGreaterEqual(out.min(), 0)

        if self.device.type != 'cpu':
            for i in EEGsleep_args:
                model = models.TinySleepNet(**i).to(device=self.device)
                out = model(self.x2)
                self.assertEqual(torch.isnan(out).sum(),0)
                self.assertEqual(out.shape[1], i['nb_classes'] if i['nb_classes']>2 else 1)
                if not(i['return_logits']):
                    self.assertLessEqual(out.max(), 1)
                    self.assertGreaterEqual(out.min(), 0)
        print('   TinySleepNet OK: tested', len(EEGsleep_args), ' combinations of input arguments')

if __name__ == '__main__':
    unittest.main()
