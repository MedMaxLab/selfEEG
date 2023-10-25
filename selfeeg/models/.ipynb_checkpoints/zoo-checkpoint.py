import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ =['ConstrainedConv2d', 'ConstrainedDense','DepthwiseConv2d', 'SeparableConv2d', 
          'EEGNetEncoder', 'EEGNet', 
          'DeepConvNetEncoder', 'DeepConvNet',
          'EEGInceptionEncoder', 'EEGInception',
          'TinySleepNetEncoder', 'TinySleepNet', 
          'StagerNetEncoder', 'StagerNet', 
          'ShallowNetEncoder', 'ShallowNet', 
          'BasicBlock1', 'ResNet1DEncoder', 'ResNet1D',
          'STNetEncoder', 'STNet',
          'EEGSymEncoder', 'EEGSym'
         ]


# ### Special Kernels not implemented in pytorch
class DepthwiseConv2d(nn.Conv2d):
    def __init__(self, in_channels, depth_multiplier, kernel_size, 
                 stride=1, padding='same', dilation=1, bias=False, max_norm=None
                ):
        super(DepthwiseConv2d, self).__init__(in_channels, depth_multiplier*in_channels, kernel_size, 
                                              groups=in_channels,stride=stride, padding=padding, 
                                              dilation=dilation, bias=bias)
        if max_norm is not None:
            if max_norm <=0:
                raise ValueError('max_norm can\'t be lower or equal than 0')
            else:
                self.max_norm=max_norm
        else:
            self.max_norm= max_norm

    @torch.no_grad()
    def scale_norm(self, eps=1e-9):
        """
        CONSIDERING THE DESCRIPTION PROVIDED IN TENSORFLOW 
        
        integer, axis along which to calculate weight norms. For instance, in a Dense layer the weight 
        matrix has shape (input_dim, output_dim), set axis to 0 to constrain each weight vector of 
        length (input_dim,). In a Conv2D layer with data_format="channels_last", the weight tensor has 
        shape (rows, cols, input_depth, output_depth), set axis to [0, 1, 2] to constrain the weights 
        of each filter tensor of size (rows, cols, input_depth).
        """
        # calcuate the norm of each filter of size (row, cols, input_depth), here (1, kernel_size)
        if self.kernel_size[1]>1:
            norm= self.weight.norm(dim=2, keepdim=True).norm(dim=3,keepdim=True)
        else:
            norm = self.weight.norm(dim=2, keepdim=True)

        # rescale only those filters which have a norm bigger than the maximum allowed
        if (norm>self.max_norm).sum()>self.max_norm:
            desired = torch.clamp(norm, 0, self.max_norm)
            self.weight = torch.nn.Parameter(self.weight*desired/ (eps + norm))
    
    def forward(self, input):
        if self.max_norm is not None:
            self.scale_norm(self.max_norm)
        return self._conv_forward(input, self.weight, self.bias)
    

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', 
                 dilation=1, bias=False, depth_multiplier=1, depth_max_norm=None):
        super(SeparableConv2d, self).__init__()
        self.depthwise = DepthwiseConv2d( in_channels, depth_multiplier, kernel_size, 
                                          stride, padding, dilation, bias, max_norm=None)
        self.pointwise = nn.Conv2d(in_channels*depth_multiplier, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class ConstrainedDense(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, max_norm=None):
        super(ConstrainedDense, self).__init__(in_features, out_features, bias, device, dtype)
        
        if max_norm is not None:
            if max_norm <=0:
                raise ValueError('max_norm can\'t be lower or equal than 0')
            else:
                self.max_norm=max_norm
        else:
            self.max_norm= max_norm
        
    @torch.no_grad()
    def scale_norm(self, eps=1e-9):
        """
        CONSIDERING THE DESCRIPTION PROVIDED IN TENSORFLOW 
        
        integer, axis along which to calculate weight norms. For instance, in a Dense layer the weight matrix has shape 
        (input_dim, output_dim), set axis to 0 to constrain each weight vector of length (input_dim,). 
        In a Conv2D layer with data_format="channels_last", the weight tensor has shape (rows, cols, input_depth, output_depth), 
        set axis to [0, 1, 2] to constrain the weights of each filter tensor of size (rows, cols, input_depth).
        """
        # calcuate the norm of each filter of size (row, cols, input_depth), here (1, kernel_size)
        norm = self.weight.norm(dim=1, keepdim=True)

        # rescale only those filters which have a norm bigger than the maximum allowed
        if (norm>self.max_norm).sum()>self.max_norm:
            desired = torch.clamp(norm, 0, self.max_norm)
            self.weight = torch.nn.Parameter(self.weight*desired/ (eps + norm))
    
    
    def forward(self, input):
        if self.max_norm is not None:
            self.scale_norm(self.max_norm)
        return F.linear(input, self.weight, self.bias)
    

class ConstrainedConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', 
                 dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None, 
                 max_norm=None):
        super(ConstrainedConv2d, self).__init__(in_channels, out_channels, kernel_size, 
                                                stride, padding, dilation, groups, bias, 
                                                padding_mode, device, dtype)
        
        if max_norm is not None:
            if max_norm <=0:
                raise ValueError('max_norm can\'t be lower or equal than 0')
            else:
                self.max_norm=max_norm
        else:
            self.max_norm= max_norm

    @torch.no_grad()
    def scale_norm(self, eps=1e-9):
        # calcuate the norm of each filter of size (row, cols, input_depth), here (1, kernel_size)
        if self.kernel_size[1]>1:
            norm= self.weight.norm(dim=2, keepdim=True).norm(dim=3,keepdim=True)
        else:
            norm = self.weight.norm(dim=2, keepdim=True)

        # rescale only those filters which have a norm bigger than the maximum allowed
        if (norm>self.max_norm).sum()>self.max_norm:
            desired = torch.clamp(norm, 0, self.max_norm)
            self.weight = torch.nn.Parameter(self.weight*desired/ (eps + norm))
    
    def forward(self, input):
        if self.max_norm is not None:
            self.scale_norm(self.max_norm)
        return self._conv_forward(input, self.weight, self.bias)



# ------------------------------
#           EEGNet
# ------------------------------
class EEGNetEncoder(nn.Module):
    """
    Pytorch Implementation of the EEGnet Encoder. For more information see the following paper:
        Lawhern et al., EEGNet: a compact convolutional neural network for EEG-based 
        brain–computer interfaces. Journal of Neural Engineering. 2018
    
    Keras implementation of the full EEGnet (updated version) with more info at:
        https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py
    
    NOTE: This implementation referres to the latest version of EEGNet which can be found in the official repository
    """
    def __init__(self, Chans, kernLength = 64, dropRate = 0.5, F1 = 8, 
                 D = 2, F2 = 16, dropType = 'Dropout', ELUalpha=1,
                 pool1=4, pool2=8, separable_kernel=16, depthwise_max_norm=1.
                ):
        
        
        if dropType not in ['SpatialDropout2D','Dropout']:
            raise ValueError('implemented Dropout types are \'Dropout\' or \'SpatialDropout2D \'')
        
        super(EEGNetEncoder, self).__init__()

        # Layer 1
        self.conv1      = nn.Conv2d(1, F1, (1, kernLength), padding = 'same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1, False)
        
        # Layer 2
        self.conv2      = DepthwiseConv2d(F1, D, (Chans, 1), padding='valid', bias=False, 
                                          max_norm=depthwise_max_norm)
        self.batchnorm2 = nn.BatchNorm2d(D*F1, False)
        self.elu2       = nn.ELU(alpha=ELUalpha)
        self.pooling2   = nn.AvgPool2d((1,pool1))
        self.drop2      = nn.Dropout(p=dropRate) if dropType.lower()=='dropout' else nn.Dropout2d(p=dropRate)

        # Layer 3
        self.sepconv3   = SeparableConv2d(D*F1, F2, (1, separable_kernel ), bias=False, padding='same')
        self.batchnorm3 = nn.BatchNorm2d(F2, False)
        self.elu3       = nn.ELU(alpha=ELUalpha)
        self.pooling3   = nn.AvgPool2d((1,pool2))
        self.drop3      = nn.Dropout(p=dropRate) if dropType.lower()=='dropout' else nn.Dropout2d(p=dropRate)
        self.flatten3   = nn.Flatten()

    def forward(self, x):
        
        # Layer 1
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        
        # Layer 2
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.elu2(x)
        x = self.pooling2(x)
        x = self.drop2(x)
        
        # Layer 3
        x = self.sepconv3(x)
        x = self.batchnorm3(x)
        x = self.elu3(x)
        x = self.pooling3(x)
        x = self.drop3(x)
        x = self.flatten3(x)
        
        return x

    
class EEGNet(nn.Module):
    """
    Pytorch Implementation of EEGnet. For more information see the following paper:
        Lawhern et al., EEGNet: a compact convolutional neural network for EEG-based 
        brain–computer interfaces. Journal of Neural Engineering. 2018
    
    Keras implementation of the full EEGnet (updated version) with more info at:
        https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py
    
    NOTE: This implementation referres to the latest version of EEGNet which can be found in the 
    """
    def __init__(self, nb_classes, Chans, Samples, kernLength = 64, dropRate = 0.5, F1 = 8, 
                 D = 2, F2 = 16, norm_rate = 0.25, dropType = 'Dropout', ELUalpha=1,
                 pool1=4, pool2=8, separable_kernel=16, depthwise_max_norm=1.0,
                 return_logits=True
                ):
        
        super(EEGNet, self).__init__()

        self.nb_classes = nb_classes
        self.return_logits = return_logits
        self.encoder    = EEGNetEncoder(Chans, kernLength, dropRate, F1, D, F2, dropType, ELUalpha,
                                        pool1, pool2, separable_kernel, depthwise_max_norm)
        self.Dense      = ConstrainedDense( F2*(Samples//int(pool1*pool2)), 1 if nb_classes<=2 else nb_classes, max_norm=norm_rate)
    
    def forward(self, x):
        x=self.encoder(x)
        x=self.Dense(x)
        if not(self.return_logits):
            if self.nb_classes<=2:
                x=torch.sigmoid(x)
            else:
                x=F.softmax(x,dim=1)
        return x


# ------------------------------
#          DeepConvNet
# ------------------------------
class  DeepConvNetEncoder(nn.Module):

    def __init__(self, Chans, kernLength = 10, F = 25, Pool = 3, stride = 3, 
                 max_norm=2., batch_momentum=0.9, ELUalpha=1, dropRate=0.5):
        
        super(DeepConvNetEncoder, self).__init__()

        self.conv1 = ConstrainedConv2d(1, F, (1, kernLength), padding='valid', 
                                       stride=(1,1), max_norm=max_norm)
        self.conv2 = ConstrainedConv2d(F, F, (Chans, 1), stride=(1,1), padding='valid', 
                                       max_norm=max_norm)
        self.BN1   = nn.BatchNorm2d(F, momentum=batch_momentum )
        self.ELU   = nn.ELU(alpha=ELUalpha)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, Pool), stride=(1,stride))
        self.drop1 = nn.Dropout(p=dropRate)

        self.conv3 = ConstrainedConv2d(F, F*2, (1, kernLength), padding='valid', 
                                       stride=(1,1), max_norm=max_norm)
        self.BN2   = nn.BatchNorm2d(F*2, momentum=batch_momentum )
        self.pool2 = nn.MaxPool2d(kernel_size=(1, Pool), stride=(1,stride))
        self.drop2 = nn.Dropout(p=dropRate)

        self.conv4 = ConstrainedConv2d(F*2, F*4, (1, kernLength), padding='valid', 
                                       stride=(1,1), max_norm=max_norm)
        self.BN3   = nn.BatchNorm2d(F*4, momentum=batch_momentum )
        self.pool3 = nn.MaxPool2d(kernel_size=(1, Pool), stride=(1,stride))
        self.drop3 = nn.Dropout(p=dropRate)

        self.conv5 = ConstrainedConv2d(F*4, F*8, (1, kernLength), padding='valid', 
                                       stride=(1,1), max_norm=max_norm)
        self.BN4   = nn.BatchNorm2d(F*8, momentum=batch_momentum )
        self.pool4 = nn.MaxPool2d(kernel_size=(1, Pool), stride=(1,stride))
        self.drop4 = nn.Dropout(p=dropRate)
        
        self.flatten = nn.Flatten()
    
    def forward(self,x):
        
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.BN1(x)
        x = self.ELU(x) 
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.conv3(x)
        x = self.BN2(x)
        x = self.ELU(x)
        x = self.pool2(x)
        x = self.drop2(x)
        x = self.conv4(x)
        x = self.BN3(x)
        x = self.ELU(x)
        x = self.pool3(x)
        x = self.drop3(x)
        x = self.conv5(x)
        x = self.BN4(x)
        x = self.ELU(x)
        x = self.pool4(x)
        x = self.drop4(x)
        x = self.flatten(x)
        return x

class  DeepConvNet(nn.Module):

    def __init__(self,nb_classes, Chans, Samples, kernLength = 10, F = 25, Pool = 3, stride = 3, 
                 max_norm=2., batch_momentum=0.9, ELUalpha=1, dropRate=0.5, max_dense_norm = 1.,
                 return_logits=True, 
                ):
        super(DeepConvNet, self).__init__()

        self.nb_classes = nb_classes
        self.return_logits = return_logits
        self.encoder    = DeepConvNetEncoder(Chans, kernLength, F, Pool, stride, max_norm, 
                                             batch_momentum, ELUalpha, dropRate)
        k = kernLength-1
        Dense_input     = F*8*((((( (Samples-k)//stride ) -k)//stride) -k )//stride -k)//stride
        self.Dense = ConstrainedDense( Dense_input, 
                                      1 if nb_classes<=2 else nb_classes, 
                                      max_norm=max_dense_norm)

    def forward(self, x):
        x=self.encoder(x)
        x=self.Dense(x)
        if not(self.return_logits):
            if self.nb_classes<=2:
                x=torch.sigmoid(x)
            else:
                x=F.softmax(x,dim=1)
        return x 


class EEGInceptionEncoder(nn.Module):

    def __init__(self, Chans, F1=8, D=2, kernel_size=64, pool=4, dropRate=0.5, 
                 ELUalpha=1.0, bias=True, batch_momentum=0.1, max_depth_norm=1.):

        super(EEGInceptionEncoder, self).__init__()
        self.inc1 = nn.Sequential( nn.Conv2d(1, F1, (1,kernel_size), padding='same', bias=bias),
                                   nn.BatchNorm2d(F1, momentum=batch_momentum),
                                   nn.ELU(alpha=ELUalpha),
                                   nn.Dropout(dropRate),
                                   DepthwiseConv2d(F1, D, kernel_size=(Chans,1), 
                                                      padding='valid', max_norm=max_depth_norm ),
                                   nn.BatchNorm2d(F1*D, momentum=batch_momentum),
                                   nn.ELU(alpha=ELUalpha),
                                   nn.Dropout(dropRate)
                                 )
        self.inc2 = nn.Sequential( nn.Conv2d(1, F1, (1,int(kernel_size//2)), 
                                             padding='same', bias=bias),
                                   nn.BatchNorm2d(F1, momentum=batch_momentum),
                                   nn.ELU(alpha=ELUalpha),
                                   nn.Dropout(dropRate),
                                   DepthwiseConv2d(F1, D, kernel_size=(Chans,1), 
                                                      padding='valid', max_norm=max_depth_norm ),
                                   nn.BatchNorm2d(F1*D, momentum=batch_momentum),
                                   nn.ELU(alpha=ELUalpha),
                                   nn.Dropout(dropRate)
                                 )
        self.inc3 = nn.Sequential( nn.Conv2d(1, F1, (1,int(kernel_size//4)), 
                                             padding='same', bias=bias),
                                   nn.BatchNorm2d(F1, momentum=batch_momentum),
                                   nn.ELU(alpha=ELUalpha),
                                   nn.Dropout(dropRate),
                                   DepthwiseConv2d(F1, D, kernel_size=(Chans,1), 
                                                      padding='valid', max_norm=max_depth_norm ),
                                   nn.BatchNorm2d(F1*D, momentum=batch_momentum),
                                   nn.ELU(alpha=ELUalpha),
                                   nn.Dropout(dropRate)
                                 )
        # concatenate inc1 inc2 e inc3 on filter size
        self.pool1 = nn.AvgPool2d((1,pool))
        
        self.inc4 = nn.Sequential( nn.Conv2d(F1*D*3, F1, (1,int(kernel_size//4)), 
                                             padding='same', bias=bias),
                                   nn.BatchNorm2d(F1, momentum=batch_momentum),
                                   nn.ELU(alpha=ELUalpha),
                                   nn.Dropout(dropRate),
                                 )
        self.inc5 = nn.Sequential( nn.Conv2d(F1*D*3, F1, (1,int(kernel_size//8)), 
                                             padding='same', bias=bias),
                                   nn.BatchNorm2d(F1, momentum=batch_momentum),
                                   nn.ELU(alpha=ELUalpha),
                                   nn.Dropout(dropRate),
                                 )
        self.inc6 = nn.Sequential( nn.Conv2d(F1*D*3, F1, (1,int(kernel_size//16)), 
                                             padding='same', bias=bias),
                                   nn.BatchNorm2d(F1, momentum=batch_momentum),
                                   nn.ELU(alpha=ELUalpha),
                                   nn.Dropout(dropRate),
                                 )
        # concatenate inc4 inc5 e inc6 on filter size
        self.pool2 = nn.AvgPool2d((1,int(pool//2)))
        self.out1 = nn.Sequential( nn.Conv2d(F1*3, int((F1*3)/2), (1,int(kernel_size//8)), 
                                             padding='same', bias=bias),
                                   nn.BatchNorm2d(int((F1*3)/2), momentum=batch_momentum),
                                   nn.ELU(alpha=ELUalpha),
                                   nn.Dropout(dropRate),
                                 )
        self.pool3 = nn.AvgPool2d((1,int(pool//2)))
        self.out2 = nn.Sequential( nn.Conv2d(int((F1*3)/2), int((F1*3)/4), (1,int(kernel_size//16)), 
                                             padding='same', bias=bias),
                                   nn.BatchNorm2d(int((F1*3)/4), momentum=batch_momentum),
                                   nn.ELU(alpha=ELUalpha),
                                   nn.Dropout(dropRate),
                                 )
        self.pool4 = nn.AvgPool2d((1,int(pool//2)))
        self.flatten = nn.Flatten()
    
    def forward(self,x):
        x = torch.unsqueeze(x,1)
        x1 = self.inc1(x)
        x2 = self.inc2(x)
        x3 = self.inc3(x)
        x_conc = torch.cat((x1,x2,x3), dim=1)
        x_conc = self.pool1(x_conc)
        x1 = self.inc4(x_conc)
        x2 = self.inc5(x_conc)
        x3 = self.inc6(x_conc)
        xout = torch.cat((x1,x2,x3), dim=1)
        xout = self.pool2(xout)
        xout = self.out1(xout)
        xout = self.pool3(xout)
        xout = self.out2(xout)
        xout = self.pool4(xout)
        xout = self.flatten(xout)
        return xout

class EEGInception(nn.Module):
    def __init__(self, nb_classes, Samples, Chans, F1=8, D=2, kernel_size=64, 
                 pool=4, dropRate=0.5, ELUalpha=1.0, bias=True, batch_momentum=0.1, 
                 max_depth_norm=1., return_logits=True):
        super(EEGInception, self).__init__()
        self.nb_classes    = nb_classes
        self.return_logits = return_logits
        self.encoder       = EEGInceptionEncoder(Chans, F1=8, D=2, kernel_size=64, pool=4, 
                                                 dropRate=0.5, ELUalpha=1.0, bias=True, 
                                                 batch_momentum=0.1, max_depth_norm=1.)
        self.Dense  = nn.Linear( int((F1*3)/4)*int((Samples//(pool*(int(pool//2)**3))) ), 
                                1 if nb_classes<=2 else nb_classes)
    
    def forward(self, x):
        x=self.encoder(x)
        x=self.Dense(x)
        if not(self.return_logits):
            if self.nb_classes<=2:
                x=torch.sigmoid(x)
            else:
                x=F.softmax(x,dim=1)
        return x 






# ------------------------------
#         TinySleepNet
# ------------------------------
class  TinySleepNetEncoder(nn.Module):

    def __init__(self, Chans, Fs, F1 = 128, kernlength=8, pool=8, dropRate=0.5, batch_momentum=0.1):
        """
        input (BxCxS)
        """
        
        super(TinySleepNetEncoder, self).__init__()
        self.conv1 = nn.Conv1d( Chans, F, int(Fs//2), stride=int(Fs//16), padding='valid')
        self.BN1   = nn.BatchNorm1d(F, momentum=batch_momentum )
        self.Relu   = nn.ReLU()
        self.pool1 = nn.MaxPool1d( pool, stride=pool)
        self.drop1 = nn.Dropout1d(dropRate)
        
        self.conv2 = nn.Conv1d( F, F, 8, stride=1, padding='valid')
        self.BN2   = nn.BatchNorm1d(F1, momentum=batch_momentum )
        #ReLU()
        self.conv3 = nn.Conv1d( F, F, 8, stride=1, padding='valid')
        self.BN3   = nn.BatchNorm1d(F1, momentum=batch_momentum )
        #ReLU()
        self.conv4 = nn.Conv1d( F, F, 8, stride=1, padding='valid')
        self.BN4   = nn.BatchNorm1d(F, momentum=batch_momentum )
        #ReLU()
        
        self.pool2 = nn.MaxPool1d( pool//2, stride=pool//2)
        self.drop2 = nn.Dropout1d(dropRate)

        self.lstm1 = nn.LSTM( input_size=F , hidden_size=128, num_layers=1)
        self.flatten = nn.Flatten()


    def forward(self,x):

        x = self.conv1(x)
        x = self.BN1(x)
        x = self.Relu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.BN2(x)
        x = self.Relu(x)
        x = self.conv3(x)
        x = self.BN3(x)
        x = self.Relu(x)
        x = self.conv4(x)
        x = self.BN4(x)
        x = self.Relu(x)
        
        x = self.pool2(x)
        x = self.drop2(x)
        print(x.shape)

        x = torch.permute(x, (2,0,1))
        print(x.shape)

        out, (ht, ct) = self.lstm1(x)
        
        return ht[-1]


class  TinySleepNet(nn.Module):

    def __init__(self,nb_classes, Chans, Fs, F = 128, kernlength=8, pool=8, 
                 dropRate=0.5, batch_momentum=0.1, max_dense_norm=2., return_logits=True
                ):
        super(TinySleepNet, self).__init__()

        self.nb_classes = nb_classes
        self.return_logits = return_logits
        self.encoder    = TinySleepNetEncoder(Chans, Fs, F, kernlength, pool, 
                                              dropRate, batch_momentum)

        self.drop3 = nn.Dropout1d(dropRate)
        self.Dense = ConstrainedDense( F, 1 if nb_classes<=2 else nb_classes, 
                                      max_norm=max_dense_norm)

    def forward(self, x):
        x=self.encoder(x)
        x=self.drop3(x) 
        x=self.Dense(x)
        if not(self.return_logits):
            if self.nb_classes<=2:
                x=torch.sigmoid(x)
            else:
                x=F.softmax(x,dim=1)
        return x  



# ------------------------------
#          StagerNet
# ------------------------------
class StagerNetEncoder(nn.Module):
    """
    Pytorch implementation of the StagerNet Encoder. For more information see the following papaer:
    Chambon et al., A deep learning architecture for temporal sleep stage classification 
    using multivariate and multimodal time series, arXiv:1707.03321
    """
    
    def __init__(self, Chans, kernLength = 64, F = 8, Pool = 16):
        
        super(StagerNetEncoder, self).__init__()
        
        self.conv1      = nn.Conv2d(1, Chans, (Chans,1), stride=(1,1), bias=True)
        self.conv2      = nn.Conv2d(1, F, (1,kernLength), stride=(1,1), padding='same')
        self.pooling2   = nn.MaxPool2d((1,Pool), stride=(1,Pool))
        self.conv3      = nn.Conv2d(F, F, (1,kernLength), stride=(1,1), padding='same')
        self.pooling3   = nn.MaxPool2d((1,Pool), stride=(1,Pool))
        self.flatten3   = nn.Flatten()
    
    def forward(self,x):
        
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = torch.permute(x, (0,2,1,3))
        x = F.relu(self.conv2(x))
        x = self.pooling2(x)
        x = F.relu(self.conv3(x))
        x = self.pooling3(x)
        x = self.flatten3(x) 
        return x
        
        
        
    
class StagerNet(nn.Module):    
    """
    Pytorch implementation of the StagerNet Encoder. For more information see the following papaer:
    Chambon et al., A deep learning architecture for temporal sleep stage classification 
    using multivariate and multimodal time series, arXiv:1707.03321
    """
    def __init__(self, nb_classes, Chans, Samples, dropRate = 0.5, 
                 kernLength = 64, F = 8, Pool = 16, return_logits=True):
        
        super(StagerNet, self).__init__()
        self.nb_classes = nb_classes
        self.return_logits = return_logits
        self.encoder    = StagerNetEncoder(Chans, kernLength=kernLength, F = F, Pool = Pool)
        
        self.drop       = nn.Dropout(p=dropRate)
        self.Dense      = nn.Linear(Chans*(Samples//256)*F, 1 if nb_classes<=2 else nb_classes )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.drop(x)
        x=self.Dense(x)
        if not(self.return_logits):
            if self.nb_classes<=2:
                x=torch.sigmoid(x)
            else:
                x=F.softmax(x,dim=1)
        return x    


# ------------------------------
#         ShallowNet
# ------------------------------
class ShallowNetEncoder(nn.Module):
    """
    Pytorch implementation of the Shallow ConvNet Encoder. For more information see the following papaer:
    Schirrmeister et al., Deep Learning with convolutional neural networks for decoding and visualization 
    of EEG pathology, arXiv:1708.08012
    
    NOTE= In this implementation, the number of channels is an argument. However, in the original paper
          they preprocess EEG data by selecting a subset of only 21 channels. Since the net is pretty
          minimalist, we suggest to follow author notes

    """
    
    def __init__(self, Chans, F = 8, K1 = 25, Pool = 75):
        
        super(ShallowNetEncoder, self).__init__() 
        self.conv1      = nn.Conv2d(1, F, (1, K1), stride=(1,1))
        self.conv2      = nn.Conv2d(F, F, (Chans, 1), stride=(1,1))
        self.pool2      = nn.AvgPool2d((1, Pool), stride=(1,15))
        self.flatten2   = nn.Flatten()
    
    def forward(self,x):
        
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = torch.log(x)
        x = self.flatten2(x) 
        return x
    
    
class ShallowNet(nn.Module):    
    """
    Pytorch implementation of the Shallow ConvNet Encoder. For more information see the following papaer:
    Schirrmeister et al., Deep Learning with convolutional neural networks for decoding and visualization 
    of EEG pathology, arXiv:1708.08012
    
    NOTE= In this implementation, the number of channels is an argument. However, in the original paper
          they preprocess EEG data by selecting a subset of only 21 channels. Since the net is pretty
          minimalist, we suggest to follow author notes
    """
    def __init__(self, nb_classes, Chans, Samples, F = 40, K1 = 25, Pool = 75, return_logits=True):
        
        super(ShallowNet, self).__init__()

        self.nb_classes = nb_classes
        self.return_logits = return_logits
        self.encoder   = ShallowNetEncoder(Chans, F = F, K1 = K1, Pool = Pool)
        self.Dense     = nn.Linear(F*((Samples-K1+1-Pool)//15 +1), 1 if nb_classes<=2 else nb_classes )
    
    def forward(self, x):
        
        x = self.encoder(x)
        x=self.Dense(x)
        if not(self.return_logits):
            if self.nb_classes<=2:
                x=torch.sigmoid(x)
            else:
                x=F.softmax(x,dim=1)
        return x   



# ------------------------------
#         ResNet 1D
# ------------------------------
class BasicBlock1(nn.Module):
    """
    BasicBlock implemented in Zheng et al.
    """
    def __init__(self, inplanes, planes, kernLength = 7, stride = 1):
        
        super(BasicBlock1, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(1, kernLength), stride=(1, stride), 
                               padding=(0, kernLength//2), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(1, kernLength), stride=(1, 1), 
                               padding=(0, kernLength//2), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        if inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=(1, kernLength), stride=(1, stride), 
                          padding=(0, kernLength//2), bias=False),
                nn.BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x

    
    def forward(self, x):
        residual = self.downsample(x)
        # print('residual: ', residual.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print('out 1: ', out.shape)
        out = self.conv2(out)
        out = self.bn2(out)
        # print('out 2: ', out.shape)
        
        out += residual
        out = self.relu(out)

        return out




class ResNet1DEncoder(nn.Module):
    def __init__(self, 
                 Chans, 
                 block: nn.Module = BasicBlock1, 
                 Layers: "list of 4 int"=[2, 3, 4, 6], 
                 inplane: int=16, 
                 kernLength: int=7,
                 addConnection: bool=True,
                 preBlock: nn.Module=None,
                 postBlock: nn.Module=None
                ):
        
        super(ResNet1DEncoder, self).__init__()
        self.inplane = inplane
        self.kernLength = kernLength
        self.connection = addConnection
        
        
        #   PRE-RESIDUAL
        if preBlock is None:
            self.preBlocks = nn.Sequential(nn.Conv2d(1, self.inplane, 
                                                     kernel_size=(1, kernLength), stride=(1, 2),
                                                    padding=(0, kernLength//2), bias=False),
                                           nn.BatchNorm2d(self.inplane),
                                           nn.ReLU(inplace=True)
                                          )
        else: 
            self.preBlocks = preBlock
        
        #  RESIDUAL BLOCKS
        self.layer1  = self._make_layer(block, self.inplane, Layers[0], kernLength=kernLength, stride=1)
        self.layer2  = self._make_layer(block, self.inplane * 2, Layers[1], 
                                        kernLength=kernLength, stride=2)
        self.layer3  = self._make_layer(block, self.inplane * 2, Layers[2], 
                                        kernLength=kernLength, stride=2)
        self.layer4  = self._make_layer(block, self.inplane * 2, Layers[3], 
                                        kernLength=kernLength, stride=2)

        #  POST-RESIDUAL
        if postBlock is None:
            self.postBlocks = nn.Sequential(nn.Conv2d(self.inplane, inplane, 
                                                      kernel_size=(1, kernLength), stride=(1, 1),
                                                      padding=(0, 0), bias=False),
                                            nn.AdaptiveAvgPool2d((Chans,1))
                                           )
        else:
            self.postBlocks = postBlock
        
        # RESIDUAL SKIP CONNECTION
        if self.connection:
            self.conv3   = nn.Conv2d(inplane, 2, kernel_size=(Chans, kernLength), stride=(1, 3),
                             padding=(0, 0), bias=False)
        
        # WEIGHT INITIALIZATION
        self.initialize()
        


    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, kernLength=7, stride=1, **kwarg):

        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, kernLength, stride))
            self.inplane = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)

        x1 = self.preBlocks(x)
        
        x2 = self.layer1(x1)
        x2 = self.layer2(x2)
        x2 = self.layer3(x2)
        x2 = self.layer4(x2)
        x2 = self.postBlocks(x2)
        out1 = x2.view(x2.size(0), -1)
        
        if self.connection:
            out2 = self.conv3(x1)             
            out2 = out2.view(out2.size(0), -1)
            embeddings = torch.cat((out1, out2), dim=-1)
        else:
            embeddings = out1

        return embeddings



    
class ResNet1D(nn.Module):
    
    def __init__(self, 
                 nb_classes, 
                 Chans, 
                 Samples, 
                 block: nn.Module, 
                 Layers: "list of 4 int" = [0, 0, 0, 0],
                 inplane: int=16, 
                 kernLength: int=7,
                 addConnection: bool=False,
                 preBlock: nn.Module=None,
                 postBlock: nn.Module=None,
                 classifier: nn.Module=None,
                 return_logits=True
                ):
        
        super(ResNet1D, self).__init__()
        self.nb_classes=nb_classes
        self.return_logits= return_logits
        # Encoder
        self.encoder    = ResNet1DEncoder(Chans=Chans, block=block, Layers=Layers, inplane= inplane, 
                                          kernLength= kernLength, addConnection=addConnection, 
                                          preBlock= preBlock, postBlock=postBlock)
        # Classifier
        if classifier is None:
            if addConnection:
                self.Dense = nn.Linear(Chans*inplane + (((Samples+1)//2 - kernLength)//3 + 1)*2, 
                                       1 if nb_classes<=2 else nb_classes)
            else:
                self.Dense = nn.Linear(Chans*inplane, 1 if nb_classes<=2 else nb_classes)
        else:
            self.Dense = classifier
    
    def forward(self, x):
        x=self.encoder(x)
        x=self.Dense(x)
        if not(self.return_logits):
            if self.nb_classes<=2:
                x=torch.sigmoid(x)
            else:
                x=F.softmax(x,dim=1)
        return x


# ------------------------------
#            STNet
# ------------------------------
class STNetInceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(STNetInceptionBlock, self).__init__()
        self.convBig    = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                                    stride=1, padding='same', bias=bias)
        self.convMedium = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size-2, 
                                    stride=1, padding='same', bias=bias)
        self.convSmall  = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size-4, 
                                    stride=1, padding='same', bias=bias)
        
    
    def forward(self, x):
        return self.convBig(x) + self.convMedium(x) + self.convSmall(x)
        

class  STNetEncoder(nn.Module):

    def __init__(self, Samples, F = 256, kernlength=5, dropRate=0.5, bias=True):
        """
        input (sample x grid x grid ) grid is a 2d matrix with channels rearranged
        """
        
        super(STNetEncoder, self).__init__()

        self.conv1  = nn.Conv2d(Samples, F, kernel_size=kernlength-2, stride=1, 
                                padding='same', bias=bias)
        self.selu   = nn.SELU()
        self.drop1  = nn.Dropout(dropRate)
        self.conv2  = nn.Conv2d(F, int(F/2), kernel_size=kernlength, stride=1, 
                                padding='same', bias=bias)
        self.drop2  = nn.Dropout(dropRate)
        self.conv3  = nn.Conv2d( int(F/2), int(F/4), kernel_size=kernlength, stride=1, 
                                padding='same', bias=bias)
        self.drop3  = nn.Dropout(dropRate)
        self.sep1   = SeparableConv2d(int(F/4), int(F/8), kernel_size=kernlength, 
                                          stride=1, padding='same', bias=bias)
        self.drop4  = nn.Dropout(dropRate)
        self.inception = STNetInceptionBlock(int(F/8), int(F/16), kernlength, bias)
        self.drop5  = nn.Dropout(dropRate)
        self.flatten  = nn.Flatten()


    def forward(self,x):

        x = self.conv1(x)
        x = self.selu(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.selu(x)
        x = self.drop2(x)
        x = self.conv3(x)
        x = self.selu(x)
        x = self.drop3(x)
        x = self.sep1(x)
        x = self.selu(x)
        x = self.drop4(x)
        x = self.inception(x)
        x = self.selu(x)
        x = self.drop5(x)
        x = self.flatten(x)
        return x


class  STNet(nn.Module):

    def __init__(self, nb_classes, Samples, grid_size=9, F = 256, kernlength=5, dropRate=0.5, 
                 bias=True, dense_size=1024, return_logits=True
                ):
        super(STNet, self).__init__()

        self.nb_classes    = nb_classes
        self.return_logits = return_logits
        self.encoder       = STNetEncoder(Samples, F, kernlength, dropRate, bias)

        x = self.lin1(x)
        x = self.drop_selu(x)
        x = self.lin2(x)
        self.Dense  = nn.Sequential(nn.Linear( int(F/16)*(grid_size**2),dense_size),
                                    nn.Dropout( dropRate),
                                    nn.SELU(),
                                    nn.Linear( dense_size, 1 if nb_classes<=2 else nb_classes)
                                   )

    def forward(self, x):
        x=self.encoder(x)
        x=self.drop3(x) 
        x=self.Dense(x)
        if not(self.return_logits):
            if self.nb_classes<=2:
                x=torch.sigmoid(x)
            else:
                x=F.softmax(x,dim=1)
        return x  


# ------------------------------
#            EEGSym
# ------------------------------
class EEGSymInput(nn.Module):
    def __init__(self, Chans = 8, lateral_chans=3, first_left=True):
        super(EEGSymInput,self).__init__()
        self.lateral =lateral_chans
        self.central =  Chans - lateral_chans*2
        self.hemichan = self.lateral + self.central
        self.left= first_left
    
    def forward(self,x):
        # expand dimension: 
        # new tensor will be 5D with ( batch x filter x hemisphere x channel x samples )
        x = x.unsqueeze(1).unsqueeze(1)
        left = x[...,:self.hemichan,:]
        right = x[...,-self.hemichan:,:]
        if not(self.left):
            left, right = right, left
        x = torch.cat((left,right), 2)
        return x


class EEGSymInception(nn.Module):
    def __init__(self, in_channels, out_channels, kernels=[16, 32, 64],
                 spatial_kernel = 5, pool=2, dropRate=0.5, ELUalpha=1.0,
                 bias=True, residual = True, 
                ):
        super(EEGSymInception,self).__init__()
        self.branch1 = nn.Sequential( nn.Conv3d(in_channels, out_channels, (1,1,kernels[0]), 
                                                padding='same', bias=bias),
                                      nn.BatchNorm3d(out_channels),
                                      nn.ELU(alpha=ELUalpha),
                                      nn.Dropout3d(dropRate)
                                    )
        self.branch2 = nn.Sequential( nn.Conv3d(in_channels, out_channels, (1,1,kernels[1]), 
                                                padding='same', bias=bias),
                                      nn.BatchNorm3d(out_channels),
                                      nn.ELU(alpha=ELUalpha),
                                      nn.Dropout3d(dropRate)
                                    )
        self.branch3 = nn.Sequential( nn.Conv3d(in_channels, out_channels, (1,1,kernels[2]), 
                                                padding='same', bias=bias),
                                      nn.BatchNorm3d(out_channels),
                                      nn.ELU(alpha=ELUalpha),
                                      nn.Dropout3d(dropRate)
                                    )
        # concatenation
        # add residual
        self.pool = nn.AvgPool3d( (1,1,pool) )
        self.spatial = nn.Sequential( nn.Conv3d(out_channels*3, out_channels*3, 
                                                 (1,spatial_kernel,1), padding='valid', 
                                                  groups=out_channels*3 ,bias=bias),
                                       nn.BatchNorm3d(out_channels*3),
                                       nn.ELU(alpha=ELUalpha),
                                       nn.Dropout3d(dropRate)
                                     )
        # add residual
        
    def forward(self,x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_conc = torch.cat((x1,x2,x3), dim=1)
        x_conc = x_conc + x
        x_conc = self.pool(x_conc)
        xout = self.spatial(x_conc)
        xout = xout + x_conc
        return xout


class EEGSymResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 spatial_kernel = 5, pool=2, dropRate=0.5, ELUalpha=1.0,
                 bias=True
                ):
        super(EEGSymResBlock,self).__init__()
        self.temporal1 = nn.Sequential( nn.Conv3d(in_channels, out_channels, (1,1,kernel_size), 
                                                padding='same', bias=bias),
                                        nn.BatchNorm3d(out_channels),
                                        nn.ELU(alpha=ELUalpha),
                                        nn.Dropout3d(dropRate)
                                      )
        self.temporal2 = nn.Sequential( nn.Conv3d(in_channels, out_channels, (1,1,1), 
                                                padding='same', bias=bias),
                                        nn.BatchNorm3d(out_channels),
                                        nn.ELU(alpha=ELUalpha),
                                        nn.Dropout3d(dropRate)
                                      )
        # add t1 and t2
        self.pool = nn.AvgPool3d( (1,1,pool) )
        self.spatial = nn.Sequential( nn.Conv3d(out_channels, out_channels, 
                                                 (1,spatial_kernel,1), padding='valid', 
                                                 bias=bias),
                                       nn.BatchNorm3d(out_channels),
                                       nn.ELU(alpha=ELUalpha),
                                       nn.Dropout3d(dropRate)
                                     )

    def forward(self,x):
        x1 = self.temporal1(x)
        x2 = self.temporal2(x)
        xout = x1+x2
        xout = self.pool(xout)
        x_spa = self.spatial(xout)
        xout = xout + x_spa
        return xout


class  EEGSymEncoder(nn.Module):

    def __init__(self, Samples, Chans, Fs, scales_time=(500, 250, 125),
                 lateral_chans=3, first_left=True, F = 8, pool=2, dropRate=0.5, 
                 ELUalpha=1.0, bias=True, residual=True
                ):
        
        super(EEGSymEncoder, self).__init__()
        
        self.input_samples = int(Samples * Fs / 1000)
        self.scales_samples = [int(s * Fs / 1000) for s in scales_time]
        self.scales_samples_2 = [int(s/4) for s in self.scales_samples]
        
        self.symInput   = EEGSymInput(Chans, lateral_chans, first_left)
        self.inception1 = EEGSymInception(1, F*3, self.scales_samples, self.symInput.hemichan,
                                          pool, dropRate, ELUalpha, bias, residual  )
        self.inception2 = EEGSymInception(F*9, F*3, self.scales_samples_2, self.symInput.hemichan,
                                          pool, dropRate, ELUalpha, bias, residual  )
        
        self.resblock1  = EEGSymResBlock( F*9, int((F*9)/2), self.scales_samples_2[0],
                                          self.symInput.hemichan, pool, dropRate, ELUalpha,
                                          bias=True ) 
        self.resblock2  = EEGSymResBlock( int((F*9)/2), int((F*9)/2), self.scales_samples_2[1],
                                          self.symInput.hemichan, pool, dropRate, ELUalpha,
                                          bias=True ) 
        self.resblock3  = EEGSymResBlock( int((F*9)/2), int((F*9)/4), self.scales_samples_2[2],
                                          self.symInput.hemichan, pool, dropRate, ELUalpha,
                                          bias=True )

        self.tempend    = nn.Sequential( nn.Conv3d(int((F*9)/4), int((F*9)/4), 
                                                   (1,1,self.scales_samples_2[2]), 
                                                    padding='same', bias=bias),
                                         nn.BatchNorm3d(int((F*9)/4)),
                                         nn.ELU(alpha=ELUalpha),
                                         nn.Dropout3d(dropRate)
                                       )
        #add tempend and resblock3
        self.pool1      = nn.AvgPool3d( (1,1,pool) )
        
        self.temp1     = nn.Sequential( nn.Conv3d(int((F*9)/4), int((F*9)/4), 
                                                   (1,1,self.scales_samples_2[2]), 
                                                    padding='same', bias=bias),
                                         nn.BatchNorm3d(int((F*9)/4)),
                                         nn.ELU(alpha=ELUalpha),
                                         nn.Dropout3d(dropRate)
                                       )
        # add temp1 and pool1
        self.temp2     = nn.Sequential( nn.Conv3d(int((F*9)/4), int((F*9)/4), 
                                                   (1,1,self.scales_samples_2[2]), 
                                                    padding='same', bias=bias),
                                         nn.BatchNorm3d(int((F*9)/4)),
                                         nn.ELU(alpha=ELUalpha),
                                         nn.Dropout3d(dropRate)
                                       )
        # add temp2 and temp1
        self.merge1     = nn.Sequential( nn.Conv3d(int((F*9)/4), int((F*9)/4), 
                                                   (2,self.symInput.hemichan,1), 
                                                    groups=int((F*9)/8), padding='valid', bias=bias),
                                         nn.BatchNorm3d(int((F*9)/4)),
                                         nn.ELU(alpha=ELUalpha),
                                         nn.Dropout3d(dropRate)
                                       )
        self.temp3     = nn.Sequential( nn.Conv3d(int((F*9)/4), int((F*9)/4), 
                                                   (1,1,self.scales_samples_2[2]), 
                                                    padding='same', bias=bias),
                                         nn.BatchNorm3d(int((F*9)/4)),
                                         nn.ELU(alpha=ELUalpha),
                                         nn.Dropout3d(dropRate)
                                       )
        # add temp3 and merge1
        self.merge2     = nn.Sequential( nn.Conv3d(int((F*9)/4), int((F*9)/2), 
                                                   (1,1, int(Samples//(pool**6))),
                                                    groups=int((F*9)/4), padding='valid', bias=bias),
                                         nn.BatchNorm3d(int((F*9)/2)),
                                         nn.ELU(alpha=ELUalpha),
                                         nn.Dropout3d(dropRate)
                                       )
        
        
        self.out1     = nn.Sequential( nn.Conv3d(int((F*9)/2), int((F*9)/2), (1,1,1), 
                                                 padding='same', bias=bias),
                                         nn.BatchNorm3d(int((F*9)/2)),
                                         nn.ELU(alpha=ELUalpha),
                                         nn.Dropout3d(dropRate)
                                       )
        # add out1 and merge2
        self.out2     = nn.Sequential( nn.Conv3d(int((F*9)/2), int((F*9)/2), (1,1,1), 
                                                 padding='same', bias=bias),
                                         nn.BatchNorm3d(int((F*9)/2)),
                                         nn.ELU(alpha=ELUalpha),
                                         nn.Dropout3d(dropRate)
                                       )
        # add out2 and out1
        self.out3     = nn.Sequential(nn.Conv3d(int((F*9)/2), int((F*9)/2), (1,1,1), 
                                                 padding='same', bias=bias),
                                         nn.BatchNorm3d(int((F*9)/2)),
                                         nn.ELU(alpha=ELUalpha),
                                         nn.Dropout3d(dropRate)
                                       )
        # add out2 and out3
        self.out4     = nn.Sequential( nn.Conv3d(int((F*9)/2), int((F*9)/2), (1,1,1), 
                                                 padding='same', bias=bias),
                                         nn.BatchNorm3d(int((F*9)/2)),
                                         nn.ELU(alpha=ELUalpha),
                                         nn.Dropout3d(dropRate)
                                       )
        # add out4 and out3
        self.flatten = nn.Flatten()
        

    def forward(self,x):
        x = self.symInput(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x1 = self.tempend(x)
        x = x + x1
        x = self.pool1(x)
        x1 = self.temp1(x)
        x = x + x1
        x1 = self.temp2(x)
        x = x + x1
        x = self.merge1(x)
        x1 = self.temp3(x)
        x = x + x1
        x = self.merge2(x)
        x1 = self.out1(x)
        x = x + x1
        x1 = self.out2(x)
        x = x + x1
        x1 = self.out3(x)
        x = x + x1
        x1 = self.out4(x)
        x = x + x1
        x = self.flatten(x)
        return x

class EEGSym(nn.Module):
    def __init__(self, nb_classes, Samples, Chans, Fs, scales_time=(500, 250, 125),
                 lateral_chans=3, first_left=True, F = 8, pool=2, dropRate=0.5, 
                 ELUalpha=1.0, bias=True, residual=True, return_logits=True):
        super(EEGSym, self).__init__()
        self.nb_classes    = nb_classes
        self.return_logits = return_logits
        self.encoder       = EEGSymEncoder(Samples, Chans, Fs, scales_time, lateral_chans, 
                                           first_left, F, pool, dropRate, ELUalpha, 
                                           bias, residual)
        self.Dense  = nn.Linear( int((F*9)/2), 1 if nb_classes<=2 else nb_classes)
    
    def forward(self, x):
        x=self.encoder(x)
        x=self.Dense(x)
        if not(self.return_logits):
            if self.nb_classes<=2:
                x=torch.sigmoid(x)
            else:
                x=F.softmax(x,dim=1)
        return x   