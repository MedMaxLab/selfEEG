import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ =['ConstrainedConv2d', 'ConstrainedDense','DepthwiseConv2d', 'SeparableConv2d', 
          'EEGNetEncoder', 'EEGNet', 'StagerNetEncoder', 'StagerNet', 'ShallowNetEncoder', 
          'ShallowNet', 'BasicBlock1', 'ResNet1DEncoder', 'ResNet1D'
         ]

## -------------------------------------------------------------------- ##
# TO DO: ADD MORE MODELS (DEEPCONVNET, EFFICIENTNET, TRANSFORMER ETC.)
## -------------------------------------------------------------------- ##


# ### Special Kernels not implemented in pytorch
class DepthwiseConv2d(nn.Conv2d):
    def __init__(self, in_channels, depth_multiplier, kernel_size, 
                 stride=1, padding='same', dilation=1, bias=False, max_norm=None
                ):
        super(DepthwiseConv2d, self).__init__(in_channels, depth_multiplier*in_channels, kernel_size, groups=in_channels,
                                              stride=stride, padding=padding, dilation=dilation, bias=bias)
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
    def forward(self, input):
        return F.conv2d(input, self.weight.clamp(min=0, max=1.0), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)



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
        self.conv2      = DepthwiseConv2d(F1, D, (Chans, 1), padding='valid', bias=False, max_norm=depthwise_max_norm)
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
                ):
        
        super(EEGNet, self).__init__()

        self.nb_classes=nb_classes
        self.encoder    = EEGNetEncoder(Chans, kernLength, dropRate, F1, D, F2, dropType, ELUalpha,
                                        pool1, pool2, separable_kernel, depthwise_max_norm)
        self.Dense      = ConstrainedDense( F2*(Samples//int(pool1*pool2)), 1 if nb_classes<=2 else nb_classes, max_norm=norm_rate)
    
    def forward(self, x):
        x=self.encoder(x)
        if self.nb_classes<=2:
            x=torch.sigmoid(self.Dense(x))
        else:
            x=F.softmax(self.Dense(x),dim=1)
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
    def __init__(self, nb_classes, Chans, Samples, dropRate = 0.5, kernLength = 64, F = 8, Pool = 16):
        
        super(StagerNet, self).__init__()
        
        self.encoder    = StagerNetEncoder(Chans, kernLength=kernLength, F = F, Pool = Pool)
        
        self.drop       = nn.Dropout(p=dropRate)
        self.Dense      = nn.Linear(Chans*(Samples//256)*F, nb_classes )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.drop(x)
        x = F.softmax(self.Dense(x), dim=1)
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
    def __init__(self, nb_classes, Chans, Samples, F = 40, K1 = 25, Pool = 75):
        
        super(ShallowNet, self).__init__()
        
        self.encoder   = ShallowNetEncoder(Chans, F = F, K1 = K1, Pool = Pool)
        self.Dense     = nn.Linear(F*((Samples-K1+1-Pool)//15 +1), nb_classes )
    
    def forward(self, x):
        
        x = self.encoder(x)
        x = F.softmax(self.Dense(x), dim=1)
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
            self.preBlocks = nn.Sequential(nn.Conv2d(1, self.inplane, kernel_size=(1, kernLength), stride=(1, 2),
                                                    padding=(0, kernLength//2), bias=False),
                                           nn.BatchNorm2d(self.inplane),
                                           nn.ReLU(inplace=True)
                                          )
        else: 
            self.preBlocks = preBlock
        
        #  RESIDUAL BLOCKS
        self.layer1  = self._make_layer(block, self.inplane, Layers[0], kernLength=kernLength, stride=1)
        self.layer2  = self._make_layer(block, self.inplane * 2, Layers[1], kernLength=kernLength, stride=2)
        self.layer3  = self._make_layer(block, self.inplane * 2, Layers[2], kernLength=kernLength, stride=2)
        self.layer4  = self._make_layer(block, self.inplane * 2, Layers[3], kernLength=kernLength, stride=2)

        #  POST-RESIDUAL
        if postBlock is None:
            self.postBlocks = nn.Sequential(nn.Conv2d(self.inplane, inplane, kernel_size=(1, kernLength), stride=(1, 1),
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
                ):
        
        super(ResNet1D, self).__init__()
        self.nb_classes=nb_classes
        # Encoder
        self.encoder    = ResNet1DEncoder(Chans=Chans, block=block, Layers=Layers, inplane= inplane, 
                                          kernLength= kernLength, addConnection=addConnection, 
                                          preBlock= preBlock, postBlock=postBlock)
        # Classifier
        if classifier is None:
            if addConnection:
                self.Dense = nn.Linear(Chans*inplane + (((Samples+1)//2 - kernLength)//3 + 1)*2, nb_classes)
            else:
                self.Dense = nn.Linear(Chans*inplane, nb_classes)
        else:
            self.Dense = classifier
    
    def forward(self, x):
        x=self.encoder(x)
        if self.nb_classes==1:
            x=torch.sigmoid(self.Dense(x))
        else:
            x=F.softmax(self.Dense(x), dim=1)
        return x

