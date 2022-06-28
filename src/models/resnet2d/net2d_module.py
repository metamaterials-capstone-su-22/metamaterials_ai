import os
import shutil
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

class LambdaLayer(nn.Module):
    
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    
    def forward(self, x):
        return self.lambd(x)

class BasicConvBlock(nn.Module):
    
    ''' The BasicConvBlock takes an input with in_channels, applies some blocks of convolutional layers 
    to reduce it to out_channels and sum it up to the original input. 
    If their sizes mismatch, then the input goes into an identity. 
    
    Basically The BasicConvBlock will implement the regular basic Conv Block + 
    the shortcut block that does the dimension matching job (option A or B) when dimension changes between 2 blocks
    '''
    
    def __init__(self, in_channels, out_channels, stride=1, option='A'):
        super(BasicConvBlock, self).__init__()
        
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)),
            ('bn1', nn.BatchNorm2d(out_channels)),
            ('act1', nn.ReLU()),
            ('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn2', nn.BatchNorm2d(out_channels))
        ]))

        self.shortcut = nn.Sequential()
        
        '''  When input and output spatial dimensions don't match, we have 2 options, with stride:
            - A) Use identity shortcuts with zero padding to increase channel dimension.    
            - B) Use 1x1 convolution to increase channel dimension (projection shortcut).
         '''
        if stride != 1 or in_channels != out_channels:
            if option == 'A':
                # Use identity shortcuts with zero padding to increase channel dimension.
                pad_to_add = out_channels//4
                ''' ::2 is doing the job of stride = 2
                F.pad apply padding to (W,H,C,N).
                
                The padding lengths are specified in reverse order of the dimensions,
                F.pad(x[:, :, ::2, ::2], (0,0, 0,0, pad,pad, 0,0))

                [width_beginning, width_end, height_beginning, height_end, channel_beginning, channel_end, batchLength_beginning, batchLength_end ]

                '''
                self.shortcut = LambdaLayer(lambda x:
                            F.pad(x[:, :, ::2, ::2], (0,0, 0,0, pad_to_add, pad_to_add, 0,0)))
            if option == 'B':
                self.shortcut = nn.Sequential(OrderedDict([
                    ('s_conv1', nn.Conv2d(in_channels, 2*out_channels, kernel_size=1, stride=stride, padding=0, bias=False)),
                    ('s_bn1', nn.BatchNorm2d(2*out_channels))
                ]))
        
    def forward(self, x):
        out = self.features(x)
        # sum it up with shortcut layer
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    """
        ResNet-56 architecture for CIFAR-10 Dataset of shape 32*32*3
    """
    def __init__(self, block_type, num_blocks, in_channels=14, dim=512, out_channels=14, kernel_size=1, stride=1, padding=0):
        super(ResNet, self).__init__()
        
        self.in_channels = in_channels
        self.conv0 = nn.Conv2d(in_channels, dim=512, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn0 = nn.BatchNorm2d(14)
        
        self.block1 = self.__build_layer(block_type, out_channels, num_blocks[0], starting_stride=1)
        
        self.block2 = self.__build_layer(block_type, out_channels, num_blocks[1], starting_stride=1)
        
        self.block3 = self.__build_layer(block_type, out_channels, num_blocks[2], starting_stride=1)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(64, 10)
    
    def __build_layer(self, block_type, out_channels, num_blocks, starting_stride):
        
        strides_list_for_current_block = [starting_stride] + [1]*(num_blocks-1)
        ''' Above line will generate an array whose first element is starting_stride
        And it will have (num_blocks-1) more elements each of value 1
         '''
        # print('strides_list_for_current_block ', strides_list_for_current_block)
        
        layers = []
        
        for stride in strides_list_for_current_block:
            layers.append(block_type(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn0(self.conv0(x)))
        out = self.block1(out)
        out = self.block2(out)        
        out = self.block3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out