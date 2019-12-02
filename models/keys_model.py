#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data

import torchvision.transforms as transforms
import cv2 
import numpy as np
import torch.nn.functional as F
import os
import argparse
from PIL import Image
    

class ResidualBlock(nn.Module):
    def __init__(self,inchannels,outchannels,stride = 1,need_shortcut = False):
        super(ResidualBlock,self).__init__()
        self.right = nn.Sequential(
            nn.Conv2d(inchannels,outchannels,kernel_size = 3,stride = stride,padding = 1),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(True),
            nn.Conv2d(outchannels,outchannels,kernel_size = 3,stride = 1,padding = 1),
            nn.BatchNorm2d(outchannels)
         )
        if need_shortcut:
            self.short_cut = nn.Sequential(
                nn.Conv2d(inchannels,outchannels,kernel_size = 1,stride = stride),
                nn.BatchNorm2d(outchannels)
            )
        else:
            self.short_cut = nn.Sequential()
    
    def forward(self,x):
        out = self.right(x)
        out += self.short_cut(x)
        return F.relu(out)

class ResNet18(nn.Module):
    def __init__(self,input_channel = 3,base_channel=64,num_classes=2):
        super(ResNet18,self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(input_channel, base_channel, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(base_channel),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.block_1 = ResidualBlock(base_channel,base_channel,stride=1,need_shortcut=True)
        self.block_2 = ResidualBlock(base_channel,base_channel,stride=1,need_shortcut=False)
        self.block_3 = ResidualBlock(base_channel,base_channel*2,stride=2,need_shortcut=True)
        self.block_4 = ResidualBlock(base_channel*2,base_channel*2,stride=1,need_shortcut=False)
        self.block_5 = ResidualBlock(base_channel*2,base_channel*4,stride=2,need_shortcut=True)
        self.block_6 = ResidualBlock(base_channel*4,base_channel*4,stride=1,need_shortcut=False)
        self.block_7 = ResidualBlock(base_channel*4,base_channel*8,stride=2,need_shortcut=True)
        self.block_8 = ResidualBlock(base_channel*8,base_channel*8,stride=1,need_shortcut=False)
        self.avepool = nn.AvgPool2d(kernel_size=(7,7),stride=1)
        self.fc = nn.Linear(base_channel*8,num_classes)
        self.num_classes = num_classes

    def forward(self,x):
        out = self.pre_layer(x)
        out = self.block_2(self.block_1(out))
        out = self.block_4(self.block_3(out))
        out = self.block_6(self.block_5(out))
        out = self.block_8(self.block_7(out))
        out = self.avepool(out)
        out = out.view(-1,self.num_flatters(out))
        return self.fc(out)

    def num_flatters(self,x):
        sizes = x.size()[1:]
        result = 1
        for size in sizes:
            result *= size
        return result
