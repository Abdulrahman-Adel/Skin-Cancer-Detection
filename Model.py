# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 12:37:58 2021

@author: Abdelrahman
"""

import torch.nn as nn


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        
        #Defining 1st convolutional layer
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        #Defining 2nd convolutional layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
         #Defining 3rd convolutional layer
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
         #Defining 4th convolutional layer
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        #Defining Linear layer
        self.linear_layer = nn.Sequential(
            nn.Linear(50176, 1024),
            nn.ReLU(inplace = True),
            nn.Linear(1024, 2)
        )
        
        
   
    def forward(self, x):
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = x.view(x.size(0), -1)
        #print(x.size())
        
        x = self.linear_layer(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
    