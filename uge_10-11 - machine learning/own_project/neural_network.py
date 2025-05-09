# -*- coding: utf-8 -*-
"""
Created on Wed May  7 10:59:03 2025

@author: spac-30
"""
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.network_stack = nn.Sequential(
        nn.Linear(in_features = 31812, out_features = 612),
        nn.BatchNorm1d(612), # nromaliser aktivering
        nn.ReLU(), # tilføjer non-linearitet
        nn.Linear(in_features = 612, out_features = 410),
        nn.BatchNorm1d(410), # nromaliser aktivering
        nn.ReLU(), # tilføjer non-linearitet
        nn.Linear(in_features = 410, out_features = 210),
        nn.BatchNorm1d(210), # nromaliser aktivering
        nn.ReLU(), # tilføjer non-linearitet
        nn.Linear(in_features = 210, out_features = 5)
        )
        
    
    def forward(self,x):
        x = self.flatten(x)
        output = self.network_stack(x)
        return output

class CNN(nn.Module):
    def __init__(self) -> None:
        return self

    def forward(self,x):
        return x

class RNN(nn.Module):
    def __init__(self) -> None:
        
        return self

    def forward(self,x):
        return x