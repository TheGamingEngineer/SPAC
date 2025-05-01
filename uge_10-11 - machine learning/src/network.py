# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 13:28:12 2025

@author: spac-30
"""
import torch.nn as nn
import torch.nn.functional as F


class neural_network(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.network_stack = nn.Sequential(
            nn.Linear(in_features = 28*28, out_features = 512),
        nn.ReLU(),
        nn.Linear(in_features = 512, out_features = 10)
        )
        
    
    def forward(self,x):
        x = self.flatten(x)
        output = self.network_stack(x)
        return output
    