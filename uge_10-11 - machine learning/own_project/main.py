# -*- coding: utf-8 -*-
"""
Created on Mon May  5 13:48:00 2025

@author: spac-30
"""

import os
from dotenv import load_dotenv, dotenv_values
import torchvision
from torch.utils.data import DataLoader, Subset
from torch.accelerator import current_accelerator, is_available
from torch.optim import SGD
import torch.nn as nn
import torch.nn.functional as F
import torch 
import pandas as pd
import numpy as np
from neural_network import *

Epochs = 50
Learning_rate = 1e-3
Batch_size = 128 

device = current_accelerator().type if is_available() else "cpu"
print(f"using device: {device}")
model = neural_network().to(device)


promoter_Data=pd.read_csv("promoters_eukaryot_2.csv",sep=";")    

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



