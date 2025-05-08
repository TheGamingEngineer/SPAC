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
from neural_network import neural_network

Epochs = 50
Learning_rate = 1e-3
Batch_size = 128 

device = current_accelerator().type if is_available() else "cpu"
print(f"using device: {device}")
model = neural_network().to(device)


promoter_Data=pd.read_csv("promoters_eukaryot_2.csv",sep=";")    





