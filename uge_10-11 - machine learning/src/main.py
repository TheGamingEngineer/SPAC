# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 11:10:22 2025

@author: spac-30
"""

import os
from dotenv import load_dotenv, dotenv_values
import torchvision
from torch.utils.data import DataLoader, Subset
from torch.accelerator import current_accelerator, is_available
from torch.optim import SGD
import torch.nn as nn
from network import neural_network

load_dotenv()

root="..\data"
transformation=torchvision.transforms.ToTensor()
targeted_transformation=None
downloaded=False


Epochs = int(os.getenv("Epochs"))
Learning_rate = float(os.getenv("Learning_rate"))
Batch_size = int(os.getenv("Batch_size"))
    
    
def subset_loader(root, train, transform, target_transform,download, size, shuffle=True):
    data=torchvision.datasets.FashionMNIST(root = root, 
                                           train = train, 
                                           transform = transform, 
                                           target_transform = target_transform, 
                                           download= download)
    
    subset_dataset = Subset(data,list(range(len(data))))
    
    return DataLoader(subset_dataset, batch_size=size, shuffle=shuffle)
    



train_dataset = subset_loader(root = root, 
                              train = True, 
                              transform = transformation, 
                              target_transform = targeted_transformation, 
                              download = downloaded,
                              size = Batch_size)

test_dataset = subset_loader(root = root, 
                             train = False, 
                             transform = transformation, 
                             target_transform = targeted_transformation, 
                             download = downloaded,
                             size = Batch_size)



device = current_accelerator().type if is_available() else "cpu"

model = neural_network.to(device)

optimizer = SGD(params = model.parameters(), lr = Learning_rate)

loss_fn = nn.CrossEntropyLoss()


