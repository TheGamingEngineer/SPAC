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
import torch 
from network import neural_network
import matplotlib.pyplot as plt
import numpy as np

save_model=True

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
    
    
    
    return DataLoader(data, batch_size=size)
    



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
print(f"using device: {device}")
model = neural_network().to(device)

optimizer = SGD(params = model.parameters(), lr = Learning_rate)

loss_fn = nn.CrossEntropyLoss()


def training_loop(dataloader, model, loss_fn, optimizer, batch_size=Batch_size):
    size = len(dataloader.dataset)
    
    model.train()
    losses=[]
    for batch, (X,y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0: 
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
        losses.append(loss)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    average_loss=sum(losses)/len(losses)
    


def testing_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    test_loss = 0
    correct = 0
    num_batches = len(dataloader)
    
    # evaluering uden gradienter
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /=size
    
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")
    return correct*100
    

epoch_labels = [x+1 for x in range(Epochs)]
training_loss = []
testing_loss = []
accuracies = []

for t in range(Epochs):
    titel=f"### Epoch {t+1} ###"
    print(f"{'#'*len(titel)}\n{titel}\n{'#'*len(titel)}")
    
    training_loop(train_dataset, model, loss_fn, optimizer)
    correct = testing_loop(test_dataset, model, loss_fn)
    accuracies += [correct]


plt.plot(list(range(Epochs+1))[1:],accuracies)
plt.xlabel('Epochs')
plt.ylabel('Accuracies')
plt.title(f'Accuracy progression given {Epochs} epochs, learning rate={Learning_rate} and batch size of {Batch_size}')    
plt.tight_layout()
plt.show()
"""
model_file = "model.pt"

if not save_model:
    model = neural_network()
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file, weights_only=True))
        model.eval()
    torch.save(model.state_dict,model_file)
else:
    if os.path.exists(model_file):
        model = torch.load(model_file,weights_only=False)
    else:
        model = neural_network()
    torch.save(model,model_file)    
"""   
    
