# -*- coding: utf-8 -*-
"""
Created on Mon May  5 13:48:00 2025

@author: spac-30
"""

import os
from dotenv import load_dotenv, dotenv_values
import torchvision
from torch.utils.data import DataLoader, Subset, Dataset
from torch.accelerator import current_accelerator, is_available
from torch.optim import SGD
import torch.nn as nn
import torch.nn.functional as F
import torch 
import pandas as pd
import numpy as np

Epochs = 50
Learning_rate = 1e-3
Batch_size = 128 
save_model=False

test_Data=pd.read_csv("test_eukaryot_curated.csv",sep=";") 
train_Data=pd.read_csv("training_eukaryot_curated.csv",sep=";")    
val_Data=pd.read_csv("validation_eukaryot_curated.csv",sep=";") 

input_size = 4
hidden_size = 32


class RNN(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(RNN,self).__init__()
        self.hidden_size = hidden_size
        
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.promoter_out = nn.Linear(hidden_size,1)
        self.org_out = nn.Linear(hidden_size, 5)
        
        self.softmax = nn.LogSoftmax(dim = 1)
                
    def forward(self,x):
        _, hidden = self.rnn(x)
        hidden = hidden.squeeze(0)
        return self.promoter_out(hidden), self.org_out(hidden)


device = current_accelerator().type if is_available() else "cpu"
print(f"using device: {device}")
model = RNN(input_size, hidden_size).to(device)

def seq_one_hot(seq):
    nucleotide = {"A":[1,0,0,0],
                  "T":[0,1,0,0],
                  "G":[0,0,1,0],
                  "C":[0,0,0,1],
                  "W":[0.5,0.5,0,0],
                  "S":[0,0,0.5,0.5],
                  "M":[0.5,0,0,0.5],
                  "K":[0,0.5,0.5,0],
                  "R":[0.5,0,0.5,0],
                  "Y":[0,0.5,0,0.5],
                  "B":[0,0.33,0.33,0.33],
                  "D":[0.33,0.33,0.33,0],
                  "H":[0.33,0.33,0,0.33],
                  "V":[0.33,0,0.33,0.33],
                  "N":[0.25,0.25,0.25,0.25]
                  }
    one_hot_list=[]
    
    
    for i in seq:
        one_hot_list.append(nucleotide[i])
    
    return torch.tensor(one_hot_list)
    
class Sequence_dataset(Dataset):
    def __init__(self, dataframe, max_len=1000):
        self.dataframe = dataframe
        self.max_len = max_len
        self.sequences = dataframe["sequence"]
        self.promoter_labels = dataframe["promoter"]
        
        # konverter organisme navne til heltal for algoritme bearbejdning
        self.organism_to_index = {label:idx for idx, label in enumerate(sorted(self.dataframe["organism"].unique()))}
        self.organism_labels = self.dataframe["organism"].map(self.organism_to_index)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences.iloc[idx][:self.max_len]
        one_hot_seq = seq_one_hot(seq)
        padded = torch.zeros(self.max_len,4)
        padded[:len(one_hot_seq)] = one_hot_seq
        promoter_label = torch.tensor(self.promoter_labels.iloc[idx],dtype=torch.long)
        org_label = torch.tensor(self.organism_labels.iloc[idx],dtype=torch.long)
        
        return padded, promoter_label, org_label
        
train_dataset = Sequence_dataset(train_Data, max_len=1000)
val_dataset = Sequence_dataset(val_Data, max_len=1000)
test_dataset = Sequence_dataset(test_Data, max_len=1000)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


optimizer = torch.optim.Adam(model.parameters(),lr=Learning_rate)

promoter_loss_func=nn.BCEWithLogitsLoss()
organism_loss_func=nn.CrossEntropyLoss()

def training_loop(dataloader, model, optimizer, batch_size=Batch_size, pro_fn=promoter_loss_func, org_fn=organism_loss_func):
    
    model.train()
    promoter_losses=[]
    organism_losses=[]
    total_losses=[]
    
    n=0
    for x, y_promoter, y_org in dataloader:
        x = x.to(device).float()
        y_promoter = y_promoter.to(device).float().unsqueeze(1)
        #y_promoter = y_promoter.unsqueeze(1)        
        
        y_org = y_org.to(device)
        
        ## generere forudsigelse
        pred_promoter, pred_org = model(x)
        
        ## udregner tab
        promoter_loss = pro_fn(pred_promoter, y_promoter)
        org_loss = org_fn(pred_org, y_org)
        total_loss = promoter_loss + org_loss
        
        ## opdater model
        total_loss.backward()
        optimizer.zero_grad()
        optimizer.step()
        
        
        if n % 100 == 0: 
            ## udregner promoter-forudsigelses nøjagtigheden
            probs = torch.sigmoid(pred_promoter)
            correct_pro = ((probs > 0.5) == y_promoter).sum().item()
            promoter_acc = correct_pro / y_promoter.size(0)
            
            ## udregner organisme-forudsigelses nøjagtigheden
            pred_class = pred_org.argmax(dim=1)
            correct_org = (pred_class == y_org).sum().item()
            org_acc = correct_org / y_org.size(0)
            
            print(f"promoter-loss: {promoter_loss.item():>7f}; organism-loss: {org_loss.item():>7f}]")
            print(f"promoter-accuracy: {promoter_acc:>7f}; organism-accuracy: {org_acc:>7f}]")
            
        promoter_losses.append(promoter_loss.item())
        organism_losses.append(org_loss.item())
        total_losses.append(total_loss)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        n+=1
    
    average_promoter_loss=sum(promoter_losses)/len(promoter_losses)
    average_organism_loss=sum(organism_losses)/len(organism_losses)
    average_total_loss=sum(total_losses)/len(total_losses)
    
    
    return average_promoter_loss, average_organism_loss, average_total_loss


def testing_loop(dataloader, model, pro_fn=promoter_loss_func, org_fn=organism_loss_func):
    size = len(dataloader.dataset)
    model.eval()
    promoter_loss = 0
    org_loss = 0
    total_loss=0
    correct_pro = 0
    correct_org = 0
    correct_total=0
    num_batches = len(dataloader)
    
    # evaluering uden gradienter
    with torch.no_grad():
        for X, y_promoter, y_org in dataloader:
            y_promoter = y_promoter.to(device).float().unsqueeze(1)
            y_org = y_org.to(device)
            
            pro_pred, org_pred = model(X)

            promoter_loss += pro_fn(pro_pred, y_promoter).item()
            org_loss += org_fn(org_pred, y_org).item()
            total_loss += pro_fn(pro_pred, y_promoter).item() + org_fn(org_pred, y_org).item()
            
            # Beregn korrekthed for promoter
            probs = torch.sigmoid(pro_pred)
            correct_pro += ((probs > 0.5) == y_promoter).sum().item()
            
            # Beregn korrekthed for organism
            correct_org += (org_pred.argmax(1) == y_org).sum().item()
            
    
    promoter_loss /= num_batches
    org_loss /= num_batches
    total_loss /= num_batches
    
    correct_total = (correct_pro + correct_org) / (size)
    correct_pro /=size
    correct_org /=size
    
    print(f"Average Test Losses: Promoters={promoter_loss:>7f}; Organisms={org_loss:>7f}; Total={total_loss:>7f}")
    print(f"Accuracies: Promoters={100*correct_pro:>7f}; Organisms={100*correct_org:>7f}; Total={100*correct_total:>7f}\n")
    return correct_pro*100, promoter_loss, correct_org*100, org_loss, correct_total*100, total_loss



epoch_labels = [x+1 for x in range(Epochs)]
training_P_losses = []
training_O_losses = []
training_T_losses = []

testing_P_losses = []
testing_O_losses = []
testing_T_losses = []

accuracies_P = []
accuracies_O = []
accuracies_T = []

for t in range(Epochs):
    titel=f"### Epoch {t+1} ###"
    print(f"{'#'*len(titel)}\n{titel}\n{'#'*len(titel)}")
    
    training_P_loss, training_O_loss, training_T_loss = training_loop(train_loader, model, optimizer)
    correct_pro, testing_P_loss, correct_org, testing_O_loss, correct_total, testing_T_loss = testing_loop(test_loader, model)
    
    training_P_losses += [float(training_P_loss)]
    training_O_losses += [float(training_O_loss)]
    training_T_losses += [float(training_T_loss)]
    
    testing_P_losses += [float(testing_P_loss)]
    testing_O_losses += [float(testing_O_loss)]
    testing_T_losses += [float(testing_T_loss)]
    
    
    accuracies_P += [float(correct_pro)]
    accuracies_O += [float(correct_org)]
    accuracies_T += [float(correct_total)]

run_data={
    "Epoch":epoch_labels,
    "Promoter training losses":training_P_losses,
    "Promoter test losses":testing_P_losses,
    "Promoter accuracy":accuracies_P,
    "Organism training losses":training_O_losses,
    "Organism_test losses":testing_O_losses,
    "Organism accuracy":accuracies_O,
    "Total training losses":training_T_losses,
    "Total test losses":testing_T_losses,
    "Total accuracy":accuracies_T
    }
df = pd.DataFrame(run_data)

df.to_excel("promoter_models.xlsx",index=False)

model_file = "model.pt"

if not save_model:
    torch.save(model.state_dict(),model_file)
else:
    model = torch.load(model_file,weights_only=False)
    torch.save(model,model_file)    








