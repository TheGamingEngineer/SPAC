# -*- coding: utf-8 -*-
"""
Created on Wed May 14 09:23:37 2025

@author: spac-30
"""

import torch
import matplotlib.pyplot as plt
import os
import seaborn
import pandas as pd
from main import RNN, Sequence_dataset, seq_one_hot, read_large_jsonl
from torch.utils.data import DataLoader
from torch.accelerator import current_accelerator, is_available

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = current_accelerator().type if is_available() else "cpu"
print(f"using device: {device}")
input_size = 4
hidden_size = 64


model = RNN(input_size, hidden_size)
model.load_state_dict(torch.load("promoter_learner.pt",map_location=device))
model = model.to(device)
model.eval()


## indlæser valideringsdatasæt
val_Data = read_large_jsonl("validation_eukaryot_2025-07-15.jsonl")
#val_set = Sequence_dataset(val_Data, max_len=1000)
val_set = Sequence_dataset(val_Data)
val_loader = DataLoader(val_set, batch_size=128, shuffle=False)


ordered_organism_dict = {v: k for k, v in val_set.organism_to_index.items()}
organism_labels = [label for label, _ in sorted(val_set.organism_to_index.items(), key=lambda x: x[1])]

organism_konfusion_matrix_data = {str(j):[0 for i in range(len(organism_labels))] for j in range(len(organism_labels))}

promoter_konfusion_matrix_data = {"0":[0,0],
                                  "1":[0,0]}

with torch.no_grad():
    for x, y_pro, y_org in val_loader:
        x = x.to(device).float()
        y_pro = y_pro.to(device).float().unsqueeze(1)
        y_org = y_org.to(device)
        
        # forudsigelse via modellen
        pro_pred, org_pred = model(x)
        
        # sandsynlighedsberegning af promoter forudsigelse
        pro_probs = torch.sigmoid(pro_pred)
        pro_preds = (pro_probs > 0.5).squeeze().long()
        true_pro_labels = y_pro.squeeze().long()
        
        # sandsynlighed for organism forudsigelse
        pred_class = org_pred.argmax(dim=1)
        
        for x, y in zip(true_pro_labels.tolist(), pro_preds.tolist()):
            promoter_konfusion_matrix_data[str(x)][y] += 1
        
        for x,y in zip(y_org.tolist(),pred_class.tolist()):
            organism_konfusion_matrix_data[str(x)][y] += 1
            

def procent_normal(dict_of_list,key_list):
    pred_amount = []
    
    for x in range(len(key_list)):
        n=0
        for key in dict_of_list.keys():
            n+=dict_of_list[key][x]
        pred_amount.append(n)
    
    for y in range(len(pred_amount)):
        for key in dict_of_list.keys():
            dict_of_list[key][y] /= pred_amount[y]
            dict_of_list[key][y] *= 100
    
    return dict_of_list

organism_matrix = procent_normal(organism_konfusion_matrix_data, organism_labels)
promoter_matrix = procent_normal(promoter_konfusion_matrix_data, [0,1])

org_df = pd.DataFrame(organism_matrix)
pro_df = pd.DataFrame(promoter_matrix)

plt.figure(figsize=(8,6))
or_hot = seaborn.heatmap(org_df,cmap = "RdYlGn", annot = True, fmt=".1f", xticklabels = organism_labels, yticklabels = organism_labels)
or_hot.tick_params(axis="x", labelrotation = 45)

# Tilføj labeller på længde- og breddeaksen
plt.xlabel("Modelens forudsigelse")
plt.ylabel("Rigtig label")

# Valgfrit: titel
plt.title("Organisme Forudsigelse: Aktuel vs. Model")
plt.savefig("Organisme Forudsigelse.png")
plt.show()

plt.figure(figsize=(8,6))
seaborn.heatmap(pro_df,cmap = "RdYlGn", annot = True, xticklabels = ["0","1"], yticklabels = ["0","1"])

# Tilføj labeller på længde- og breddeaksen
plt.xlabel("Modelens forudsigelse")
plt.ylabel("Rigtig label")

# Valgfrit: titel
plt.title("Promoter Forudsigelse: Aktuel vs. Model")
plt.savefig("Promoter Forudsigelse.png")
plt.show()