# -*- coding: utf-8 -*-
"""
Created on Fri May  2 10:32:22 2025

@author: spac-30
"""

import torch
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import struct
import os
from network import neural_network
import seaborn
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


path = r"..\data\FashionMNIST\raw\t10k-images-idx3-ubyte"
label_path = r"..\data\FashionMNIST\raw\t10k-labels-idx1-ubyte"

transformation=torchvision.transforms.ToTensor()



targeted_transformation=None

with open(path, "rb") as f:
    magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
    images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)

with open(label_path, "rb") as f:
    magic, num_items = struct.unpack(">II", f.read(8))
    labels = np.frombuffer(f.read(), dtype=np.uint8)

model_instance = neural_network()
model_instance.load_state_dict(torch.load("model.pt"))
model_instance.eval()    

data={"0":[0 for x in range(10)],
      "1":[0 for x in range(10)],
      "2":[0 for x in range(10)],
      "3":[0 for x in range(10)],
      "4":[0 for x in range(10)],
      "5":[0 for x in range(10)],
      "6":[0 for x in range(10)],
      "7":[0 for x in range(10)],
      "8":[0 for x in range(10)],
      "9":[0 for x in range(10)]}

df_label=[str(i) for i in range(10)]

for i in range(10000):
    # Vælg fx billede nummer 0
    billede = images[i]
    label = labels[i]
    
    plt.imshow(billede, cmap="gray")
    plt.title(f"FashionMNIST billede #{i}")
    plt.axis("off")
    plt.show()
    
    img = Image.fromarray(billede)
    img.save(f"fashionmnist_{i}.png")
    
    
    img = Image.open(f"fashionmnist_{i}.png").convert("L")
    img_array = images[i]
    
    x = transformation(Image.fromarray(img_array)).unsqueeze(0)
    x = transformation(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model_instance(x)
        pred = model_instance(x).argmax(1).item()
    
    #print(f"Billede {i}: Sand label = {label}, Forudsigelse = {pred}")
    #plt.imshow(img_array, cmap="gray")
    #plt.title(f"Rigtig: {label}, Model: {pred}")
    #plt.show()
    data[str(label)][pred]+=1
    os.remove(f"fashionmnist_{i}.png")

df = pd.DataFrame(data)

plt.figure(figsize=(8,6))
seaborn.heatmap(df,cmap = "RdYlGn", annot = True, xticklabels = df_label, yticklabels = df_label)

# Tilføj labeller på længde- og breddeaksen
plt.xlabel("Modelens forudsigelse")
plt.ylabel("Rigtig label")

# Valgfrit: titel
plt.title("Konfusionsmatrix: aktuel vs. model")

plt.show()


