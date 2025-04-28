# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 11:10:22 2025

@author: spac-30
"""

import os
from dotenv import load_dotenv, dotenv_values


load_dotenv()

print(os.getenv("Epochs"))
print(os.getenv("Learning_rate"))
print(os.getenv("Batch_size"))





