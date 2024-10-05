# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:36:40 2024

@author: Allan
"""

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

A = pd.read_csv("A.csv")
L = pd.read_csv("L.csv")


# Set up the regularization parameter
alpha = 0.01  # Adjust this value based on your needs

methods = ['trf', 'lsmr', 'svd']

for method in methods:
    print(f"\nTesting method '{method}':")
    if method == 'trf':
        result = lsq_linear(A, L, method=method, regularize=alpha)
    else:
        result = lsq_linear(A, L, method=method)

    if result.success:
        X = result.x
        print("Solution X:")
        print(X)
        # Optional: Check residuals or other metrics
    else:
        print("Failed to find a solution.")