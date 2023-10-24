import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2, truncnorm
from tqdm import trange
import ite7
# 1
def Covariate_Shift_Weight(x, z, v):
    return 1

def Model_X(z, v):
    return z + v**2 + np.random.normal(0, 1, 1)

#gi
def T_statistic(y, x, z, v):
    return (y - 1.5*x - z * v) ** 2

def Conterfeits(y, x, z, v, L = 5, K = 10):
    M = L * K - 1
    cnt = 0
    
    for i in range(M):
        x_ = Model_X(z, v)
        if T_statistic(y, x, z, v) > T_statistic(y, x_, z, v):
            cnt=cnt+1
            
    return cnt // K

def PCRtest(Y, X, Z, V, L = 5, K = 20, covariate_shift = True):
    n = Y.size
    W = np.array([0]*L)
    
    for j in range(n):
        y, x, z, v = Y[j], X[j], Z[j], V[j]
        if covariate_shift == True:
            W[Conterfeits(y, x, z, v, L, K)] += Covariate_Shift_Weight(x, z, v)
        if covariate_shift == False:
            W[Conterfeits(y, x, z, v, L, K)] += 1
            
    return W, L/n * np.dot(W - n/L, W - n/L)

n=500
Z = np.random.normal(1, 1, n)
V = np.random.normal(1, 1, n)
X = Z + V**2 + np.random.normal(0, 1, n)
Y = Z * V + np.random.normal(0, 1, n) + 0.5*X

PCRtest(Y, X, Z, V, L = 5, K = 20, covariate_shift = True)
