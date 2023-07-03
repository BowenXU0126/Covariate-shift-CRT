#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 21:20:51 2023

@author: bowenxu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2, truncnorm
from tqdm import trange
from sklearn import linear_model
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from densratio import densratio
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

def Model_X(z, v):
    return z @ u + np.random.normal(0, 1, 1)

def T_statistic(y, x, z, v = 0):
    d_y = reg.predict(z.reshape(1,20))
    d_x = z @ u
    return np.abs((y-d_y)*(x-d_x))

def Conterfeits(y, x, z, v = 0, L = 5, K = 10):
    M = L * K - 1
    cnt = 0
    t_stat = T_statistic(y, x, z, v)
    
    for i in range(M):
        x_ = Model_X(z, v)
        if t_stat > T_statistic(y, x_, z, v):
            cnt=cnt+1
            
    return cnt // K

def PCRtest(Y, X, Z, V = 0, L = 5, K = 20, covariate_shift = True):
    n = Y.size
    W = np.array([0.0]*L)

    for j in range(n):
        y, x, z, v = Y[j], X[j], Z[j], V[j]
        if covariate_shift == True:
            ind = Conterfeits(y, x, z, v, L, K)
            W[ind] += sample_density_ratio[j]
        if covariate_shift == False:
            W[Conterfeits(y, x, z, v, L, K)] += 1     
    return W, L/n * np.dot(W - n/L, W - n/L)

def generate_cov_matrix(Y, X, Z, V = 0, L = 5, K = 20):
    """
    Generate a covariance matrix for quadratic form normal rv.

    Parameters:
    - L (int): The size of the covariance matrix.

    Returns:
    - covariance_matrix (ndarray): The generated covariance matrix.
    """
    n = Y.size
    diag = np.array([0.0]*L)
    
    for j in range(n):
        y, x, z, v = Y[j], X[j], Z[j], V[j]
        diag[Conterfeits(y, x, z, v, L, K)] += (sample_density_ratio[j]**2)
    diag = L*(diag/n)- 1/L
    covariance_matrix = np.full((L, L), -1/L)  # Fill all entries with 1/L
    np.fill_diagonal(covariance_matrix, diag)  # Set diagonal entries to 1 - 1/L^2
    return covariance_matrix

import scipy.stats as stats

def chi_squared_p_value(chi_squared_statistic, df):
    """
    Calculate the p-value from a chi-squared distribution.

    Parameters:
    - chi_squared_statistic (float): The observed chi-squared test statistic.
    - df (int): The degrees of freedom.

    Returns:
    - p_value (float): The calculated p-value.
    """
    p_value = 1 - stats.chi2.cdf(chi_squared_statistic, df)
    return p_value

def monte_carlo_p_value(n_samples, covariance_matrix, L, quantile):
    """
    Calculate the probability corresponding to a given quantile using the Monte Carlo method.

    Parameters:
    - n_samples (int): The number of Monte Carlo samples to generate.
    - covariance_matrix (ndarray): The covariance matrix of the random vector X.
    - L (int): The number of components to sum.
    - quantile (float): The quantile value.

    Returns:
    - probability (float): The estimated probability corresponding to the quantile.
    """
    count = 0
    for _ in range(n_samples):
        sample = np.random.multivariate_normal(np.zeros(L), covariance_matrix)
        squared_sum = np.sum(sample**2)
        if squared_sum <= quantile:
            count += 1

    probability = count / n_samples
    return 1-probability


# Generate Data

def generate(n, p, s, t, u, Alpha = 0):
    Z_source, Z_target = np.zeros((n, p)), np.zeros((n, p))
    V_source, V_target = 0, 0
    for i in range(n):
        Z_source[i] = np.random.normal(0, 1, p)
        Z_target[i] = np.random.normal(0.1, 1, p)
        
    X_source = Z_source @ u + np.random.normal(0, 1, n)
    X_target = Z_target @ u + np.random.normal(0, 1, n)
    
    V_source = Z_source @ s - X_source + np.random.normal(0, 1, n)
    V_target = Z_target @ t + X_target + np.random.normal(0, 1, n)
    
    Y_source, Y_target = np.zeros(n), np.zeros(n)
    for i in range(n):
        Y_source[i] = np.sin(Z_source[i].sum() + X_source[i] + V_source[i]) + np.random.normal(0, 1, 1) + Alpha * X_source[i]
        Y_target[i] = np.sin(Z_target[i].sum() + X_target[i] + V_target[i]) + np.random.normal(0, 1, 1) + Alpha * X_target[i]
    return Y_source.reshape(-1,1), X_source.reshape(-1,1), V_source.reshape(-1,1), Z_source,\
           Y_target.reshape(-1,1), X_target.reshape(-1,1), V_target.reshape(-1,1), Z_target
           

if __name__ == '__main__':
    K = 20
    L = 10
    n, p = 1000, 20
    s = np.random.normal(0, 1, p)
    t = s + np.random.normal(0, 0.1, p)
    u = np.random.normal(0, 1, p)
    
    #generate data
    Y_source, X_source, V_source, Z_source, Y_target, X_target, V_target, Z_target = generate(n, p, s, t, u, 0)
    
    # calculate density ratio
    D_s = np.concatenate((X_source, Z_source, V_source), axis = 1)
    D_t = np.concatenate((X_target, Z_target, V_target), axis = 1)
    densratio_obj = densratio(D_t, D_s)
    
    #calculate density ratio for each sample
    sample_density_ratio = densratio_obj.compute_density_ratio(D_s)
    
    # distill information of Z on Y
    reg = LassoCV().fit(Z_source,Y_source)
    
    # verification by the p value
    for j in range(1000):
        cov1 = generate_cov_matrix(Y_source, X_source, Z_source,V_source, L = 10, K = 20)
        print(monte_carlo_p_value(100000, cov1, 10, PCRtest(Y_source, X_source, Z_source,V_source, L = 10, K = 20, covariate_shift = True)[1]))
