#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 21:20:51 2023

@author: bowenxu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2, truncnorm, multivariate_normal
from tqdm import trange
from sklearn import linear_model
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from densratio import densratio
from numpy import linalg as la


from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict


def density_ratio_estimate_prob_RF(D_nu, D_de, n_estimators=100):
    l_nu = np.ones(len(D_nu))
    l_de = np.zeros(len(D_de))
    
    l = np.concatenate((l_nu, l_de))
    D = np.concatenate((D_nu, D_de))
    
    # Fit Random Forest model
    model = LogisticRegressionCV(penalty='l1', solver='liblinear', cv=5)
    model.fit(D, l)
    
    # Get density ratios for all samples
    density_ratios = (model.predict_proba(D_de)[:, 1] / model.predict_proba(D_de)[:, 0]) * (len(D_de) / len(D_nu))
    
    return density_ratios


def density_ratio_estimate_prob_LR(D_nu, D_de):
    l_nu = np.ones(len(D_nu))
    l_de = np.zeros(len(D_de))
    
    l = np.concatenate((l_nu, l_de))
    D = np.concatenate((D_nu, D_de))
    
    #fit losgistic model
    C = 0.5
    model = LogisticRegression(penalty= 'l1', C= C,solver='liblinear')
    model.fit(D, l)
    
    # get density ratios for all samples
    density_ratios = (model.predict_proba(D_de)[:, 1] / model.predict_proba(D_de)[:, 0])*(len(D_de)/len(D_nu))
    
    return density_ratios
    


# def Covariate_Shift_Weight(x, z, v = 0):
#     return np.exp(((x - z @ s)**2 - (x - z @ t)**2)/2)

def Model_X(z, v, u):
    return z[:5] @ u + np.random.normal(0, 5, 1)

## group samples together

def T_statistic(y, x, z, v, u,s, t,regr):
    d_y = regr.predict(z.reshape(1, z.shape[0]))
  
    d_x = z[:5]@u
    
    return np.abs((y-d_y)*(x-d_x))


def Conterfeits(y, x, z, v, u,s, t, L, K, regr):
    M = L * K - 1
    cnt = 0
    t_stat = T_statistic(y, x, z, v, u,s, t,regr)

    for i in range(M):
        x_ = Model_X(z, v, u)
        if t_stat > T_statistic(y, x_, z, v, u,s, t, regr):
            cnt=cnt+1
            
    return cnt // K

def PCRtest(Y, X, Z, V, u, s, t, L, K, covariate_shift, density_ratio, regr):
    n = Y.size
    W = np.array([0.0]*L)

    for j in range(n):
        y, x, z, v = Y[j], X[j], Z[j], V[j]
        if covariate_shift == True:
            ind = Conterfeits(y, x, z, v, u,s, t, L, K, regr)
            W[ind] += density_ratio[j]
            # W[ind] += true_density_ratio(x, z, v, s, t, p)
        if covariate_shift == False:
            W[Conterfeits(y, x, z, v, u,s, t, L, K, regr)] += 1
    return W, L/n * np.dot(W - n/L, W - n/L)

def generate_cov_matrix(Y, X, Z, V,u, s, t, L, K, density_ratio, regr):
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
        diag[Conterfeits(y, x, z, v, u,s,t,L, K, regr)] += (density_ratio[j]**2)
        # diag[Conterfeits(y, x, z, v,u,s,t, L, K, regr)] += (true_density_ratio(x,z,v,s,t,p)**2)
    diag = L*(diag/n)- 1/L
    covariance_matrix = np.full((L, L), -1/L)  # Fill all entries with 1/L
    np.fill_diagonal(covariance_matrix, diag)  # Set diagonal entries to 1 - 1/L^2
    return covariance_matrix


 #get p values

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


import numpy as np

def generate(ns, nt, p,q, s, t, u, Alpha=0):
    Zs_null = np.random.normal(1,0.1, (ns, q))
    Zt_null = np.random.normal(1,0.1, (nt, q))
    
    Z_source = np.hstack((np.random.normal(0, 1, (ns, p)) , Zs_null))
    Z_target = np.hstack((np.random.normal(0.5, 1, (nt, p)) , Zt_null))
    
    X_source = Z_source[:, :p] @ u + np.random.normal(0, 5, ns)
    X_target = Z_target[:, :p] @ u + np.random.normal(0, 5, nt)

    V_source = Z_source[:, :p] @ s - X_source + np.random.normal(0, 1, ns)
    V_target = Z_target[:, :p] @ t + X_target + np.random.normal(0, 1, nt)
    
    Y_source = Z_source[:, :p].sum(axis=1) + X_source + V_source + np.random.normal(0, 1, ns) + Alpha * X_source
    Y_target = Z_target[:, :p].sum(axis=1) + X_target + V_target + np.random.normal(0, 1, nt) + Alpha * X_target
    
    return Y_source.reshape(-1, 1), X_source.reshape(-1, 1), V_source.reshape(-1, 1), Z_source,\
           Y_target.reshape(-1, 1), X_target.reshape(-1, 1), V_target.reshape(-1, 1), Z_target


def true_density_ratio(X, Z, V, s, t,p,q):
    zs_prob = multivariate_normal.pdf(Z[:p], mean=np.zeros(p), cov= np.identity(p))
    vs_prob = norm.pdf(V, loc=Z[:p]@s - X, scale=5)
    zt_prob = multivariate_normal.pdf(Z[:p], mean= 0.5*np.ones(p), cov=np.identity(p))
    vt_prob = norm.pdf(V, loc=Z[:p]@t +2* X, scale=5)
    return (zt_prob*vt_prob)/(zs_prob*vs_prob)




# Power Simulation part
if __name__ == "__main__":
    ns,nt, p,q = 1000,5000, 5, 50
    s = np.random.normal(0, 1, p)
    t = np.random.normal(0, 1, p)
    u = np.random.normal(1, 1, p)
    # verificaion by the p value
    l = 20
    count = 0
    #calculate covariance matrix
    pvalue_lst = []
    for j in trange(1000):
        #generate data
        Y_source, X_source, V_source, Z_source, Y_target, X_target, V_target, Z_target = generate(ns,nt, p,q, s, t, u, 0)

        # calculate density ratio
        D_s = np.concatenate((X_source, Z_source, V_source), axis = 1)
        D_t = np.concatenate((X_target, Z_target, V_target), axis = 1)
        # densratio_obj = densratio(D_t, D_s)

        # #calculate density ratio for each sample

        sample_density_ratio2 = density_ratio_estimate_prob_LR(D_t, D_s)

        #fit Lasso model to distill Y on Z
        reg = LassoCV().fit(Z_source,Y_source)
        
        # Compute the covariance matrix
        cov1 = generate_cov_matrix(Y_source, X_source, Z_source,V_source,\
                                u,s,t, L = l, K = 40, density_ratio = sample_density_ratio2, regr = reg)
        # Do PCR test
        w, statistic = PCRtest(Y_source, X_source, Z_source,V_source,\
                                u,s,t, L = l, K = 40, covariate_shift = True, density_ratio = sample_density_ratio2, regr = reg)
    
        p_value = monte_carlo_p_value(100000, cov1, l, statistic)
        pvalue_lst.append(p_value)
        if p_value < 0.1:
            count += 1
    probability = count/1000
