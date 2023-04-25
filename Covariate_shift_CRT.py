#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 12:06:36 2023

@author: bowenxu
Package for Covariate shift CRT

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2, truncnorm
from tqdm import trange
from sklearn import linear_model
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso

class Covariate_shift_CRT:
    def __init__(self, X_source, X_target,
                 Z_source, Z_target, V_source, V_target,Y_source,
                 source_distribution,band_width = 0.1,L=5, K = 10, T_statistic= 'LASSO',
                 weight_estimate = 'Kernel density', covariate_shift = True):
        
        self.covariate_shift = True
        self.L = L
        self.K = K
        self.source_distribution = source_distribution
        self.weight_estimate = weight_estimate
        self.band_width = band_width
        self.T_statistic = T_statistic
        self.X_s = X_source
        self.X_t = X_target
        self.Z_s = Z_source
        self.Z_t = Z_target
        self.V_t = V_target
        self.V_s = V_source
        self.Y_s = Y_source
        
        
    def Covariate_Shift_Weight(self, x, z, v = 0):
        if self.weight_estimate == 'Kernel density':
            pass
        if callable(self.weight_estimate):
            return self.weight_estimate(x,z,v)
    
    def Model_X(self, z, v = 0):
        return self.source_distribution(z,v).rvs(size = 1)
    
    def T_statistic(self, y, x, z, v = 0):
        if self.T_statistic == 'LASSO':
            a = np.c_[self.Z_s,self.X_s]
            reg = LassoCV().fit(a,self.Y_s)
            b = np.append(self.Z_s[0],self.X_s[0])
            c = reg.predict(b.reshape(1,21))
            return (y - 1.5*x - z @ z) ** 2
        else:
            raise ValueError('Method not included.')
    
    def Conterfeits(self, y, x, z, v = 0):
        M = self.L * self.K - 1
        rank = 1
        
        for i in range(M):
            x_ = self.Model_X(z, v)
            if self.T_statistic(y, x, z, v) >= self.T_statistic(y, x_, z, v):
                rank = rank+1
                
        return rank // self.K
    
    def PCRtest(self, L = 5, K = 20, covariate_shift = True):
        n = self.Y_s.size
        W = np.array([0.0]*self.L)
        
        for j in range(n):
            y, x, z, v = self.Y_s[j], self.X_s[j], self.Z_s[j], 0
            if covariate_shift == True:
                W[self.Conterfeits(y, x, z, v)] += self.Covariate_Shift_Weight(x, z, v)
            if covariate_shift == False:
                W[self.Conterfeits(y, x, z, v)] += 1
                
        return W, self.L/n * np.dot(W - n/self.L, W - n/self.L)
        