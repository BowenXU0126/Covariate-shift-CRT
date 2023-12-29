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
    def __init__(self,L=3, K = 20, X_cond_model = None, score_fnc= 'residual correlation',
                 covariate_shift = True, est_dr = True, dr = None, power_enhance = True):
        # L: Int. number of bins binning the p-values in the test (tuning parameter). L=3 is recommended(best performance in simulation)
        # K: Int. Controls the number of couterfeits for each test, M = LK-1
        # X_cond_model: Function. Should be a conditional model X|Z.
        #              Input: Z value
        #              Output: The probability distribution of X|Z
        # score_fnc: Function.
        self.covariate_shift = True
        self.L = L
        self.K = K
        self.X_cond_model  = X_cond_model
        self.t_statistic = score_fnc
        self.covariate_shift = True
        self.power_enhance = power_enhance
        self.est_dr = est_dr
        self.dr = None
        if self.est_dr:
            self.dr = dr

            

        
        
    def Covariate_Shift_Weight(self, x, z, v = 0):
        if self.weight_estimate == 'Kernel density':
            pass
        if callable(self.weight_estimate):
            return self.weight_estimate(x,z,v)
    
    def Counterfeits(self, z, v = 0):
        return self.source_distribution(z,v).rvs(size = 1)
    
    def Score(self, y, x, z, v = 0):
        if self.T_statistic == 'LASSO':
            a = np.c_[self.Z_s,self.X_s]
            reg = LassoCV().fit(a,self.Y_s)
            b = np.append(self.Z_s[0],self.X_s[0])
            c = reg.predict(b.reshape(1,21))
            return (y - 1.5*x - z @ z) ** 2
        else:
            raise ValueError('Method not included.')
    
    def generate_pvalue(self, y, x, z, v = 0):
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
        