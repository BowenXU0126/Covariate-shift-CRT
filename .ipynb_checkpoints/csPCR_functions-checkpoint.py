import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2, multivariate_normal
from sklearn import linear_model
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from tqdm import trange
from densratio import densratio
from numpy import linalg as la
import momentchi2 as mchi
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

# Function for estimate the conditional model of V|X,Z
from sklearn.linear_model import ElasticNetCV


def est_v_ratio(X_s, Z_s, V_s, Y_s, X_t, Z_t, V_t, test_size=0.5):
    '''
    Input:
    the X,Z,V data on both source and target (ndarray)
    Return:
    v_ratio_test: the density ratio for V|X,Z(ndarray)
    X_s_test,Z_s_test, V_s_test, Y_s_test :the (X,Z,V,Y)s data used for the test(ndarray)
    '''
    # Train-test split for source domain
    X_s_train, X_s_test, Z_s_train, Z_s_test, V_s_train, V_s_test, Y_s_train, Y_s_test = train_test_split(
        X_s, Z_s, V_s, Y_s, test_size=test_size, random_state=42
    )
   
    # Concatenate the X and Z data
    D_s_train = np.concatenate((Z_s_train, X_s_train), axis=1)
    model_s = LassoCV(cv=5)
    model_s.fit(D_s_train, V_s_train.ravel())
    print(model_s.coef_)
    
    # Estimate the variance of the V|X,Z model for testing data
    D_s_test = np.concatenate((Z_s_test, X_s_test), axis=1)
    V_pred_s_test = model_s.predict(D_s_test)
    residual_s_test = V_s_test.ravel() - V_pred_s_test
    est_var_s_test = np.var(residual_s_test)

    # Estimate the V probability for each sample in the testing set
    V_s_prob_test = norm.pdf(V_s_test.ravel(), loc=V_pred_s_test, scale=np.sqrt(est_var_s_test))

    # No need to test train split, use all samples in the target domain
    X_t_train, X_t_test, Z_t_train, Z_t_test, V_t_train, V_t_test = train_test_split(
        X_t, Z_t, V_t, test_size=0.01, random_state=42
    )

    # Train the conditional model V|X,Z on target
    D_t_train = np.concatenate((Z_t_train, X_t_train), axis=1)
    model_t = LassoCV(cv=5)
    model_t.fit(D_t_train, V_t_train.ravel())
    print(model_t.coef_)
    
    # Estimate the variance of the V|X,Z model for testing data
    D_t_test = np.concatenate((Z_t_test, X_t_test), axis=1)
    V_pred_t_test = model_t.predict(D_t_test)
    residual_t_test = V_t_test.ravel() - V_pred_t_test
    est_var_t_test = np.var(residual_t_test)

    V_pred_st_test = model_t.predict(D_s_test)

    V_t_prob_test = norm.pdf(V_s_test.ravel(), loc=V_pred_st_test, scale=np.sqrt(est_var_t_test))

    v_ratio_test = V_t_prob_test / V_s_prob_test

    return v_ratio_test, X_s_test, Z_s_test, V_s_test, Y_s_test

# Function for CRT statistic calculation for each sample
def T_statistic(y, x, z, v, E_X):
    '''
    Input:
    - y, x, z, v: Sample data
    - E_X: Expectation function E_X(z, v)

    Return:
    - Test statistic for the sample
    '''
    d_x = E_X(z,v)
    
    # Return the test statistic
    return y*x

# Function for ranking the pvalues of all conterfeits and assign the sample the bin index
def Bin_pvalue(y, x, z, v, model_X, E_X, L, K):
    '''
    Input:
    - y, x, z, v: Sample data
    - model_X: Model for generating X
    - E_X: Expectation function E_X(z, v)
    - L: Number of bins
    - K: Number of counterfeits per bin

    Return:
    - Bin index for the sample
    '''
        
    # The total number of bins
    M = L * K - 1
    cnt = 0
    
    # Calculate the test statistic of current sample
    t_stat = T_statistic(y, x, z, v, E_X)
    
    # Generate M counterfeits
    for i in range(M):
        x_ = model_X(z, v)
        if t_stat > T_statistic(y, x_, z, v, E_X):
            cnt=cnt+1
    # Find the bin index for the current sample 
    return cnt // K


# The main function for csPCR test
def PCRtest( Y, X, Z, V,model_X, E_X,density_ratio, L, K,covariate_shift):
    '''
    Input:
    - Y, X, Z, V: Data arrays
    - model_X: Model for generating X
    - E_X: Expectation function E_X(z, v)
    - density_ratio: Density ratio for V|X,Z
    - L: Number of bins
    - K: Number of counterfeits per bin
    - covariate_shift: Boolean indicating whether to consider covariate shift

    Return:
    - W: Array of weights in each bin
    - Test statistic for csPCR test
    '''
    n = Y.size
    # initialize the weight in each bin
    W = np.array([0.0]*L)

    # Loop over all samples
    for j in range(n):
        y, x, z, v = Y[j], X[j], Z[j], V[j]
        
        # With Covariate shift
        if covariate_shift == True:
            ind = Bin_pvalue(y, x, z, v,model_X, E_X, L, K)
            W[ind] += density_ratio[j]
           
        # Normal PCR test
        if covariate_shift == False:
            W[Bin_pvalue(y, x, z, v, model_X, E_X, L, K)] += 1
    
    # Return the weights and the test statistic for csPCR test
    return W, L/n * np.dot(W - n/L, W - n/L)


# Function for generating the covariance matrix of the test statistic distribution
def generate_cov_matrix(Y, X, Z, V, model_X, E_X, density_ratio,L, K):
    '''
    Input:
    - Y, X, Z, V: Data arrays
    - model_X: Model for generating X
    - E_X: Expectation function E_X(z, v)
    - density_ratio: Density ratio for V|X,Z
    - L: Number of bins
    - K: Number of counterfeits per bin

    Return:
    - Covariance matrix for the test statistic distribution
    '''
    
    n = Y.size
    diag = np.array([0.0]*L)
    
    # Loop over all samples and add corresponding weights
    for j in range(n):
        y, x, z, v = Y[j], X[j], Z[j], V[j]
        diag[Bin_pvalue(y, x, z, v,model_X, E_X,L, K)] += (density_ratio[j]**2)
        
    diag = L*(diag/n)- 1/L
     # Fill all entries with 1/L
    covariance_matrix = np.full((L, L), -1/L)
    
    # Set diagonal entries to 1 - 1/L^2
    np.fill_diagonal(covariance_matrix, diag) 
    
    # Return the 
    return covariance_matrix


import scipy.stats as stats


# Calculate chi-squared p-value
def chi_squared_p_value(chi_squared_statistic, df):
    '''
    Input:
    - chi_squared_statistic: Observed chi-squared test statistic
    - df: Degrees of freedom

    Return:
    - Calculated p-value
    '''

    p_value = 1 - stats.chi2.cdf(chi_squared_statistic, df)
    return p_value


# Calculate the normal quadratic form p-value
def moment_chi_pvalue(statistic, cov1):
    '''
    Input:
    - statistic: Test statistic
    - cov1: Covariance matrix

    Return:
    - Calculated p-value using momentchi2 library
    '''
    weight = la.eigh(cov1)[0]

    p_value = 1-mchi.hbe(coeff=abs(weight), x=statistic)
    
    return p_value


# Function for the testing procedure
def Test(X_source, Z_source, V_source, Y_source, X_target, Z_target, V_target, model_X, E_X, xz_ratio, L=3, K=20, test_size = 0.5):
    '''
    Input:
    - X_source, Z_source, V_source, Y_source: Source domain data
    - X_target, Z_target, V_target: Target domain data
    - model_X: Model for generating X
    - E_X: Expectation function E_X(z, v)
    - xz_ratio: Function for ratio X|Z
    - L: Number of bins
    - K: Number of counterfeits per bin

    Return:
    - p_value: Resulting p-value from the csPCR test
    '''
    
    # Estimate the density ratio by the V|X,Z conditional model using Lasso
    v_dr, X_source, Z_source, V_source, Y_source = est_v_ratio(X_source, Z_source, V_source,Y_source, X_target, Z_target, V_target)
    
    # Calculate the xz_ratio by the given function
    xz_dr = xz_ratio(X_source,Z_source)
    # Calculate the estimated density ratio
    est_dr = v_dr * xz_dr
    
    print('max dr: ' + str(max(est_dr)))
    # Estimate the covariance matrix for p-value calculation
    cov1 = generate_cov_matrix(Y_source, X_source, Z_source,V_source,model_X, E_X, L = L, K = K, density_ratio = est_dr)
    
    # Get the csPCR test statistic
    w, statistic = PCRtest(Y_source, X_source, Z_source,V_source,model_X, E_X, L = L, K = K, covariate_shift = True, density_ratio = est_dr)
    
    #print(w)
    # Call moment chi function to get the final p-value for the test
    p_value = moment_chi_pvalue(statistic, cov1)
    
    return p_value



def Test_true_dr(X_source, Z_source, V_source, Y_source, X_target, Z_target, V_target, model_X, E_X,L=3, K=20, true_dr = None):
    '''
    Input:
    - X_source, Z_source, V_source, Y_source: Source domain data
    - X_target, Z_target, V_target: Target domain data
    - model_X: Model for generating X
    - E_X: Expectation function E_X(z, v)
    - xz_ratio: Function for ratio X|Z
    - L: Number of bins
    - K: Number of counterfeits per bin

    Return:
    - p_value: Resulting p-value from the csPCR test
    '''
    
    # Calculate the true density ratio
    est_dr = true_dr(X_source, Z_source, V_source)
    
    print(max(est_dr))
    # Estimate the covariance matrix for p-value calculation
    cov1 = generate_cov_matrix(Y_source, X_source, Z_source,V_source,model_X, E_X, L = L, K = K, density_ratio = est_dr)
    
    # Get the csPCR test statistic
    w, statistic = PCRtest(Y_source, X_source, Z_source,V_source,model_X, E_X, L = L, K = K, covariate_shift = True, density_ratio = est_dr)
    # print(statistic)
    # Call moment chi function to get the final p-value for the test
    #p_value = moment_chi_pvalue(statistic, cov1)
    p_value = chi_squared_p_value(statistic, 2)
    return p_value
    
    



# Function for power enhancement version PCR test
def PCRtest_Powen(Y, X, Z, V, X_, Z_, V_, model_X, E_X, L, K, density_ratio):

    y_ind, v_ind, c = [], [], []
    W = np.array([0.0]*L)
    ns, nt = V.size, V_.size
    
    g_lst = np.zeros(L)
        
    for j in range(ns):
        y, x, z, v = Y[j], X[j], Z[j], V[j]
        ind_y = Bin_pvalue(y, x, z, v, model_X, E_X, L, K)
        ind_v = Bin_pvalue(v, x, z, v, model_X, E_X, L, K)
        y_ind.append(ind_y)
        v_ind.append(ind_v)
    
    y_ind = np.array(y_ind)
    v_ind = np.array(v_ind)
        
    density_ratio=np.array(density_ratio).ravel()
#     for l in range(L):
#         a = np.array([1 if x == l else 0 for x in y_ind])
#         b = np.array([1 if x == l else 0 for x in v_ind])
#         a_d = a-(a@density_ratio.T)/density_ratio.sum()
#         b_d = b-(b@density_ratio.T)/density_ratio.sum()

#         g_lst[l] = ((density_ratio*a_d)@b_d.T)/((density_ratio*b_d)@b_d.T)
    
#     print(g_lst)
    
    

        
    for j in range(nt):
        x_, z_, v_ = X_[j], Z_[j], V_[j]
        ind_v_ = Bin_pvalue(v_, x_, z_, v_,model_X, E_X, L, K)
        W[ind_v_] += (ns/nt)*g_lst[ind_v_]
        c.append(ind_v_)

    c = np.array(c)
    for j in range(ns):
        W[y_ind[j]] += density_ratio[j]
        W[v_ind[j]] -= density_ratio[j]*g_lst[v_ind[j]]   

    return W, L/ns * np.dot(W - ns/L, W - ns/L),y_ind, v_ind, c, g_lst


def I(a, b):
    if a == b:
        return 1
    else:
        return 0
    

# Generate covariance matrix for the power enhancement version
def generate_cov_matrix_powen(ind_y_source, ind_v_source, ind_v_target ,g_lst, L, K, density_ratio):
    ns = ind_y_source.size
    nt = ind_v_target.size
    
    ad = []
    num_row = ns + nt
    num_col = L
    for l in range(L):
        row = []
        for s in range(ns):
            row.append(density_ratio[s]*(I(l, ind_y_source[s]) - g_lst[l]*I(l, ind_v_source[s])))
        for t in range(nt):
            row.append(ns/nt*g_lst[l]*I(l, ind_v_target[t]))
        ad.append(row)
    ad = np.array(ad)
    cov_matrix_s = np.cov(ad[:,:ns], rowvar=True)*(ns)
    cov_matrix_t = np.cov(ad[:,ns+1:], rowvar=True)*(nt)
    #print(cov_matrix_s, cov_matrix_t)
    cov_matrix = cov_matrix_s + cov_matrix_t
    #print(cov_matrix_s*L/ns, '\n', cov_matrix_t*L/ns, '\n', ad[:,ns+1:].shape)
    return cov_matrix*L/ns


# Function for power enhancement implementation
def Test_pe(X_source, Z_source, V_source, Y_source, X_target, Z_target, V_target, model_X, E_X, xz_ratio, L=3, K=20, test_size = 0.5):
    '''
    Input:
    - X_source, Z_source, V_source, Y_source: Source domain data
    - X_target, Z_target, V_target: Target domain data
    - model_X: Model for generating X
    - E_X: Expectation function E_X(z, v)
    - xz_ratio: Function for ratio X|Z
    - L: Number of bins
    - K: Number of counterfeits per bin

    Return:
    - p_value: Resulting p-value from the csPCR test
    '''
    
    # Estimate the density ratio by the V|X,Z conditional model using Lasso
    v_dr, X_source, Z_source, V_source, Y_source = est_v_ratio(X_source, Z_source, V_source,Y_source, X_target, Z_target, V_target, test_size = test_size)
    # Calculate the xz_ratio by the given function
    xz_dr = xz_ratio(X_source,Z_source)
    # Calculate the estimated density ratio
    est_dr = v_dr * xz_dr
    
    
    #print(max(est_dr))
    WV, statistic, a,b,c,g = PCRtest_Powen(Y_source, X_source, Z_source, V_source, X_target, Z_target, V_target, model_X, E_X, L, K, est_dr)
    print(WV)
    cov = generate_cov_matrix_powen(a, b, c, g, L, K, density_ratio = est_dr)

    p_value = moment_chi_pvalue(statistic, cov)

    return p_value



def Test_pe_true_dr(X_source, Z_source, V_source, Y_source, X_target, Z_target, V_target, model_X, E_X,L=3, K=20, true_dr = None):
    '''
    Input:
    - X_source, Z_source, V_source, Y_source: Source domain data
    - X_target, Z_target, V_target: Target domain data
    - model_X: Model for generating X
    - E_X: Expectation function E_X(z, v)
    - xz_ratio: Function for ratio X|Z
    - L: Number of bins
    - K: Number of counterfeits per bin

    Return:
    - p_value: Resulting p-value from the csPCR test
    '''
    
    # Calculate the true density ratio
    est_dr = true_dr(X_source, Z_source, V_source).reshape(-1)
    
    #print(est_dr)
    
    WV, statistic, a,b,c,g = PCRtest_Powen(Y_source, X_source, Z_source, V_source, X_target, Z_target, V_target, model_X, E_X, L, K, est_dr)
    print(WV)
    cov = generate_cov_matrix_powen(a, b, c, g, L, K, density_ratio = est_dr)

    p_value = moment_chi_pvalue(statistic, cov)
    
    return p_value
    