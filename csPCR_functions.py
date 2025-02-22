import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2, multivariate_normal
from sklearn import linear_model
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from tqdm import trange
from numpy import linalg as la
import momentchi2 as mchi
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


# Function for estimate the conditional model of V|X,Z
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neural_network import MLPClassifier




def est_z_ratio(Z_e, Z_s, Z_t, method = 'LR', gamma=0.1,hidden_layer_sizes=(10,10), alpha=1e-4):
    N_e = Z_e.shape[0]
    N_t = Z_t.shape[0]
    N_ratio = N_e/N_t
    y_e = np.zeros(N_e)
    y_t = np.ones(N_t)
    z = np.concatenate((Z_e, Z_t), axis=0)
    y = np.concatenate((y_e, y_t), axis = 0)
    if method == 'LR':
        clf = LogisticRegression(random_state=0).fit(z, y)
        training_accuracy = clf.score(z, y)
        class_prob = clf.predict_proba(Z_s)
        prob_ratio = class_prob[:, 1:2] / class_prob[:, 0:1]
        density_ratio = prob_ratio * N_ratio
        #print(f'Training Accuracy for Z DR: {training_accuracy}')
    elif method == 'KLR':
        # Compute RBF kernel between Z_e and Z_t
        K = rbf_kernel(z, z, gamma=gamma)
        
        # Fit Kernel Ridge Regression
        clf = LogisticRegression(random_state=0).fit(K,y)
        # Compute kernel between Z_s and combined dataset Z_e + Z_t
        K_s = rbf_kernel(Z_s, z, gamma=gamma)

        # Predict using Kernel Ridge Regression
        class_prob = clf.predict_proba(K_s)
        prob_ratio = class_prob[:, 1:2] / class_prob[:, 0:1]
        density_ratio = prob_ratio * N_ratio
    if method == 'NN':
        # Define and train a feedforward neural network
        clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, random_state=0)
        clf.fit(z, y)
        
        # Predict probabilities for Z_s using the neural network
        class_prob = clf.predict_proba(Z_s)
        
        # Calculate probability ratio and density ratio
        prob_ratio = class_prob[:, 1:2] / class_prob[:, 0:1]
        density_ratio = prob_ratio * N_ratio
        
    return density_ratio
        

def est_x(X_e, Z_e, datatype='continuous', method = 'LR'):
    std = 0
    if method == 'LR':
        if datatype == 'binary':
            model = LogisticRegression(random_state=0).fit(Z_e, X_e)
            # print(f'Training accuracy for X|Z: {model.score(Z_e, X_e)}')
            return model
        elif datatype == 'continuous':# 'continuous'
            model = LassoCV(cv=5).fit(Z_e, X_e)

            X_pred = model.predict(Z_e)
            #print(f'Training accuracy for X|Z: {model.score(Z_e, X_e)}')
            residuals = X_e - X_pred
            std = np.std(residuals)
            return model, std

# def model_X(Z, model, method='binary', std=1):
#     Z = Z.reshape(1, -1)
#     if method == 'binary':
#         prob_X_is_1 = model.predict_proba(Z)[:, 1]
#         sampled_X = np.random.binomial(1, prob_X_is_1)
#         return sampled_X + np.random.normal(0, 0.1, 1)
#     else:  # 'continuous'
#         sampled_X = model.predict(Z)+np.random.normal(0,std,1)
#         return sampled_X
    
    


def model_X_binary(Z, model):
    Z = Z.reshape(1,-1)
    prob_X_is_1 = model.predict_proba(Z)[:, 1]

        # Sample new binary values for X based on the predicted probabilities
    sampled_X = np.random.binomial(1, prob_X_is_1)

    return sampled_X + np.random.normal(0,0.1,1)


def create_model_X_continuous(std):
    def model_X_continuous(Z, model):
        Z = Z.reshape(1, -1)
        return model.predict(Z) + np.random.normal(0, std, 1)
    return model_X_continuous


def model_X_continuous(Z, model, std):
    Z = Z.reshape(1,-1)
    return model.predict(Z)+np.random.normal(0,std,1)
    
def model_X_true(Z, model=None):
    u = np.array([ 0, -1, 0.5, -0.5, 1])
    x = Z[:5] @ u + np.random.normal(0, 1, 1)
    return x
        
def est_v_ratio(X_e, Z_e, V_e, X_s, Z_s, V_s, Y_s, X_t, Z_t, V_t):
    '''
    Input:
    the X,Z,V data on both source and target (ndarray)
    Return:
    v_ratio_test: the density ratio for V|X,Z(ndarray)
    X_s_test,Z_s_test, V_s_test, Y_s_test :the (X,Z,V,Y)s data used for the test(ndarray)
    '''
    
    # Concatenate the X and Z data
    D_e = np.concatenate((Z_e, X_e), axis=1)
    model_s = ElasticNetCV(cv=5)
    model_s.fit(D_e, V_e.ravel())
    
    # Estimate the variance of the V|X,Z model for testing data
    D_s = np.concatenate((Z_s, X_s), axis=1)
    V_pred_s = model_s.predict(D_s)
    residual_s = V_s.ravel() - V_pred_s
    est_var_s = np.var(residual_s)

    # Estimate the V probability for each sample in the testing set
    V_s_prob = norm.pdf(V_s.ravel(), loc=V_pred_s, scale=np.sqrt(est_var_s))

    # Train the conditional model V|X,Z on target
    D_t = np.concatenate((Z_t, X_t), axis=1)
    model_t = ElasticNetCV(cv=5)
    model_t.fit(D_t, V_t.ravel())

    # Estimate the variance of the V|X,Z model for testing data
    D_t = np.concatenate((Z_t, X_t), axis=1)
    V_pred_t = model_t.predict(D_t)
    residual_t = V_t.ravel() - V_pred_t
    est_var_t = np.var(residual_t)

    V_pred_st = model_t.predict(D_s)

    V_t_prob = norm.pdf(V_s.ravel(), loc=V_pred_st, scale=np.sqrt(est_var_t))

    v_ratio = V_t_prob / V_s_prob

    return v_ratio



# Function for CRT statistic calculation for each sample
def T_statistic(y, x, z, v):
    '''
    Input:
    - y, x, z, v: Sample data
    - E_X: Expectation function E_X(z, v)

    Return:
    - Test statistic for the sample
    '''
    u = np.array([ 0, -1, 0.5, -0.5, 1])
    # d_x = E_X(z,v)
    # Return the test statistic
    return y*(x- z[:5]@u)+ np.random.normal(0,0.1,1)



# Function for ranking the pvalues of all conterfeits and assign the sample the bin index
def Bin_pvalue(y, x, z, v, model_X, model, L, K):
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
    t_stat = T_statistic(y, x, z, v)
    
    # Generate M counterfeits
    for i in range(M):
        x_ = model_X(z, model)
        if t_stat > T_statistic(y, x_, z, v):
            cnt=cnt+1
    # Find the bin index for the current sample 
    return cnt // K


# The main function for csPCR test
def PCRtest( Y, X, Z, V,model_X, model, L, K,covariate_shift,density_ratio=None):
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
            ind = Bin_pvalue(y, x, z, v,model_X, model,L, K)
            W[ind] += density_ratio[j]
           
        # Normal PCR test
        if covariate_shift == False:
            W[Bin_pvalue(y, x, z, v,model_X,model, L, K)] += 1
    
    # Return the weights and the test statistic for csPCR test
    return W, L/n * np.dot(W - n/L, W - n/L)


# Function for generating the covariance matrix of the test statistic distribution
def generate_cov_matrix(Y, X, Z, V, model_X,model,density_ratio,L, K):
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
        diag[Bin_pvalue(y, x, z, v,model_X,model,L, K)] += (density_ratio[j]**2)
        
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


def PCR_test(X, Z, V, Y, L=3, K=20, datatype='binary'):
    '''
    Input:
    - X, Z, V, Y: Data for PCR test
    - L: Number of bins
    - K: Number of counterfeits per bin

    Return:
    - p_value: Resulting p-value from the csPCR test
    '''
    # Preprocess data
    # Add noise to the value to break ties in the test
    proportion = 0.5
    num = int(proportion * X.shape[0])
    Z_e = Z[:num]
    X_e = X[:num]
    X = X[num+1:]
    Z = Z[num+1:]
    Y = Y[num+1:]
    V = V[num+1:]
    
    noise_std = 0.1
    noise_x = np.random.normal(0, noise_std, X.shape)  # Make sure to use X instead of X_source
    X = X.astype(float)
    X += noise_x

    # Adding random Gaussian noise to Y if it's continuous
    noise_y = np.random.normal(0, noise_std, Y.shape)  # Make sure to use Y instead of Y_source
    Y = Y.astype(float)
    Y += noise_y
    if datatype== 'binary':
        model = est_x(X_e, Z_e, datatype = 'binary')

        w, statistic = PCRtest(Y, X, Z, V, model_X_binary, model=model, L=L, K=K, covariate_shift=False)
        p_value = chi_squared_p_value(statistic, L-1)
         
    
    return p_value

        
    
    
# Function for the testing procedure
def Test(X_e, Z_e, V_e, X_source, Z_source, V_source, Y_source, X_target, Z_target, V_target, L=3, K=20, datatype = 'binary'):
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
    #Preprocess data
    # ADD noise to the value so that it break ties in the test
    noise_std = 0.1 
    noise_x = np.random.normal(0, noise_std, X_source.shape)
    X_source = X_source.astype(float)
    X_source += noise_x

    # Adding random Gaussian noise to Y_source if it's continuous
    noise_y = np.random.normal(0, noise_std, Y_source.shape)
    Y_source = Y_source.astype(float)
    Y_source += noise_y
    
    #Estimate the density ratio of Z
    z_dr = np.squeeze(est_z_ratio(Z_e, Z_source, Z_target))
    
    # Estimate the density ratio by the V|X,Z conditional model using Lasso
    v_dr= est_v_ratio(X_e, Z_e, V_e, X_source, Z_source, V_source,Y_source, X_target, Z_target, V_target)
    
    # Calculate the xz_ratio by the given function
    # xz_dr = xz_ratio(X_source,Z_source)
    # Calculate the estimated density ratio
    est_dr = v_dr * z_dr
    est_dr = est_dr / est_dr.mean()

    #est_dr = est_dr/np.mean(est_dr)               
    if datatype == 'binary':
        model = est_x(X_target, Z_target, datatype = 'binary')
    
        # Estimate the covariance matrix for p-value calculation
        cov1 = generate_cov_matrix(Y_source, X_source, Z_source,V_source,model_X_binary, model=model,L = L, K = K, density_ratio = est_dr)

        # Get the csPCR test statistic
        w, statistic = PCRtest(Y_source, X_source, Z_source,V_source,model_X_binary, model=model, L = L, K = K, covariate_shift = True, density_ratio = est_dr)
        print(f'weight distribution:{w}, test statistic:{statistic}')
        print(f'Max density ratio:{max(est_dr)}')
        print(f'X model coef:{model.coef_}')
        print(f'covariance matrix{cov1}')
    else:
        model,std = est_x(X_target, Z_target, datatype = 'continuous')
        
        model_X = create_model_X_continuous(std)
        # Estimate the covariance matrix for p-value calculation
        cov1 = generate_cov_matrix(Y_source, X_source, Z_source,V_source,model_X_true, model=model,L = L, K = K, density_ratio = est_dr)

        # Get the csPCR test statistic
        w, statistic = PCRtest(Y_source, X_source, Z_source,V_source,model_X_true, model=model, L = L, K = K, covariate_shift = True, density_ratio = est_dr)
        
    # print(f'The weighted rank distribution: {w}')
    #print(w)
    # Call moment chi function to get the final p-value for the test
    p_value = moment_chi_pvalue(statistic, cov1)
    
    return p_value





def Test_true_dr(X_source, Z_source, V_source, Y_source, X_target, Z_target, V_target, model_X,model,L=3, K=20, true_dr = None):
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
    cov1 = generate_cov_matrix(Y_source, X_source, Z_source,V_source,model_X, model,L = L, K = K, density_ratio = est_dr)
    
    # Get the csPCR test statistic
    w, statistic = PCRtest(Y_source, X_source, Z_source,V_source,model_X,model, L = L, K = K, covariate_shift = True, density_ratio = est_dr)
    print(w)
    # Call moment chi function to get the final p-value for the test
    p_value = moment_chi_pvalue(statistic, cov1)
    
    return p_value
    
    



# Function for power enhancement version PCR test
def PCRtest_Powen(Y, X, Z, V, X_, Z_, V_, model_X, model, L, K, density_ratio):

    y_ind, v_ind, c = [], [], []
    W = np.array([0.0]*L)
    ns, nt = V.size, V_.size
    
    g_lst = np.zeros(L)
        
    for j in range(ns):
        y, x, z, v = Y[j], X[j], Z[j], V[j]
        ind_y = Bin_pvalue(y, x, z, v, model_X, model, L, K)
        ind_v = Bin_pvalue(v, x, z, v, model_X, model, L, K)
        y_ind.append(ind_y)
        v_ind.append(ind_v)
    
    y_ind = np.array(y_ind)
    v_ind = np.array(v_ind)
        
    density_ratio=np.array(density_ratio).ravel()
    for l in range(L):
        a = np.array([1 if x == l else 0 for x in y_ind])
        b = np.array([1 if x == l else 0 for x in v_ind])
        a_d = a-(a@density_ratio.T)/density_ratio.sum()
        b_d = b-(b@density_ratio.T)/density_ratio.sum()

        g_lst[l] = ((density_ratio*a_d)@b_d.T)/((density_ratio*b_d)@b_d.T)
    
    print(g_lst)

        
    for j in range(nt):
        x_, z_, v_ = X_[j], Z_[j], V_[j]
        ind_v_ = Bin_pvalue(v_, x_, z_, v_,model_X, model,L, K)
        W[ind_v_] += (ns/nt)*g_lst[ind_v_]
        c.append(ind_v_)

    c = np.array(c)
    for j in range(ns):
        W[y_ind[j]] += density_ratio[j]
        W[v_ind[j]] -= density_ratio[j]*g_lst[v_ind[j]]   

    return W, L/ns * np.dot(W - ns/L, W - ns/L),y_ind, v_ind, c, g_lst
# def PCRtest_Powen(Y, X, Z, V, X_, Z_, V_, model_X, model, L, K, density_ratio):
#     ns = V.size
#     nt = V_.size

#     # Compute y_ind and v_ind using list comprehension
#     y_ind = np.array([Bin_pvalue(Y[j], X[j], Z[j], V[j], model_X, model, L, K) for j in range(ns)])
#     v_ind = np.array([Bin_pvalue(V[j], X[j], Z[j], V[j], model_X, model, L, K) for j in range(ns)])

#     density_ratio = np.array(density_ratio).ravel()  # Ensure it's a 1D array

#     # Vectorize over L to compute g_lst
#     L_values = np.arange(L)  # Shape (L,)

#     # Create indicator matrices A and B
#     A = (y_ind == L_values[:, None]).astype(int)  # Shape (L, ns)
#     B = (v_ind == L_values[:, None]).astype(int)  # Shape (L, ns)

#     sum_density_ratio = density_ratio.sum()
#     A_density = A * density_ratio  # Element-wise multiplication
#     B_density = B * density_ratio

#     sum_A_density = A_density.sum(axis=1)  # Sum over ns
#     sum_B_density = B_density.sum(axis=1)

#     # Compute adjusted values a_d and b_d
#     a_d = A - (sum_A_density[:, None] / sum_density_ratio)
#     b_d = B - (sum_B_density[:, None] / sum_density_ratio)

#     # Compute numerator and denominator for g_lst
#     numerator = np.sum(A_density * b_d, axis=1)
#     denominator = np.sum(B_density * b_d, axis=1)
#     g_lst = numerator / denominator

#     # Initialize W and compute c using list comprehension
#     W = np.zeros(L)
#     c = np.array([Bin_pvalue(V_[j], X_[j], Z_[j], V_[j], model_X, model, L, K) for j in range(nt)])

#     # Update W based on c and g_lst
#     W_increment = np.bincount(c, weights=(ns / nt) * g_lst[c], minlength=L)
#     W += W_increment

#     # Update W using np.add.at for in-place addition
#     np.add.at(W, y_ind, density_ratio)
#     np.add.at(W, v_ind, -density_ratio * g_lst[v_ind])

#     # Compute the test statistic
#     test_stat = L / ns * np.dot(W - ns / L, W - ns / L)

#     return W, test_stat, y_ind, v_ind, c, g_lst


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
    
    cov_matrix = np.cov(ad, rowvar=True)
    return cov_matrix*L/ns*(ns+nt)



# Function for power enhancement implementation
def Test_pe(X_e, Z_e, V_e, X_source, Z_source, V_source, Y_source, X_target, Z_target, V_target, L=3, K=20, datatype = 'binary', score = 'pos'):
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
    
    
     #Preprocess data
    # ADD noise to the value so that it break ties in the test
    noise_std = 0.1 
    noise_x = np.random.normal(0, noise_std, X_source.shape)
    X_source = X_source.astype(float)
    X_source += noise_x

    # Adding random Gaussian noise to Y_source if it's continuous
    noise_y = np.random.normal(0, noise_std, Y_source.shape)
    Y_source = Y_source.astype(float)
    Y_source += noise_y
    
    #Estimate the density ratio of Z
    z_dr = np.squeeze(est_z_ratio(Z_e, Z_source, Z_target))
    # Estimate the density ratio by the V|X,Z conditional model using Lasso
    v_dr= est_v_ratio(X_e, Z_e, V_e, X_source, Z_source, V_source,Y_source, X_target, Z_target, V_target)
    
    # Calculate the xz_ratio by the given function
    # xz_dr = xz_ratio(X_source,Z_source)
    # Calculate the estimated density ratio
    est_dr = v_dr * z_dr
    # print(f'Max of DR: {np.max(est_dr)}')
    # print(f'Mean of DE: {np.mean(est_dr)}')
    est_dr = est_dr/np.mean(est_dr)
    if score == 'neg':
        V_source = -V_source
        V_target = -V_target
        
        
    if datatype == 'binary':
    #print(max(est_dr))
        model = est_x(X_target, Z_target, datatype = 'binary')
        WV, statistic, a,b,c,g = PCRtest_Powen(Y_source, X_source, Z_source, V_source, X_target, Z_target, V_target, model_X_binary, model, L, K, est_dr)
        # print(WV)
        cov = generate_cov_matrix_powen(a, b, c, g, L, K, density_ratio = est_dr)
    else:
        model,std = est_x(X_target, Z_target, datatype = 'continuous')
        WV, statistic, a,b,c,g = PCRtest_Powen(Y_source, X_source, Z_source, V_source, X_target, Z_target, V_target, model_X_true, model, L, K, est_dr)
        # print(WV)
        cov = generate_cov_matrix_powen(a, b, c, g, L, K, density_ratio = est_dr)
    p_value = moment_chi_pvalue(statistic, cov)

    return p_value



def Test_pe_true_dr(X_source, Z_source, V_source, Y_source, X_target, Z_target, V_target, model_X,L=3, K=20, true_dr = None):
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
    
    WV, statistic, a,b,c,g = PCRtest_Powen(Y_source, X_source, Z_source, V_source, X_target, Z_target, V_target, model_X, L, K, est_dr)
    print(WV)
    cov = generate_cov_matrix_powen(a, b, c, g, L, K, density_ratio = est_dr)

    p_value = moment_chi_pvalue(statistic, cov)
    
    return p_value
    