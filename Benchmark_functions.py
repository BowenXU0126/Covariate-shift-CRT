import numpy as np
from New_functions import *

def model_X_true(Z, model=None):
    u = np.array([ 0, -1, 0.5, -0.5, 1])
    x = Z[:5] @ u + np.random.normal(0, 1, 1)
    return x
        

def resample(Y_source, X_source, Z_source, V_source, M, density_ratio = None):
    n = Y_source.shape[0]
    Y_, X_, Z_, V_ = [], [], [], []
    for i in range(n):
        value = density_ratio[i]
        u = np.random.random()
        if value >= u * M:
            Y_.append(Y_source[i])
            X_.append(X_source[i])
            Z_.append(Z_source[i])
            V_.append(V_source[i])
    Y_, X_, Z_, V_ = np.array(Y_) , np.array(X_), np.array(Z_), np.array(V_)
    return Y_, X_, Z_, V_


def resample_DR(num_samples, density_ratio, rate = lambda n: n**0.5, replacement='NO-REPL-gibbs', m=None):
        """
        Resampling function that returns a weighted resample of X
        Return the index of the resampled data
        """

        # Compute sample and resample size
        n = num_samples
        m = int(rate(num_samples)) if m is None else m

        # Draw weights
        w = np.array(density_ratio[:num_samples]).ravel()
        w /= w.sum()
        
        # Resample with modified replacement scheme:
        # Sample w replace, but reject if non-distinct
        if replacement == "REPL-reject":
            idx = np.random.choice(n, size=m, p=w, replace=True)
            count = 0
            while count < 100 and (len(np.unique(idx)) != len(idx)):
                count += 1
                idx = np.random.choice(n, size=m, p=w, replace=True)

            print(f"Rejections: {count}")
            raise ValueError("Unable to draw sample from REPL rejection sampler")

        elif replacement == "NO-REPL-gibbs":
            # Initialize space
            space = np.arange(n)
            # Initialize Gibbs sampler in NO-REPL distribution and shuffle to mimick dist
            idx = np.random.choice(space, size=m, p=w, replace=False)
            np.random.shuffle(idx)

            # Loop, sampling from conditional
            for _ in range(10):
                for j, i in (enumerate(idx)):
                    retain = np.delete(idx, j)
                    vacant = np.setdiff1d(space, retain)
                    idx[j] = np.random.choice(vacant, 1, p=w[vacant]/w[vacant].sum())

        elif replacement == "NO-REPL-reject":
            # Denominator for rejection sampler is smallest weights
            m_smallest = np.cumsum(w[np.argsort(w)][:(m-1)])

            # Sample from proposal, and compute bound p/Mq
            count = 0
            idx = np.random.choice(n, size=m, p=w, replace=False)
            bound = np.prod(1 - np.cumsum(w[idx])[:-1])/np.prod(1 - m_smallest)

            while ((np.random.uniform() > bound) and count < 100):
                count += 1
                idx = np.random.choice(n, size=m, p=w, replace=False)
                bound = np.prod(1 - np.cumsum(w[idx])[:-1])/np.prod(1 - m_smallest)

            if count == 100:
                raise ValueError("Unable to draw sample from NO-REPL rejection sampler")
                

        # If nothing else, just sample with or without replacement
        else:
            idx = np.random.choice(n, size=m, p=w, replace=replacement)


        return idx
    
    
def IS_test(X_e, Z_e, V_e, X_source, Z_source, V_source, Y_source, X_target, Z_target, V_target, L=3, K=20, datatype = 'binary'):
    '''
    Input:
    - X_source, Z_source, V_source, Y_source: Source domain data
    - L: Number of bins
    - K: Number of counterfeits per bin

    Return:
    - p_value: Resulting p-value from the benchmark test
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
    
    if datatype == 'binary':
        model = est_x(X_target, Z_target, datatype = 'binary')
    
        M = np.percentile(est_dr, 98)
        Y_, X_, Z_, V_ = resample(Y_source, X_source, Z_source, V_source, M = M, density_ratio=est_dr)
        _,statistic = PCRtest(Y_, X_, Z_, V_, L = L, K = K, model_X = model_X_binary, model = model,covariate_shift = False, density_ratio = None)
        p_value = chi_squared_p_value(statistic, L-1)
    else:
        model = est_x(X_target, Z_target, datatype = 'continuous')
    
        M = np.percentile(est_dr, 98)
        Y_, X_, Z_, V_ = resample(Y_source, X_source, Z_source, V_source, M = M, density_ratio=est_dr)
        _,statistic = PCRtest(Y_, X_, Z_, V_, L = L, K = K, model_X = model_X_true,model = model,covariate_shift = False, density_ratio = None)
        p_value = chi_squared_p_value(statistic, L-1)
    return p_value