
import numpy as np
from scipy.stats import norm, chi2, multivariate_normal
from sklearn.linear_model import LogisticRegressionCV
import numpy.linalg as la
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score

import momentchi2 as mchi



# Logistic regression density ratio estimation
def density_ratio_estimate_prob_LR(D_nu, D_de):
    l_nu = np.ones(len(D_nu))
    l_de = np.zeros(len(D_de))
    
    l = np.concatenate((l_nu, l_de))
    D = np.concatenate((D_nu, D_de))
    
    # Adding cross terms
    cross_terms_nu = np.prod(D_nu, axis=1)  # Taking the element-wise product of features in D_nu
    cross_terms_de = np.prod(D_de, axis=1)  # Taking the element-wise product of features in D_de
    D_nu_with_cross_terms = np.hstack((D_nu, cross_terms_nu[:, np.newaxis]))
    D_de_with_cross_terms = np.hstack((D_de, cross_terms_de[:, np.newaxis]))
    D_with_cross_terms = np.concatenate((D_nu_with_cross_terms, D_de_with_cross_terms))
    
    # Fit logistic model with cross terms
    C = 0.1
    model = LogisticRegressionCV(penalty='l1', solver='liblinear', cv=5)
    model.fit(D_with_cross_terms, l)
    
    # Get density ratios for all samples
    density_ratios = (model.predict_proba(D_de_with_cross_terms)[:, 1] / model.predict_proba(D_de_with_cross_terms)[:, 0]) * (len(D_de) / len(D_nu))
    
    return density_ratios

def est_v_ratio(X_s, Z_s, V_s, X_t, Z_t, V_t):
    # Generate cross-terms between X_s and Z_s
    # concatenate the data and fit the lasso model
    D_s = np.concatenate((Z_s, X_s), axis=1)
    model_s = LassoCV(cv=5)
    model_s.fit(D_s, V_s.ravel())

    # Estimate the variance of the V|X,Z model
    V_pred_s = model_s.predict(D_s)
    residual_s = V_s.ravel() - V_pred_s
    est_var_s = np.var(residual_s)
   
    # Compute the R^2 score for the source model
    r2_s = r2_score(V_s, V_pred_s)
    
    # Estimate the V probability for each sample
    V_s_prob = norm.pdf(V_s.ravel(), loc=V_pred_s, scale=np.sqrt(est_var_s))
    
    # Generate cross-terms between X_t and Z_t
    
    D_t = np.concatenate((Z_t, X_t), axis=1)
    model_t = LassoCV(cv=5)
    model_t.fit(D_t, V_t.ravel())
    
    V_pred_t = model_t.predict(D_t)
    residual_t = V_t.ravel() - V_pred_t
    est_var_t = np.var(residual_t)
    
    V_pred_st = model_t.predict(D_s)
    # Compute the R^2 score for the target model
    r2_t = r2_score(V_t, V_pred_t)

    V_t_prob = norm.pdf(V_s.ravel(), loc=V_pred_st, scale=np.sqrt(est_var_t))
    
    v_ratio = V_t_prob / V_s_prob
    
    return v_ratio

# normal distribution density ration estimation(directly calculate covarince and mean)
def norm_est_ratio(D_s, D_t):
    ns = D_s.shape[0]
    nt = D_t.shape[0]
    p = D_s.shape[1]
    
    ED_s = np.mean(D_s, axis=0)
    ED_t = np.mean(D_t, axis=0)
    
    CS = np.cov(D_s.T)
    CT = np.cov(D_t.T)
    
    pdf_s = multivariate_normal.pdf(D_s, mean=ED_s, cov=CS)
    pdf_t = multivariate_normal.pdf(D_s, mean=ED_t, cov=CT)
    
    true_ = pdf_t / pdf_s
    
    return true_

# generate x couterfeits
def Model_X(z, v, u):
    return z[:5] @ u + np.random.normal(0, 1, 1)


# Calculate T-statistics
def T_statistic(y, x, z, v, u,s, t,regr):
#     d_y = regr.predict(z.reshape(1, z.shape[0]))
#     # d_y = (1+s)@z
    d_x = z[:5]@u
    
#     return np.abs(((y-d_y)*(x-d_x)))
    return y*(x-d_x)
    # return coef[0]

# Calculate p-value for each sample
def Conterfeits(y, x, z, v, u,s, t, L, K, regr):
    M = L * K - 1
    cnt = 0
    t_stat = T_statistic(y, x, z, v, u,s, t,regr)

    for i in range(M):
        x_ = Model_X(z, v, u)
        if t_stat > T_statistic(y, x_, z, v, u,s, t, regr):
            cnt=cnt+1
            
    return cnt // K


# The PCRtest procedure
def PCRtest( Y, X, Z, V, u, s, t, L, K, covariate_shift, density_ratio, regr):
    n = Y.size
    W = np.array([0.0]*L)

    for j in range(n):
        y, x, z, v = Y[j], X[j], Z[j], V[j]
        if covariate_shift == True:
            ind = Conterfeits(y, x, z, v, u,s, t, L, K, regr)
            W[ind] += density_ratio[j]
                
        if covariate_shift == False:
            W[Conterfeits(y, x, z, v, u,s, t, L, K, regr)] += 1
    return W, L/n * np.dot(W - n/L, W - n/L)



# generate covarianve matrix
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
    diag = L*(diag/n)- 1/L
    covariance_matrix = np.full((L, L), -1/L)  # Fill all entries with 1/L
    np.fill_diagonal(covariance_matrix, diag)  # Set diagonal entries to 1 - 1/L^2
    return covariance_matrix

def PCRtest_Powen(Y, X, Z, V, Y_, X_, Z_, V_, u, s, t, L, K, density_ratio, regr):
    a, b, c = [], [], []
    W = np.array([0.0]*L)
    ns, nt = Y.size, Y_.size
    for j in range(ns):
        y, x, z, v = Y[j], X[j], Z[j], V[j]
        ind_y = Conterfeits(y, x, z, v, u, s, t, L, K, regr)
        ind_v = Conterfeits(v, x, z, v, u, s, t, L, K, regr)
        a.append(ind_y)
        b.append(ind_v)
    
    a = np.array(a)
    b = np.array(b)
    density_ratio=np.array(density_ratio).ravel()

    g = (a*density_ratio)@(b*density_ratio)/((b*density_ratio)@((b*density_ratio).T))
    
    for j in range(nt):
        y_, x_, z_, v_ = Y_[j], X_[j], Z_[j], V_[j]
        ind_v_ = Conterfeits(v_, x_, z_, v_, u, s, t, L, K, regr)
        W[ind_v_] += ns/nt*g
        c.append(ind_v_)

    c = np.array(c)
    for j in range(ns):
        W[a[j]] += density_ratio[j]
        W[b[j]] -= density_ratio[j]*g    

    return W, L/ns * np.dot(W - ns/L, W - ns/L), a, b, c, g

def I(a, b):
    if a == b:
        return 1
    else:
        return 0
    

def generate_cov_matrix_powen(ind_y_source, ind_v_source, ind_v_target ,gamma, L, K, density_ratio):
    ns = ind_y_source.size
    nt = ind_v_target.size

    cov_matrix = np.zeros((L, L))
    E_t = []
    E_s = []
    V_t = []
    for l in range(L):
        e = 0
        f = 0
        for i in range(nt):
            e += I(ind_v_target[i], l)
        for j in range(ns):
            f += float(density_ratio[j]*(I(ind_y_source[j], l) - gamma*I(ind_v_source[j], l)))
        E_t.append(e*gamma*ns/nt)
        V_t.append(e*(nt - e) * gamma**2 *ns**2 / (nt**3))
        E_s.append(f)

    for l_1 in range(L):
        for l_2 in range(l_1+1):
            e = 0
            for j in range(ns):
                ind_v_source[j]
                e += density_ratio[j]**2*(I(ind_y_source[j], l_1) - gamma*I(ind_v_source[j], l_1))*(I(ind_y_source[j], l_2) - gamma*I(ind_v_source[j], l_2))

            cov_matrix[l_1, l_2] = e - E_s[l_1]*E_s[l_2]/ns + I(l_1, l_2)*V_t[l_1] - (1 - I(l_1, l_2))*E_t[l_1]*E_t[l_2]/(nt**2)
            cov_matrix[l_2, l_1] = e - E_s[l_1]*E_s[l_2]/ns + I(l_1, l_2)*V_t[l_1] - (1 - I(l_1, l_2))*E_t[l_1]*E_t[l_2]/(nt**2)
    
#    print(V_t)
#    print(ind_y_source.sum(), ind_v_source.sum(), ind_v_target.sum())
    return cov_matrix*L/ns
   


# Define the data genaration process

def generate(ns, nt, p,q, s, t, u, Alpha_s=0, Alpha_t = 2):
    Zs_null = np.random.normal(0,0.1, (ns, q))
    Zt_null = np.random.normal(0,0.1, (nt, q))
    
    Z_source = np.hstack((np.random.normal(0, 1, (ns, p)) , Zs_null))
    Z_target = np.hstack((np.random.normal(0.1, 1, (nt, p)) , Zt_null))
    
    X_source = Z_source[:, :p] @ u + np.random.normal(0, 1, ns)
    X_target = Z_target[:, :p] @ u + np.random.normal(0, 1, nt)

    V_source = Z_source[:, :p] @ s + Alpha_s * X_source + np.random.normal(0, 5, ns)
    V_target = Z_target[:, :p] @ t + Alpha_t * X_target + np.random.normal(0, 5, nt)
    
    # V_source = Z_source[:, :p] @ s + 2*X_source 
    # V_target = Z_target[:, :p] @ t - 2*X_target
    
    Y_source = (Z_source[:, :p].sum(axis=1))**2 + V_source + np.random.normal(0, 1, ns) 
    Y_target = (Z_target[:, :p].sum(axis=1))**2 + V_target + np.random.normal(0, 1, nt) 
    
    
    return Y_source.reshape(-1, 1), X_source.reshape(-1, 1), V_source.reshape(-1, 1), Z_source,\
           Y_target.reshape(-1, 1), X_target.reshape(-1, 1), V_target.reshape(-1, 1), Z_target


def true_density_ratio(X, Z, V, s, t,p,q, Alpha_s = 0, Alpha_t = 2):
    ratios = []
    size = V.size
    for i in range(size):
        zs_prob = multivariate_normal.pdf(Z[i][:p], mean = 0*np.ones(p), cov= 1*np.identity(p))
        vs_prob = norm.pdf(V[i], loc=Z[i][:p]@s + Alpha_s*X[i], scale =5)
        zt_prob = multivariate_normal.pdf(Z[i][:p], mean = 0.1*np.ones(p), cov= 1*np.identity(p))
        vt_prob = norm.pdf(V[i], loc=Z[i][:p]@t + Alpha_t*X[i], scale =5)
        ratios.append((zt_prob*vt_prob)/(zs_prob*vs_prob))
    # zs_probs = multivariate_normal.pdf(Z[:, :p], mean=0*np.ones(p), cov=np.identity(p))
    # vs_probs = norm.pdf(V, loc=(Z[:, :p] @ s).reshape(-1,1) + 2*X, scale=5)
    # zt_probs = multivariate_normal.pdf(Z[:, :p], mean=0.2*np.ones(p), cov=np.identity(p))
    # vt_probs = norm.pdf(V, loc=(Z[:, :p] @ t).reshape(-1,1) - 2*X, scale=5)
    # ratios = (zt_probs * vt_probs) / (zs_probs * vs_probs)
    return ratios

def xz_ratio(X, Z, V, s, t,p,q):
    ratios = []
    size = V.size
    for i in range(size):
        zs_prob = multivariate_normal.pdf(Z[i][:p], mean = 0*np.ones(p), cov= 1*np.identity(p))
        zt_prob = multivariate_normal.pdf(Z[i][:p], mean = 0.1*np.ones(p), cov= 1*np.identity(p))
        ratios.append((zt_prob)/(zs_prob))
    
    return ratios


ns,nt, p,q = 5000,5000, 5, 50
n_labeled = 1000
s = np.array([-0.56228753, -1.01283112,  0.31424733, -0.90802408, -1.4123037 ])
t = np.array([ 1.46564877, -0.2257763 ,  0.0675282 , -1.42474819, -0.54438272])
u = np.array([ 0.11092259, -1.15099358,  0.37569802, -0.60063869, -0.29169375])
# s = np.random.normal(0, 1, p)
# t = np.random.normal(0, 1, p)
# u = np.random.normal(0, 1, p)



# import sys

# # 获取从命令行传递的l值参数
# if len(sys.argv) != 2:
#     print("Usage: python Power_simulation.py <l>")
#     sys.exit(1)

l = 3
effect_s = 0
effect_t = 2

power_enhance = False

if power_enhance ==  False:
    count = 0
    #calculate covariance matrix

    for j in range(500):
        #generate data
        Y_source, X_source, V_source, Z_source, Y_target, X_target, V_target, Z_target = \
        generate(ns,nt, p,q, s, t, u, Alpha_s = effect_s, Alpha_t = effect_t)

        # # concatenate X,Z,V together
        # D_s = np.concatenate((X_source, Z_source, V_source), axis = 1)
        # D_t = np.concatenate((X_target, Z_target, V_target), axis = 1)



        reg = 1
        v_dr = est_v_ratio(X_source, Z_source, V_source, X_target, Z_target, V_target)
        xz_dr = xz_ratio(X_source,Z_source, V_source, s,t,p,q)
        # densratio_obj = densratio(D_t, D_s)
        # #calculate density ratio for each sample
        # sample_density_ratio1 = densratio_obj.compute_density_ratio(D_s)
        true_dr = true_density_ratio(X_source, Z_source, V_source,s,t,p,q,Alpha_s = effect_s, Alpha_t = effect_t)[:n_labeled]
        est_dr = v_dr[:n_labeled] * xz_dr[:n_labeled]
    
        
        cov1 = generate_cov_matrix(Y_source[:n_labeled], X_source[:n_labeled], Z_source[:n_labeled],
                                V_source[:n_labeled],u,s,t, L = l, K = 20, density_ratio = est_dr, regr = reg)
        w, statistic = PCRtest(Y_source[:n_labeled], X_source[:n_labeled], Z_source[:n_labeled],
                            V_source[:n_labeled],u,s,t, L = l, K = 20, covariate_shift = True, density_ratio = est_dr, regr = reg)
        weight = la.eigh(cov1)[0]
        # print([w,statistic])

        # calculate the p value as weighted sum of chi squared random variables
        p_value = 1-mchi.hbe(coeff=weight, x=statistic)
        #print(p_value)
        if p_value < 0.1:
            count += 1
    probability = count/500

    with open("Results/Power_eff_est_n1000_0.1.txt", "a+") as text_file:
        text_file.write("Effect: %s, power: %s\n" % (effect_t, probability))

else:
    count = 0
    #calculate covariance matrix
    probability= 0
    for j in range(1000):
        #generate data
        Y_source, X_source, V_source, Z_source, Y_target, X_target, V_target, Z_target = generate(ns,nt, p,q, s, t, u, Alpha_s = effect_s, Alpha_t = effect_t)

        # calculate density ratio
        D_s = np.concatenate((X_source, Z_source, V_source), axis = 1)
        D_t = np.concatenate((X_target, Z_target, V_target), axis = 1)

        v_dr = est_v_ratio(X_source, Z_source, V_source, X_target, Z_target, V_target)
        xz_dr = xz_ratio(X_source,Z_source, V_source, s,t,p,q)
        reg=1
        
        true_dr = true_density_ratio(X_source[:n_labeled], Z_source[:n_labeled], V_source[:n_labeled],s,t,p,q,Alpha_s = effect_s, Alpha_t = effect_t)
        est_dr = v_dr[:n_labeled] * xz_dr[:n_labeled]
        
        WV, statistic, a, b, c, g = PCRtest_Powen(Y_source[:n_labeled], X_source[:n_labeled], Z_source[:n_labeled], V_source[:n_labeled], Y_target[:n_labeled], X_target[:n_labeled], Z_target[:n_labeled], V_target[:n_labeled], u, s, t, l, 20, true_dr, reg)
        cov = generate_cov_matrix_powen(a, b, c, g, l, 20, density_ratio = true_dr)
        weight = la.eigh(cov)[0]
        p_value = 1-mchi.lpb4(coeff=weight, x=statistic)
        

        if p_value < 0.1:
            count += 1
        probability = count/(j+1)

    with open("/gpfsnyu/home/bx2038/Covariate-shift-CRT/Results/Size_enhance_L_est_n2000_0.1.txt", "a+") as text_file:
        text_file.write("L: %s, size: %s\n" % (l, probability))