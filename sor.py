import numpy as np
import pandas as pd
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import time
import os

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.optim import Optimizer
import scipy
from scipy.stats import pearsonr
from scipy.fft import fft, ifft

from shared import *

import seaborn as sns

import statsmodels.api as sm









def simpleGWAS(snps, phenotype):

    n_snps = snps.shape[1]
    # Prepare a DataFrame to store the GWAS results (beta coefficient and p-value for each SNP)
    results = pd.DataFrame({
        "SNP": np.arange(n_snps),
        "beta": np.empty(n_snps),
        "pvalue": np.empty(n_snps)
    })

    # Loop over each SNP and run a simple linear regression using statsmodels.
    # This tests the association between the SNP (independent variable) and the phenotype (dependent variable).
    for i in range(n_snps):
        # The SNP genotype for the current marker
        X = snps[:, i]
        # Add a constant term to include an intercept in the model
        X = sm.add_constant(X)
        
        # Fit the OLS model
        model = sm.OLS(phenotype, X)
        res = model.fit()
        
        # Store the effect size (beta) and the p-value for the SNP
        results.loc[i, "beta"] = res.params[1]
        results.loc[i, "pvalue"] = res.pvalues[1]


    betas = results['pvalue']
    pvals = results['pvalue']

    # Display the first few GWAS results
    #print(results.head())

    return betas, pvals








def deconvolver1(X):


    from scipy.optimize import minimize
    from scipy.linalg import expm

    # Problem dimensions
    #N = 2              # A is N x N
    N = X.shape[0]
    M = X.shape[1]            # X is N x M; adjust as needed

    # Generate a random X (you can replace this with your data)
    #X = np.random.randn(N, M)

    # Number of parameters for the skew-symmetric matrix: N(N-1)/2
    num_params = N * (N - 1) // 2

    def skew_from_vector(theta, N):
        """
        Construct an N x N skew-symmetric matrix S from a vector theta
        of length N*(N-1)/2.
        """
        S = np.zeros((N, N))
        idx = 0
        for i in range(N):
            for j in range(i+1, N):
                S[i, j] = theta[idx]
                S[j, i] = -theta[idx]
                idx += 1
        return S

    def A_from_theta(theta, N):
        """
        Compute the orthonormal matrix A = expm(S), where S is skew-symmetric.
        """
        S = skew_from_vector(theta, N)
        A = expm(S)
        return A

    def objective(theta, X, N):
        """
        Objective: the L1 norm of A @ X.
        """
        A = A_from_theta(theta, N)
        AX = A @ X

        score = np.sum(AX ** 2) ** 0.5
        score = -1 * score / np.sum(np.abs(AX))

        #return np.sum(np.abs(AX))
        return score

    # Initial guess for theta: zeros (i.e. A = expm(0) = identity)
    theta0 = np.zeros(num_params)

    # Optimize using SciPy's minimize.
    res = minimize(objective, theta0, args=(X, N), method='Nelder-Mead')

    theta_opt = res.x
    A_opt = A_from_theta(theta_opt, N)

    print("Optimized orthonormal matrix A:")
    print(A_opt)


    #print (np.sum(np.abs(A_opt ** 2), axis=1))
    #print("L1 norm of A*X:", objective(theta_opt, X, N))

    AX = A_opt @ X 

    #print (AX)

    return A_opt



def getAllCoefs(Y_mean, X):

    #Ntrait = 5

    Ntrait = Y_mean.shape[1]
    coef_all = np.zeros((Ntrait,  X.shape[1] ))
    
    for a in range(Ntrait):

        Y = Y_mean[:, a]

        N = X.shape[0]
        M = X.shape[1]

        # Simulated kinship matrix (normally computed from genome-wide SNPs)
        K = np.corrcoef(X.T)  # Approximate relatedness

        # Fit a GWAS-style model (one SNP at a time)
        p_values = []
        for i in range(M):
            snp = X[:, i]  # Single SNP
            df = pd.DataFrame({"Y": Y, "SNP": snp, "individual": np.arange(N)})
            # Mixed model: SNP as fixed effect, polygenic background as random effect
            model = MixedLM(df["Y"], sm.add_constant(df["SNP"]), groups=df["individual"])
            result = model.fit()
            # Store p-value of SNP
            p_values.append(result.pvalues["SNP"])

        
        print(p_values)
        quit()


        coef_all[a] = gamma_hat

    return coef_all




def deconvolver1(X):


    from scipy.optimize import minimize
    from scipy.linalg import expm

    # Problem dimensions
    #N = 2              # A is N x N
    N = X.shape[0]
    M = X.shape[1]            # X is N x M; adjust as needed

    # Generate a random X (you can replace this with your data)
    #X = np.random.randn(N, M)

    # Number of parameters for the skew-symmetric matrix: N(N-1)/2
    num_params = N * (N - 1) // 2

    def skew_from_vector(theta, N):
        """
        Construct an N x N skew-symmetric matrix S from a vector theta
        of length N*(N-1)/2.
        """
        S = np.zeros((N, N))
        idx = 0
        for i in range(N):
            for j in range(i+1, N):
                S[i, j] = theta[idx]
                S[j, i] = -theta[idx]
                idx += 1
        return S

    def A_from_theta(theta, N):
        """
        Compute the orthonormal matrix A = expm(S), where S is skew-symmetric.
        """
        S = skew_from_vector(theta, N)
        A = expm(S)
        return A

    def objective(theta, X, N):
        """
        Objective: the L1 norm of A @ X.
        """
        A = A_from_theta(theta, N)
        AX = A @ X

        score = np.sum(AX ** 2) ** 0.5
        score = -1 * score / np.sum(np.abs(AX))

        #return np.sum(np.abs(AX))
        return score

    # Initial guess for theta: zeros (i.e. A = expm(0) = identity)
    theta0 = np.zeros(num_params)

    # Optimize using SciPy's minimize.
    res = minimize(objective, theta0, args=(X, N), method='Nelder-Mead')

    theta_opt = res.x
    A_opt = A_from_theta(theta_opt, N)

    print("Optimized orthonormal matrix A:")
    print(A_opt)


    #print (np.sum(np.abs(A_opt ** 2), axis=1))
    #print("L1 norm of A*X:", objective(theta_opt, X, N))

    AX = A_opt @ X 

    #print (AX)

    return A_opt





class HardSparseDictionaryLearning:
    def __init__(self, n_components, max_iter=50, tol=1e-4, force_orthogonal=False, random_state=None):
        """
        Custom Dictionary Learning with exactly one active atom per sample.
        Optionally enforces D to be orthogonal.

        Parameters:
        - n_components: Number of dictionary atoms (m)
        - max_iter: Maximum number of iterations
        - tol: Convergence threshold
        - force_orthogonal: If True, forces D to be an orthogonal matrix
        - random_state: Seed for reproducibility
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.force_orthogonal = force_orthogonal
        self.random_state = random_state

    def fit(self, X):

        
        """
        Learns the dictionary and sparse representation.

        Parameters:
        - X: (n, d) data matrix

        Returns:
        - D: (n, m) dictionary (optionally orthogonal)
        - Z: (m, d) sparse coefficients (each column has exactly one nonzero value)
        """
        np.random.seed(self.random_state)
        n, d = X.shape
        m = self.n_components

        # Step 1: Initialize Dictionary randomly
        D = np.random.randn(n, m)
        D /= np.linalg.norm(D, axis=0, keepdims=True)  # Normalize columns

        Z = np.zeros((m, d))  # Sparse coefficient matrix
        prev_error = float('inf')  # Track error for convergence

        for iteration in range(self.max_iter):
            # -------- Step 1: Sparse Coding (Fix D, Solve for Z) -------- #
            for i in range(d):
                x_i = X[:, i]

                # Compute projection onto each dictionary atom
                projections = D.T @ x_i  # (m,)
                
                # Select the best atom (largest absolute projection)
                best_atom_idx = np.argmax(np.abs(projections))
                
                # Assign the best coefficient value
                Z[:, i] = 0  # Reset Z for this column
                Z[best_atom_idx, i] = projections[best_atom_idx]  # Store the projection

            # Ensure no zero columns in Z
            if np.all(Z == 0):
                print("Warning: All-zero Z encountered, reinitializing.")
                Z = np.random.randn(m, d) * 1e-3  # Small noise instead of zero

            # -------- Step 2: Dictionary Update (Fix Z, Solve for D) -------- #
            try:
                D = X @ np.linalg.pinv(Z)  # Least squares update for D
            except np.linalg.LinAlgError:
                print("Warning: Singular matrix encountered in np.linalg.pinv(Z), using regularized SVD.")
                U, S, Vt = np.linalg.svd(Z, full_matrices=False)
                S = np.maximum(S, 1e-6)  # Regularization
                D = X @ (Vt.T @ np.diag(1 / S) @ U.T)

            # -------- Option: Enforce Orthogonality -------- #
            if self.force_orthogonal and m <= n:
                U, _, Vt = np.linalg.svd(D, full_matrices=False)
                print ('orth')
                D = U @ Vt  # Projection onto the closest orthogonal matrix

            # Normalize columns to avoid scaling issues
            norms = np.linalg.norm(D, axis=0, keepdims=True)
            D[:, norms[0] > 0] /= norms[:, norms[0] > 0]  # Avoid division by zero

            # -------- Convergence Check -------- #
            reconstruction_error = np.linalg.norm(X - D @ Z, 'fro')
            if iteration > 0 and abs(prev_error - reconstruction_error) < self.tol:
                break  # Stop if the improvement is below tolerance
            prev_error = reconstruction_error

        self.D_ = D
        self.Z_ = Z
        return D, Z





def ridge_regression_cov(X, y, cov_matrix):
    """
    Perform ridge regression with a penalty given by a covariance matrix.

    Parameters:
    X : ndarray of shape (n_samples, n_features)
        The input data.
    y : ndarray of shape (n_samples,) or (n_samples, n_targets)
        The target values.
    cov_matrix : ndarray of shape (n_features, n_features)
        The covariance matrix used as the penalty.

    Returns:
    beta : ndarray of shape (n_features,) or (n_features, n_targets)
        The estimated coefficients.
    """
    n_features = X.shape[1]
    XtX = X.T @ X
    XtY = X.T @ y

    ridge_penalty = np.linalg.inv(cov_matrix)

    # Solve the ridge regression system
    beta = np.linalg.solve(XtX + ridge_penalty, XtY)
    return beta



def specialOptimizer(X, y, regTerm):

    X = torch.tensor(X).float()
    y = torch.tensor(y).float()

    # Initialize coefficients
    beta = torch.randn(X.shape[1], requires_grad=True)
    lambda_reg = 0.1  # Regularization strength

    # Optimization loop
    optimizer = torch.optim.Adam([beta], lr=0.01)

    for _ in range(1000):
        optimizer.zero_grad()
        
        # Compute loss
        mse_loss = torch.mean((y - X @ beta) ** 2)
        rms_penalty = torch.sqrt(torch.mean(beta ** 2)) * regTerm
        loss = mse_loss + lambda_reg * rms_penalty
        
        # Compute gradients and update
        loss.backward()
        optimizer.step()

    #print("Optimized Coefficients:", beta.detach().numpy())

    return beta.detach().numpy()



def batchCorrelation(Y, Y_pred):

    # Compute mean and standard deviation
    Y_true_mean = np.mean(Y, axis=0)
    Y_pred_mean = np.mean(Y_pred, axis=0)

    Y = Y - Y_true_mean.reshape((1, -1))
    Y_pred = Y_pred - Y_pred_mean.reshape((1, -1))
    
    Y_true_std = np.std(Y, axis=0)
    Y_pred_std = np.std(Y_pred, axis=0)

    Y = Y / Y_true_std.reshape((1, -1))
    Y_pred = Y_pred / Y_pred_std.reshape((1, -1))

    
    # Compute covariance
    covariance = np.mean(Y * Y_pred, axis=0)
    
    return covariance







#import numpy as np
#from scipy.linalg import cho_factor, cho_solve
#from scipy.optimize import minimize

def reml_log_likelihood(params, y, X, K):
    sigma_g2, sigma_e2 = params
    if sigma_g2 <= 0 or sigma_e2 <= 0:
        return np.inf

    n = len(y)
    V = sigma_g2 * K + sigma_e2 * np.eye(n)

    try:
        L, lower = cho_factor(V, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return np.inf

    # Solve V^{-1}y and V^{-1}X
    Vinv_y = cho_solve((L, lower), y, check_finite=False)
    Vinv_X = cho_solve((L, lower), X, check_finite=False)

    XtVinvX = X.T @ Vinv_X
    XtVinvy = X.T @ Vinv_y

    try:
        beta_hat = np.linalg.solve(XtVinvX, XtVinvy)
    except np.linalg.LinAlgError:
        return np.inf

    res = y - X @ beta_hat
    Vinv_res = cho_solve((L, lower), res, check_finite=False)
    res_term = res.T @ Vinv_res

    # log(det(V)) = 2 * sum(log(diag of Cholesky))
    log_detV = 2 * np.sum(np.log(np.diag(L)))
    log_det_XtVinvX = np.log(np.linalg.det(XtVinvX))

    neg_loglik = 0.5 * (log_detV + log_det_XtVinvX + res_term)
    return neg_loglik


def runExampleKin():
    # Example data
    #np.random.seed(42)
    n = 500
    X = np.ones((n, 1))  # intercept only

    def simulate_genotype_matrix(n, m, maf_range=(0.05, 0.5)):
        maf = np.random.uniform(*maf_range, size=m)
        G = np.random.binomial(2, maf, size=(n, m)).astype(float)
        G -= G.mean(axis=0)  # center
        G /= G.std(axis=0) + 1e-6  # scale
        return G

    G = simulate_genotype_matrix(n=n, m=1000)
    K = G @ G.T / G.shape[1]
    K = K / np.trace(K) * n  # normalize trace to n

    # Simulate phenotype
    true_sigma_g2 = 0.6
    true_sigma_e2 = 0.4
    V = true_sigma_g2 * K + true_sigma_e2 * np.eye(n)
    y = np.random.multivariate_normal(np.zeros(n), V)

    # Initial guess
    init_params = [0.5, 0.5]

    # Bounds to ensure positive variance components
    bounds = [(1e-5, 10.0), (1e-5, 10.0)]

    result = minimize(
        reml_log_likelihood,
        init_params,
        args=(y, X, K),
        method='L-BFGS-B',
        bounds=bounds,
        options={"disp": True}
    )

    # Output
    if result.success:
        sigma_g2_est, sigma_e2_est = result.x
        h2_est = sigma_g2_est / (sigma_g2_est + sigma_e2_est)

        print(f"\nEstimated σ_g²: {sigma_g2_est:.4f}")
        print(f"Estimated σ_e²: {sigma_e2_est:.4f}")
        print(f"Estimated h²:    {h2_est:.4f}")
    else:
        print("Optimization failed:", result.message)

    quit()





def checkFourier():


    if False:
        model = torch.load('./data/plant/models/linear_crossVal_reg4_' + str(0) + '_fourier.pt')

        coef = getMultiModelCoef(model, multi=True)

        coef = coef[:5]
        fourier = np.fft.rfft(coef, axis=0)

        plt.plot(np.abs(fourier).T)
        plt.show()

        print (coef.shape)
        quit()


    file_fam = './data/software/data/WEST_original_copy.fam'
    data = np.loadtxt(file_fam, delimiter='\t', dtype=str)
    names_SNP = data[:, 0]


    file_save = './data/plant/SNP/allChr.npz'
    SNPdata = loadnpz(file_save)
    SNPdata = np.sum(SNPdata, axis=2)
    SNPdata = SNPdata.T
    

    #if False:
    #    fourier = np.fft.rfft(SNPdata, axis=0)
    #    np.savez_compressed('./data/temp/fourier.npz', fourier)
    #    print ("SAVED")
    #else:
    #    fourier = loadnpz('./data/temp/fourier.npz')


    Y = loadnpz('./data/plant/syntheticTraits/linear_crossVal_reg4_' + str(0) + '_mod10.npz')
    names_Y = loadnpz('./data/plant/processed/sor/names.npz')

    #fourier = fourier[:, np.isin(names_SNP, names_Y)]
    SNPdata = SNPdata[np.isin(names_SNP, names_Y)]
    names_SNP = names_SNP[np.isin(names_SNP, names_Y)]
    
    Y_mean = np.zeros((names_SNP.shape[0], Y.shape[1]))
    for a in range(names_SNP.shape[0]):
        args1 = np.argwhere(names_Y == names_SNP[a])[:, 0]
        Y_mean[a] = np.mean(Y[args1], axis=0)

    Y_mean = Y_mean - np.mean(Y_mean, axis=0).reshape((1, -1))
    Y_mean = Y_mean / (np.mean(Y_mean**2, axis=0).reshape((1, -1)) ** 0.5)


    SNP_projected = np.matmul(SNPdata.T, Y_mean)

    meanPart = np.mean(SNP_projected, axis=0).reshape((1, -1))
    phase = np.mean(SNP_projected * meanPart, axis = 1)
    phase_binary = np.sign(phase)
    SNP_projected = SNP_projected * phase_binary.reshape((-1, 1))


    SNP_dummy = np.zeros((  SNP_projected.shape[0], 5 ))
    midPoint = SNP_dummy.shape[0] // 2
    SNP_dummy[midPoint, 0] = 1
    SNP_dummy[midPoint:midPoint+5, 1] = 1
    SNP_dummy[midPoint:midPoint+20, 2] = 1
    SNP_dummy[midPoint:midPoint+100, 3] = 1


    fourier_dummy = np.fft.rfft(SNP_dummy, axis=0)
    plt.plot( np.cumsum(np.abs(fourier_dummy[-1::-1]) ** 2, axis=0) )
    plt.show()
    quit()
    

    fourier = np.fft.rfft(SNP_projected, axis=0)

    #print (SNP_projected.shape)
    #print (fourier.shape)


    fourier = fourier / np.mean(np.abs(fourier), axis=0).reshape((1, -1))


    #M = 50
    #M = 100
    M = 30
    fourier_mod = fourier[1:]
    #fourier_mod = np.abs(fourier_mod[:(fourier_mod.shape[0]//M)*M]) ** 2
    #fourier_mod = fourier_mod.reshape((fourier_mod.shape[0] // M, M, fourier_mod.shape[1]))
    #fourier_mod = np.mean(fourier_mod, axis=1)

    fourier_mod = np.cumsum(  np.abs(fourier_mod[-1::-1]) ** 2, axis=0)


    
    for a in range(10):
        plt.plot( np.arange(fourier_mod.shape[0]),  fourier_mod[:, a] )#,  s=4)
    plt.show()
    quit()


    if True:
        corMatrix = np.zeros((fourier.shape[0], Y_mean.shape[1]))
        for a in range(10000):#fourier.shape[0]):
            for b in range(Y_mean.shape[1]):
                cor1 = scipy.stats.pearsonr(np.real(fourier[a]), Y_mean[:, b])[0]
                corMatrix[a, b] = cor1


    sns.heatmap(np.abs(corMatrix[:1000]))
    plt.show()
    quit()





    print (SNPdata[:, 0])

    for a in range(fourier.shape[1]):
        print (a)
        if a % 2 == 0:
            ar1 = np.real(fourier[:, a])
        else:
            ar1 = np.imag(fourier[:, a])


        ar1 = np.abs(ar1)

        M = 100
        ar1 = ar1[:(ar1.shape[0]//M)*M]
        ar1 = ar1.reshape((ar1.shape[0] // M, M))
        ar1 = np.mean(ar1, axis=1)

        plt.plot(ar1)
        plt.show()
    quit()



    np.fft.fft(np.sin(t))
    quit()



#checkFourier()
#quit()




def checkSNP():

    import joblib

    Y = loadnpz('./data/plant/syntheticTraits/linear_crossVal_reg4_' + str(0) + '_mod10.npz')
    names_new = loadnpz('./data/plant/processed/sor/names.npz')





    file_save = './data/plant/SNP/allChr.npz'
    SNPdata = loadnpz(file_save)


    related = np.loadtxt('./data/plant/GWAS/relatedness_matrix.cXX.txt')


    U, S, Vh = np.linalg.svd(related, full_matrices=False)
    S = S / np.mean(S)

    #np.savez_compressed('./data/plant/syntheticTraits/deconv_0.npz', Vh[:,  np.array([50, 10, 500, 1, 20, 100, 0 ]) ] )
    #print ('saved')
    #quit()

    file_fam = './data/software/data/WEST_original_copy.fam'
    data = np.loadtxt(file_fam, delimiter='\t', dtype=str)
    names = data[:, 0]

    #perm3 = np.random.permutation(SNPdata.shape[0])[:100000]
    #perm3 = np.random.permutation(SNPdata.shape[0])[:1000000]
    perm3 = np.random.permutation(SNPdata.shape[0])[:1000]
    SNPdata = SNPdata[perm3]
    SNPdata = np.sum(SNPdata, axis=2).T

    SNPdata = SNPdata - np.mean(SNPdata, axis=0).reshape((1, -1))




    if False:
        from sklearn.decomposition import DictionaryLearning, MiniBatchDictionaryLearning
        dict_learner = DictionaryLearning(
        n_components=20, transform_algorithm='lasso_cd', fit_algorithm='lars', transform_alpha=1e-2,
        random_state=42) #SNPdata.shape[1]
        #dict_learner = MiniBatchDictionaryLearning(n_components=100, alpha=1e-4, batch_size=128, random_state=0, transform_algorithm='lasso_cd')
        

        dict_learner = dict_learner.fit(SNPdata.T)


        joblib.dump(dict_learner, './data/temp/dictionary_model8.pkl')

        quit()


    dict_learner = joblib.load('./data/temp/dictionary_model8.pkl')

    #print (coef_all.shape)
    dict1 = dict_learner.components_
   
    print (dict1.shape)

    dict1 = dict1.T

    dict1 = dict1[:, np.array([10, 5, 1, 0, 19], dtype=int)]


    np.savez_compressed('./data/plant/syntheticTraits/deconv_0.npz', dict1)

    quit()


    SNPdata = SNPdata[np.isin(names, names_new)]
    related = related[np.isin(names, names_new)][:, np.isin(names, names_new)]
    names = names[np.isin(names, names_new)]




    Y_mean = np.zeros((names.shape[0], Y.shape[1]))
    for a in range(names.shape[0]):
        args1 = np.argwhere(names_new == names[a])[:, 0]
        Y_mean[a] = np.mean(Y[args1], axis=0)




    #array1 = np.array([[ 1.00613215, -0.79956175,  0.64392144, -0.42191636,  2.92162782, -0.25167191, 0.71745439, -0.61206298, -0.42237066, -0.57932251]])

    #Y_mean = np.matmul(Y_mean, array1.T)
    np.random.seed(0)
    #random1 = np.random.randn(10, 10000)
    random0 = loadnpz('./data/temp/random1.npz')
    random2 = np.random.randn(50, 10000)
    random1 = np.matmul(random0, random2)
    Y_mean = np.matmul(Y_mean, random1)
    Y_mean = Y_mean - np.mean(Y_mean, axis=0).reshape((1, -1))
    Y_mean = Y_mean / (np.mean(Y_mean**2, axis=0).reshape((1, -1)) ** 0.5)


    Y_transform = np.matmul(Y, random1)
    Y_transform = Y_transform - np.mean(Y_transform, axis=0).reshape((1, -1))
    Y_transform = Y_transform / (np.mean(Y_transform**2, axis=0).reshape((1, -1)) ** 0.5)

    
    SNPdata = SNPdata - np.mean(SNPdata, axis=0).reshape((1, -1))
    randomPhenotypes = np.random.randn(SNPdata.shape[1], 1000)
    SNPdata_random = np.matmul(SNPdata, randomPhenotypes)

    SNPdata = SNPdata[:, :1000]
    SNPdata_random = SNPdata_random / (np.mean( (SNPdata_random ** 2), axis=0).reshape((1, -1)) ** 0.5)
    SNPdata = SNPdata / (np.mean( (SNPdata ** 2), axis=0).reshape((1, -1)) ** 0.5)


    U, S, Vh = np.linalg.svd(related, full_matrices=False)
    S = S / np.mean(S)



    Vh = Vh[:100]
    S = S[:100]
    

    
    projection = np.matmul(Vh, SNPdata)
    S_proj = np.mean(projection ** 2, axis=1)
    
    
    Y_projection = np.matmul(Vh, Y_mean)
    #Y_projection = np.matmul(Vh, SNPdata_random)
    #Y_projection = np.matmul(Vh, SNPdata)


    #rescaled =  np.matmul( np.diag(1.0 / (S_root + 1e-3)  )  , projection) #/ projection.shape[0]
    
    #Y_p_full = np.zeros(Y_projection.shape[1]) - 10000

    #PastedPVals = np.zeros(( 10,  S.shape[0], Y_mean.shape[1]  ))

    PastedPVals =  np.zeros((S.shape[0], Y_projection.shape[1]  )) + 1e5
    #PastedPVals =  np.zeros((S.shape[0], Y_projection.shape[1]  )) + 10

    for a in range(10):

        propGenetic = (a + 1) * 0.1
        propResidual = (10 - a) * 0.1

        S_mod = (S * propGenetic) + propResidual
        #S_mod = S + 1
        S_root = S_mod ** 0.5

        Y_rescaled = np.matmul( np.diag(1.0 / (S_root + 1e-3)  )  , Y_projection)
        
        #Y_p = np.max(np.abs(Y_rescaled), axis=0)
        Y_p = np.abs(Y_rescaled)
        Y_p = 2 * scipy.stats.norm.cdf( -1 * Y_p)

        #print (Y_rescaled.shape)
        #quit()



        # Compute log(1 - p_min) safely
        Y_p = np.log1p(-Y_p)
        # Then compute log( (1 - p_min)^n ) = n * log(1 - p_min)
        Y_p = Y_rescaled.shape[0] * Y_p
        # Now compute p_overall = 1 - exp(log_prob), in a numerically stable way
        Y_p = -np.expm1(Y_p)


        #print (Y_p)
        #quit()


        Y_p = 1.0 / Y_p

        PastedPVals[Y_p - PastedPVals < 0] = Y_p[Y_p - PastedPVals < 0]
        #PastedPVals[a] = Y_p

        #Y_p_full = np.max( np.array([  Y_p, Y_p_full ]) , axis=0 )

        #print (Y_p)
    

    #print (PastedPVals.shape)

    #PastedPVals = np.min(PastedPVals, axis=0)


    #sns.heatmap(Y_projection)
    #plt.show()

    #PastedPVals[PastedPVals < 2] = 0

    sns.heatmap(PastedPVals)
    plt.show()

    phenotypeScore = np.max(PastedPVals, axis=0)
    #argBest = np.argmax(phenotypeScore)

    argBest = np.argsort(  -1 * phenotypeScore )[:5]


    #np.savez_compressed('./data/temp/random1.npz', random1[:, np.argsort(  -1 * phenotypeScore )[:50]  ])
    #quit()


    sns.heatmap(PastedPVals[:, np.argsort(  -1 * phenotypeScore )[:20] ])
    plt.show()


    Y_transform = Y_transform[:, argBest]

    print (Y_transform.shape)

    #np.savez_compressed('./data/plant/syntheticTraits/deconv_0.npz', Y_transform)

    quit()


    max_S = np.max(PastedPVals, axis=1)


    argTop = np.argwhere(max_S > 2)[:, 0]
    argTop = argTop[np.argsort(max_S[argTop])]


    

    Y_projGood = np.matmul( U[argTop], Y_mean  )

    print (Y_projGood)
    quit()

    

    Y_transform = np.matmul(  Y , Y_projGood.T )

    print (Y_transform.shape)


    np.savez_compressed('./data/plant/syntheticTraits/deconv_0.npz', Y_transform)

    
    quit()

    plt.plot(max_S)
    plt.show()

    sns.heatmap(PastedPVals)
    plt.show()

    plt.hist(PastedPVals.reshape((-1,)), bins=100)
    plt.show()

    


    #plt.hist(Y_p_full, bins=100)
    #plt.show()

    #plt.scatter( np.arange(Y_p_full.shape[0])+1, Y_p_full )
    #plt.show()
    quit()


    #YS_proj = np.mean(Y_projection ** 2, axis=1)

    for a in range(Y_projection.shape[1]):
        plt.scatter(S_root, Y_projection[:, a])
        plt.show()


    

    plt.hist(Y_p, bins=100)
    plt.show()
    quit()


    

    #std1 = np.mean(np.abs(rescaled)) * ((np.pi / 2) ** 0.5)
    #rescaled = rescaled / std1

    #plt.plot( np.mean(projection**2, axis=1) )
    #plt.show()

    #plt.plot( np.mean(rescaled**2, axis=1) )
    #plt.show()

    line1 = (np.arange(1000) / 100) - 5

    print ("STD", np.mean(rescaled ** 2))

    plt.hist( rescaled.reshape((-1,)), bins=100 )
    plt.plot(line1 , 36000 * np.exp( -1 * (line1 ** 2) / 2  ))
    plt.show()

    #sns.heatmap(rescaled)
    #plt.show()


    pValue = np.max(np.abs(rescaled), axis=0)
    pValue[pValue > np.log(1e8)] = np.log(1e8)
    #pValue = np.exp(pValue) / rescaled.shape[0]

    pValue = scipy.stats.norm.cdf( -1 *  pValue) * rescaled.shape[0]


    
    pValue = np.log10(pValue)
    #pValue = 1.0/pValue



    print (np.min(pValue))

    print (np.argwhere(pValue > 1000 * 20).shape)


    plt.hist(pValue, bins=100)
    plt.show()




    
    quit()


    

    SNPdata = SNPdata - np.mean(SNPdata, axis=0).reshape((1, -1))
    randomPhenotypes = np.random.randn(SNPdata.shape[1], 1000)
    SNPdata_random = np.matmul(SNPdata, randomPhenotypes)

    SNPdata = SNPdata[:, :1000]
    SNPdata_random = SNPdata_random / (np.mean( (SNPdata_random ** 2), axis=0).reshape((1, -1)) ** 0.5)
    SNPdata = SNPdata / (np.mean( (SNPdata ** 2), axis=0).reshape((1, -1)) ** 0.5)

    dict_learner = joblib.load('./data/temp/dictionary_model6.pkl')

    #print (coef_all.shape)
    dict1 = dict_learner.components_
    Z = dict_learner.transform(SNPdata.T)
    Z_random = dict_learner.transform(SNPdata_random.T)
    Z_Y = dict_learner.transform(Y_mean.T)

    Z = Z / (np.mean(Z**2, axis=0)**0.5).reshape((1, -1))
    Z_random = Z_random / (np.mean(Z_random**2, axis=0)**0.5).reshape((1, -1))
    Z_Y = Z_Y / (np.mean(Z_Y**2, axis=0)**0.5).reshape((1, -1))

    print (Z.shape)

    


    plt.plot(np.max(np.abs(Z), axis=1))
    plt.plot(np.max(np.abs(Z_random), axis=1))
    plt.plot(np.max(np.abs(Z_Y), axis=1))
    plt.show()


    quit()


    





    sns.clustermap(rescaled, row_cluster=False)
    plt.show()



#checkSNP()
#quit()







def checkRNA():


    #TODO !!!! Try kinship in addition to PCA


    doRelated = True


    related = np.loadtxt('./data/plant/GWAS/relatedness_matrix.cXX.txt')

    #print (related.shape)
    #quit()


    file_fam = './data/software/data/WEST_original_copy.fam'
    data = np.loadtxt(file_fam, delimiter='\t', dtype=str)
    names = data[:, 0]
    

    
    file_save = './data/plant/SNP/allChr.npz'
    SNPdata = loadnpz(file_save)

    perm3 = np.random.permutation(SNPdata.shape[0])[:2000]
    SNPdata = SNPdata[perm3]
    SNPdata = np.sum(SNPdata, axis=2).T

    GWASPCA = np.loadtxt('./data/software/data/WEST_original_PCA' + str(5) + '.txt')
    GWASPCA = GWASPCA[:, 1]



    RNA_leaf = SNPdata




    if False:
        for a in range(10):
            Y0 = loadnpz('./data/plant/syntheticTraits/linear_crossVal_reg4_' + str(a) + '_mod10.npz')
            if a == 0:
                Y = np.zeros((Y0.shape[0], 10))
            Y[:, a] = Y0[:, 1]
        
        for a in range(1, 10):
            cor1 = scipy.stats.pearsonr(Y[:, a], Y[:, 0])[0]
            if cor1 < 0:
                Y[:,a] = Y[:,a] * -1
        names_new = loadnpz('./data/plant/processed/sor/names.npz')


    if False:
        Y = loadnpz('./data/plant/syntheticTraits/linear_crossVal_reg4_' + str(0) + '_mod10.npz')
        names_new = loadnpz('./data/plant/processed/sor/names.npz')
    


    


    if True:
        simulationName = 'sameHeritSep100'
        #simulationName = 'seperate100SNP'
        folder0 = './data/plant/simulations/encoded/' + simulationName + '/'
        #simIndex = 0
        #simIndex = 1
        simIndex = 2
        #simIndex = 3 #Interesting
        folder1 = folder0 + str(simIndex)


        dataFolder =  './data/plant/simulations/simPhen/' + simulationName + '/' + str(simIndex) 
        dataFile = dataFolder + '/'
        files1 = os.listdir(dataFolder)
        for file1 in files1:
            if '.txt' not in dataFile:
                if 'Simulated_Data_' in file1:
                    dataFile = dataFile + file1
        X_original = np.loadtxt(dataFile, dtype=str)
        X_original = X_original[1:, 1:-1].astype(float) 


        Y = loadnpz(folder1 + '/H2Opt_predValues.npz')
        names_new = loadnpz(folder1 + '/names.npz')
        envirement = np.zeros((Y.shape[0], 0))


        trainSplitFile = folder1 + '/trainSplit.npz'
        trainTest = loadnpz(trainSplitFile)

    Y = Y[:, :3]
    #Y = Y[:, :5]

    Y_tensor = torch.tensor(Y).float()

    #argNow = np.argwhere(trainTest == 1)[:, 0]

    #N_tensor = Y_tensor.shape[1]
    #cor1, _, _ = coherit(Y_tensor[argNow][:, np.arange(N_tensor*N_tensor) // N_tensor], Y_tensor[argNow][:, np.arange(N_tensor*N_tensor) % N_tensor], names_new[argNow], envirement[argNow], geneticCor=False)
    #cor1 = cor1.reshape((N_tensor, N_tensor))
    #cor1 = cor1.data.numpy()
    #U, S, Vh = np.linalg.svd(cor1, full_matrices=False)

    #print (cor1)
    #quit()

    


    #Y_transform = np.matmul(Y, U)


    


    #Y = Y * (np.array( [0.48392549,  0.38444868,  0.45406613,  0.08554688,  0.03440499]).reshape((1, -1)) / .48392)

    
    

    if False:

        names_SNP = np.copy(names)

        RNA_gp = np.loadtxt('./data/plant/RNA/exp_gp.txt', dtype=str)
        RNA_leaf = np.loadtxt('./data/plant/RNA/exp_leaf.txt', dtype=str) 

        geneNames_gp = RNA_gp[0]
        geneNames_leaf = RNA_leaf[0]
        intersect1 = np.intersect1d(geneNames_gp, geneNames_leaf)
        args_gp = np.argwhere(np.isin(geneNames_gp , intersect1))[:, 0]
        args_leaf = np.argwhere(np.isin(geneNames_leaf , intersect1))[:, 0]
        RNA_gp = RNA_gp[:, args_gp]
        RNA_leaf = RNA_leaf[:, args_leaf]

        for a in range(RNA_gp.shape[1]):
            assert RNA_gp[0, a] == RNA_leaf[0, a]

        names = RNA_gp[1:, 0]

        RNA_gp = RNA_gp[1:, 1:].astype(float)
        RNA_leaf = RNA_leaf[1:, 1:].astype(float)

        


        if doRelated:
            RNA_leaf = RNA_leaf[np.isin(names, names_SNP)]
            names = names[np.isin(names, names_SNP)]


            subsetOfNames = np.zeros(names.shape[0], dtype=int)
            for a in range(names.shape[0]):
                arg1 = np.argwhere(names_SNP == names[a])[0, 0]
                subsetOfNames[a] = arg1 

            SNPdata = SNPdata[subsetOfNames]
            related = related[subsetOfNames][:, subsetOfNames]


        U, S, Vh = np.linalg.svd(related, full_matrices=False)
        U_related, S, Vh_related = np.linalg.svd(related, full_matrices=False)


        
        #projector = np.matmul(U[:, :5], Vh[:5, :])
        #projector = np.matmul(U[:, :10], Vh[:10, :])
        #projector = np.matmul(U[:, :20], Vh[:20, :])
        #projector = np.matmul(U[:, :40], Vh[:40, :])

        #S_mod = np.copy(S)
        #S_mod[:10] = S_mod[10]
        #related_mod = np.matmul(U, np.matmul(  np.diag(S_mod) , Vh))


        #SNPdata = SNPdata[:, np.random.permutation(SNPdata.shape[1])]
        #RNA_leaf = RNA_leaf[:, np.random.permutation(RNA_leaf.shape[1])]

        #print (np.mean(RNA_leaf**2))

        #RNA_leaf = np.matmul(projector, RNA_leaf)
        #RNA_leaf = np.matmul(projector, SNPdata)
        #RNA_leaf = SNPdata
        #RNA_leaf = projector
        #RNA_leaf = related_mod
        #RNA_leaf = U[:, :20]

        #print (np.mean(RNA_leaf**2))
        #quit()



        #SNPdata_proj = np.matmul(U, SNPdata)
        #RNA_leaf_proj = np.matmul(U, RNA_leaf)


        #sns.clustermap(  SNPdata_proj , row_cluster=False)
        #plt.title("SNP")
        #plt.show()

        #sns.clustermap(  RNA_leaf_proj , row_cluster=False)
        #plt.title("RNA")
        #plt.show()
        #quit()

        if False:

            rand1 = np.random.random(size=SNPdata.shape[0]) * 0.001
            rand2 = np.random.random(size=SNPdata.shape[0]) * 0.001

            pValList = np.zeros((2000, 100))
            for a in range(pValList.shape[0]):
                for b in range(pValList.shape[1]):
                    
                    #pValList[a, b] = scipy.stats.pearsonr(SNPdata[:, a] + rand1,  RNA_leaf[:, b] + rand2 )[1]
                    pValList[a, b] = scipy.stats.pearsonr(SNPdata[:, a] + rand1,  U[:, b] + rand2 )[1]
                    #pValList[a, b] = scipy.stats.pearsonr(RNA_leaf[:, a] + rand1,  U[:, b] + rand2 )[1]
            pValList = np.array(pValList) + 1e-50
            #print (np.min(pValList), np.max(pValList))
            pValList = np.log10(pValList)

        
        

        


        #print (SNPdata.shape)
        #print (RNA_leaf.shape)
        #quit()

    


    #RNA_leaf = (np.random.random(size=RNA_leaf.shape) * 0.01) + np.mean(RNA_leaf, axis=1).reshape((-1, 1)) #TODO REMOVE!!!!
 
    

    #RNA_gp = RNA_gp[np.isin(names, names_new)]
    RNA_leaf = RNA_leaf[np.isin(names, names_new)]
    #GWASPCA = GWASPCA[np.isin(names, names_new)]

    if doRelated:
        related = related[np.isin(names, names_new)]
        related = related[:, np.isin(names, names_new)]
    
    

    names = names[np.isin(names, names_new)]
    
    


    
    Y_mean = np.zeros((names.shape[0], Y.shape[1]))
    #PCA1 =  np.zeros(names.shape[0])
    #trainTest2 = np.zeros(names.shape[0])
    for a in range(names.shape[0]):
        args1 = np.argwhere(names_new == names[a])[:, 0]
        Y_mean[a] = np.mean(Y[args1], axis=0)
        #trainTest2[a] = trainTest[args1[0]]

    try:
        X_original_mean = np.zeros((names.shape[0], X_original.shape[1]))
        for a in range(names.shape[0]):
            args1 = np.argwhere(names_new == names[a])[:, 0]
            X_original_mean[a] = np.mean(X_original[args1], axis=0)

        

    except:
        True
        

    #Y_mean = Y_mean - np.mean(Y_mean, axis=0).reshape((1, -1))
    #meanModifier = np.mean(Y_mean**2, axis=0).reshape((1, -1)) ** 0.5
    #Y_mean = Y_mean / meanModifier

    #Y = Y / meanModifier



    if False:
        related_U, related_S, related_UT = np.linalg.svd(related, full_matrices=False)
        CompCut = np.argwhere(related_S > 0.5).shape[0]
        remap_proj = np.matmul( related_U[:CompCut].T, related_U[:CompCut] )
        Y_mean_proj = np.matmul(remap_proj, Y_mean)
        #Y_mean = Y_mean - Y_mean_proj


    RNA_leaf = RNA_leaf - np.mean(RNA_leaf, axis=0).reshape((1, -1))
    RNA_leaf = RNA_leaf / np.mean(RNA_leaf**2, axis=0).reshape((1, -1)) ** 0.5


    
    #perm2 = np.random.permutation(RNA_leaf.shape[1])[:1000]
    #perm2 = np.random.permutation(RNA_leaf.shape[1])[:100]
    #RNA_leaf = RNA_leaf[:, perm2]
    
    #RNA_leaf_proj = np.matmul(Y_mean.T, RNA_leaf).T



    if True:

        import statsmodels.api as sm

        #RNA_leaf = np.array([RNA_leaf[np.random.permutation(RNA_leaf.shape[0]), i] for i in range(RNA_leaf.shape[1])]).T

        coef_all = np.zeros((  Y_mean.shape[1], RNA_leaf.shape[1] ))
        pValList = np.zeros(RNA_leaf.shape[1])
        Y_mean_con = sm.add_constant(Y_mean)  # Add intercept
        
        for a in range(RNA_leaf.shape[1]):
            model = sm.OLS(RNA_leaf[:, a], Y_mean_con).fit()
            pValList[a] = model.f_pvalue
            coefficients = model.params[1:] #Remove intercept with [1:]
            coef_all[:, a] = coefficients



        coef_all = coef_all * np.log10(pValList).reshape((1, -1))

        #plt.hist(np.log10(pValList), bins=100)
        #plt.show()



        #coef_all = coef_all[:, np.log10(pValList) < -10]
        #coef_all = coef_all[:, np.log10(pValList) < 0.05]

        #coef_all = coef_all[:,  np.argsort(  -1*np.sum(coef_all**2, axis=1) )[:3] ]


        #coef_all0 = np.copy(coef_all)


        mean1 = np.mean(coef_all, axis=1).reshape((-1, 1))

        #coef_all = coef_all + (20 * mean1)

        print (coef_all.shape)


        '''
        
        Y_mod = np.matmul(Y_mean, mean1)

        pValList2 = np.zeros(RNA_leaf.shape[1])
        corList = np.zeros(RNA_leaf.shape[1])
        for a in range(RNA_leaf.shape[1]):
            corBoth = scipy.stats.pearsonr(Y_mod[:, 0], RNA_leaf[:, a])
            corList[a] = corBoth[0]
            pValList2[a] = corBoth[1]


        #argStat = np.argwhere(pValList2 < 0.001)[:, 0]
        argStat = np.argwhere(np.logical_and(pValList2 < 0.001,  corList > 0 ))[:, 0]
        coef_all = np.zeros((  Y_mean.shape[1], argStat.shape[0] ))
        pValList3 = np.zeros(argStat.shape[0])
        for a in range(coef_all.shape[1]):
            #model = sm.OLS(RNA_leaf[:, argStat[a]] * corList[argStat[a]], Y_mean_con).fit()
            model = sm.OLS(RNA_leaf[:, argStat[a]], Y_mean_con).fit()
            pValList3[a] = model.f_pvalue
            coefficients = model.params[1:] #Remove intercept with [1:]
            coef_all[:, a] = coefficients

        #plt.hist(np.log10(pValList3), bins=100)
        #plt.show()

        #coef_all = coef_all[:, pValList3 < 0.01]

        #print (np.mean(coef_all, axis=1) )

        mean1 = np.mean(coef_all, axis=1).reshape((-1, 1))

        #print (mean1[:, 0])
        #quit()

        #'''


    U, S, Vh = np.linalg.svd(coef_all, full_matrices=False)


    #U, S, Vh = np.linalg.svd(related, full_matrices=False)


    #Y_mean_proj  = np.matmul( np.diag(S), np.matmul( Vh, Y_mean))
    #Y_mean_proj = Y_mean

    #U, S, Vh = np.linalg.svd(Y_mean[:], full_matrices=False)

    print (S)
    print (Vh.T)

    #Y_transform = np.matmul(Y, Vh.T)
    Y_transform = np.matmul(Y, U)


    #np.savez_compressed('./data/plant/syntheticTraits/deconv_0.npz', Y_transform)
    #quit()



    corMatrix_before = np.zeros((X_original.shape[1], Y.shape[1]))
    for a in range(X_original.shape[1]):
        for b in range(Y.shape[1]):
            corMatrix_before[a, b] = abs(scipy.stats.pearsonr(X_original[:, a], Y[:, b])[0])

    for a in range(5):
        print ('')

    print ("HELLO!")

    corMatrix_after = np.zeros((X_original.shape[1], Y_transform.shape[1]))
    for a in range(X_original.shape[1]):
        for b in range(Y_transform.shape[1]):
            corMatrix_after[a, b] = abs(scipy.stats.pearsonr(X_original[:, a], Y_transform[:, b])[0])

    



    for a in range(2):
        corMatrix = [corMatrix_before, corMatrix_after][a]

        plt.imshow(corMatrix)
        for i in range(corMatrix.shape[0]):
            for j in range(corMatrix.shape[1]):
                text = plt.text(j, i, f'{corMatrix[i, j]:.2f}',
                            ha="center", va="center", color="white")
        if a == 0:
            plt.title('correlations pre-deconvolution')
        else:
            plt.title('correlations post-deconvolution')
        plt.xlabel('synthetic trait')
        plt.ylabel('ground-truth trait')
        plt.show()


    
    quit()



    #U_related, S, Vh_related = np.linalg.svd(related, full_matrices=False)


    #RNA_related = np.matmul(RNA_leaf, RNA_leaf.T)
    #U_RR, S_RR, Vh_RR = np.linalg.svd(RNA_related, full_matrices=False)


    #Y_mean_proj = np.matmul(RNA_leaf.T, Y_mean)
    #Y_mean_proj = np.matmul(Vh_RR, Y_mean)
    #Y_mean_proj = np.matmul(U_RR[:20], Y_mean)

    Y_mean_proj = Y_mean

    Y_mean_proj = Y_mean_proj / (np.sum(Y_mean_proj ** 2, axis=0).reshape((1, -1)) ** 0.5)

    U, S, Vh = np.linalg.svd(Y_mean_proj, full_matrices=False)
    U = Vh.T

    print (U)
    print (S)
    quit()

    #Y_mean_proj = np.matmul(Y_mean_proj, U)


    X_original_mean_proj = np.matmul(RNA_leaf.T, X_original_mean)

    X_original_mean_proj = X_original_mean_proj / (np.sum(X_original_mean_proj ** 2, axis=0).reshape((1, -1)) ** 0.5)

    
    print (np.matmul( Y_mean_proj.T, Y_mean_proj ))
    print ('')
    print (np.matmul( X_original_mean_proj.T, X_original_mean_proj ))


    #sns.clustermap(Y_mean_proj, row_cluster=True)
    #plt.show()


    #sns.clustermap(X_original_mean_proj, row_cluster=True)
    #plt.show()
    #quit()

    


    #plt.scatter(coef_all[0], coef_all[1])
    #plt.show()
    #plt.scatter(coef_all[0], coef_all[2])
    #plt.show()
    #plt.scatter(coef_all[1], coef_all[2])
    #plt.show()


    print (U)
    #quit()

    

    
    Y_transform = np.matmul(Y, U)
    #Y_transform = np.matmul(Y, mean1)
    #Y_transform = np.matmul(Y, coef_all)

    


    





    #print (scipy.stats.pearsonr( Y_mean_proj[:, 0], Y_mean_proj[:, 1]  ))

    #sns.heatmap(Y_mean_proj2)
    #plt.show()
    #quit()



    #GWASPCA = GWASPCA - np.mean(GWASPCA)
    #GWASPCA = GWASPCA / (np.mean(GWASPCA**2) ** 0.5)
    #Y_mean[:, 0] = Y_mean[:, 0] - (GWASPCA * np.mean(GWASPCA * Y_mean[:, 0]))
    #Y_mean[:, 1] = Y_mean[:, 1] - (GWASPCA * np.mean(GWASPCA * Y_mean[:, 1]))
    #Y_mean[:, 2] = Y_mean[:, 2] - (GWASPCA * np.mean(GWASPCA * Y_mean[:, 2]))



    




    related_U, related_S, related_UT = np.linalg.svd(related, full_matrices=False)
    

    #print (scipy.stats.pearsonr( Y_mean[:, 0], Y_mean[:, 1]  ))

    #print (related_S)
    Y_mean_proj = np.matmul(related_UT, Y_mean)
    #Y_mean_proj = np.matmul( np.diag( (1 / related_S)  + 0.5 ), Y_mean_proj )
    

    U, S, Vh = np.linalg.svd(Y_mean_proj.T, full_matrices=False)
    


    


    Y_transform = np.matmul(Y, U.T)


    




    from sklearn.decomposition import DictionaryLearning
    dict_learner = DictionaryLearning(
    n_components=Y.shape[1], transform_algorithm='omp', transform_alpha=1e-5,
    random_state=42)
    dict_learner = dict_learner.fit(coef_all.T)

    #print (coef_all.shape)
    dict1 = dict_learner.components_
    Z = dict_learner.transform(coef_all.T)

    scale1 = np.mean(np.abs(Z), axis=0)
    print (scale1)
    print (dict1)
    dict1 = dict1[np.argsort(scale1 * -1)] * 50


    if False:
        bounds = [(1e-5, 10.0), (1e-5, 10.0)]
        init_params = [0.5, 0.5]

        X = np.ones((RNA_leaf.shape[0], 1))

        result = minimize(
            reml_log_likelihood,
            init_params,
            args=(Y_mean[:, 1], X, related),
            method='L-BFGS-B',
            bounds=bounds,
            options={"disp": True}
        )

        # Output
        if result.success:
            sigma_g2_est, sigma_e2_est = result.x
            h2_est = sigma_g2_est / (sigma_g2_est + sigma_e2_est)

            print(f"\nEstimated σ_g²: {sigma_g2_est:.4f}")
            print(f"Estimated σ_e²: {sigma_e2_est:.4f}")
            print(f"Estimated h²:    {h2_est:.4f}")
        else:
            print("Optimization failed:", result.message)
        quit()

        for a in range(Y_mean.shape[1]):

            # Simulate data
            X = sm.add_constant(np.ones(Y_mean.shape[0]))  # Intercept only


            print (Y_mean.shape, X.shape, related.shape)
            
            gls_model = sm.GLS(Y_mean[:, a], X, sigma=related)
            gls_results = gls_model.fit()
            print ('')
            print ('')
            print (a)
            print ('')
            print ('')

            print (gls_results)
        quit()

    
    

    sns.heatmap(X_transformed)
    plt.show()
    quit()


    quit()







    if False:
        #alphas = [1e1, 1e2, 5e2, 1e3, 2e3, 1e4, 1e5, 1e6, 1e7]
        alphas = ['1e0', '1e1', '1e2', '5e2', '1e3', '2e3', '1e4', '1e5', '1e6', '1e7']

        for alpha_str in alphas:
            alpha = float(alpha_str)

            testSize = 20
            testTrue = np.zeros((testSize, RNA_leaf.shape[1] ))
            testPred = np.zeros((testSize, RNA_leaf.shape[1] ))

            for a in range(RNA_leaf.shape[1]):
                train1 = np.random.permutation(RNA_leaf.shape[0])
                train1, test1 = train1[:-testSize], train1[-testSize:]
                #clf = linear_model.Ridge(alpha=1e3)
                clf = linear_model.Ridge(alpha=alpha)
                clf.fit(Y_mean[train1], RNA_leaf[train1, a])
                pred = clf.predict(Y_mean)
                testTrue[:, a] = RNA_leaf[test1, a]
                testPred[:, a] = pred[test1]

            #testTrue = testTrue - np.mean(testTrue, axis=0).reshape((1, -1))
            #testPred = testPred - np.mean(testPred, axis=0).reshape((1, -1))

            print (alpha_str)
            print (np.mean(np.abs(testTrue - testPred)))
            print (scipy.stats.pearsonr(testTrue.reshape((-1,)),  testPred.reshape((-1,)) ))
        quit()



    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(Y_mean, RNA_leaf)  # Fits all Y variables simultaneously
    pred1 = model.predict(Y_mean)
    coef_all = model.coef_


    corList = batchCorrelation(RNA_leaf, pred1)

    RNA_leaf_random = np.array([RNA_leaf[np.random.permutation(RNA_leaf.shape[0]), i] for i in range(RNA_leaf.shape[1])]).T

    model_random = LinearRegression()
    model_random.fit(Y_mean, RNA_leaf_random)  # Fits all Y variables simultaneously
    pred_random = model_random.predict(Y_mean)
    #coef = model.coef_

    corList_random = batchCorrelation(RNA_leaf_random, pred_random)

    corList = np.abs(corList)
    corList_random = np.abs(corList_random)

    cutOff = np.sort(corList_random)[-corList_random.shape[0] // 20]
    #plt.hist(corList, bins=100)
    #plt.hist(corList_random, bins=100)
    #plt.show()
    #quit()

    #print (coef_all.shape)
    #print (corList.shape)

    coef_all = coef_all[corList > cutOff].T


    if False:
        coef_all0 = np.zeros((  Y_mean.shape[1], RNA_leaf.shape[1] ))
        for a in range(RNA_leaf.shape[1]):
            clf = linear_model.Ridge(alpha=1e1)
            clf.fit(Y_mean, RNA_leaf[:, a])
            coef = clf.coef_
            coef_all0[:, a] = coef

        coef_all = np.copy(coef_all0)
        for loops1 in range(10):
            cov_matrix = np.cov(coef_all.T, rowvar=False)
            print (cov_matrix)
            cov_matrix = cov_matrix * 1e-4 / np.mean(np.abs(cov_matrix))
            coef_all = np.zeros((  Y_mean.shape[1], RNA_leaf.shape[1] ))
            for a in range(RNA_leaf.shape[1]):
                beta = ridge_regression_cov(Y_mean, RNA_leaf[:, a], cov_matrix)
                coef_all[:, a] = beta



    import cvxpy as cp

    #regTerm = 1e-1
    #regTerm = 1e0
    #lambda_reg = 1e-5
    #lambda_reg = 1e-2
    #lambda_reg = 1e-1

    lambda_reg = 1e-5
    #regTerm = 1e
    coef_all = np.zeros((  Y_mean.shape[1], RNA_leaf.shape[1] ))

    cor_list = np.zeros(RNA_leaf.shape[1])
    for a in range(RNA_leaf.shape[1]):
        print (a, RNA_leaf.shape[1])

        # Define variables
        beta = cp.Variable(Y_mean.shape[1])
        #lambda_reg = 0.1  # Regularization strength

        # Define objective
        mse_loss = cp.sum_squares(RNA_leaf[:, a] - Y_mean @ beta) / len(RNA_leaf[:, a])
        #mse_loss = cp.norm(RNA_leaf[:, a] - Y_mean @ beta) / np.sqrt(len(RNA_leaf[:, a]))
        #rms_penalty = cp.norm(beta, 2) / np.sqrt(beta.shape[0])  # Root mean square
        rms_penalty = cp.sum_squares(beta) / np.sqrt(beta.shape[0])  # Root mean square
        objective = cp.Minimize(mse_loss + lambda_reg * rms_penalty)

        # Solve
        problem = cp.Problem(objective)
        problem.solve()

        coef = beta.value


        pred1 = Y_mean @ coef
        #print (pred1.shape)
        #print (RNA_leaf[:, a].shape)
        #quit()
        cor1 = scipy.stats.pearsonr(pred1, RNA_leaf[:, a])[0]

        cor_list[a] = cor1
    
        coef_all[:, a] = coef


    #U, S, Vh = np.linalg.svd(coef_all, full_matrices=False)
    #sns.clustermap(  np.matmul(U.T,coef_all))
    #plt.show()
    sum1 =  np.sum(coef_all ** 2, axis=0)    
    

    #spreader1 = np.arange(RNA_leaf.shape[0]).repeat(RNA_leaf.shape[1])
    #groups = np.arange(RNA_leaf.shape[1]).repeat(RNA_leaf.shape[0])
    #groups = groups.reshape((RNA_leaf.shape[1], RNA_leaf.shape[0])).T.reshape((-1,))



    
    
    

    
    


    quit()




    import bambi as bmb

    
    synthNames = []
    for a in range(Y.shape[1]):
        name1 = 'synth_' + str(a+1)
        synthNames.append(name1)


    df = {}
    spreader1 = np.arange(RNA_leaf.shape[0]).repeat(RNA_leaf.shape[1])
    groups = np.arange(RNA_leaf.shape[1]).repeat(RNA_leaf.shape[0])
    groups = groups.reshape((RNA_leaf.shape[1], RNA_leaf.shape[0])).T.reshape((-1,))
    #spreader1 = spreader1.reshape(RNA_leaf.shape)
    for a in range(len(synthNames)):
        df[synthNames[a]] = Y_mean[spreader1, a]
    
    df['RNA'] = RNA_leaf.reshape((-1,))
    df['groups'] = groups

    # Load Data (assuming df is already created)
    df['groups'] = df['groups'].astype(str)  # Ensure groups are categorical

    df = pd.DataFrame(df)

    

    #print (bmb.inference_methods.names)
    #quit()

    # Define the model in Bambi
    #model = bmb.Model("RNA ~ 1 + (synth_1 + synth_2 + synth_3|groups)", data=df)
    model = bmb.Model("RNA ~ 1 + (synth_1 + synth_2 + synth_3|groups)", data=df, noncentered=True)

    
    #model = bmb.Model("RNA ~ 1 + (synth_1|groups) + (synth_2|groups) + (synth_3|groups)", data=df)

    # Fit the model using MCMC (Markov Chain Monte Carlo)
    #results = model.fit(draws=2000, chains=4, target_accept=0.95)
    results = model.fit(inference_method="vi", draws=100, chains=4)#, target_accept=0.95)
    #results = model.fit(inference_method="blackjax_nuts", draws=100, chains=4)#, target_accept=0.95)
    
    #results = model.fit("vi", draws=1000)


    print(results.params_dict.keys())

    print(results.params_dict["mu"].eval().shape)
    quit()

    print (dir(results))
    quit()

    #print (dir(results.model))

    print (results.cov.eval().shape)
    quit()

    print (dir(results.model))
    quit()
    print (dir(results.model['synth_1|groups']))
    print (type(results.model['synth_1|groups']))

    print (results.model['synth_1|groups'][:])
    
    print (np.array( results.model['synth_1|groups']  )[:])
    quit()


    print (dir(results.params[0]))
    print (dir(results.params[1]))

    print(results.params)  # Check available model parameters
    print(results.params_dict)  # Check available model parameters

    #print (results.params_dict.mu)
    #print (results.params_dict.rho)

    quit()

    # Print summary of the results
    #print(type(results.mean))
    #print (results.mean['mu'])

    print(results.model)


    print (results.mean.eval().shape)

    vi_means = {key: value.eval() for key, value in results.mean.items()}

    print (vi_means)
    
    quit()

    # Extract posterior estimates for the random slopes
    posterior = results.posterior  # Bayesian posterior samples

    #print (posterior)
    #print (posterior.keys())
    #quit()

    for a in range(5):
        print ('')
    #print (results.posterior.keys())
    synth1 = np.array(results.posterior['synth_1|groups'])
    synth2 = np.array(results.posterior['synth_2|groups'])
    synth3 = np.array(results.posterior['synth_3|groups'])

    synth1_mean = np.mean(synth1, axis=(0, 1))
    synth2_mean = np.mean(synth2, axis=(0, 1))
    synth3_mean = np.mean(synth3, axis=(0, 1))

    coef_all = np.array([synth1_mean, synth2_mean, synth3_mean])


    #sns.clustermap(coef_all, row_cluster=False)
    #plt.show()

    #cov_matrix = np.cov(synthAll, rowvar=False)

    U, S, Vh = np.linalg.svd(coef_all, full_matrices=False)


    

    quit()
    
    quit()

    re_formula = '~' + ' + '.join(synthNames)
    

    md = smf.mixedlm("RNA ~ 1", df, groups=df["groups"], re_formula=re_formula)

    mdf = md.fit(method='powell')

    random_effects = mdf.random_effects
    fixed_effects = mdf.fe_params 

    #print (fixed_effects)

    array = np.zeros((RNA_leaf.shape[1], Y_mean.shape[1]))
    for a in range(RNA_leaf.shape[1]):
        for b in range(len(synthNames)):
            coef1 = random_effects[a][synthNames[b]]
            array[a, b] = coef1 

    #for b in range(len(synthNames)):
    #    array[:, b] += fixed_effects[synthNames[b]]


    print(mdf.summary())

    #sns.heatmap(array)

    sns.clustermap(array, col_cluster=False)
    plt.show()
    quit()

    #mdf = md.fit(method='nm')  # Nelder-Mead (gradient-free)
    #mdf = md.fit(method='powell')  # Powell’s method (gradient-free)
    #mdf = md.fit(method='cg')  # Conjugate gradient


    #print(pd.DataFrame(df[['synth_1', 'synth_2', 'synth_3']]).corr())
    #print(df['RNA'].var())
    #print(df['groups'].nunique())

    #mdf = md.fit(method='bfgs')

    

    quit()

    

    for a in range(RNA_leaf.shape[1]):
        #train1 = np.random.permutation(RNA_leaf.shape[0])
        #train1, test1 = train1[:-5], train1[-5:]
        clf = linear_model.Ridge(alpha=1e8)
        #clf = linear_model.Ridge(alpha=1e3)
        clf.fit(Y_mean, RNA_leaf[:, a])
        coef = clf.coef_
        coef_all[:, a] = coef


    #sns.heatmap(coef_all)
    #plt.show()

    



    


    '''
    corList = []

    pVals = np.zeros((  Y_mean.shape[1], RNA_leaf.shape[1] ))


    for b in range(Y_mean.shape[1]):
        print (b)
        for a in range(RNA_leaf.shape[1]):
            pVals[b, a] = np.log10(scipy.stats.pearsonr(Y_mean[:, b], RNA_leaf[:, a])[1]) * -1

    pVals = np.array(pVals)

    max1 = np.max(pVals, axis=0)
    '''
    

    RNA_projection = np.zeros(RNA_leaf.shape)
    from sklearn.linear_model import LinearRegression

    for a in range(RNA_leaf.shape[1]):
        clf = linear_model.Ridge(alpha=1e2).fit(Y_mean, RNA_leaf[:, a])
        #clf = LinearRegression().fit(Y_mean, RNA_leaf[:, a])
        predRNA = clf.predict(Y_mean)
        RNA_projection[:, a] = predRNA


    print (RNA_projection.shape)
    print ('')
    U, S, Vh = np.linalg.svd(RNA_projection, full_matrices=False)
    RNA_comp = U[:, :Y_mean.shape[1]]

    coef_all = np.zeros((Y_mean.shape[1], Y_mean.shape[1]))
    for a in range(Y_mean.shape[1]):
        clf = LinearRegression().fit(Y_mean, RNA_comp[:, a])
        print (np.mean( np.abs(  RNA_comp[:, a] - clf.predict(Y_mean)  )  ))
        coef_all[:, a] = clf.coef_

    predCheck = np.matmul(Y_mean, coef_all)
    predCheck = predCheck - np.mean(predCheck, axis=0).reshape((1, -1)) + np.mean(RNA_comp, axis=0).reshape((1, -1))


    assert np.mean(np.abs(predCheck - RNA_comp)) < 0.001

    

    

    

    quit()




    

        #for b in range(Y_mean.shape[1]):
        #    cor_all[b, a] = scipy.stats.pearsonr(RNA_leaf[:, a], Y_mean[:, b])[0]




    U, S, Vh = np.linalg.svd(coef_all, full_matrices=False)

    #A = deconvolver1(cor_all)
    #U, S, Vh = np.linalg.svd(cor_all, full_matrices=False)

    #U.T cor_all = 


    #print (A)

    #RNA_leaf = Y_mean * coef_all
    #RNA_leaf = Y_mean * U, S, Vh

    #print (S)
    #print (U)

    #sns.clustermap(coef_all[:, :])
    #plt.show()
    #quit()

    
    quit()

    plt.imshow(corMatrix_after)
    plt.show()

    
    quit()
    np.savez_compressed('./data/plant/syntheticTraits/deconv_0.npz', Y_transform)
    quit()

    

    #sns.clustermap(Vh[:3])
    #plt.show()
    #quit()

    sign1 = np.sign(coef_all[0]).reshape((1, -1))
    coef_all = coef_all * sign1

    print (np.mean(coef_all, axis=1))

    


    U, S, Vh = np.linalg.svd(coef_all, full_matrices=False)

    np.savez_compressed('./data/temp/deconv.npz', U.T)


    print (S)

    plt.imshow(U)
    plt.show()

    print (U.shape, S.shape, Vh.shape)
    quit()
    

    Ntrait = Y_mean.shape[1]
    for a in range(Ntrait-1):
        for b in range(a+1, Ntrait):
            print (a, b)
            print (scipy.stats.pearsonr(coef_all[a], coef_all[b]))




    quit()


    # Optional: Plot a simple "Manhattan plot" (using -log10(pvalue))
    plt.scatter(np.arange(pvals.shape[0]), -np.log10(pvals), c="blue", s=10)
    plt.plot(  np.arange(pvals.shape[0]), np.zeros(pvals.shape[0]) + np.log10(pvals.shape[0] * 20) )
    plt.xlabel("SNP index")
    plt.ylabel("-log10(p-value)")
    plt.title("Manhattan Plot of GWAS (Random Data)")
    plt.show()


    sorted_pvalues = np.sort(pvals)
    # Compute the expected p-values assuming a uniform distribution under the null.
    expected = np.arange(1, len(sorted_pvalues) + 1) / (len(sorted_pvalues) + 1)
    # Transform to -log10 scale
    expected_log = -np.log10(expected)
    observed_log = -np.log10(sorted_pvalues)


    plt.scatter(expected_log, observed_log, color="blue", s=10)
    # Plot the y=x line for reference
    plt.plot([expected_log[0], expected_log[-1]], [expected_log[0], expected_log[-1]], color="red", linestyle="--")
    plt.xlabel("Expected -log10(p)")
    plt.ylabel("Observed -log10(p)")
    plt.title("QQ Plot of GWAS p-values")
    plt.show()

    
    quit()

    Y_mean, RNA_leaf = torch.tensor(Y_mean).float(), torch.tensor(RNA_leaf).float()

    


    

    Ntrait = 5
    
    for a in range(Ntrait-1):
        for b in range(a+1, Ntrait):
            print (a, b)
            print (scipy.stats.pearsonr(coef_all[a], coef_all[b]))

    A = deconvolver1(coef_all)

    A = torch.tensor(A).float()
    Y_new = torch.matmul(A, Y_mean[:, :5].T).T

    coef_new = getAllCoefs(Y_new, RNA_leaf)

    range_min, range_max = np.min((coef_new.min(), coef_all.min())), np.max((coef_new.max(), coef_all.max()))
    range1 = (range_min, range_max)

    plt.hist(coef_all.reshape((-1,)), bins=100, alpha=0.5, range=range1)
    plt.hist(coef_new.reshape((-1,)), bins=100, alpha=0.5, range=range1)
    plt.show()

    #for a in range(Ntrait-1):
    #    for b in range(a+1, Ntrait):
    #        print (a, b)
    #        print (scipy.stats.pearsonr(coef_all[a], coef_all[b]))





    quit()



    
    if False:
        Ntrait = 10
        coef_all = np.zeros((Ntrait, perm2.shape[0]))
        for a in range(Ntrait):
            print (a)
            results_df, random_effects_df = run_lmm_random_slope(Y_mean[:, a], RNA_leaf[:, perm2])
            coef = random_effects_df['Random_Effect'].to_numpy()
            coef_all[a] = coef

        np.savez_compressed('./data/temp/mixedRNAcoef.npz', coef_all)
        quit()


    coef_all = loadnpz('./data/temp/mixedRNAcoef.npz')
    A = deconvolver1(coef_all)

    Y_mean = Y_mean[:, :5]

    Y_new = np.matmul(A, Y_mean.T).T

    print (perm2.shape)


    Ntrait = 10
    coef_all = np.zeros((Ntrait, perm2.shape[0]))
    for a in range(Ntrait):
        print (a)
        print (Y_new[:, a].shape, RNA_leaf[:, perm2].shape)
        results_df, random_effects_df = run_lmm_random_slope(Y_new[:, a], RNA_leaf[:, perm2])
        coef = random_effects_df['Random_Effect'].to_numpy()
        coef_all[a] = coef

    np.savez_compressed('./data/temp/mixedRNAcoef_after.npz', coef_all)

    quit()

    AX_cor = np.matmul(AX.T, RNA_leaf)

    min1 = min(np.min(AX_cor), np.min(cor1))
    max1 = max(np.max(AX_cor), np.min(cor1))

    plt.hist(cor1.reshape((-1,)), bins=100, alpha=0.5, range=(min1, max1))
    plt.hist(AX_cor.reshape((-1,)), bins=100, alpha=0.5, range=(min1, max1))
    plt.show()
    quit()



    quit()

    M = 5
    for a in range(M-1):
        for b in range(a+1, M):
            print (a, b)
            print (scipy.stats.pearsonr(cor1[a], cor1[b]))
            print (scipy.stats.pearsonr(abs1[a], abs1[b]))

    sns.heatmap(cor1)
    plt.show()
    quit()


#checkRNA()
#quit()





def autoRegressiveTrain(X, trainTest2, modelName, Niter = 10000, doPrint=True, regScale=1e-8, learningRate=1e-4, Nlatent=1):

    X = torch.tensor(X).float()


    numWavelengths = X.shape[1]

    model = autoEncoder(numWavelengths, Nlatent)


    optimizer = torch.optim.Adam(model.parameters(), lr = learningRate, betas=(0.9, 0.99))

    

    argTrain = np.argwhere(trainTest2 == 0)[:, 0]

    batch_size = -1#1000
    #print (X.shape)
    #quit()
    #print (np.unique(trainTest2, return_counts=True))
    #quit()


    if batch_size < 0:
        batch_size = argTrain.shape[0]




    for a in range(Niter):
        #print ("C")

        #print (argTrain.shape[0] ,batch_size)
        #quit()

        for batch_index in range(argTrain.shape[0] // batch_size):
            #print ("A")

            
            argNow = argTrain[batch_index*batch_size:(batch_index+1)*batch_size]

            

            X_pred = model(X[argNow])


            loss = torch.mean(  (X[argNow] - X_pred) ** 2 ) * X.shape[1]
            
            #reg1 = 0
            #for param in model.parameters():
            #    reg1 += torch.mean(torch.abs(param))

            if doPrint:

                if batch_index == 0:

                    if a % 100 == 0:
                        print ('iter', a // 100)

                        X_pred = model(X)



                        loss1 = torch.sum(  (X - X_pred) ** 2 , axis=1)
                        print (torch.mean(loss1[trainTest2 == 0]))
                        print (torch.mean(loss1[trainTest2 == 1]))

                        meanPercent = torch.mean(torch.abs((X_pred[trainTest2==1] / X[trainTest2==1]) - 1))
                        print ('meanPercent', meanPercent)

            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.save(model, modelName)




def trainAuto():

    X = loadnpz('./data/plant/processed/sor/X.npz')
    names = loadnpz('./data/plant/processed/sor/names.npz')
    
    name_unique, name_inverse, name_counts = np.unique(names, return_inverse=True, return_counts=True)

    np.random.seed(0)
    trainTest2 = np.random.randint(3, size=name_unique.shape[0])
    trainTest2 = trainTest2[name_inverse]


    trainTest3 = np.copy(trainTest2)
    trainTest3[trainTest3 == 0] = 100
    trainTest3[trainTest3!=100] = 0
    trainTest3[trainTest3 == 100] = 1

    #Nlatent = 1
    Nlatent = 5
    #Nlatent = 10
    #modelName = './data/plant/models/autoencode_2.pt'
    modelName = './data/plant/models/autoencode_fake.pt'
    regScale = 1e-20

    learningRate = 5e-5
    #learningRate = 1e-4

    autoRegressiveTrain(X, trainTest3, modelName, Niter = 100000, doPrint=True, regScale=1e-8, learningRate=learningRate, Nlatent=Nlatent)
    quit()

#trainAuto()
#quit()



def printAutoError():

    X = loadnpz('./data/plant/processed/sor/X.npz')
    names = loadnpz('./data/plant/processed/sor/names.npz')

    #print (X.shape)
    #quit()
    
    name_unique, name_inverse, name_counts = np.unique(names, return_inverse=True, return_counts=True)

    np.random.seed(0)
    trainTest2 = np.random.randint(3, size=name_unique.shape[0])
    trainTest2 = trainTest2[name_inverse]


    trainTest3 = np.copy(trainTest2)
    trainTest3[trainTest3 == 0] = 100
    trainTest3[trainTest3!=100] = 0
    trainTest3[trainTest3 == 100] = 1

    Nlatent = 5

    X = torch.tensor(X).float()


    numWavelengths = X.shape[1]

    modelName = './data/plant/models/autoencode_2.pt'
    model = torch.load(modelName)

    argTrain = np.argwhere(trainTest2 == 0)[:, 0]

    batch_size = -1#1000
    
        
    X_pred = model(X)



    loss1 = torch.sum(  (X - X_pred) ** 2 , axis=1)
    print (torch.mean(loss1[trainTest2 == 0]))
    print (torch.mean(loss1[trainTest2 == 1]))

    meanPercent = torch.mean(  torch.abs(  (X_pred[trainTest2==0] / X[trainTest2==0]) - 1)  )
    print ('meanPercent', meanPercent)


    meanPercent = torch.mean(torch.abs((X_pred[trainTest2==1] / X[trainTest2==1]) - 1))
    print ('meanPercent', meanPercent)


    X_norm = X / torch.mean(X, axis=1).reshape((-1, 1))
    X_pred_norm = X_pred / torch.mean(X_pred, axis=1).reshape((-1, 1))


    meanPercent = torch.mean(  torch.abs(  X_norm[trainTest2==0] - X_pred_norm[trainTest2==0] )  )
    print ('meanPercent', meanPercent)


    meanPercent = torch.mean(  torch.abs(  X_norm[trainTest2==1] - X_pred_norm[trainTest2==1] )  )
    print ('meanPercent', meanPercent)


    quit()


#printAutoError()
#quit()


def checkAutoDistribute():


    modelName = './data/plant/models/autoencode_2.pt'
    model = torch.load(modelName)
    X = loadnpz('./data/plant/processed/sor/X.npz')

    X = torch.tensor(X).float()

    Y = model.encode(X)

    for a in range(5):
        plt.hist(Y[:, a].data.numpy(), bins=100)
        plt.show()


#checkAutoDistribute()
#quit()




def trainModel(model, X, names, envirement, trainTest2, modelName, Niter = 10000, doPrint=True, regScale=1e-8, learningRate=1e-4, NphenStart=0, Nphen=1,  noiseLevel=0.1):

    

    X = torch.tensor(X).float()


    numWavelengths = X.shape[1]
    
    argTrain = np.argwhere(trainTest2 == 0)[:, 0]


    for phenNow in range(NphenStart, Nphen):

        print ('X shape', X.shape)


        if phenNow > 0:
            subset1 = np.arange(phenNow)

            Y_background = model(X, subset1)
            Y_background = Y_background.detach()
            Y_background = normalizeIndependent(Y_background)


        else:
            Y_background = torch.zeros((X.shape[0]), 0)

        subset_phen = np.zeros(1, dtype=int) + phenNow

        optimizer = torch.optim.RMSprop(model.parameters(), lr = learningRate)
        

        for a in range(Niter):
            
            X_train = X[trainTest2 == 0]
            
            #rand1 = torch.randn(X_train.shape) * noiseLevel
            rand1 = torch.rand(size=X_train.shape) * noiseLevel
            X_train = X_train + rand1
        
            
            Y = model(X_train, subset_phen)

            Y_abs = torch.mean(torch.abs(Y -  torch.mean(Y, axis=0).reshape((1, -1))   ))

            Y = removeIndependence(Y, Y_background[trainTest2 == 0])

            Y = normalizeIndependent(Y, cutOff=2) #Include for now

            
            heritability_now = cheapHeritability(Y, names[trainTest2 == 0], envirement[trainTest2 == 0])#, device=mps_device )
            loss = -1 * torch.mean(heritability_now)


            count1 = 0
            for param in model.parameters():
                if len(param.shape) == 2:
                    if count1 == phenNow:
                        diff1 = param[:, 1:] - param[:, :-1]
                        reg1 = torch.sum(torch.abs(diff1), axis=1)
                    count1 += 1
            regLoss = torch.mean(reg1) / Y_abs
            loss = loss + (regLoss * regScale)
            

            if a % 100 == 0:
                
                print ('iter:', a)

                with torch.no_grad():
                    Y = model(X, subset_phen)
                    Y = removeIndependence(Y, Y_background)

                    Y = normalizeIndependent(Y, cutOff=2) #Include for now

                    heritability_train = cheapHeritability(Y[trainTest2 == 0], names[trainTest2 == 0], envirement[trainTest2 == 0] )
                    if 1 in trainTest2:
                        heritability_test = cheapHeritability(Y[trainTest2 == 1], names[trainTest2 == 1], envirement[trainTest2 == 1] )
                

                print ('subset_phen', subset_phen)
                print (heritability_train.data.numpy())
                if 1 in trainTest2:
                    print (heritability_test.data.numpy())


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if a % 10 == 0:
                torch.save(model, modelName)
        
    




def trainNoSplit():


    X = loadnpz('./data/plant/processed/sor/X.npz')
    names = loadnpz('./data/plant/processed/sor/names.npz')
    envirement = loadnpz('./data/plant/processed/sor/set1.npz')
    

    envirement = envirement.reshape((-1, 1))
    name_unique, name_inverse, name_counts = np.unique(names, return_inverse=True, return_counts=True)
    
    trainTest3 = np.zeros(names.shape[0], dtype=int)

    Nphen = 20

    args = [X.shape[1], 1]
    model = multiConv(Nphen, args, simpleModel)
    
    #modelName = './data/plant/models/linear_trainAll2.pt'

    modelName = './data/plant/models/linear_trainAll_fake.pt'

    #noiseLevel = 0.005 #Good
    noiseLevel = 0.0
    #regScale = 0.0
    regScale = 1e-4 #Seems good

    Niter = 100000
    Nphen = 10
    NphenStart = 0
    learningRate = 1e-5


    trainModel(model, X, names, envirement, trainTest3, modelName, Niter=Niter, doPrint=True, regScale=regScale, NphenStart=NphenStart, Nphen=Nphen, learningRate=learningRate, noiseLevel=noiseLevel)
        

#trainNoSplit()
#quit()


def trainMultiplePhenotypes():


    X = loadnpz('./data/plant/processed/sor/X.npz')
    names = loadnpz('./data/plant/processed/sor/names.npz')
    envirement = loadnpz('./data/plant/processed/sor/set1.npz')
    traitAll = loadnpz('./data/plant/processed/sor/traits.npz')


    print (X.shape)

    print (names.shape)
    print (np.unique(names).shape)
    quit()

    #X = np.fft.rfft(X, axis=1)
    #X = np.real(X)
    #X = np.concatenate(( np.real(X), np.imag(X) ), axis=1)
    

    #print (names[:10])
    #quit()


    print (np.max(X))
    print (np.min(X))
    quit()
        
    Nitrogen = traitAll[:, 0]
    SLA = traitAll[:, 2]
    PN = traitAll[:, 3]
    PS = traitAll[:, 5]

    #trait1 = SLA
    #trait1 = Nitrogen
    #trait1 = PS
    #trait1 = PN

    

    
    #mean1 = np.mean(trait1[np.isnan(trait1) == False])
    #trait1[np.isnan(trait1)] = mean1
    
    #trait1 = torch.tensor(trait1).float()

    envirement = envirement.reshape((-1, 1))
    name_unique, name_inverse, name_counts = np.unique(names, return_inverse=True, return_counts=True)

    #name_counts = name_counts[name_inverse]
    #X = X[name_counts >= 2]
    #names = names[name_counts >= 2]
    #envirement = envirement[name_counts >= 2]
    #name_unique, name_inverse, name_counts = np.unique(names, return_inverse=True, return_counts=True)
    

    

    np.random.seed(0)
    #np.random.seed(2)
    #trainTest2 = np.random.randint(3, size=name_unique.shape[0])
    trainTest2 = np.random.randint(10, size=name_unique.shape[0])
    trainTest2 = trainTest2[name_inverse]



    #for splitIndex in range(0, 10):
    for splitIndex in range(10):

        for b in range(10):
            print ('')
        print ('splitIndex', splitIndex)
        for b in range(10):
            print ('')



        trainTest3 = np.copy(trainTest2)
        trainTest3[trainTest3 == splitIndex] = 100
        trainTest3[trainTest3!=100] = 0
        trainTest3[trainTest3 == 100] = 1
        
        #trainTest3[:] = 0



        Nphen = 20

        args = [X.shape[1], 1]
        #model = torch.load('./data/plant/models/linear_crossVal_' + str(splitIndex) + '_mod2.pt')
        #model = multiConv(Nphen, args, simpleModel)
        model = multiConv(Nphen, args, convModel)
        #model = torch.load('./data/plant/models/linear_6.pt')
        #model = torch.load('./data/plant/models/linear_crossVal_reg4_' + str(splitIndex) + '_mod2.pt')
        

        #modelName = './data/plant/models/linear_1.pt'
        #modelName = './data/plant/models/linear_8.pt'

        #modelName = './data/plant/models/linear_crossVal_' + str(splitIndex) + '.pt'
        #modelName = './data/plant/models/linear_crossVal_' + str(splitIndex) + '_mod10.pt'
        #modelName = './data/plant/models/linear_crossVal_reg4_' + str(splitIndex) + '.pt'
        #modelName = './data/plant/models/linear_crossVal_reg4_' + str(splitIndex) + '_mod10.pt'

        modelName = './data/plant/models/linear_crossVal_reg4_' + str(splitIndex) + '_convOverfit2.pt'


        overFit = True

        noiseLevel = 0.005 #Good
        if overFit:
            noiseLevel = 0.0
        #regScale = 1e-8
        #regScale = 0.0
        #regScale = 1e-7

        
        #regScale = 1e-5
        regScale = 1e-4 #Seems good
        if overFit:
            regScale = 0.0
        #regScale = 1e-3
        #regScale = 1e-2
        #regScale = 1e0

        #regScale = 1e-2
        #Niter = 10000
        #Niter = 100000
        #(heritability_train, heritability_test) = trainModel(X, names, envirement, trainTest3, modelName, Niter=Niter, doPrint=True, regScale=regScale, Nphen=Nphen, learningRate=5e-5, corVar=trait1)
        #trainModel(X, names, envirement, trainTest3, modelName, Niter=Niter, doPrint=True, regScale=regScale, Nphen=Nphen, learningRate=5e-5)
        #trainModel(X, names, envirement, trainTest3, modelName, Niter=Niter, doPrint=True, regScale=regScale, Nphen=Nphen, learningRate=5e-5)

        #trainModel(X, names, envirement, trainTest3, modelName, Niter=Niter, doPrint=True, regScale=regScale, Nphen=Nphen, learningRate=1e-5)

        #Niter = 10000
        #Niter = 100000
        #Niter = 50000 #GOOd
        Niter = 10000
        Nphen = 10
        NphenStart = 0
        learningRate = 1e-5
        #learningRate = 1e-4
        if overFit:
            learningRate = 1e-3


        trainModel(model, X, names, envirement, trainTest3, modelName, Niter=Niter, doPrint=True, regScale=regScale, NphenStart=NphenStart, Nphen=Nphen, learningRate=learningRate, noiseLevel=noiseLevel)
        quit()



#trainMultiplePhenotypes()
#quit()



def evaluateHerits():

    X = loadnpz('./data/plant/processed/sor/X.npz')
    names = loadnpz('./data/plant/processed/sor/names.npz')
    envirement = loadnpz('./data/plant/processed/sor/set1.npz')
    traitAll = loadnpz('./data/plant/processed/sor/traits.npz')


    #print (names.shape)
    #print (np.unique(names).shape)
    #print (X.shape)
    #quit()

    envirement = envirement.reshape((-1, 1))
    name_unique, name_inverse, name_counts = np.unique(names, return_inverse=True, return_counts=True)

    np.random.seed(0)
    #np.random.seed(2)
    #trainTest2 = np.random.randint(3, size=name_unique.shape[0])
    trainTest3 = np.random.randint(10, size=name_unique.shape[0])
    trainTest3 = trainTest3[name_inverse]

    trainInfos = np.zeros((3, 10, 10))
    testInfos = np.zeros((3, 10, 10))


    for splitIndex in range(10):
        #for splitIndex in range(5):

        trainTest2 = np.zeros(names.shape[0], dtype=int)
        trainTest2[trainTest3 == splitIndex] = 1
        #trainTest2[trainTest2!=100] = 0
        #trainTest2[trainTest2 == 100] = 1



        #simulationNames = ['uncorSims', 'random3SNP', 'random100SNP']

        #simulationNames = ['uncorSims']
        #simulationNames = ['seperate100SNP']
        #simulationNames = ['random100SNP']

        #simulationNames = ['uncorSims', 'random100SNP']

        #methodNames = ['H2Opt', 'maxWave', 'PCA']

        methodNames = ['H2Opt', 'maxWave', 'PCA']
        #methodNames = ['NDVI']

        #methodNames = ['H2Opt']

        #methodNames = ['H2Opt-Conv']

        

        synthUsed = 10

        



        #for methodName in methodNames:
        for methodIndex in range(len(methodNames)):
            methodName = methodNames[methodIndex]

            print ('methodName', methodName)

            
            

            ######envirement = np.zeros((names.shape[0], 0))


            if 'H2Opt' in methodName:
                #modelName = './data/plant/models/linear_6.pt'
                #modelName = './data/plant/models/linear_7.pt'
                #modelName = './data/plant/models/linear_crossVal_' + str(splitIndex) + '_mod10.pt'

                #modelName = './data/plant/models/linear_crossVal_reg4_' + str(splitIndex) + '_mod1.pt'
                modelName = './data/plant/models/linear_crossVal_reg4_' + str(splitIndex) + '_mod10.pt'


                model = torch.load(modelName)
                subset1 = np.arange(synthUsed)

                Y = model(torch.tensor(X).float(), subset1)
                Y = normalizeIndependent(Y)
                Y_np = Y.data.numpy()

            if methodName == 'PCA':
                from sklearn.decomposition import PCA
                pca = PCA(n_components=synthUsed)
                pca.fit(X)
                Y_np = pca.transform(X)
                Y = torch.tensor(Y_np).float()

            if methodName == 'maxWave':

                #print ("A")
                #quit()
                X_copy = np.copy(X[trainTest2 == 0])
                X_copy = X_copy - np.mean(X_copy, axis=0).reshape((1, -1))
                argBest_list = []
                for a in range(synthUsed):
                    heritability_wave = cheapHeritability(torch.tensor(X_copy).float() , names[trainTest2 == 0], envirement[trainTest2 == 0] )
                    heritability_wave = heritability_wave.data.numpy()

                    #plt.plot(heritability_wave)
                    #plt.show()
                    #print (heritability_wave)

                    if a > 0:
                        heritability_wave[np.array(argBest_list)] = 0
                    #print (np.max(heritability_wave))
                    argBest = np.argmax(heritability_wave)
                    argBest_list.append(argBest)
                    X_copy = remove_projections(X_copy, np.copy(X_copy[:, argBest:argBest+1]))
                    X_copy[:, argBest] = 1

                    #plt.plot(X_copy[:, 0])
                    #plt.plot(X_copy[:, 10])
                    #plt.show()
                    #quit()


                argBest_list = np.array(argBest_list)
                Y = np.copy(X[:, argBest_list])

                #Y = normalizeIndependent( torch.tensor(Y).float() , cutOff=2 )
                Y = normalizeIndependent( torch.tensor(Y).float() )
                Y_np = Y.data.numpy()


            print (Y.shape)

            phenoFile = './data/plant/syntheticTraits_baseline/' + str(methodName) + '_' + str(splitIndex) + '.npz'
            print (phenoFile)
            np.savez_compressed(phenoFile, Y.data.numpy())


            heritability_train = cheapHeritability(Y[trainTest2 == 0], names[trainTest2 == 0], envirement[trainTest2 == 0] )
            heritability_test = cheapHeritability(Y[trainTest2 == 1], names[trainTest2 == 1], envirement[trainTest2 == 1] )

            #heritBoth = np.array([ heritability_train.data.numpy(),  heritability_test.data.numpy() ])
            #filesave  = './data/plant/eval/herit/' + methodName + '.npz' 
            #print (heritBoth)
            #np.savez_compressed(filesave, heritBoth)

            testInfos[methodIndex, splitIndex] = heritability_test.data.numpy()
            trainInfos[methodIndex, splitIndex] = heritability_train.data.numpy()
            


    print (np.mean(trainInfos[:, :], axis=1))
    print (np.mean(testInfos[:, :], axis=1))

    #print (np.mean(testInfos, axis=1))
    quit()
    
    #print (testInfos[0, 0])

    #print (np.mean(testInfos[0, :, 0 ]))
    #print (np.mean(trainInfos[0, :, 0]))
    #quit()

    filesave  = './data/plant/eval/herit/crossValid_train_proper.npz' 
    np.savez_compressed(filesave, trainInfos)
    filesave  = './data/plant/eval/herit/crossValid_test_proper.npz' 
    np.savez_compressed(filesave, testInfos)

    quit()
    
    filesave  = './data/plant/eval/herit/crossValid_train.npz' 
    np.savez_compressed(filesave, trainInfos)
    filesave  = './data/plant/eval/herit/crossValid_test.npz' 
    np.savez_compressed(filesave, testInfos)

    

    quit()

    #filesave  = './data/plant/eval/herit/' + methodName + '.npz' 

    
    #quit()
    #print (np.median(testInfos, axis=1)[:, :3])

    trainInfos = np.mean(trainInfos, axis=1)[:, :3]
    testInfos = np.mean(testInfos, axis=1)[:, :3]

    print (trainInfos)
    print (testInfos)


    methodNames = ['H2Opt', 'maxWave', 'PCA']

    colorList = ['tab:blue', 'tab:orange', 'tab:green']

    for methodIndex in range(len(methodNames)):
        
        train1 = trainInfos[methodIndex]
        test1 = testInfos[methodIndex]

        color1 = colorList[methodIndex]

        arange1 = np.arange(test1.shape[0]) + 1
        plt.plot(arange1, train1, c=color1, linestyle=':')
        plt.plot(arange1, test1, c=color1)
        plt.scatter(arange1, train1, c=color1)
        plt.scatter(arange1, test1, c=color1)
        plt.xticks(arange1)
        plt.ylim(bottom=0)
    
    plt.xlabel('sythetic trait number')
    plt.ylabel("heritability")
    plt.legend(['training set', 'test set'])
    #plt.gcf().set_size_inches(7, 2)
    plt.tight_layout()
    plt.show()



evaluateHerits()
quit()


def saveNVDI():

    X = loadnpz('./data/plant/processed/sor/X.npz')
    names = loadnpz('./data/plant/processed/sor/names.npz')

    
    envirement = loadnpz('./data/plant/processed/sor/set1.npz')
    envirement = envirement.reshape((-1, 1))
    #traitAll = loadnpz('./data/plant/processed/sor/traits.npz')

    #: MODIS red, 620–670 nm; TM/ETM +  NIR, 760–900 nm
    #red_value = np.sum(X[:, 620-350:670-350 ], axis=1) 
    #NIR_value = np.sum(X[:, 760-350:900-350 ], axis=1) 
    red_value =  np.mean(X[:, 670:690 ], axis=1)
    NIR_value = np.mean(X[:, 790:810 ], axis=1)

    Y = (NIR_value - red_value) / (NIR_value + red_value)
    Y = torch.tensor(Y).float().reshape((-1, 1))

    print (Y.shape)

    heritability = cheapHeritability(Y, names, envirement)

    print (heritability)
    quit()


saveNVDI()
quit()


def makeIndep(Y, Y_old):

    cor1 = torch.mean(Y - Y_old)
    Y = Y - (cor1 * Y_old)

    return Y 



def mixedTrainModel( variables_X, variables_cat, spreader, wavelengthIntensity, trainTest,  modelName, learningRate=1e-4, regScale=1e-2, Niter=1000):

    #modelName


    modelOld_name = './data/plant/models/mixed_1.pt'
    modelOld = torch.load(modelOld_name)

    modelOld_name2 = './data/plant/models/mixed_2.pt'
    modelOld2 = torch.load(modelOld_name2)

    X = torch.tensor(wavelengthIntensity).float()

    with torch.no_grad():
        Y_old = modelOld(X)
        Y_old = Y_old - torch.mean(Y_old)
        Y_old = Y_old / (torch.mean(Y_old ** 2) ** 0.5)

        Y_old2 = modelOld2(X)
        Y_old2 = Y_old2 - torch.mean(Y_old2)
        Y_old2 = Y_old2 / (torch.mean(Y_old2 ** 2) ** 0.5)


    numWavelengths = X.shape[1]
    model = simpleModel(numWavelengths, 1)
    

    optimizer = torch.optim.Adam(model.parameters(), lr = learningRate, betas=(0.9, 0.99))



    with torch.no_grad():
        Y_train = model(X[trainTest==0])

        Y_train = makeIndep(Y_train, Y_old[trainTest==0])
        Y_train = makeIndep(Y_train, Y_old2[trainTest==0])

        Y_train = Y_train - torch.mean(Y_train)
        Y_abs = torch.mean(torch.abs(Y_train))
        Y_train = Y_train / (torch.mean(Y_train**2)**0.5)
        totalVariance = torch.mean(Y_train ** 2)
        prior_variances = torch.ones(variables_cat.shape[1])
        prior_residual = torch.ones(1)

        initialIter = 50
        for a in range(initialIter):
            #print (a)
            new_variances, new_residual = singleUpdate( Y_train , prior_variances, prior_residual, variables_X[trainTest==0], variables_cat[trainTest==0], spreader )
            prior_variances, prior_residual = new_variances, new_residual
            #print (prior_variances)

        
    
        Y_test = model(X[trainTest==1])

        Y_test = makeIndep(Y_test, Y_old[trainTest==1])
        Y_test = makeIndep(Y_test, Y_old2[trainTest==1])

        Y_test = Y_test - torch.mean(Y_test)
        Y_test = Y_test / (torch.mean(Y_test**2)**0.5)
        totalVariance = torch.mean(Y_test ** 2)
        test_variances = torch.ones(variables_cat.shape[1])
        test_residual = torch.ones(1)
        
        initialIter = 50
        for a in range(initialIter):
            #print (a)
            new_variances_test, new_residual_test = singleUpdate( Y_test , test_variances, test_residual, variables_X[trainTest==1], variables_cat[trainTest==1], spreader )
            test_variances, test_residual = new_variances_test, new_residual_test


        

    


    error_train = []
    error_test = []

    for a in range(Niter):


        Y_train = model(X[trainTest == 0])

        Y_train = makeIndep(Y_train, Y_old[trainTest==0])
        Y_train = makeIndep(Y_train, Y_old2[trainTest==0])

        Y_train = Y_train - torch.mean(Y_train)
        Y_train = Y_train / (torch.mean(Y_train**2)**0.5)
        totalVariance = torch.mean(Y_train ** 2)

        if False:
            Y = normalizeIndependent(Y)

        #Y_abs = torch.mean(torch.abs(Y - torch.mean(Y, axis=0).reshape((1, -1)) ), axis=0)

        #heritability_train = cheapHeritability(Y[trainTest2 == 0], names[trainTest2 == 0], envirement[trainTest2 == 0].reshape((-1, 1)) )

        new_variances, new_residual = singleUpdate( Y_train , prior_variances, prior_residual, variables_X[trainTest==0], variables_cat[trainTest==0], spreader )

        for param in model.parameters():
            if len(param.shape) == 2:
                #print (param.shape)
                diff1 = param[:, 1:] - param[:, :-1]
                reg1 = torch.sum(torch.abs(diff1), axis=1)
                #reg2 = torch.mean(torch.abs(param), axis=1)


        train_herit = new_variances[0] / totalVariance

        loss = -1 * train_herit
        

        regLoss = torch.mean(reg1) / Y_abs
        loss = loss + (regLoss * regScale)



        optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-6)
        optimizer.step()


        torch.save(model, modelName)

        if a % 10 == 0:
            with torch.no_grad(): 
                

                Y_test = model(X[trainTest == 1])

                Y_test = makeIndep(Y_test, Y_old[trainTest==1])
                Y_test = makeIndep(Y_test, Y_old2[trainTest==1])

                Y_test = Y_test - torch.mean(Y_test)
                Y_test = Y_test / (torch.mean(Y_test**2)**0.5)
                totalVariance_test = torch.mean(Y_test ** 2)

                test_variances, test_residual = singleUpdate( Y_test , test_variances, test_residual, variables_X[trainTest==1], variables_cat[trainTest==1], spreader )

                test_herit = test_variances[0] / totalVariance_test


                print (train_herit, test_herit)





        prior_variances, prior_residual = new_variances.detach(), new_residual.detach()

    
    return train_herit, test_herit


    


def simpleMixedTrain():


    wavelengthIntensity = loadnpz('./data/plant/processed/sor/X.npz')
    names = loadnpz('./data/plant/processed/sor/names.npz')
    env_loc = loadnpz('./data/plant/processed/sor/LOC1.npz')
    env_set = loadnpz('./data/plant/processed/sor/set1.npz')
    env_block = loadnpz('./data/plant/processed/sor/block1.npz')
    env_plot = loadnpz('./data/plant/processed/sor/PlotID.npz')





    _, genotype_1 = np.unique(names, return_inverse=True)
    _, env_loc = np.unique(env_loc, return_inverse=True)
    _, env_set = np.unique(env_set, return_inverse=True)
    _, env_block = np.unique(env_block, return_inverse=True)


    inverse_1 = env_loc
    inverse_2 = uniqueValMaker( np.array([env_loc, env_set]).T )
    inverse_3 = uniqueValMaker( np.array([env_loc, env_set, env_block]).T )

    variables_cat = np.array([genotype_1, inverse_1, inverse_2, inverse_3]).T.astype(int)

    variables_X, spreader = makeCatBoolVariables(variables_cat)
    variables_X = torch.tensor(variables_X).float()

    wavelengthIntensity = torch.tensor(wavelengthIntensity).float()


    trainTest = np.random.randint(3, size=genotype_1.shape[0])
    #trainTest[trainTest<=1] = 0
    #trainTest[trainTest>=2] = 1
    #trainTest = trainTest[genotype_1]
    trainTest[:] = 0

    modelName = './data/plant/models/mixed_3.pt'
    

    mixedTrainModel(variables_X, variables_cat, spreader, wavelengthIntensity, trainTest, modelName, learningRate=1e-4, regScale=1e-1, Niter=100000)
    quit()

    model = torch.load(modelName, weights_only=False)

    Y = model(wavelengthIntensity)
    herit = estimateHerit(Y[trainTest==1], variables_X[trainTest==1], variables_cat[trainTest==1], spreader )

    print (herit)
    coef1 = getModelCoef(model)
    

    envirement_now = loadnpz('./data/processed/sor/set1.npz').reshape((-1, 1))
    herit_cheap = cheapHeritability(Y[trainTest==1], names[trainTest==1], envirement_now[trainTest==1], returnVariance=False)

    print (herit_cheap)

    #print (coef1.shape)

    plt.plot( coef1 )
    plt.show()
    quit()


#simpleMixedTrain()
#quit()





def predictPhenotypes():

    #modelName = './data/plant/models/mixed_3.pt'
    #modelName = './data/plant/models/conv_1.pt'
    #modelName = './data/plant/models/related_3.pt'

    #modelName = './data/plant/models/linear_overfit.pt'
    #modelName = './data/plant/models/real_linearAug.pt'
    #modelName = './data/plant/models/linear_6.pt'
    #modelName = './data/plant/models/linear_7.pt'


    for splitIndex in range(10):
        print (splitIndex)
        #Y = loadnpz('./data/plant/syntheticTraits/linear_trainAll2.npz')
        #modelName = './data/plant/models/linear_crossVal_reg4_' + str(splitIndex) + '_mod10.pt'
        #modelName = './data/plant/models/linear_crossVal_reg4_' + str(splitIndex) + '_fourier.pt'
        modelName = './data/plant/models/linear_crossVal_reg4_' + str(splitIndex) + '_convOverfit2.pt'


        #modelName = './data/plant/models/linear_trainAll2.pt'



        model = torch.load(modelName)

        X = loadnpz('./data/plant/processed/sor/X.npz')

        X = np.fft.rfft(X, axis=1)
        X = np.concatenate(( np.real(X), np.imag(X) ), axis=1)

        #X = X / np.mean(X, axis=1).reshape((-1, 1))

        X = torch.tensor(X).float()

        try:
            Y = model(X)
        except:
            Y = model(X, np.arange(10))

        Y = normalizeIndependent(Y)
        

        Y = Y.data.numpy()

        
        #np.savez_compressed('./data/plant/syntheticTraits/linear_trainAll2.npz', Y)
        #np.savez_compressed('./data/plant/syntheticTraits/linear_crossVal_reg4_' + str(splitIndex) + '_fourier.npz', Y)
        np.savez_compressed('./data/plant/syntheticTraits/linear_crossVal_reg4_' + str(splitIndex) + '_convOverfit2.npz', Y)
        
        True


#predictPhenotypes()
#quit()



def compareTraits():

    def remove_projections(v, u):
        if len(v.shape) == 1:
            for a in range(u.shape[1]):
                v =  v - ((np.dot(v, u[:, a]) / np.dot(u[:, a], u[:, a])) * u[:, a])
            return v
        else:
            for a in range(v.shape[1]):
                v[:, a] = remove_projections(v[:, a])
    
    


    names = loadnpz('./data/plant/processed/sor/names.npz')

    

    envirement = loadnpz('./data/plant/processed/sor/set1.npz').reshape((-1, 1))


    traitData = np.loadtxt('./data/plant/processed/traits/SorgumTraits.tsv', delimiter='\t', dtype=str)

    traitData = traitData[2:, :-4]

    genotype = traitData[1:, 0]
    traitValues = traitData[1:, 1:]
    traitNames = traitData[0, 1:]

    argGood = np.argwhere(np.isin(genotype, names))[:, 0]
    traitValues, genotype = traitValues[argGood], genotype[argGood]


    corMatrix = np.zeros((10, 10, traitValues.shape[1]))
    

    for splitIndex in range(10):
        #Y = loadnpz('./data/plant/syntheticTraits/linear_trainAll2.npz')
        Y = loadnpz('./data/plant/syntheticTraits/linear_crossVal_reg4_' + str(splitIndex) + '_mod10.npz')

        

        

        

        Y_means = np.zeros(( genotype.shape[0], Y.shape[1] ))
        for a in range(genotype.shape[0]):
            args1 = np.argwhere(names == genotype[a])[:, 0]
            Y_now = np.mean(Y[args1], axis=0)
            Y_means[a] = Y_now

        
        if False:
            traitValues_early = traitValues[:, :-2]
            argBad = np.argwhere(traitValues_early == '')[:, 0]
            argBad = np.unique(argBad)
            argGood = np.argwhere( np.isin(np.arange(traitValues.shape[0]),  argBad   ) == False )[:, 0]

            traitValues_early = traitValues_early[argGood].astype(float)
            Y_means = Y_means[argGood]

            traitValues_early = traitValues_early - np.mean(traitValues_early, axis=0).reshape((1, -1))
            traitValues_early = traitValues_early / (np.mean(traitValues_early**2, axis=0).reshape((1, -1)) ** 0.5)

            traitValues_early = traitValues_early[:, -2:-1]


        #print (Y_means.shape)
        #print (traitValues.shape)
        #quit()


        
        for a in range(10):
            #print ('')
            #print ('')
            #print ('')
            #print (a)
            #print ('')
            #print ('')
            #print ('')
            Y_current = Y_means[:,a]
            for b in range(traitValues.shape[1]):
                #print (b, traitNames[b])
                trait1 = traitValues[:, b]
                args1 = np.argwhere(trait1 != '')[:, 0]
                trait1 = trait1[args1].astype(float)
                Y_subset = Y_current[args1]

                #print (scipy.stats.pearsonr( Y_subset, trait1 ))
                cor1 = scipy.stats.pearsonr( Y_subset, trait1 )[0]


                #cor1 = scipy.stats.pearsonr( Y_subset, trait1 )[0]
                #Y_subset = torch.tensor(Y_subset).float().reshape((-1, 1))
                #trait1 = torch.tensor(trait1).float().reshape((-1, 1))

                #cor1, _, _ = coherit(Y_subset, trait1, names[args1], envirement[args1], geneticCor=True)
                #cor1 = cor1[0].data.numpy()

                corMatrix[splitIndex, a, b] = cor1
                #plt.scatter(Y_subset, trait1)
                #plt.show()

   

    np.savez_compressed('./data/plant/eval/traitCompare.npz', corMatrix)

    corMatrix = np.median(np.abs(corMatrix), axis=0)
    corMatrix = corMatrix.T

    ax = sns.heatmap( corMatrix, annot=True)#, cmap='bwr')
    #sns.diverging_palette(220, 20, as_cmap=True)
    ax.set_yticklabels(  traitNames , rotation=0)
    ax.set_xticklabels(   np.arange(10)+1 )
    plt.xlabel('synthetic trait')
    plt.show()
    quit()


    for a in range(Y_means.shape[1]):
        Y_now = Y_means[:, a]

        print (a)
        print (np.mean(Y_now ** 2) ** 0.5)

        print (np.mean(Y_now * traitValues_early[:, 0]))
        print (scipy.stats.pearsonr( Y_now, traitValues_early[:, 0] ))
        quit()
        

        #for b in range(traitValues_early.shape[1]):
        #    print (scipy.stats.pearsonr( Y_now, traitValues_early[:, b] ))

        Y_now = remove_projections(Y_now, traitValues_early)
        print (np.mean(Y_now ** 2) ** 0.5)

        quit()

    quit()



    


    from sklearn.linear_model import LinearRegression
    

    if True:
        for b in range(traitValues.shape[1]):
            print (traitNames[b])
            trait1 = traitValues[:, b]
            args1 = np.argwhere(trait1 != '')[:, 0]
            trait1 = trait1[args1].astype(float)
            Y_subset = Y_means[args1]

            Y_subset = Y_subset[:, np.array([0, 4])]

            #print (scipy.stats.pearsonr( Y_subset, trait1 ))

            
            reg = LinearRegression().fit(Y_subset, trait1)
            predict = reg.predict(Y_subset)

            print (scipy.stats.pearsonr( predict, trait1 )) 

    quit()


    for a in range(10):
        print ('')
        print ('')
        print ('')
        print (a)
        print ('')
        print ('')
        print ('')
        Y_current = Y_means[:,a]
        for b in [10]:# range(traitValues.shape[1]):
            print (b, traitNames[b])
            trait1 = traitValues[:, b]
            args1 = np.argwhere(trait1 != '')[:, 0]
            trait1 = trait1[args1].astype(float)
            Y_subset = Y_current[args1]

            print (scipy.stats.pearsonr( Y_subset, trait1 ))

            #plt.scatter(Y_subset, trait1)
            #plt.show()




    #print (genotype.shape)


    #print (np.intersect1d(genotype, names).shape)
    quit()


    print (traitData.shape)

    print (traitData[0])
    print (traitData[-1])
    quit()

    print (traitData[:5, :5])


#compareTraits()
#quit()





def predictNitro():


    traitAll = loadnpz('./data/plant/processed/sor/traits.npz')


    Nitrogen = traitAll[:, 0]
    SLA = traitAll[:, 2]
    PN = traitAll[:, 3]
    PS = traitAll[:, 5]

    trait1 = traitAll[:, np.array([0, 2, 3, 5])]
    for a in range(trait1.shape[1]):
        mean1 = np.mean(trait1[np.isnan(trait1[:, a]) == False, a])
        trait1[np.isnan(trait1[:, a]), a] = mean1
    

    


    np.savez_compressed('./data/plant/syntheticTraits/sor_traits.npz', trait1)


#predictNitro()
#quit()





def predictCheat():

    def remove_projections(v, u):

        for a in range(u.shape[1]):
            v =  v - ((np.dot(v, u[:, a]) / np.dot(u[:, a], u[:, a])) * u[:, a])
        return v


    names = loadnpz('./data/plant/processed/sor/names.npz')


    #head1 = np.loadtxt('./data/plant/SNP/WEST_original.txt', delimiter='\t', dtype=str)
    #head_names = head1[9:]
    #for a in range(head_names.shape[0]):
    #    head_names[a] = head_names[a].split('_')[0]

    file_fam = '../software/data/WEST_original_copy.fam'
    data_fam = np.loadtxt(file_fam, delimiter='\t', dtype=str)
    #head_names = data_fam[:, 0]


    if True:
        nameKey = np.loadtxt('../software/data/nameKey_corrected.tsv', delimiter='\t', dtype=str)
        names_unique = np.unique(names)
        name_fam = data_fam[:, 0]
        
        for a in range(nameKey.shape[0]):
            nameKey[a, 2] = nameKey[a, 2].replace(' ', '')
        
        for a in range(data_fam.shape[0]):
            name1 = name_fam[a]
            arg1 = np.argwhere(nameKey[:, 0] == name1)[0, 0]
            name2 = nameKey[arg1, 2]
            name_fam[a] = name2


        head_names = name_fam

    


    file_SNP = './data/plant/SNP/allChr.npz'

    data = loadnpz(file_SNP)
    data = np.sum(data, axis=2)
    data = data.astype(int)

    data = data.T

    np.random.seed(0)
    #data = data[:, np.random.permutation( data.shape[1] )[:100]  ]
    data = data[:, :10  ]


    #from sklearn.decomposition import PCA
    #pca = PCA(n_components=20)
    #related = pca.fit_transform(data)

    #related_new = np.zeros((names.shape[0], related.shape[1]))

    Y = np.zeros((names.shape[0], data.shape[1]))
    nameCheck = np.copy(names)
    nameCheck[:] = ''
    for a in range(names.shape[0]):
        if names[a] in head_names:
            Y[a] = data[ head_names == names[a] ][0]
            nameCheck[a] = head_names[ head_names == names[a] ][0]
            #related_new[a] = related[ head_names == names[a] ][0]


    for a in range(names.shape[0]):
        if nameCheck[a] != '':
            assert names[a] == nameCheck[a]

    #args1 = np.argwhere(names == names[1])[:, 0]
    #print (Y[args1])
    #quit()


    

    #for a in range(Y.shape[1]):

    #    Y[:, a] = remove_projections(Y[:, a], related_new)




    #for b in range(Y.shape[1]):
    #    plt.hist(Y[:, b])
    #    plt.show()

    #Y = normalizeIndependent(torch.tensor(Y).float()).data.numpy()

    #for a in range(Y.shape[1]):

    #    plt.hist(Y[:, a], bins=100)
    #    plt.show()
    #quit()


    np.savez_compressed('./data/plant/syntheticTraits/sor_cheat.npz', Y)


#predictCheat()
#quit()



def compareTraits():

    Y_simple = loadnpz('./data/plant/syntheticTraits/sor_simple_1.npz')
    #Y_conv = loadnpz('./data/plant/syntheticTraits/sor_conv_1.npz')
    Y_conv = loadnpz('./data/plant/syntheticTraits/sor_A_4.npz')

    #Y_simple = Y_conv

    for a in range(Y_conv.shape[1]):
        Y_conv[:, a] = Y_conv[:, a] - np.mean(Y_conv[:, a])
        Y_conv[:, a] = Y_conv[:, a] / (np.mean(Y_conv[:, a] ** 2) ** 0.5)

    print (Y_simple.shape)
    print (Y_conv.shape)


    corList = np.zeros((10, 10))

    for a in range(10):
        for b in range(10):
            cor1 = scipy.stats.pearsonr( Y_simple[:, a], Y_conv[:, b] )
            corList[a, b] = cor1[0]

    corList = np.abs(corList)
    
    plt.imshow(corList)
    #plt.title('standard conv net')
    plt.title('low regularization conv net')
    plt.colorbar()
    plt.xlabel('trait in linear model')
    plt.ylabel('trait in conv net')
    plt.show()




#compareTraits()
#quit()


def investigateSNP():

    Y = loadnpz('./data/plant/syntheticTraits/sor_cheat.npz')
    names = loadnpz('./data/plant/processed/sor/names.npz')

    envirement = np.zeros((Y.shape[0], 0))

    Y = torch.tensor(Y).float()

    #print (Y.shape)
    #quit()

    heritability = cheapHeritability_related(Y[:, :], names, envirement)

    if True:
        related = np.loadtxt('../software/output/relatedness_matrix.cXX.txt', delimiter='\t', dtype=str)
        related = related.astype(float)
        file_fam = '../software/data/WEST_original_copy.fam'
        data_fam = np.loadtxt(file_fam, delimiter='\t', dtype=str)
        name_fam = data_fam[:, 0]

        argGood = np.argwhere(np.isin(name_fam, names))[:, 0]
        related = related[argGood][:, argGood]
        name_fam = name_fam[argGood]


        _, count1 = np.unique(names, return_counts=True)
        _, inverse1 = np.unique(names, return_inverse=True)
        count1 = count1[inverse1]


        argGood = np.argwhere(np.isin(names, name_fam))[:, 0]
        argGood = argGood[count1[argGood] == 2]
        names = names[argGood]
        #X = X[argGood]
        envirement = envirement[argGood]
        Y = Y[argGood]
        #traitAll = traitAll[argGood]

        related_new = []
        for a in range(names.shape[0]):
            name1 = names[a]
            arg1 = np.argwhere(name_fam == name1)[0, 0]
            related_new.append(np.copy(related[arg1]))
        related = np.array(related_new)



        #related

        from sklearn.decomposition import PCA
        #pca = PCA(n_components=2)
        #pca = PCA(n_components=5)
        pca = PCA(n_components=20)
        #pca = PCA(n_components=related.shape[1])
        related = pca.fit_transform(related)

        #plt.plot(np.mean(np.abs(related), axis=0))
        #plt.show()
        #quit()


    print (related.shape)


    for a in range(related.shape[1]):
        print (scipy.stats.pearsonr( Y[:, 2], related[:, a] ))


    
    quit()


#investigateSNP()
#quit()


def checkRelated():

    import seaborn as sns

    #wavelengthIntensity = loadnpz('./data/plant/processed/sor/X.npz')
    names = loadnpz('./data/plant/processed/sor/names.npz')
    Y0 = loadnpz('./data/plant/syntheticTraits/sor_H_4.npz')
    #Y0 = loadnpz('./data/plant/syntheticTraits/sor_related_8.npz')

    
    related = np.loadtxt('../software/output/relatedness_matrix.cXX.txt', delimiter='\t', dtype=str)
    related = related.astype(float)
    file_fam = '../software/data/WEST_original_copy.fam'
    data_fam = np.loadtxt(file_fam, delimiter='\t', dtype=str)
    name_fam = data_fam[:, 0]

    argGood = np.argwhere(np.isin(name_fam, names))[:, 0]
    related = related[argGood][:, argGood]
    name_fam = name_fam[argGood]


    from sklearn.decomposition import PCA
    #pca = PCA(n_components=2)
    pca = PCA(n_components=5)
    #pca = PCA(n_components=related.shape[1])
    related_pca = pca.fit_transform(related)


    Y = []
    for a in range(name_fam.shape[0]):
        name1 = name_fam[a]
        args1 = np.argwhere(names == name1)[:, 0]
        Y.append(np.mean(Y0[args1], axis=0))
    Y = np.array(Y)

    for a in range(Y.shape[1]):
        Y[:, a] = Y[:, a] - np.mean(Y[:, a])
        Y[:, a] = Y[:, a] / (np.mean(Y[:, a] ** 2) ** 0.5)

    if True:
        for phenIndex in range(Y.shape[1]):
            Y_now = Y[:, phenIndex]
            print ('phenIndex', phenIndex + 1)
            for b in range(1):

                print (scipy.stats.pearsonr( related_pca[:, 0], Y_now ))
                print (scipy.stats.pearsonr( related_pca[:, 1], Y_now ))
                print (scipy.stats.pearsonr( related_pca[:, 2], Y_now ))
                print (scipy.stats.pearsonr( related_pca[:, 3], Y_now ))
                #plt.scatter(related_pca[:, b], Y_now)
                #plt.xlabel('relatedness matrix PCA componenet ' + str(b + 1))
                #plt.ylabel('phenotype ' + str(phenIndex + 1) + ' value')
                #plt.tight_layout()
                #plt.show()

        quit()

    for phenIndex in range(Y.shape[1]):
        Y_now = Y[:, phenIndex]
        diff1 = Y_now.reshape((-1, 1)) - Y_now.reshape((1, -1))
        diff1 = np.abs(diff1)
        #diff1 = np.abs(diff1) ** 2

        #sns.clustermap(diff1)
        #plt.show()



        print (scipy.stats.pearsonr(  diff1.reshape((-1,)) , related.reshape((-1,)) ))
        print (scipy.stats.spearmanr(  diff1.reshape((-1,)) , related.reshape((-1,)) ))

        
        diff1 = diff1.reshape((-1,))
        related = related.reshape((-1,)) 
        perm1 = np.random.permutation( diff1.shape[0])
        #perm1 = perm1[:100000]


        plt.scatter(   diff1 , related , alpha=0.005)
        plt.xlabel('trait difference')
        plt.ylabel('genetic similarity')
        plt.show()

    #print (Y.shape)
    #print (related.shape)
    quit()





    print (name_fam.shape)
    print (related.shape)
    quit()

    data_fam0 = data_fam0[:, :6]


    import seaborn as sns
    sns.clustermap(data)
    plt.show()
    quit()

#checkRelated()
#quit()


def predictPCA():


    X = loadnpz('./data/plant/processed/sor/X.npz')


    from sklearn.decomposition import PCA
    pca = PCA(n_components=20)
    X = pca.fit_transform(X)

    np.savez_compressed('./data/plant/syntheticTraits/sor_PCA_1.npz', X)


#predictPCA()
#quit()



def predictSingleWave():


    X = loadnpz('./data/plant/processed/sor/X.npz')

    np.random.seed(0)
    perm1 = np.random.permutation(X.shape[1])[:20]
    X = X[:, perm1]

    X_copy = np.copy(X)


    X = normalizeIndependent(torch.tensor(X).float()).data.numpy()

    #print (scipy.stats.pearsonr(X[:, 0], X_copy[:, 0]))
    #print (scipy.stats.pearsonr(X[:, 1], X_copy[:, 1]))
    #quit()

    



    np.savez_compressed('./data/plant/syntheticTraits/sor_singleWave_2.npz', X)


#predictSingleWave()
#quit()






def showMultiplePhenotypes():

    
    
    if True:
        trainHerit = [0.74023,    0.5400625,  0.54466534, 0.49058506, 0.3577672,  0.3960388,
                     0.4131205,  0.3169396,  0.14191557, 0.2510371 ]
        testHerit = [0.71502084, 0.5777815,  0.44731605, 0.4697126,  0.40201804, 0.33444914, 
                     0.41091684, 0.35417044, 0.25673062, 0.20181043]
        #testHerit = testHerit[:10]
        plt.plot( np.arange(len(testHerit))+1,  trainHerit)
        plt.plot( np.arange(len(testHerit))+1,  testHerit)
        plt.xlabel('Phenotype number')
        plt.ylabel('Heritability')
        plt.legend(['test set heritability', 'train set heritability'])
        plt.show()
        quit()


    modelName = './data/models/1.pt'
    model = torch.load(modelName)

    coef = getModelCoef(model, multi=True)
    

    '''
    for a in range(coef.shape[0]):
        coef[a] = coef[a] - np.mean(coef[a])
        coef[a] = coef[a] / (np.mean(coef[a] ** 2) ** 0.5)


    cor1 = np.matmul(coef[:3], coef[:3].T)

    print (cor1)    

    for a in range(1, coef.shape[0]):
        for b in range(0, a):
            cor1 = np.mean(coef[a] * coef[b])
            coef[a] = coef[a] - (cor1 * coef[b])
            coef[a] = coef[a] / (np.mean(coef[a] ** 2) ** 0.5)
    '''
    

    #quit()

    

    #print (coef.shape)
    #quit()
        
    delta1 = 0.1
    x1 = np.arange(coef[0].shape[0])+350
    #plt.plot(x1, coef[0])
    #plt.plot(x1, coef[1], c='orange')
    plt.plot(x1, coef[2], c='green')
    #plt.plot(x1, coef[1])
    #plt.plot(x1, coef[2])
    #plt.plot(x1, coef[3])
    #plt.plot(x1, coef[4])
    #plt.plot(x1, coef[5])
    plt.xlabel('wavelength')
    plt.ylabel('coefficient')
    plt.gcf().set_size_inches(8, 2.5)
    plt.tight_layout()
    plt.xscale('log')
    plt.show()


#showMultiplePhenotypes()
#quit()
    

def analyzeMultiplePhenotypes():

    modelName = './data/models/H_1.pt'
    model = torch.load(modelName)

    #coef = getModelCoef(model, multi=True)

    #print (coef.shape)


    corMaxs = []
    traitNames = ['N', 'SLA', 'PN', 'PS']
    for b in range(4):
        modTrait = traitNames[b]
        corMatrix = loadnpz('./data/pairWave/fastHerit/matrix_covCor_'  + modTrait + '.npz')
        corMatrix = np.abs(corMatrix)
        corMatrix[np.isnan(corMatrix)] = 0.0
        corMaxs.append(np.max(corMatrix))





    if True:

        X = loadnpz('./data/processed/sor/X.npz')
        names = loadnpz('./data/processed/sor/names.npz')
        envirement = loadnpz('./data/processed/sor/set1.npz')
        traitAll = loadnpz('./data/processed/sor/traits.npz')


        Nitrogen = traitAll[:, 0]
        SLA = traitAll[:, 2]
        PN = traitAll[:, 3]
        PS = traitAll[:, 5]

        trait1 = traitAll[:, np.array([0, 2, 3, 5])]

        #trait1 = SLA
        #trait1 = Nitrogen
        #trait1 = PS

        for a in range(trait1.shape[1]):
            #if trait1[a, np.isnan(trait1[a].shape[0] > 0:
            #print (trait1[np.isnan(trait1[:, a]) == False, a].shape)
            mean1 = np.mean(trait1[np.isnan(trait1[:, a]) == False, a])
            trait1[np.isnan(trait1[:, a]), a] = mean1



    #print (trait1.shape)
    #print (X.shape)
    #quit()
    


    #trait1 = trait1 - np.mean(trait1)
    #trait1 = trait1 / (np.mean(trait1 ** 2.0) ** 0.5)
    #trait1 = trait1.reshape((-1, 1))

    trait1 = trait1 - np.mean(trait1, axis=0).reshape((1, -1))
    trait1 = trait1 / (np.mean(trait1 ** 2, axis=0) ** 0.5).reshape((1, -1))





    geneticVar_N, totalVar_N = batch_cheapHeritability(trait1, names, envirement , returnVariance=True )


    X = torch.tensor(X).float()
    Y = model(X)

    Y = normalizeIndependent(Y)

    Y = Y.data.numpy()


    #Y = np.copy(X[:, a:a+1] / X[:, arange2])
    Y = Y - np.mean(Y, axis=0).reshape((1, -1))
    Y = Y / (np.mean(Y ** 2, axis=0) ** 0.5).reshape((1, -1))

    


    geneticVar_Y, totalVar_Y = batch_cheapHeritability(Y, names, envirement , returnVariance=True )
    herit = geneticVar_Y / totalVar_Y



    colors = ['blue', 'orange', 'green', 'red']
    for b in range(trait1.shape[1]):
        #print (herit)

        geneticVar_NY, totalVar_NY = batch_cheapHeritability( Y + trait1[:, b:b+1], names, envirement , returnVariance=True )
        covarience = geneticVar_NY - geneticVar_Y - geneticVar_N[b]
        #print (geneticVar_NY[0], geneticVar_Y[0], geneticVar_N)
        
        cor1 = covarience / 2
        
        #print (cor1)
        cor1 = np.abs(cor1)

        plt.plot(cor1, c=colors[b])

    for b in range(4):
        plt.plot( (cor1 * 0) + corMaxs[b] , c=colors[b], linestyle='dashed' )

    
    
    print (corMaxs)
    plt.legend(['N', 'SLA', 'PN', 'PS'])
    plt.xlabel("phenotype number")
    plt.ylabel('coheritability')
    plt.show()




#analyzeMultiplePhenotypes()
#quit()







def otherPlantTrain():





    if False:

        X = loadnpz('./data/processed/setaria/X.npz')
        names = loadnpz('./data/processed/setaria/names.npz')
        machine1 = loadnpz('./data/processed/setaria/machine1.npz')
        project1 = loadnpz('./data/processed/setaria/project1.npz')

        _, names_inverse, counts = np.unique(names, return_inverse=True, return_counts=True)
        counts = counts[names_inverse]
        argGood = np.argwhere(counts != 1)[:, 0]
        X = X[argGood]
        names = names[argGood]
        machine1 = machine1[argGood]
        project1 = project1[argGood]

        envirement = np.array([machine1, project1]).T


    if True:

        X = loadnpz('./data/processed/yendrek/X.npz')
        names = loadnpz('./data/processed/yendrek/names.npz')

        #print (names)
        #quit()
        year1 = loadnpz('./data/processed/yendrek/year1.npz')
        ozone1 = loadnpz('./data/processed/yendrek/ozone1.npz')


        #_, names_inverse, counts = np.unique(names, return_inverse=True, return_counts=True)
        #print (np.unique(counts, return_counts=True))

        argGood = np.argwhere(np.isnan(X[:, 0]) == False)[:, 0]
        X = X[argGood]
        names = names[argGood]
        year1 = year1[argGood]
        ozone1 = ozone1[argGood]


        #print (np.argwhere(np.isnan(X)))
        #quit()



        #envirement = np.array([year1, ozone1]).T
        envirement = year1

        #quit()



    if False:


        X = loadnpz('./data/processed/west/X.npz')
        names = loadnpz('./data/processed/west/names.npz')
        envirement = loadnpz('./data/processed/west/field1.npz')



        _, names_inverse, counts = np.unique(names, return_inverse=True, return_counts=True)
        counts = counts[names_inverse]
        argGood = np.argwhere(counts != 1)[:, 0]
        X = X[argGood]
        names = names[argGood]
        envirement = envirement[argGood]

    name_unique, name_inverse = np.unique(names, return_inverse=True)

    trainTest3 = np.random.randint(3, size=name_unique.shape[0])
    trainTest3[trainTest3!=1] = 0
    trainTest3 = trainTest3[name_inverse]

    print (np.unique(trainTest3))


    modelName = './data/models/setaria_1.pt'
    learningRate = 1e-4
    regScale = 5e-8
    Niter = 10000
    (heritability_train, heritability_test) = trainModel(X, names, envirement, trainTest3, modelName, Niter=Niter, doPrint=True, regScale=regScale, learningRate=learningRate)


#otherPlantTrain()
#quit()


def plotAccuracy():






    regScales = [1e-9, 1e-8, 5e-8, 1e-7, 2e-7]


    heritVals = loadnpz('./data/results/crossFoldHerit.npz')


    #plt.plot(heritVals[:, 0, 0])
    #plt.show()

    print (np.mean(heritVals[:, :, 0], axis=1))
    print (np.mean(heritVals[:, :, 1], axis=1))

    plt.plot(regScales, np.mean(heritVals[:, :, 0], axis=1))
    plt.plot(regScales, np.mean(heritVals[:, :, 1], axis=1))
    plt.plot(regScales, 0.3774 + np.zeros(len(regScales)))
    plt.scatter(regScales, np.mean(heritVals[:, :, 0], axis=1))
    plt.scatter(regScales, np.mean(heritVals[:, :, 1], axis=1))
    plt.xscale('log')
    plt.xlabel("regularization")
    plt.ylabel('heritability')
    plt.legend(['training set heritability', 'test set heritability', 'wavelength pair baseline'])
    plt.savefig('./images/regularizationLoss.pdf')
    plt.show()


    quit()


    X = loadnpz('./data/processed/X.npz')
    names = loadnpz('./data/processed/names.npz')
    X = torch.tensor(X).float()

    name_unique, name_inverse = np.unique(names, return_inverse=True)

    trainTest1 = loadnpz('./data/processed/trainSplit3.npz')
    trainTest2 = trainTest1[name_inverse]

    modelNames = []
    modelNames.append('./data/models/1e9_0.pt')
    modelNames.append('./data/models/1e8_0.pt')
    #modelNames.append('./data/models/1e8_1.pt')
    #modelNames.append('./data/models/1e7_0.pt')
    modelNames.append('./data/models/15.pt')
    #modelNames.append('./data/models/1e7_1.pt')
    modelNames.append('./data/models/2e7_0.pt')
    #modelNames.append('./data/models/2e7_1.pt')


    train_acc = []
    test_acc = []

    regScales = [1e-9, 1e-8, 5e-8, 1e-7, 2e-7]

    for modelName in modelNames:

        model = torch.load(modelName)
        Y = model(X)
        Y = Y[:, 0]

        heritability_train = cheapHeritability(Y[trainTest2 == 0], names[trainTest2 == 0])
        heritability_test = cheapHeritability(Y[trainTest2 == 1], names[trainTest2 == 1])

        train_acc.append(heritability_train.data.numpy())
        test_acc.append(heritability_test.data.numpy())

    plt.plot(reglist, train_acc)
    plt.plot(reglist, test_acc)
    plt.plot(reglist, 0.625 + np.zeros(len(train_acc)))
    plt.xscale('log')
    plt.xlabel("regularization")
    plt.ylabel('heritability')
    plt.legend(['training set heritability', 'test set heritability', 'wavelength pair baseline'])
    plt.savefig('./images/regularizationLoss.pdf')
    plt.show()


#plotAccuracy()
#quit()





def plotCoefSim():

    import scipy.stats



    X = loadnpz('./data/processed/X.npz')
    #names = loadnpz('./data/processed/names.npz')
    X = torch.tensor(X).float()


    #modelNames1 = []
    #modelNames2 = []

    #regScales = [1e-9, 1e-8, 3e-8, 1e-7]

    regScales = [1e-9, 1e-8, 5e-8, 1e-7, 2e-7]
    #regScales = [1e-9, 1e-8, 3e-8, 1e-7, 2e-7, 5e-7, 1e-6]




    #for a in range()

    #modelNames1.append('./data/models/reg_1e-08_fold_0.pt')
    #modelNames1.append('./data/models/1e8_0.pt')
    #modelNames1.append('./data/models/1e7_0.pt')
    #modelNames1.append('./data/models/2e7_0.pt')


    #modelNames2.append('./data/models/1e9_1.pt')
    #modelNames2.append('./data/models/1e8_1.pt')
    #modelNames2.append('./data/models/1e7_all.pt')
    #modelNames2.append('./data/models/2e7_1.pt')

    #plotNames = ['1e9', '1e8', '1e7', '2e7']

    corList = []
    phenCorList = []


    for a in range(len(regScales)):

        modelName1 = './data/models/folds/L5_reg_' + str(regScales[a]) + '_fold_' + str(0) + '.pt'
        modelName2 = './data/models/folds/L5_reg_' + str(regScales[a]) + '_fold_' + str(1) + '.pt'
        modelName3 = './data/models/folds/L5_reg_' + str(regScales[a]) + '_fold_' + str(2) + '.pt'

        model1 = torch.load(modelName1)
        model2 = torch.load(modelName2)
        model3 = torch.load(modelName3)
        coef1 = getModelCoef(model1)
        coef2 = getModelCoef(model2)
        coef3 = getModelCoef(model3)

        (cor_12, pval) = scipy.stats.pearsonr(coef1, coef2)
        (cor_13, pval) = scipy.stats.pearsonr(coef1, coef3)
        (cor_23, pval) = scipy.stats.pearsonr(coef2, coef3)

        Y1 = model1(X)[:, 0].data.numpy()
        Y2 = model2(X)[:, 0].data.numpy()
        Y3 = model3(X)[:, 0].data.numpy()

        (cor_phen_12, pval_phen) = scipy.stats.pearsonr(Y1, Y2)
        (cor_phen_13, pval_phen) = scipy.stats.pearsonr(Y1, Y3)
        (cor_phen_23, pval_phen) = scipy.stats.pearsonr(Y2, Y3)

        cor = (abs(cor_12) + abs(cor_13) + abs(cor_23)) / 3
        cor_phen = (abs(cor_phen_12) + abs(cor_phen_13) + abs(cor_phen_23)) / 3



        print(abs(cor))
        print(abs(cor_phen))

        plotName = str(regScales[a])
        plotName = './images/similarity_' + plotName + '.pdf'

        if True:#a == 2:

            plt.plot(np.arange(2501-350)+350, coef1)
            plt.plot(np.arange(2501-350)+350, coef2 * np.sign(cor_12))
            plt.plot(np.arange(2501-350)+350, coef3 * np.sign(cor_13))
            plt.ylabel('coefficient')
            plt.xlabel('wavelength')
            plt.savefig(plotName)
            #plt.legend(['on training set', 'on full dataset'])
            plt.savefig('./images/similarity_all.pdf')
            plt.show()

        corList.append(abs(cor))
        phenCorList.append(abs(cor_phen))




    #reglist = [1e-9, 1e-8, 1e-7, 2e-7]

    plt.plot(regScales, corList)
    plt.plot(regScales, phenCorList)
    plt.scatter(regScales, corList)
    plt.scatter(regScales, phenCorList)
    plt.xlabel('regularization')
    plt.ylabel('pearson correlation')
    plt.legend(['coefficient correlation', 'phenotype correlation'])
    plt.xscale('log')
    plt.savefig('./images/correlation.pdf')
    plt.show()

    quit()

#plotCoefSim()
#quit()




def phenotypeAnalysis():

    X = loadnpz('./data/processed/X.npz')
    names = loadnpz('./data/processed/names.npz')
    X = torch.tensor(X).float()

    #model = torch.load('./data/models/1e7_0.pt')
    #model = torch.load('./data/models/1e9_0.pt')
    model = torch.load('./data/models/1.pt')

    Y = model(X)[:, 0].data.numpy()


    names_unique, names_inverse, counts = np.unique(names, return_inverse=True, return_counts=True)


    envirement = loadnpz('./data/processed/set1.npz')


    #names_unique = names_unique[counts >= 64]
    #counts = counts[counts >= 64]

    names_unique = names_unique[counts <= 4]
    counts = counts[counts <= 4]

    Y = Y - np.mean(Y)

    #plt.hist(Y, bins=100)
    #plt.show()

    #plt.hist(Y[np.isin(names, names_unique) == False], bins=100)
    #plt.show()


    #quit()

    #Y[Y >= 0] = 1
    #Y[Y < 0] = 0


    #print (np.unique(setVar[Y==0], return_counts=True))
    #print (np.unique(setVar[Y==1], return_counts=True))
    #quit()

    max1 = np.max(Y)
    min1 = np.min(Y)


    #plt.hist(Y[counts[names_inverse]==4], bins=100, range=(min1, max1), density=True, alpha=0.5)
    #plt.hist(Y[counts[names_inverse]!=4], bins=100, range=(min1, max1), density=True, alpha=0.5)
    #plt.show()
    #quit()



    print (cheapHeritability(torch.tensor(Y[np.isin(names, names_unique)]   ).float(), names[np.isin(names, names_unique)], envirement[np.isin(names, names_unique)]   ))
    print (cheapHeritability(torch.tensor(Y[np.isin(names, names_unique)==False]   ).float(), names[np.isin(names, names_unique)==False] , envirement[np.isin(names, names_unique)==False]  ))
    quit()

    means = []
    stds = []

    #counter = np.zeros(5)
    counter = np.zeros(70)

    for a in range(len(names_unique)):
        #plt.hist(Y[names==names_unique[a]],  bins=100, range=(min1, max1))

        ar1 = Y[names==names_unique[a]]

        num1 = int(np.sum(ar1))
        counter[num1] += 1
        #mean1 = np.mean(ar1)
        #std1 = np.mean( (ar1 - mean1) ** 2  ) ** 0.5

        #print (mean1, std1)

        #means.append(mean1)
        #stds.append(std1)


    print (counter)
    quit()

    min2, max2 = np.min(means), np.max(means)
    plt.hist(means, bins=100, range=(min2, max2))
    plt.hist(stds, bins=100, range=(min2, max2), alpha=0.5)
    plt.xlabel('phenotype')
    plt.show()




#phenotypeAnalysis()
#quit()
    

def plotMultiPhenotype():

    trainHerit = [0.75053406, 0.5892931,  0.5784627,  0.48287487, 0.4428065,  0.41483742, 0.35569015, 0.30058062, 0.277605,   0.23101078] 
    testHerit = [0.6718362,  0.32926393, 0.43835434, 0.36878222, 0.3576192,  0.31716603, 0.31549278, 0.1479807,  0.1510361,  0.11493391]


    plt.plot(np.arange(10)+1, trainHerit)
    plt.plot(np.arange(10)+1, testHerit)
    plt.legend(['Training set heritability', 'Test set heritability'])
    plt.xlabel('Phenotype')
    plt.ylabel("Heritability")
    plt.gcf().set_size_inches(8, 4)
    plt.tight_layout()    
    plt.show()
    quit()

    model1 = torch.load('./data/models/H_1.pt')
    coef1 = getModelCoef(model1, multi=True)    

    plt.plot(np.arange(2501-350)+350,  coef1[:3].T)
    plt.xlabel('wavelength')
    plt.ylabel('coefficient')
    plt.gcf().set_size_inches(8, 4)
    plt.tight_layout()    
    plt.show()

    quit()

#plotMultiPhenotype()
#quit()



def plotSparseModel():

    #7, 8 #1e-7
    #9, 10 #2e-7
    #11, 12 #1e-8

    #a = 0
    #model1 = torch.load('./data/models/folds/L5_reg_5e-08_fold_0.pt')
    #model1 = torch.load('./data/models/setaria_1.pt')

    traitName = 'SLA'

    model1 = torch.load('./data/models/' + traitName + '_1.pt')
    coef1 = getModelCoef(model1)


    plt.plot(np.arange(2501-350)+350,  coef1)
    plt.xlabel('wavelength')
    plt.ylabel('coefficient')
    plt.gcf().set_size_inches(8, 4)
    plt.tight_layout()

    plt.savefig('./images/coef_' + traitName + '.png')
    #plt.savefig('./images/coefSingle.pdf')
    
    plt.show()

    quit()

#plotSparseModel()
#quit()




def saveANOVAherit():

    X = loadnpz('./data/plant/processed/sor/X.npz')
    names = loadnpz('./data/plant/processed/sor/names.npz')
    envirement = loadnpz('./data/plant/processed/sor/set1.npz')
    #traitAll = loadnpz('./data/plant/processed/sor/traits.npz')
    X = torch.tensor(X).float()
    
    heritList =  cheapHeritability(X, names, envirement  )



#saveANOVAherit()
#quit()


def findRherit(Y, subset2):


    data = pd.read_csv("./data/Phenotypes_N.txt", sep="\t")


    bool1 = np.zeros(len(data.index) )
    bool1[subset2] = 1

    data = data[bool1 == 1]

    #Y2 = np.zeros(data.to_numpy().shape[0])
    #Y2[subset2] = Y

    #data['pheno'] = Y2

    data['pheno'] = Y

    #print (len(data.index))
    #quit()

    #data.to_csv("./data/Phenotypes_Ratio.txt", sep='\t')
    data.to_csv("./data//temp/Phenotypes.txt", sep='\t')

    os.system('Rscript called.r')


    output = pd.read_csv("./data/temp/output.csv", sep="\t")
    output = output.to_numpy()[0][0]
    output = output.split(',')[1]
    output = float(output)

    return output




def savePhenoModel():


    #model = torch.load('./data/models/1.pt')
    model = torch.load('./data/models/folds/L5_reg_1e-07_fold_0.pt')
    #model = torch.load('./data/models/folds/L5_reg_1e-08_fold_0.pt')

    #X, names, subset2 = loadData()

    X = loadnpz('./data/processed/X.npz')
    names = loadnpz('./data/processed/names.npz')


    _, inverse1, count1 = np.unique(names, return_inverse=True, return_counts=True)
    count2 = count1[inverse1]
    #print (inverse1.shape)
    #print (count1.shape)
    #quit()

    X = torch.tensor(X).float()
    Y = model(X)[:, 0].data.numpy()

    matrix1 = loadnpz('./data/pairWave/originalHerit/matrix.npz')
    max1 = np.max(matrix1)
    #heritMatrix = loadnpz('./data/pairWave/fastHerit/matrix.npz')
    #heritMatrix[heritMatrix < 0.53] = 0


    #wavePair1 = np.argwhere(matrix1 == max1)[0]

    #print (wavePair1 + 350)
    #quit()

    #Y = X[:, wavePair1[0]] / X[:, wavePair1[1]]


    #argMany = np.argwhere(count2 >= 64)[:, 0]
    #argMany = np.argwhere(count2 >= -1)[:, 0]
    #argFew = np.argwhere(count2 == 4)[:, 0]
    #perm1 = np.random.permutation(argMany.shape[0])
    #perm2 = np.random.permutation(argFew.shape[0])[:argMany.shape[0]]
    #Y[argMany] = Y[argMany[perm1]]
    #Y[argMany] = Y[argFew[perm2]]



    #subset1 = np.argwhere(count1[inverse1] != 4)[:, 0]
    #subset1 = np.argwhere(count1[inverse1] == 4)[:, 0]

    #subset1 = np.argwhere(count1[inverse1] != -1)[:, 0]

    #X = X[subset1]
    #subset2 = subset2[subset1]







    #plt.imshow(heritMatrix)
    #plt.show()
    #quit()

    #heritMatrix[np.arange(heritMatrix.shape[0]), np.arange(heritMatrix.shape[0])] = 0


    #max2 = np.max(heritMatrix)

    #print (max1, max2)

    #
    #wavePair2 = np.argwhere(heritMatrix == max2)[0]

    #print (wavePair1)
    #print (wavePair2)
    #quit()

    #Y = X[:, wavePair1[0]] / X[:, wavePair1[1]]
    #Y = X[:, 185] / X[:, 353]
    output1 = findRherit(Y, np.arange(Y.shape[0]))

    print (output1)

    quit()

    #Y = X[:, wavePair2[0]] / X[:, wavePair2[1]]
    #output2 = findRherit(Y, subset2)

    #print (output1, output2)
    #quit()


    #randomList = []
    #for a in range(10):
    #    Y = np.random.random(X.shape[0])
    #    output1 = findRherit(Y, subset2)
    #    randomList.append(output1)


    #M = 2501 - 350
    #randomList = []
    #for a in range(10):
    #    r1, r2 = np.random.randint(M), np.random.randint(M)
    #    Y = X[:, r1] / X[:, r2]
    #    output1 = findRherit(Y, subset2)
    #    randomList.append(output1)

    #print ('randomList')
    #print (randomList)
    #quit()

    #Y = X[:, wavePair2[0]] / X[:, wavePair2[1]]
    #Y = X[:, 1400] / X[:, 1800]
    #output2 = findRherit(Y, subset2)

    #print ('outputs')
    #print (output1, output2)

    quit()

    data = pd.read_csv("./data/Phenotypes_N.txt", sep="\t")


    Y2 = np.zeros(data.to_numpy().shape[0])
    Y2[subset2] = Y.data.numpy()

    data['pheno'] = Y2

    data.to_csv("./data/Phenotypes_Mine.txt", sep='\t')


#savePhenoModel()
#quit()






def checkHeritOnMany():


    #X, names, subset2 = loadData()

    X = loadnpz('./data/processed/X.npz')
    names = loadnpz('./data/processed/names.npz')
    envirement = loadnpz('./data/processed/set1.npz')



    _, inverse1, count1 = np.unique(names, return_inverse=True, return_counts=True)


    subset1 = np.argwhere(count1[inverse1] != 4)[:, 0]
    #subset1 = np.argwhere(count1[inverse1] == 4)[:, 0]

    #subset1 = np.argwhere(count1[inverse1] != -1)[:, 0]

    #X = X[subset1]
    #subset2 = subset2[subset1]
    #names = names[subset1]



    matrix1 = loadnpz('./data/pairWave/originalHerit/matrix.npz')

    max1 = np.max(matrix1)

    wavePair1 = np.argwhere(matrix1 == max1)[0]

    Y = X[:, wavePair1[0]] / X[:, wavePair1[1]]


    Y = Y - np.mean(Y)
    min1, max1 = np.min(Y), np.max(Y)

    names_unique = np.unique(names)


    print (cheapHeritability(torch.tensor(Y ).float(), names, envirement))

    quit()

    #for a in range(len(names_unique)):

        #plt.hist(Y[names == names_unique[a]], bins=100, range=(min1, max1))
    #Eplt.show()

    means = []
    stds = []

    #counter = np.zeros(5)
    #counter = np.zeros(70)

    for a in range(len(names_unique)):
        #plt.hist(Y[names==names_unique[a]],  bins=100, range=(min1, max1))

        ar1 = Y[names==names_unique[a]]

        mean1 = np.mean(ar1)
        std1 = np.mean( (ar1 - mean1) ** 2  ) ** 0.5

        print (mean1, std1)

        means.append(mean1)
        stds.append(std1)


    quit()



#checkHeritOnMany()
#quit()



def savePhenoRatio():

    model = torch.load('./data/models/3.pt')

    X, names, subset2 = loadData()

    _, inverse1 = np.unique(names, return_inverse=True)
    _, count1 = np.unique(inverse1, return_counts=True)

    count2 = count1[inverse1]

    #print (np.unique(count1, return_counts=True))
    #quit()

    #subset1 = np.argwhere(count2 == 4)[:, 0]
    #subset1 = np.argwhere(count2 > 10)[:, 0]

    #print (subset1.shape)


    #185 353

    Y = X[:, 185] / X[:, 353]
    #Y = X[:, 399] / X[:, 400]
    Y = Y - np.mean(Y)
    Y = Y / np.mean(np.abs(Y))


    #Y = Y[subset1]
    #subset2 = subset2[subset1]


    output = findRherit(Y, subset2)

    print (output)

    quit()

    data = pd.read_csv("./data/Phenotypes_N.txt", sep="\t")


    Y2 = np.zeros(data.to_numpy().shape[0])
    Y2[subset2] = Y#.data.numpy()

    data['pheno'] = Y2

    #data.to_csv("./data/Phenotypes_Ratio.txt", sep='\t')
    data.to_csv("./data/Phenotypes_Ratio2.txt", sep='\t')


#savePhenoRatio()
#quit()




def generateIndependentGeneSim():


    X = loadnpz('./data/processed/X.npz')
    names = loadnpz('./data/processed/names.npz')


    mean1 = np.mean(X, axis=0)
    measuredNoise = X - mean1.reshape((1, -1))


    #measuredNoise_recon = ifft(fft(measuredNoise))
    #print (np.mean(np.abs( measuredNoise_recon - measuredNoise  )))
    #quit()

    noiseFFT = fft(measuredNoise, axis=1)
    noiseFFT = np.abs(noiseFFT)
    noiseFFT = np.mean(noiseFFT ** 2, axis=0) ** 0.5




    #print (noiseFFT.shape)
    #quit()

    #np.fft.fft(np.exp(2j * np.pi * np.arange(8) / 8))

    #plt.plot(noiseFFT[0])
    #plt.show()
    #quit()


    #std1 = np.mean(  (measuredNoise) ** 2.0   , axis=0) ** 0.5
    #std1_mean = np.mean(std1)

    noise1 = np.random.normal(size=X.shape)
    #noise1 = noise1 * std1.reshape((1, -1))
    noise1 = noise1 * noiseFFT.reshape((1, -1))
    noise1 = ifft(noise1, axis=1)
    noise1 = np.real(noise1)


    noise_mean = np.mean(noise1 ** 2) ** 0.5


    X2 = mean1.reshape((1, -1)) + (noise1 * 1)


    names_unique, names_inverse = np.unique(names, return_inverse=True)


    gene1 = np.random.randint(2, size=len(names_unique))
    gene2 = np.random.randint(2, size=len(names_unique))
    gene1 = (gene1 * 2) - 1
    gene2 = (gene2 * 2) - 1

    gene1 = gene1 * 2 * noise_mean
    gene2 = gene2 * 2 * noise_mean

    gene1 = gene1[names_inverse]
    gene2 = gene2[names_inverse]

    genes = np.array([gene1, gene2]).T

    waveperm = np.random.permutation(X.shape[1])
    wavelength1 = waveperm[0]
    wavelength2 = waveperm[1]

    #X2 = X2 * 0
    #X2[:] = 1

    X2[:, wavelength1] = X2[:, wavelength1] + gene1
    X2[:, wavelength2] = X2[:, wavelength2] + gene2

    #plt.plot(gene1)
    #plt.show()
    #quit()

    #X2 = X2 + gene1.reshape((-1, 1))

    #plt.imshow(X2)
    #plt.show()

    #intensities = np.array([std1_mean, std1_mean])

    folder1 = './data/simulation/indepSim1/5/'

    #np.savez_compressed(folder1 + 'intensity.npz', intensities)
    np.savez_compressed(folder1 + 'wavelength.npz', waveperm[:2])
    np.savez_compressed(folder1 + 'genes.npz', genes)
    np.savez_compressed(folder1 + 'X.npz', X2)




#generateIndependentGeneSim()
#quit()


def generateMultiSim():


    X = loadnpz('./data/processed/X.npz')
    names = loadnpz('./data/processed/names.npz')

    #mean1 = np.mean(X, axis=0)
    #std1 = np.mean(  (X - mean1.reshape((1, -1))) ** 2.0   , axis=0) ** 0.5
    #std1_mean = np.mean(std1)
    #noise1 = np.random.normal(size=X.shape)
    #noise1 = noise1 * std1.reshape((1, -1))

    mean1 = np.mean(X, axis=0)
    measuredNoise = X - mean1.reshape((1, -1))

    noiseFFT = fft(measuredNoise, axis=1)
    noiseFFT = np.mean(noiseFFT ** 2, axis=0) ** 0.5
    noise1 = np.random.normal(size=X.shape)
    noise1 = noise1 * noiseFFT.reshape((1, -1))
    noise1 = ifft(noise1, axis=1)
    noise1 = np.real(noise1)

    X2 = mean1.reshape((1, -1)) + (noise1 * 1)

    #print (std1_mean)
    #quit()


    names_unique, names_inverse = np.unique(names, return_inverse=True)


    gene1 = np.random.randint(2, size=len(names_unique))
    gene1 = (gene1 * 2) - 1


    gene1 = gene1 * 0.05

    gene1 = gene1[names_inverse]


    waveperm = np.random.permutation(X.shape[1])


    #X2 = X2 * 0
    #X2[:] = 1

    X2[:, waveperm[0]] = X2[:, waveperm[0]] + gene1
    X2[:, waveperm[1]] = X2[:, waveperm[1]] + gene1
    X2[:, waveperm[2]] = X2[:, waveperm[2]] + gene1
    X2[:, waveperm[3]] = X2[:, waveperm[3]] + gene1
    X2[:, waveperm[4]] = X2[:, waveperm[4]] + gene1

    #plt.plot(gene1)
    #plt.show()
    #quit()

    #X2 = X2 + gene1.reshape((-1, 1))

    #plt.imshow(X2)
    #plt.show()

    #intensities = np.array([std1_mean, std1_mean])

    folder1 = './data/simulation/multiSim1/5/'

    #np.savez_compressed(folder1 + 'intensity.npz', intensities)
    np.savez_compressed(folder1 + 'wavelength.npz', waveperm[:5])
    np.savez_compressed(folder1 + 'genes.npz', gene1)
    np.savez_compressed(folder1 + 'X.npz', X2)



#generateMultiSim()
#quit()


def generateSharedWavelengthSim():

    X = loadnpz('./data/processed/X.npz')
    names = loadnpz('./data/processed/names.npz')

    mean1 = np.mean(X, axis=0)
    measuredNoise = X - mean1.reshape((1, -1))

    noiseFFT = fft(measuredNoise, axis=1)
    noiseFFT = np.mean(noiseFFT ** 2, axis=0) ** 0.5
    noise1 = np.random.normal(size=X.shape)
    noise1 = noise1 * noiseFFT.reshape((1, -1))
    noise1 = ifft(noise1, axis=1)
    noise1 = np.real(noise1)

    

    X2 = mean1.reshape((1, -1)) + (noise1 * 1)

    

    #print (std1_mean)
    #quit()


    names_unique, names_inverse = np.unique(names, return_inverse=True)


    gene1 = np.random.randint(2, size=len(names_unique))
    gene1 = (gene1 * 2) - 1

    gene2 = np.random.randint(2, size=len(names_unique))
    gene2 = (gene2 * 2) - 1


    #gene1 = gene1 * 0.1
    #gene2 = gene2 * 0.1


    #gene1 = gene1 * 0.05
    #gene2 = gene2 * 0.05

    #gene1 = gene1 * 0.02 * 0.5
    #gene2 = gene2 * 0.01 * 0.5

    gene1 = gene1[names_inverse]
    gene2 = gene2[names_inverse]

    impact1 = gene1 * 0.01
    impact2 = gene2 * 0.01


    waveperm = np.random.permutation(X.shape[1])[:6]
    #waveperm = (np.arange(6)+0.5)*(X.shape[1]//7)
    #waveperm = waveperm[np.random.permutation(waveperm.shape[0])]
    

    '''
    masks1 = np.zeros((waveperm.shape[0], X.shape[1]))
    arange1 = np.arange(X.shape[1])
    width1 = 50
    for a in range(waveperm.shape[0]):
        masks1[a] = np.exp(  -1 * ((waveperm[a] - arange1) / width1) ** 2  )

    X2 += masks1[0].reshape((1, -1)) * impact1.reshape((-1, 1))
    X2 += masks1[1].reshape((1, -1)) * impact1.reshape((-1, 1))
    X2 += masks1[2].reshape((1, -1)) * impact1.reshape((-1, 1))
    X2 += masks1[3].reshape((1, -1)) * impact1.reshape((-1, 1))
    X2 += masks1[4].reshape((1, -1)) * impact1.reshape((-1, 1))
    X2 += masks1[5].reshape((1, -1)) * impact1.reshape((-1, 1))

    X2 += masks1[0].reshape((1, -1)) * impact2.reshape((-1, 1))
    X2 += -1 * masks1[1].reshape((1, -1)) * impact2.reshape((-1, 1))
    X2 += masks1[2].reshape((1, -1)) * impact2.reshape((-1, 1))
    X2 += -1 * masks1[3].reshape((1, -1)) * impact2.reshape((-1, 1))
    X2 += masks1[4].reshape((1, -1)) * impact2.reshape((-1, 1))
    X2 += -1 * masks1[5].reshape((1, -1)) * impact2.reshape((-1, 1))
    '''

    X2[:, waveperm] = X2[:, waveperm] + impact1.reshape((-1, 1))

    X2[:, waveperm[0::2]] = X2[:, waveperm[0::2]] + impact2.reshape((-1, 1))
    X2[:, waveperm[1::2]] = X2[:, waveperm[1::2]] - impact2.reshape((-1, 1))

    X2 = X2 - np.mean(X2, axis=1).reshape((-1, 1))

    
    waveperm_used = waveperm[:6]

    #phenotype1 = np.zeros(X.shape[0])
    #phenotype2 = np.zeros(X.shape[0])
    #for a in range(waveperm.shape[0]):
    #    phenotype1 += np.sum(X2 * masks1[a].reshape((1, -1)), axis=1)
    #    #phenotype2 +=  (-1 ** a) * np.sum(X2 * masks1[a].reshape((1, -1)), axis=1)
    #    phenotype2 +=  (((a % 2) * 2)-1) * np.sum(X2 * masks1[a].reshape((1, -1)), axis=1)
    

    #phenotypes = np.array([phenotype1, phenotype2]).T

    

    #phenotypes[:, 0] = gene1
    #phenotypes[:, 1] = gene2

    #print (pearsonr(phenotypes[:, 0], gene1))
    #quit()

    #envirement = np.zeros((names.shape[0], 0))
    #originalHerit = batch_cheapHeritability(phenotypes, names, envirement)

    #print (originalHerit)
    #quit()

    

    genes = np.array([gene1, gene2]).T

    #intensities = np.array([std1_mean, std1_mean])

    folder1 = './data/simulation/sharedWave1/5/'

    #np.savez_compressed(folder1 + 'originalHerit.npz', originalHerit)
    #np.savez_compressed(folder1 + 'intensity.npz', intensities)
    np.savez_compressed(folder1 + 'wavelength.npz', waveperm[:6])
    np.savez_compressed(folder1 + 'genes.npz', genes)
    np.savez_compressed(folder1 + 'X.npz', X2)

    #print (genes.shape)


#generateSharedWavelengthSim()
#quit()



def simulationTrain():

    #regScale = 1e-7
    regScale = 1e-5
    #regScale = 1e-4
    #regScale = 1e-3
    #regScale = 2e-3

    #[ 0.9537, -0.0026]
    #[0.9758, 0.0133]
    #[0.9675, 0.0044]
    #[ 9.7332e-01, -3.2295e-04
    #[ 0.9601, -0.0237]

    #instance = '1'

    for instanceNum in range(5):
        instance = str(instanceNum+1)
    
        #folder2 = 'indepSim1'
        #folder2 = 'sharedWave1'
        folder2 = 'multiSim1'

        #folder1 = './data/simulation/indepSim1/5/'
        folder1 = './data/simulation/' + folder2 + '/' + instance + '/'
        #folder1 = './data/simulation/sharedWave1/1/'

        #modelName = './data/models/simulation/indepSim1/5.pt'
        modelName = './data/models/simulation/' + folder2  + '/' + instance  + '.pt'
        #modelName = './data/models/simulation/sharedWave1/1.pt'

        heritName = folder1 + 'herit.npz'

        #originalHerit = loadnpz(folder1 + 'originalHerit.npz')

        #print (originalHerit)

        X = loadnpz(folder1 + 'X.npz')

        #print (X.shape)
        #plt.plot(X[0])
        #plt.show()
        #quit()

        names = loadnpz('./data/processed/names.npz')
        envirement = np.zeros((names.shape[0], 0))

        #trainTest3 = np.zeros(names.shape[0], dtype=int)
        name_unique, name_inverse = np.unique(names, return_inverse=True)
        trainTest3 = np.random.randint(3, size=name_unique.shape[0])
        trainTest3[trainTest3!=0] = 1
        trainTest3 = trainTest3[name_inverse]

        


        #trainTest3 = np.copy(trainTest2)
        #trainTest3[trainTest3 == b] = 100
        #trainTest3[trainTest3!=100] = 0
        #trainTest3[trainTest3 == 100] = 1




        Niter = 5000
        #Niter = 1000
        #Niter = 10000
        #Niter = 100000
        (heritability_train, heritability_test) = trainModel(X, names, envirement, trainTest3, modelName, Niter=Niter, doPrint=True, regScale=regScale, Nphen=2)

        heritArray = np.array([heritability_train.data.numpy(), heritability_test.data.numpy()])

        #print (heritArray.shape)

        
        np.savez_compressed(heritName, heritArray)

        model = torch.load(modelName)

        coef = getModelCoef(model, multi=True)

        

        plt.plot(coef[0])
        plt.plot(coef[1])
        plt.show()


#simulationTrain()
#quit()


def testSimulation():


    
    corAll = []
    heritAll = []

    for instanceNum in range(5):
        instance = str(instanceNum + 1)

        #folder2 = 'indepSim1'
        #folder2 = 'sharedWave1'
        #folder2 = 'multiSim1'
        #modelName = './data/models/simulation/indepSim1/' + instance + '.pt'
        modelName = './data/models/simulation/' + folder2 + '/' + instance + '.pt'

        model = torch.load(modelName)
        coef = getModelCoef(model, multi=True)


        #plt.plot(np.arange(coef.shape[1])+350, coef[0])
        #plt.xlabel('wavelength')
        #plt.ylabel('coefficeint')
        #plt.show()
        #quit()

        #folder1 = './data/simulation/indepSim1/' + instance + '/'
        folder1 = './data/simulation/' + folder2 + '/' + instance + '/'

        waveperm = loadnpz(folder1 + 'wavelength.npz')

        genes = loadnpz(folder1 + 'genes.npz')

        #originalHerit = loadnpz(folder1 + 'herit.npz')
        herit = loadnpz(folder1 + 'herit.npz')

        X = loadnpz(folder1 + 'X.npz')
        pred = model(torch.tensor(X).float())
        pred = normalizeIndependent(pred).data.numpy()

        heritAll.append(herit)
        
        if len(genes.shape) == 1:
            genes = genes.reshape((-1, 1))

        corMatrix = np.zeros((2, genes.shape[1]))
        for a in range(2):
            for b in range(genes.shape[1]):
                corMatrix[a, b] = pearsonr(pred[:, a], genes[:, b])[0]

        corAll.append(corMatrix)

    heritAll = np.array(heritAll)
    corAll = np.array(corAll)
    #print (heritAll[:, 1])
    #print (corAll)

    if True:
        corAll_abs = np.abs(corAll)

        for a in range(5):
            plt.scatter(corAll_abs[a, :, 0], corAll_abs[a, :, 1])
        plt.legend(['instance 1', 'instance 2', 'instance 3', 'instance 4', 'instance 5'])
        plt.xlabel('correlation with gene 1 (absolute value)')
        plt.ylabel('correlation with gene 2 (absolute value)')
        plt.show()


#testSimulation()
#quit()






def checkSimpleSim():


    values_str = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']

    values = []

    pred1 = []

    for a in range(len(values_str)):

        str1 = values_str[a]

        values.append(float(str1))
    
        data = np.loadtxt('./data/simulation/simplePhen/plant3/Simulated_Data_10_Reps_Herit_' + str1 + '.txt', dtype=str)
        data = data[1:]

        names = data[:, 0]
        pheno = data[:, 1].astype(float)

        envirement = np.zeros(names.shape[0], dtype=int)

        pheno = pheno.reshape((-1, 1))


        geneticVar_NY, totalVar_NY = batch_cheapHeritability( pheno, names, envirement , returnVariance=True )

        herit =  geneticVar_NY / totalVar_NY

        pred1.append(herit)


    plt.plot(values, values)
    plt.plot(values, pred1)
    plt.scatter(values, values)
    plt.scatter(values, pred1)
    plt.legend(['ground truth', 'prediction'])
    plt.xlabel('true heritability')
    plt.ylabel('predicted heritability')
    plt.show()


#checkSimpleSim()
#quit()
    

def checkSimpleMultiSim():


    dataFull = np.zeros((0, 4), dtype=str)
    

    for trait1 in range(1, 11):

        str1 = str(trait1)

    
        #data = np.loadtxt('./data/simulation/simplePhen/plant2/Results_Pleiotropic/Phenotypes/Simulated_Data__Rep' + str1 + '_Herit_0.2_0.4_0.6.txt', dtype=str)
        data = np.loadtxt('./data/simulation/simplePhen/plant4/Results_Pleiotropic/Phenotypes/Simulated_Data__Rep' + str1 + '_Herit_0.2_0.4_0.6.txt', dtype=str)
        #
        data = data[1:]

        dataFull = np.concatenate((dataFull, data), axis=0)

        #print (data[:10])
        #quit()
    
    #print (dataFull.shape)

    names = dataFull[:, 0]
    phenoTypes = dataFull[:, 1:].astype(float)

    envirement = np.zeros(names.shape[0], dtype=int)

    geneticVar_NY, totalVar_NY = batch_cheapHeritability( phenoTypes, names, envirement , returnVariance=True )

    herit =  geneticVar_NY / totalVar_NY

    print (herit)

    values = [0.2, 0.4, 0.6]
    #values = [0.2, 0.4]

    plt.plot(values, values)
    plt.plot(values, herit)
    plt.scatter(values, values)
    plt.scatter(values, herit)
    plt.legend(['ground truth', 'prediction'])
    plt.xlabel('true heritability')
    plt.ylabel('predicted heritability')
    plt.title('Pleiotropy high dominance effect')
    plt.show()


    '''
    > test1 <-  create_phenotypes(
    geno_obj = SNP55K_maize282_maf04,
    add_QTN_num = 3,
    dom_QTN_num = 4,
    big_add_QTN_effect = c(0.1, 0.1, 0.1),
    h2 = c(0.2, 0.4, 0.6),
    add_effect = c(0.04,0.1,0.1),
    dom_effect = c(0.3,0.3,0.3),
    ntraits = 3,
    rep = 10,
    vary_QTN = FALSE,
    output_format = "multi-file",
    architecture = "pleiotropic",
    output_dir = "Results_Pleiotropic",
    to_r = TRUE,
    seed = 10,
    model = "AD",
    sim_method = "geometric",
    home_dir = '/Users/stefanivanovic/Desktop/Coding/Bio/plant/data/simulation/simplePhen/plant4'
    '''

    '''
    > create_phenotypes(
    geno_obj = SNP55K_maize282_maf04,
    add_QTN_num = 3,
    h2 = c(0.2, 0.4),
    add_effect = c(0.02, 0.05),
    rep = 5,
    seed = 200,
    output_format = "wide",
    architecture = "LD",
    output_dir = "Results_LD",
    out_geno = "BED",
    remove_QTN = TRUE,
    ld_max =0.8,
    ld_min =0.2,
    model = "A",
    ld_method = "composite",
    type_of_ld = "indirect",
    home_dir = '/Users/stefanivanovic/Desktop/Coding/Bio/plant/data/simulation/simplePhen/plant5'
    '''




#checkSimpleMultiSim()
#quit()


def checkSimpleFalse():

    data = np.loadtxt('./data/simulation/simplePhen/plant5/Results_LD/Simulated_Data_5_Reps_Herit_0.2_0.4.txt', dtype=str)

    data = data[1:]

    names = np.arange(data.shape[0]).astype(str)
    pheno1 = data[:, 1::2].astype(float)
    pheno2 = data[:, 2::2].astype(float)

    names = names.reshape((names.shape[0], 1))[:, np.zeros(pheno1.shape[1], dtype=int) ]

    names = names.reshape((-1,))
    pheno1 = pheno1.reshape((-1,))
    pheno2 = pheno2.reshape((-1,))



    envirement = np.zeros(names.shape[0], dtype=int)

    pheno = np.concatenate(( pheno1.reshape((-1, 1))  , pheno2.reshape((-1, 1)) ), axis=1)


    geneticVar_NY, totalVar_NY = batch_cheapHeritability( pheno, names, envirement , returnVariance=True )

    herit =  geneticVar_NY / totalVar_NY


    values = [0.2, 0.4]

    plt.plot(values, values)
    plt.plot(values, herit)
    plt.scatter(values, values)
    plt.scatter(values, herit)
    plt.legend(['ground truth', 'prediction'])
    plt.xlabel('true heritability')
    plt.ylabel('predicted heritability')
    plt.title('spurious pleiotropy')
    plt.show()


checkSimpleFalse()
quit()