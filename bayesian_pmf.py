import numpy as np
from scipy.stats import wishart, multivariate_normal

from hyperparam_sampler import *

# TODO
# We initialized the Gibbs sampler by setting the model parameters U and V to their 
# MAP estimates obtained by training a linear PMF model.
def init_UV(N: int, M: int, D: int):
    U = np.zeros((D, N), dtype=np.float32)
    V = np.zeros((D, M), dtype=np.float32)
    return U, V

# def 

def bayesian_PMF(
        R: np.ndarray,
        M: np.ndarray,
        R_test: np.ndarray,
        D: int,
        T: int,
        mu0=None,
        nu0=None,
        W0=None,
        beta0=None,
        seed=None,
        v=1,
    ):
    """
    R: rating matrix with dimension N*M
    M: mask matrix with dimension N*M
    D: latent matrix dimension
    T: #steps for gibbs sampling
    \Theta_0
        mu0: hyperparameter for the hyper-prior of mu_U and mu_V
        nu0: degree of freedom for Wishart Distribution
        W0: scale matrix for Wishart Distribution
    beta0: strength of prior belief in the mean \mu
    """
    (N, M) = R.shape
    _U, _V = init_UV(N, M, D)
    W0_inv = np.linalg.inv(W0)

    for t in range(T):
        ## sample from Gaussian-Wishart priors
        W0u = update_W0(W0_inv, _U, mu0, beta0)
        mu0u = update_mu0(mu0, beta0, _U)
        nu0u = update_nu0(nu0, N)
        beta0u = update_beta0(beta0, N)
        # \Theta_U^t
        (mu_u, cov_u, Lambda_u) = sample_Normal_Wishart(nu0u, mu0u, W0u, beta0u, seed)
        
        W0v = update_W0(W0_inv, _V, mu0, beta0)
        mu0v = update_mu0(mu0, beta0, _V)
        nu0v = update_nu0(nu0, M)
        beta0v = update_beta0(beta0, M)
        # \Theta_U^t
        (mu_v, cov_v, Lambda_v) = sample_Normal_Wishart(nu0v, mu0v, W0v, beta0v, seed)

        ## sample user latent D-vector
        for i in range(N):
            pass

        # sample item latent D-vector
        for j in range(M):
            pass
        pass
    pass


D = 64
nu0 = D
mu0 = 0
W0 = np.eye((D, D), dtype=np.float32)
