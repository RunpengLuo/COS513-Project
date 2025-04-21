import numpy as np
from scipy.stats import wishart, multivariate_normal

# TODO
# We initialized the Gibbs sampler by setting the model parameters U and V to their 
# MAP estimates obtained by training a linear PMF model.
def init_UV(N: int, M: int, D: int):
    U = np.zeros((D, N), dtype=np.float32)
    V = np.zeros((D, M), dtype=np.float32)
    return U, V

def update_beta0(beta0, n):
    return beta0 + n

def update_nu0(nu0, n):
    return nu0 + n

def get_mat_(mat: np.ndarray):
    return np.mean(mat, axis=0)

def get_S_(mat: np.ndarray):
    _, n = mat.shape
    assert n != 0, "invalid div0"
    return (mat @ mat.T) / n

def update_mu0(mu0, beta0, mat):
    n = mat.shape[1]
    num = beta0 * mu0 + n * get_mat_(mat)
    dom = beta0 + n
    assert dom != 0, "invalid div0"
    return num / dom

def update_W0(W0_inv: np.ndarray, mat: np.ndarray, mu0, beta0):
    """
    return: W* in D*D
    """
    _, n = mat.shape
    S_ = get_S_(mat)
    mu0_mat_ = mu0 - get_mat_(mat)
    mutc = (beta0 * n) / (beta0 + n)
    _W0_inv = W0_inv + n * S_ + mutc * mu0_mat_ @ mu0_mat_.T
    return np.linalg.inv(_W0_inv)

def sample_Normal_Wishart(nu0, mu0, W0, beta0, seed=None):
    Lambda = wishart.rvs(df=nu0, scale=W0, seed=seed)
    cov_mat = np.linalg.inv(beta0 * Lambda)
    mu = multivariate_normal(mean=mu0, cov=cov_mat)
    return mu, cov_mat, Lambda
# def 

def bayesian_PMF(
        R: np.ndarray,
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

    # updates
    mu0u = mu0, nu0u = nu0, beta0u = beta0, W0u = W0
    mu0v = mu0, nu0v = nu0, beta0v = beta0, W0v = W0

    for t in range(T):
        ## sample from Gaussian-Wishart priors
        W0u = update_W0(W0_inv, _U, mu0u, beta0u)
        mu0u = update_mu0(mu0u, beta0u, _U)
        nu0u = update_nu0(nu0u, N)
        beta0u = update_beta0(beta0u, N)
        # \Theta_U^t
        (mu_u, cov_u, Lambda_u) = sample_Normal_Wishart(nu0u, mu0u, W0u, beta0u, seed)
        
        W0v = update_W0(W0_inv, _V, mu0v, beta0v)
        mu0v = update_mu0(mu0v, beta0v, _V)
        nu0v = update_nu0(nu0v, M)
        beta0v = update_beta0(beta0v, M)
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
