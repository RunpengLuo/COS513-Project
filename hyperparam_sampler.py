import numpy as np
from scipy.stats import wishart


def update_beta0(beta0, n):
    return beta0 + n


def update_nu0(nu0, n):
    return nu0 + n


def get_mat_(mat: np.ndarray):
    return np.mean(mat, axis=1)


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
    _W0_inv = W0_inv + n * S_ + mutc * (mu0_mat_ @ mu0_mat_.T)
    return np.linalg.inv(_W0_inv)


def sample_Normal_Wishart(nu0, mu0, W0, beta0, seed=None):
    Lambda = wishart.rvs(df=nu0, scale=W0, random_state=seed)
    cov_mat = np.linalg.inv(beta0 * Lambda)
    mu = np.random.multivariate_normal(mean=mu0, cov=cov_mat)
    return mu, cov_mat, Lambda

def update_hyperparameters(mu_u, sigma_u, mu_v, sigma_v, alpha_0, beta_0):
    D, N = mu_u.shape
    _, M = mu_v.shape
    alpha_u = np.full(D, alpha_0 + 0.5 * N)
    beta_u = np.full(D, beta_0 + 0.5 * (mu_u**2 + sigma_u).sum(axis=1))
    alpha_v = np.full(D, alpha_0 + 0.5 * M)
    beta_v = np.full(D, beta_0 + 0.5 * (mu_v**2 + sigma_v).sum(axis=1))
    return alpha_u, beta_u, alpha_v, beta_v
