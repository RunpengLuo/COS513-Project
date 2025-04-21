import numpy as np
from scipy.stats import multivariate_normal

def update_mu_ui(R, M, V, Lambda_u, mu_u, a, i):
    (N, M) = R.shape
    (D, _) = V.shape
    Lambda_term = np.zeros((D, D), dtype=np.float32)
    for j in range(M):
        if M[i, j]:
            Lambda_term += np.outer(V[:, j], V[:, j])
    Lambda_i_inv = np.linalg.inv(Lambda_u + a * Lambda_term)
    
    mu_term = np.zeros((D, 1), dtype=np.float32)
    for j in range(M):
        if M[i, j]:
            mu_term += R[i, j] * V[:, j]
    mu_i = Lambda_i_inv @ (a * mu_term + Lambda_u @ mu_u)

    return Lambda_i_inv, mu_i

def sample_ui(mu_i, Lambda_i_inv):
    ui = multivariate_normal(mean=mu_i, cov=Lambda_i_inv)
    return ui

# TODO
def update_mu_vi(R, M, V, Lambda_u, mu_u, a, i):
    (N, M) = R.shape
    (D, _) = V.shape
    Lambda_term = np.zeros((D, D), dtype=np.float32)
    for j in range(M):
        if M[i, j]:
            Lambda_term += np.outer(V[:, j], V[:, j])
    Lambda_i_inv = np.linalg.inv(Lambda_u + a * Lambda_term)
    
    mu_term = np.zeros((D, 1), dtype=np.float32)
    for j in range(M):
        if M[i, j]:
            mu_term += R[i, j] * V[:, j]
    mu_i = Lambda_i_inv @ (a * mu_term + Lambda_u @ mu_u)

    return Lambda_i_inv, mu_i

# TODO
def sample_vi(mu_i, Lambda_i_inv):
    ui = multivariate_normal(mean=mu_i, cov=Lambda_i_inv)
    return ui
