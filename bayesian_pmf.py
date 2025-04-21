import numpy as np
import pandas as pd

from hyperparam_sampler import *
from feature_sampler import *

def init_UV_rand(N: int, M: int, D: int):
    U = np.zeros((D, N), dtype=np.float32)
    V = np.zeros((D, M), dtype=np.float32)

    U = np.random.normal(0, 0.1, size=(D, N))
    V = np.random.normal(0, 0.1, size=(D, M))
    return U, V

# TODO
# We initialized the Gibbs sampler by setting the model parameters U and V to their
# MAP estimates obtained by training a linear PMF model.
def init_UV_MAP(N: int, M: int, D: int):
    pass


def compute_RMSE(test_rating: np.ndarray, pred_rating: np.ndarray):
    return np.sqrt(np.mean((test_rating - pred_rating) ** 2))


def predict(
    U: np.ndarray,
    V: np.ndarray,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    mean_rating: np.float32,
):
    (n, _) = user_ids.shape
    ret = np.zeros((n, 1), dtype=np.float32)
    for i in range(n):
        ret[i] = np.dot(U[:, user_ids[i]], V[:, item_ids[i]]) + mean_rating
    return ret


def bayesian_PMF(
    R: np.ndarray,
    M: np.ndarray,
    R_test: pd.DataFrame,
    D: int,
    T: int,
    G: int,
    mu0: float,
    nu0: float,
    W0: np.ndarray,
    beta0: float,
    alpha: float,
    rate_min=0,
    rate_max=1,
    seed=None,
    v=1,
):
    """
    R: rating matrix with dimension N*M
    M: boolean mask matrix with dimension N*M
    D: latent matrix dimension
    T: #total iterations
    G: #iterations for gibbs sampling
    \Theta_0
        mu0: hyperparameter for the hyper-prior of mu_U and mu_V
        nu0: degree of freedom for Wishart Distribution
        W0: scale matrix for Wishart Distribution
    beta0: strength of prior belief in the mean \mu
    alpha: Gaussian precision
    """
    (N, M) = R.shape
    _U, _V = init_UV_rand(N, M, D)
    W0_inv = np.linalg.inv(W0)

    # extract test-information
    user_ids = R_test["user_id"].to_numpy()
    item_ids = R_test["item_id"].to_numpy()
    test = R_test["rate"].to_numpy()

    # mean rating subtraction, address global biases
    mean_rating = R[M].mean()
    print(f"Mean rating: {mean_rating}")
    R = R - mean_rating

    rmses = np.zeros((T + 1, 1), dtype=np.float32)
    # prediction
    pred = predict(_U, _V, user_ids, item_ids, mean_rating)
    pred = np.clip(pred, a_min=rate_min, a_max=rate_max)
    rmses[0] = compute_RMSE(test, pred)
    print(f"rmse={rmses[0]} pre-loop")

    for t in range(1, T + 1):
        ## sample from Gaussian-Wishart priors
        W0u = update_W0(W0_inv, _U, mu0, beta0)
        mu0u = update_mu0(mu0, beta0, _U)
        nu0u = update_nu0(nu0, N)
        beta0u = update_beta0(beta0, N)
        # \Theta_U^t
        (mu_u, _, Lambda_u) = sample_Normal_Wishart(nu0u, mu0u, W0u, beta0u, seed)

        W0v = update_W0(W0_inv, _V, mu0, beta0)
        mu0v = update_mu0(mu0, beta0, _V)
        nu0v = update_nu0(nu0, M)
        beta0v = update_beta0(beta0, M)
        # \Theta_V^t
        (mu_v, _, Lambda_v) = sample_Normal_Wishart(nu0v, mu0v, W0v, beta0v, seed)

        for g in range(G):
            print(f"t={t}\tgibbs-sampling iteration {g}")
            ## sample user latent D-vector
            for i in range(N):
                mu_i, Lambda_i_inv = update_mu_ui(R, M, _V, Lambda_u, mu_u, alpha, i)
                _U[:, i] = sample_ui(mu_i, Lambda_i_inv)

            # sample item latent D-vector
            for i in range(M):
                mu_i, Lambda_i_inv = update_mu_vi(R, M, _U, Lambda_v, mu_v, alpha, i)
                _V[:, i] = sample_vi(mu_i, Lambda_i_inv)

        ## prediction step
        pred_t = predict(_U, _V, user_ids, item_ids, mean_rating)
        pred_t = np.clip(pred, a_min=rate_min, a_max=rate_max)
        # compute Gibbs sampling approximation over <1..t> iterations
        pred = (pred * t + pred_t) / (t + 1)

        rmses[t] = compute_RMSE(test, pred)
        print(f"\trmse={rmses[t]}")

    return pred, rmses[-1]
