import sys
import numpy as np
import pandas as pd

from hyperparam_sampler import *
from feature_sampler import *
from initialization import *

import time


def bayesian_PMF(
    train_df: pd.DataFrame,
    R: np.ndarray,
    Mask: np.ndarray,
    validate: np.ndarray,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
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
    init_method="MAP"
):
    """
    R: rating matrix with dimension N*M
    Mask: boolean mask matrix with dimension N*M
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
    W0_inv = np.linalg.inv(W0)

    # mean rating subtraction, address global biases
    mean_rating = R[Mask].mean()
    print(f"Mean rating: {mean_rating}")
    R = R - mean_rating

    if init_method == "ALS":
        _U, _V = init_UV_ALS(train_df, R, Mask, mean_rating, D, epsilon=0.05, num_epoch=10)
    elif init_method == "MAP":
        _U, _V = init_UV_MAP(
            train_df, R, Mask, mean_rating, D,
            epsilon=0.005, lambda_u=0.002, lambda_v=0.002,
            momentum=0.9, num_epoch=50
        )
    else:
        raise ValueError(f"Unsupported init_method: {init_method}")

    # _U, _V = init_UV_sgd(train_df, R, Mask, mean_rating, D, 50, 0.01, 0.01, 0.8, 50)
    # _U, _V = init_UV_rand(N, M, D)
    # _U, _V = init_UV_ALS(train_df, R, Mask, mean_rating, D, 0.05, 10)

    rmses = np.zeros(T + 1, dtype=np.float128)
    # prediction
    pred = predict(_U, _V, user_ids, item_ids, mean_rating)
    pred = np.clip(pred, a_min=rate_min, a_max=rate_max)
    rmses[0] = compute_RMSE(validate, pred)
    print(f"rmse(validate)={rmses[0]} pre-loop")

    ts = time.perf_counter()
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
                mu_i, Lambda_i_inv = update_mu_ui(R, Mask, _V, Lambda_u, mu_u, alpha, i)
                _U[:, i] = sample_ui(mu_i, Lambda_i_inv)

            # sample item latent D-vector
            for i in range(M):
                mu_i, Lambda_i_inv = update_mu_vi(R, Mask, _U, Lambda_v, mu_v, alpha, i)
                _V[:, i] = sample_vi(mu_i, Lambda_i_inv)

        ## prediction step
        pred_t = predict(_U, _V, user_ids, item_ids, mean_rating)
        assert np.any(pred_t != pred)
        pred_t = np.clip(pred, a_min=rate_min, a_max=rate_max)
        # compute Gibbs sampling approximation over <1..t> iterations
        pred = (pred * t + pred_t) / (t + 1)

        rmses[t] = compute_RMSE(validate, pred)
        print(
            f"\telapsed={round(time.perf_counter() - ts, 3)}s\trmse(validate)={rmses[t]}"
        )

    return _U, _V, pred, rmses
