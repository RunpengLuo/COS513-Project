import sys
import numpy as np
import pandas as pd

from hyperparam_sampler import *
from feature_sampler import *

import time


def init_UV_rand(N: int, M: int, D: int):
    U = 0.01 * np.random.randn(D, N)
    V = 0.01 * np.random.randn(D, M)
    return U, V


# TODO
def init_UV_MAP_old(
    train_df: pd.DataFrame,
    R: np.ndarray,
    Mask: np.ndarray,
    mean_rating: float,
    D: int,
    epsilon: float,
    lambda_u: float,
    lambda_v: float,
    momentum: float,
    num_epoch: int,
):
    """
    find MAP estimate via SGD
    """
    print(f"init solution with MAP")
    (N, M) = R.shape
    U, V = init_UV_rand(N, M, D)
    U_inc = np.zeros_like(U)
    V_inc = np.zeros_like(V)

    idx = train_df[["uidx", "bidx"]].to_numpy()
    rating = (train_df["rating"] - mean_rating).to_numpy()

    _R = R - mean_rating
    for epoch in range(num_epoch):
        # compute loss
        pred = np.array([np.dot(U[:, i], V[:, j]) for [i, j] in idx])
        term1 = ((pred - rating) ** 2).sum()
        term2 = (U**2).sum()
        term3 = (V**2).sum()
        loss = 0.5 * term1 + 0.5 * lambda_u * term2 + 0.5 * lambda_v * term3

        print(loss, pred.shape)
        # compute gredient
        gd1 = pred - rating
        # gd1 = np.tile(gd1, )
        sys.exit(0)

        pass

    return

def init_UV_MAP(
    train_df: pd.DataFrame,
    R: np.ndarray,
    Mask: np.ndarray,
    mean_rating: float,
    D: int,
    epsilon: float,
    lambda_u: float,
    lambda_v: float,
    momentum: float,
    num_epoch: int,
):
    """
    find MAP estimate via SGD
    """
    print("Initializing U and V using MAP estimate via SGD...")
    (N, M) = R.shape
    U, V = init_UV_rand(N, M, D)
    U_inc = np.zeros_like(U)
    V_inc = np.zeros_like(V)

    idx = train_df[["uidx", "bidx"]].to_numpy()
    rating = (train_df["rating"] - mean_rating).to_numpy()

    for epoch in range(num_epoch):
        pred = np.array([np.dot(U[:, i], V[:, j]) for [i, j] in idx])
        err = pred - rating

        loss = 0.5 * np.sum(err ** 2)
        loss += 0.5 * lambda_u * np.sum(U ** 2)
        loss += 0.5 * lambda_v * np.sum(V ** 2)
        print(f"Epoch {epoch + 1}/{num_epoch}, Loss: {loss:.4f}")

        # gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        for n in range(len(idx)):
            i, j = idx[n]
            eij = err[n]
            dU[:, i] += eij * V[:, j]
            dV[:, j] += eij * U[:, i]

        # regularization
        dU += lambda_u * U
        dV += lambda_v * V

        # momentum-based SGD update
        U_inc = momentum * U_inc - epsilon * dU
        V_inc = momentum * V_inc - epsilon * dV
        U += U_inc
        V += V_inc

    return U, V



def init_UV_ALS(
    train_df: pd.DataFrame,
    R: np.ndarray,
    Mask: np.ndarray,
    mean_rating: float,
    D: int,
    epsilon: float,
    num_epoch: int,
):
    print(f"init solution with ALS")
    (N, M) = R.shape
    U, V = init_UV_rand(N, M, D)
    idx = train_df[["uidx", "bidx"]].to_numpy()
    rating = (train_df["rating"] - mean_rating).to_numpy()
    for epoch in range(num_epoch):
        pred = np.array([np.dot(U[:, i], V[:, j]) for [i, j] in idx])
        rmse = compute_RMSE(rating, pred)
        print(f"\tepoch={epoch}\trmse (train)={rmse}")
        if rmse <= epsilon:
            break
        for ui in range(N):
            rows = train_df.loc[train_df["uidx"] == ui, :]
            A = V[:, rows["bidx"].to_numpy()]
            b = (rows["rating"] - mean_rating).to_numpy()
            U[:, ui], _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
        for vi in range(M):
            rows = train_df.loc[train_df["bidx"] == vi, :]
            A = U[:, rows["uidx"].to_numpy()]
            b = (rows["rating"] - mean_rating).to_numpy()
            V[:, vi], _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
    return U, V


def compute_RMSE(test_rating: np.ndarray, pred_rating: np.ndarray):
    return np.sqrt(np.mean((test_rating - pred_rating) ** 2))


def predict(
    U: np.ndarray,
    V: np.ndarray,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    mean_rating: np.float32,
):
    n = user_ids.shape[0]
    ret = np.zeros(n, dtype=np.float32)
    for i in range(n):
        ret[i] = np.dot(U[:, user_ids[i]], V[:, item_ids[i]])
    ret += mean_rating
    return ret


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

    return pred, rmses
