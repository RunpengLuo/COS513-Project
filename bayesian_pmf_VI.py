import sys
import numpy as np
import pandas as pd

from hyperparam_sampler import *
from feature_sampler import *
from collections import defaultdict

import time

import numpy as np
from scipy.special import gammaln

def init_UV_rand(N: int, M: int, D: int):
    U = 0.01 * np.random.randn(D, N)
    V = 0.01 * np.random.randn(D, M)
    return U, V


# TODO
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


def compute_elbo(R, Mask, mu_u, mu_v, sigma_u, sigma_v, alpha_u, beta_u, alpha_v, beta_v):
    # Likelihood term: E_q[ -0.5*alpha * (r - u^T v)^2 ]
    alpha = 1.0  # observation precision (example)
    ll = 0.0
    idxs = np.argwhere(Mask)
    for i, j in idxs:
        mu_dot = mu_u[:, i] @ mu_v[:, j]
        var_dot = np.sum(sigma_u[:, i] * (mu_v[:, j]**2)) + np.sum(sigma_v[:, j] * (mu_u[:, i]**2)) + np.sum(sigma_u[:, i] * sigma_v[:, j])
        ll += -0.5 * alpha * ((R[i, j] - mu_dot)**2 + var_dot)
    # KL terms for U and V (Gaussian vs Gaussian prior with precision alpha_u[d]/beta_u[d])
    kl_uv = 0.0
    D, N = mu_u.shape
    _, M = mu_v.shape
    for d in range(D):
        # U factors
        prior_prec_u = alpha_u[d] / beta_u[d]
        kl_uv += 0.5 * (prior_prec_u * (np.sum(mu_u[d]**2 + sigma_u[d])) - N + N*np.log(1/prior_prec_u) + np.sum(np.log(sigma_u[d])))
        # V factors
        prior_prec_v = alpha_v[d] / beta_v[d]
        kl_uv += 0.5 * (prior_prec_v * (np.sum(mu_v[d]**2 + sigma_v[d])) - M + M*np.log(1/prior_prec_v) + np.sum(np.log(sigma_v[d])))
    return ll - kl_uv


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

def bayesian_PMF_VI(
    train_df: pd.DataFrame,
    R: np.ndarray,
    Mask: np.ndarray,
    validate: np.ndarray,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    D: int,
    T: int,
    alpha: float,
    alpha_0=1.0,
    beta_0=1.0,
    rate_min=0,
    rate_max=1,
    seed=None
):
    N, M = R.shape
    mean_rating = R[Mask].mean()

    train_df["uidx"] = train_df["uidx"].astype(int) 
    train_df["bidx"] = train_df["bidx"].astype(int) 

    # Initialize U and V using ALS
    U, V = init_UV_ALS(train_df, R, Mask, mean_rating, D, 0.05, 10)
    

    # Precompute user and item ratings
    user_ratings = defaultdict(list)
    item_ratings = defaultdict(list)
    for _, row in train_df.iterrows():
        u = int(row["uidx"])  # Explicit cast to int
        j = int(row["bidx"])  # Explicit cast to int
        r = row['rating'] - mean_rating
        user_ratings[u].append((j, r))
        item_ratings[j].append((u, r))

    mu_u = U.copy()
    mu_v = V.copy()
    sigma_u = 0.1 * np.ones((D, N))
    sigma_v = 0.1 * np.ones((D, M))

    # Initialize variational hyperparameters for lambda_u and lambda_v (Gamma distributions)
    alpha_u, beta_u, alpha_v, beta_v = update_hyperparameters(
    mu_u, sigma_u, mu_v, sigma_v, alpha_0, beta_0)

    rmses = np.zeros(T + 1)
    pred = predict(mu_u, mu_v, user_ids, item_ids, mean_rating)
    pred = np.clip(pred, rate_min, rate_max)
    rmses[0] = compute_RMSE(validate, pred)
    print(f"Initial RMSE: {rmses[0]}")

    elbo_vals = []

    for t in range(1, T + 1):
           
        # Update user latent factors U
        # ================== User Updates (U) ==================

        for i in range(N):
            jr_pairs = user_ratings.get(i, [])
            if not jr_pairs:
                continue  # Skip users with no ratings

            # Temporary storage to hold new values for all dimensions
            new_mu_u = np.zeros(D)
            new_sigma_u = np.zeros(D)
            # Precompute all residuals for this user once
            residuals = []
            for (j, r) in jr_pairs:
                full_pred = np.dot(mu_u[:, i], mu_v[:, j])
                residuals.append((j, r, full_pred))
            # Process each dimension with precomputed residuals
            for d in range(D):
                sum_terms = 0.0
                sum_vjr = 0.0
                for (j, r, full_pred) in residuals:
                    residual = r - (full_pred - mu_u[d, i] * mu_v[d, j])
                    sum_terms += (mu_v[d, j] ** 2) + sigma_v[d, j]
                    sum_vjr += mu_v[d, j] * residual
                precision = (alpha_u[d] / beta_u[d]) + alpha * sum_terms
                new_sigma_u[d] = 1.0 / precision
                new_mu_u[d] = (alpha * sum_vjr) / precision
            # Apply all updates for user i after processing all dimensions
            sigma_u[:, i] = new_sigma_u
            mu_u[:, i] = new_mu_u

        
        # Update item latent factors V
        # ================== Item Updates (V) ==================
        for j in range(M):
            ir_pairs = item_ratings.get(j, [])
            if not ir_pairs:
                continue  # Skip items with no ratings

            # Temporary storage to hold new values for all dimensions
            new_mu_v = np.zeros(D)
            new_sigma_v = np.zeros(D)
            
            # Precompute all residuals for this item once
            residuals = []
            for (i, r) in ir_pairs:
                full_pred = np.dot(mu_u[:, i], mu_v[:, j])  # Precompute full prediction
                residuals.append((i, r, full_pred))  # Store (user_idx, rating, full_pred)
            
            # Process each dimension with precomputed residuals
            for d in range(D):
                sum_terms = 0.0
                sum_uir = 0.0
                
                for (i, r, full_pred) in residuals:
                    # Compute residual by removing contribution from current dimension d
                    residual = r - (full_pred - mu_u[d, i] * mu_v[d, j])
                    
                    # Accumulate terms for Gaussian parameters
                    sum_terms += (mu_u[d, i] ** 2) + sigma_u[d, i]
                    sum_uir += mu_u[d, i] * residual  # mu_u from previous iteration
                
                # Update variational parameters for V_j^d
                precision = (alpha_v[d] / beta_v[d]) + alpha * sum_terms
                new_sigma_v[d] = 1.0 / precision
                new_mu_v[d] = (alpha * sum_uir) / precision
            
            # Apply all updates for item j after processing all dimensions
            sigma_v[:, j] = new_sigma_v
            mu_v[:, j] = new_mu_v


        # ============ Update Hyperparameters lambda_u and lambda_v ============
        for d in range(D):
            # Update lambda_u (precision for U)
            alpha_u[d] = alpha_0 + 0.5 * N
            beta_u[d] = beta_0 + 0.5 * (mu_u[d]**2 + sigma_u[d]).sum()
            
            # Update lambda_v (precision for V)
            alpha_v[d] = alpha_0 + 0.5 * M
            beta_v[d] = beta_0 + 0.5 * (mu_v[d]**2 + sigma_v[d]).sum()

        # Predict and compute RMSE
        pred_t = predict(mu_u, mu_v, user_ids, item_ids, mean_rating)
        pred_t = np.clip(pred_t, rate_min, rate_max)
        rmses[t] = compute_RMSE(validate, pred_t)
        print(f"Iteration {t}, RMSE: {rmses[t]}")

        elbo = compute_elbo(R, Mask, mu_u, mu_v, sigma_u, sigma_v, alpha_u, beta_u, alpha_v, beta_v)
        elbo_vals.append(elbo)
        print(f"Iteration {t}, ELBO = {elbo:.3f}")

    return pred_t, rmses