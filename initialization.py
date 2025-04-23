import numpy as np
import pandas as pd

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