import os
import sys
import argparse

import numpy as np
import pandas as pd

from preprocessing import leave_k_out
from bayesian_pmf import bayesian_PMF
from evaluation import plot_rmse, compare_rmse

if __name__ == "__main__":
    _, data_file, outdir = sys.argv

    os.makedirs(outdir, exist_ok=True)

    np.random.seed(42)
    df = pd.read_table(data_file, sep="\t")
    rate_min = df.stars.min()
    rate_max = df.stars.max()
    df.loc[:, "rating"] = df.apply(
        func=lambda r: (r["stars"] - rate_min) / (rate_max - rate_min), axis=1
    )
    df = df.drop_duplicates(subset=["user_cid", "business_cid"], keep="first")
    print(len(df))

    train_df, valid_df, test_df = leave_k_out(df, k=10, seed=42)

    train_mdf = pd.pivot(
        train_df, index="user_cid", columns="business_cid", values="rating"
    )
    user2index = {u: i for i, u in enumerate(train_mdf.index.tolist())}
    item2index = {i: j for j, i in enumerate(train_mdf.columns.tolist())}
    train_df.loc[:, "uidx"] = train_df["user_cid"].map(user2index)
    train_df.loc[:, "bidx"] = train_df["business_cid"].map(user2index)

    print(len(user2index), len(item2index))

    valid_uidx = valid_df["user_cid"].map(user2index).to_numpy()
    valid_bidx = valid_df["business_cid"].map(item2index).to_numpy()
    valid = valid_df["rating"].to_numpy()

    mask_train = ~train_mdf.isna()
    mask_train = mask_train.to_numpy().astype(bool)

    mat_train = train_mdf.to_numpy().astype(np.float32)

    n, m = train_mdf.shape
    num_na = train_mdf.isna().sum().sum()
    print("train matrix sparsity: ", 100 * (n * m - num_na) / (n * m), n, m)

    D = 5 # latent dimension
    T = 10 # num.epochs
    # D = 5
    # T = 3
    nu0 = D
    mu0 = 0
    W0 = np.eye(N=D, M=D, dtype=np.float32)
    G = 2
    alpha = 2
    beta0 = 2

    pred, rmses = bayesian_PMF(
        train_df,
        mat_train,
        mask_train,
        valid,
        valid_uidx,
        valid_bidx,
        D,
        T,
        G,
        mu0,
        nu0,
        W0,
        beta0,
        alpha,
        rate_min=0,
        rate_max=1,
        seed=42,
        v=1,
    )

    out_rmse = os.path.join(outdir, "rmse.png")
    plot_rmse(rmses, out_rmse)

    ### comparing MAP vs ALS
    init_methods = ["MAP", "ALS"]
    results = {}

    for method in init_methods:
        print(f"\n=== Running Bayesian PMF with {method} initialization ===\n")
        pred, rmses = bayesian_PMF(
            train_df,
            mat_train,
            mask_train,
            valid,
            valid_uidx,
            valid_bidx,
            D,
            T,
            G,
            mu0,
            nu0,
            W0,
            beta0,
            alpha,
            rate_min=0,
            rate_max=1,
            seed=42,
            v=1,
            init_method=method,
        )
        results[method] = rmses

    compare_plot_path = os.path.join(outdir, "rmse_comparison.png")
    compare_rmse(results, compare_plot_path)

