import os
import sys

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_rmse(rmses: np.ndarray, outfile: str):
    fig, ax = plt.subplots(1, 1)
    sns.lineplot(y=rmses, x=np.arange(len(rmses)), ax=ax)
    plt.title("RMSE plot")
    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.savefig(outfile)
    return

def compare_rmse(results: dict, outfile: str):
    """
    results: dict of {init_method: rmse_array}
    """
    plt.figure()
    for label, rmse in results.items():
        plt.plot(np.arange(len(rmse)), rmse, label=f"{label} init")
    plt.xlabel("Epochs")
    plt.ylabel("Validation RMSE")
    plt.title("Bayesian PMF RMSE: MAP vs ALS Initialization")
    plt.legend()
    plt.grid(True)
    plt.savefig(outfile)
    print(f"Saved RMSE comparison to {outfile}")
