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
