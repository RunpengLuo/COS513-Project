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


def plot_test_rmse(test_results, outdir):
        plt.figure(figsize=(10, 6))
        
        models = list(test_results.keys())
        init_methods = list(test_results[models[0]].keys())
        
        bar_width = 0.35
        x = np.arange(len(models))
        
        for i, init_method in enumerate(init_methods):
            rmse_values = [test_results[model][init_method] for model in models]
            positions = x + i * bar_width - (len(init_methods)-1)*bar_width/2
            bars = plt.bar(positions, rmse_values, width=bar_width, label=init_method)
            

            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}',
                        ha='center', va='bottom')

        plt.title('Test RMSE Comparison Across Models and Initialization Methods')
        plt.xlabel('Model')
        plt.ylabel('RMSE')
        plt.xticks(x, models)
        plt.legend(title='Initialization Method')
        plt.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        outpath = os.path.join(outdir, "test_rmse_comparison.png")
        plt.savefig(outpath)
        plt.close()
