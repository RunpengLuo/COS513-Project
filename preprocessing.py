import pandas as pd
import numpy as np


def leave_k_out(df: pd.DataFrame, k=1, seed=42):
    df.loc[:, "split"] = 0
    for _, group in df.groupby("user_cid"):
        group = group.sample(frac=1, random_state=seed)  # shuffle
        if len(group) < 2 * k + 1:
            continue
        test_idx = group.index[:k]
        val_idx = group.index[k : 2 * k]

        df.loc[val_idx, "split"] = 1
        df.loc[test_idx, "split"] = 2

    train = pd.DataFrame(df[df["split"] == 0])
    valid = pd.DataFrame(df[df["split"] == 1])
    test = pd.DataFrame(df[df["split"] == 2])

    print(len(train), len(valid), len(test))

    return train, valid, test
