import os
import sys
import argparse

import numpy as np

if __name__ == "__main__":
    _, dataset_path = sys.argv

    # df = json_to_df(os.path.join(dataset_path, "yelp_academic_dataset_business.json"))
    # print(",".join(df.columns.tolist()), len(df))

    # df = json_to_df(os.path.join(dataset_path, "yelp_academic_dataset_checkin.json"))
    # print(",".join(df.columns.tolist()), len(df))

    # df = json_to_df(os.path.join(dataset_path, "yelp_academic_dataset_review.json"))
    # print(",".join(df.columns.tolist()), len(df))

    # df = json_to_df(os.path.join(dataset_path, "yelp_academic_dataset_tip.json"))
    # print(",".join(df.columns.tolist()), len(df))

    # df = json_to_df(os.path.join(dataset_path, "yelp_academic_dataset_user.json"))
    # print(",".join(df.columns.tolist()), len(df))

    D = 64
    nu0 = D
    mu0 = 0
    W0 = np.eye((D, D), dtype=np.float32)
    G = 2
    alpha = 2
