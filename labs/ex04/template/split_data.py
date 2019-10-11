# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    indices = np.random.permutation(len(x))
    tr_end_idx = int(len(y) * ratio)
    # split dataset
    x_tr = x[indices[:tr_end_idx]]
    x_te = x[indices[tr_end_idx:]]
    y_tr = y[indices[:tr_end_idx]]
    y_te = y[indices[tr_end_idx:]]
    return x_tr, x_te, y_tr, y_te
