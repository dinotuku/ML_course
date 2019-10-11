# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    reg = 2 * len(tx) * lambda_ * np.identity(tx.shape[1])
    lt = np.dot(tx.T, tx) + reg
    rt = np.dot(tx.T, y)
    # solve seems to work better than inv?
    opt_weights = np.linalg.solve(lt, rt)
    return opt_weights
