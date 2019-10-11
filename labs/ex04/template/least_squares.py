# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares solution."""
    lt = np.dot(tx.T, tx)
    rt = np.dot(tx.T, y)
    # solve seems to work better than inv?
    opt_weights = np.linalg.solve(lt, rt)
    return opt_weights
