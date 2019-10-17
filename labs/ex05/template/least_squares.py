# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares."""
    lt = np.dot(tx.T, tx)
    rt = np.dot(tx.T, y)
    w = np.linalg.solve(lt, rt)
    e = y - tx.dot(w)
    mse = e.dot(e.T) / (2 * len(e))
    return mse, w
