# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np

def compute_mse(y, tx, w):
    """Calculate the mse."""
    # If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b.
    e = y - np.dot(tx, w)
    mse = 1 / 2 * np.mean(e**2)
    return mse