# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.zeros((len(x), degree + 1))
    for i in range(0, degree + 1):
        poly[:, i] = np.power(x, i)
    return poly
