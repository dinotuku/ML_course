#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main File for ML project 1
Spot the Boson
"""

import numpy as np
from proj1_helpers import load_csv_data, standardize
from implementations import least_squares_GD


def main():
    """ Main function """
    yb, tx, ids = load_csv_data('../data/train.csv')
    tx = standardize(tx)
    w, loss = least_squares_GD(yb, tx, np.zeros(tx.shape[1]), 10, 0.01)

if __name__ == '__main__':

    main()
