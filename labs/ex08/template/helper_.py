# -*- coding: utf-8 -*-
"""Some helper functions."""
import os
import shutil
import numpy as np
from scipy import misc


def load_data():
    """Load data and convert it to the metrics system."""
    path_dataset = "faithful.csv"
    data = np.loadtxt(path_dataset, delimiter=" ", skiprows=0)
    return data


def normalize_data(data):
    """normalize the data by (x - mean(x)) / std(x)."""
    mean_data = np.mean(data, axis=0)
    data = data - mean_data
    std_data = np.std(data)
    data = data / std_data
    return data


def build_dir(dir):
    """build a new dir. if it exists, remove it and build a new one."""
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)


def load_image(path):
    """use the scipy.misc to load the image."""
    return misc.imread(path)

    
def build_distance_matrix(data, mu):
    """build a distance matrix.
    input:
        data: original data with shape (num_sample, num_feature).
        a numpy array with shape (k, num_feature)
    return
        distance matrix:
            row of the matrix represents the data point,
            column of the matrix represents the k-th cluster.
    """
    distance = np.sum(data**2, axis=1)[:, np.newaxis] + np.sum(mu**2, axis=1) - 2 * np.dot(data, mu.T)
    return distance
