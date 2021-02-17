"""
Exercise 4.
This module contains the implementation of the closed walk kernel.
"""

import os
import config
import numpy as np
from tqdm import tqdm
from data_utils import get_adjacency_matrix
from numpy import linalg as LA


def compute_closed_walks(datasets, names):
    """
    computes the correspoding feature vectors using the random walk kernel on
    the passed datasets. If specified in config file, the computed vectors are
    stored for later use.

    param datasets: list of datasets to be converted into feature vectors
    param names: list of names of the datasets
    return: list of feature vectors for each dataset
    """

    if not os.path.exists(os.path.join(os.getcwd(), "data", "closed_walk")) and config.PREPROC_STORE:
        os.makedirs(os.path.join(os.getcwd(), "data", "closed_walk"))

    # convert graph to feature vector
    closed_walk_sets = []

    for i, dataset in enumerate(datasets):
        closed_walks = np.zeros([len(dataset), config.CLOSED_WALK_VECTOR_SIZE],
                                dtype=np.int64)
        for j, graph in tqdm(enumerate(dataset), total=len(dataset),
                             desc=f"converting {names[i]}"):
            closed_walks[j] = closed_walks_vector(graph,
                                                  config.CLOSED_WALK_VECTOR_SIZE)

        if config.PREPROC_STORE:
            np.save(os.path.join(os.getcwd(),
                                 "data", "closed_walk",
                                 "dataset_" + names[i] + ".npy"), closed_walks)

        closed_walk_sets.append(closed_walks)

    return closed_walk_sets


def load_closed_walks(names):
    """
    Loads the already computed and stored feature vectors from disk.

    param names: list of names of datasets to load
    return: list of feature vectors for each dataset
    """

    closed_walk_sets = []
    for name in names:
        closed_walks = np.load(os.path.join(os.getcwd(),
                                            "data", "closed_walk",
                                            "dataset_" + name + ".npy"))
        closed_walk_sets.append(closed_walks)
    return closed_walk_sets


def closed_walks_vector(x, max_i):
    """
    Function calculating the vector of dimension max_i containing the number of
    closed walks of length 2 upto max_i.
    We make use of the spectral theorem for the calculation of the closed path
    number by using the eigenvalues of the adjecency matrix.

    :param x: A NetworkX Graph
    :return: Vector of dimension max_i, index i contains the number of closed
             walks of length i+2

    """

    ##Initialization of used datastructures##
    walks_x = np.zeros(max_i, dtype=np.int32)  # Vector holding the number of closed walks
    eig_values, _ = LA.eig(get_adjacency_matrix(x))  # Vector holding the eigenvalues of the Adjecency Matrix

    for i in range(max_i):
        # Summation of the powered eigenvalues, we start with i+2
        # since closed walks start with length 2. walks_x[0] will thus
        # hold the walks of length 2, not 0.
        walks_x[i] = int((np.sum([np.power(y, i + 2) for y in eig_values])))
        if walks_x[i] < 1:  # Handling of numerical nuisance
            walks_x[i] = 0
    return walks_x
