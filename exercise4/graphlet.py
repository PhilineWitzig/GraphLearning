"""
Exercise 1.
This module contains the code for the graphlet kernel.
"""

import os
import random
import networkx as nx
import numpy as np
from tqdm import tqdm


comp_graphs = nx.generators.atlas.graph_atlas_g()[19:53]


def compute_graphlets(datasets, names):
    """
    Computes the correspoding feature vectors using the graphlet kernel on the passed datasets.
    The computed vectors are stored for later use.

    param datasets: list of datasets to be converted into feature vectors
    param names:    list of names of the datasets
    return:         list of feature vectors for each dataset
    """

    if not os.path.exists(os.path.join(os.getcwd(), "data")):
        os.makedirs(os.path.join(os.getcwd(), "data", "graphlet"))

    # convert graph to feature vector
    graphlet_sets = []

    for i, dataset in enumerate(datasets):
        graphlet = np.zeros([len(dataset), 34])
        for j, graph in tqdm(enumerate(dataset), total=len(dataset),
                             desc=f"converting {names[i]}"):
            graphlet[j] = graph_to_hist(graph)

        
        np.save(os.path.join(os.getcwd(),"data", "graphlet", "dataset_" + names[i] + ".npy"), graphlet)

        graphlet_sets.append(graphlet)

    return graphlet_sets


def load_graphlets(names):
    """
    Loads the already computed and stored feature vectors from disk.

    :param names:   list of names of datasets to load
    :return:        list of feature vectors for each dataset
    """
    graphlet_sets = []
    for name in names:
        graphlet = np.load(os.path.join(os.getcwd(),
                                        "data", "graphlet", "dataset_" + name + ".npy"))
        graphlet_sets.append(graphlet)
        
    return graphlet_sets


def graph_to_hist(x):
    """
    Creates the histogram vector for the graphlet kernel.

    :param x:   a networkX graph
    :return:    histogram vector for graph x
    """

    hist = np.zeros(34)  # histogram vector for graph x

    for i in range(1000):
        rand_nodes = random.sample(list(x), 5)
        subgraph = x.subgraph(rand_nodes)
        hist[subgraph_id(subgraph)] += 1

    return hist


def subgraph_id(x):
    """
    Identifies the type of a graph of length 5

    :param x:   a networkX graph of length 5
    :return:    an id corresponding to the way the graph looks
    """

    for i, graph in enumerate(comp_graphs):
        if nx.algorithms.isomorphism.is_isomorphic(x, graph):
            return i

    raise ValueError("Passed graph was not isomorphig to any graph with 5 nodes.")
