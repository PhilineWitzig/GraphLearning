#!/usr/bin/env python3
"""
Exercise 2
This module contains the code for the Weisfeiler Lehman Kernel.
Implementation is mainly oriented on the paper "Color Refinement and its
Applications".
"""
import os
import config
import numpy as np
from tqdm import tqdm


def compute_wl(datasets, names):
    """
    Computes the correspoding feature vectors using the wl kernel on the passed
    datasets. If specified in config file, the computed vectors are stored for
    later use.

    :param datasets:    list of datasets to be converted into feature vectors
    :param names:       list of names of the datasets
    :return:            list of feature vectors for each dataset
    """

    if not os.path.exists(os.path.join(os.getcwd(), "data", "wl")) and config.PREPROC_STORE:
        os.makedirs(os.path.join(os.getcwd(), "data", "wl"))

    wl_sets = []
    for i, dataset in enumerate(datasets):
        wl_features = []
        for j, graph in tqdm(enumerate(dataset), total=len(dataset),
                             desc=f"converting {names[i]}"):
            wl_features.append(get_wl_hist(graph))

        # make WL_feature vectors have equal size
        max_len = max([len(feature_vec) for feature_vec in wl_features])
        for j in range(len(wl_features)):
            feature_vec_ext = np.zeros([max_len])
            feature_vec_ext[0:len(wl_features[j])] = wl_features[j]
            wl_features[j] = feature_vec_ext

        if config.PREPROC_STORE:
            np.save(os.path.join(os.getcwd(),
                                 "data", "wl", "dataset_" + names[i] + ".npy"), wl_features)

        wl_sets.append(wl_features)

    return wl_sets


def load_wl(names):
    """
    Loads the already computed and stored feature vectors from disk.

    :param names:   list of names of datasets to load
    :return:        list of WL feature vectors for each dataset
    """

    wl_sets = []
    for name in names:
        wl = np.load(os.path.join(os.getcwd(),
                                  "data", "graphlet", "dataset_" + name + ".npy"))
        wl_sets.append(wl)
    return wl_sets


def get_wl_hist(G):
    """
    Computes the feature vector of a networkx graph for a Weisfeiler Lehman
    kernel. Weights are assumed to be 1.0 for all refined graphs.

    :param G:       a networkx graph
    :return:        feature vector for WL kernel
    """

    G_refined, nc = color_refinement(G, config.REFINEMENT_ITER)
    WL_histogram = np.zeros([nc])

    for iter in range(len(G_refined)):
        WL_histogram += 1.0 * color_to_hist(G_refined[iter], nc)

    return WL_histogram


def color_to_hist(C, nc):
    """
    Computes the histogram of the current graph coloring.

    :param C:   dictionary holding the current coloring, indexed by vertices
    :param nc:  number of colors used in current graph coloring
    :return:    np vector representing the histogram
    """

    hist = np.zeros([nc])

    for n, c in C.items():
        hist[c - 1] += 1

    return hist


def color_refinement(G, n_iter):
    """
    Executes a finite number of color refinement steps given a networkx graph.
    Initally all vertiex colors are set to 1.
    Implementation oriented on the paper "Color Refinement and its Applications".

    :param G:       a networkx graph
    :param n_iter:  number of refinement steps
    :return:        list of color dictionaries for the different refinement
                    steps, no. of colors used
    """

    # --------------
    # Initialization
    # --------------
    C = dict()  # associates with each vertex a color
    for v in list(G.nodes()):
        C[v] = 1

    P = dict()  # associates with each color the vertices of this color
    P[1] = list(G.nodes())
    last = 1  # last color used
    nc = 1  # number of colors used
    Q = []  # colors that will be used for refinement
    Q.append(1)

    G_refined = []  # list of refined graphs
    G_refined.append(C.copy())

    # --------------------------
    # Color refinement algorithm
    # --------------------------
    for iter in range(n_iter + 1):
        if not Q:
            break

        # 1. Get vertex degrees for next refinement color
        q = Q.pop(0)
        D = dict()  # associates with each vertex the no. of the neighbors of color q
        for v in list(G.nodes()):
            D[v] = 0
            for w in list(G.neighbors(v)):
                if C[w] == q:
                    D[v] += 1

        D_inv = dict()  # D with keys and values swapped
        {D_inv.setdefault(d, []).append(v) for v, d in D.items()}

        # 2. split nodes into color classes based on current coloring and degrees
        D_C = []
        for d, nodes in D_inv.items():
            D_C.append(nodes)

        len_D_C = len(D_C)
        for i in range(len_D_C):
            nodes = D_C[i]
            values = set(map(lambda v: C[v], nodes))
            split_nodes = [[w for w in nodes if C[w] == v] for v in values]
            D_C[i] = split_nodes[0]
            for j in range(1, len(split_nodes)):
                D_C.append(split_nodes[j])

        l = len(D_C)  # l - 1 = number of new colors required

        # 3. Update vertex colors
        for i in range(1, l):
            Q.append(last + i)

        for b in range(0, l):
            P[b] = D_C[b]
            for v in P[b]:
                C[v] = b

        # 4. Update variables
        G_refined.append(C.copy())
        last += l - 1
        nc = l

    return G_refined, nc
