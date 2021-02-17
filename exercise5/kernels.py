"""
Module containing different kernel functions.
"""
import numpy as np

import config
from utils.data_utils import get_padded_node_labels
from sklearn.metrics.pairwise import linear_kernel
from feature_extractor import degree_features, color_refinement, color_to_hist


def compute_gram_matrix(x):
    """
    Kernel function calculating the inner product.

    :param x: list of feature vectors
    :return: gram matrix
    """
    return linear_kernel(x, x, dense_output=True)


def degree_kernel(graphs):
    """
    Builds computes the gram matrix using the node degrees as feature vectors.

    :param graphs:     list of networkx graphs
    :return:           computed gram matrix
    """

    feature_vectors = degree_features(graphs)
    gram_matrix = compute_gram_matrix(feature_vectors, feature_vectors)

    return gram_matrix


def weisfeiler_lehman_kernel(graphs, steps=config.REFINEMENT_STEPS):
    """
    Computes the gram matrix using the Weisfeiler-Lehman kernel with a specfied
    number of refinement steps.

    :param graphs:      list of networkx graphs
    :param steps:       number of refinement steps

    :return:            the computed gram matrix for the WL kernel
    """

    # initial node labels will be included in the GRAM matrix
    node_labels = get_padded_node_labels(graphs)
    gram_matrix = compute_gram_matrix(np.count_nonzero(node_labels, axis=1))

    for i in range(steps):
        graphs, color_counts = color_refinement(graphs)
        no_colors = len(
            list(set(color for cur_coloring in color_counts for color in cur_coloring.keys())))
        # compute histograms from graph colors
        color_hists = color_to_hist(graphs, color_counts, no_colors)
        interm_gram_matrix = compute_gram_matrix(color_hists)
        # update gram matrix
        gram_matrix = gram_matrix + interm_gram_matrix

    return gram_matrix
