"""
This file contains the initial data loading and preprocessing.
"""
import os
import numpy as np

from utils.data_utils import get_graph_label, get_padded_adjacency, get_padded_node_labels, get_padded_node_attributes
from utils.dataset_parser import Parser


def load_dataset(name):
    """
    Loads the graph dataset from file given a dataset name in form of a string.

    :param name:    string of dataset name
    :return:        loaded networkX graphs, graph labels
    """
    dataset = Parser(os.path.join('datasets', name))

    graphs = dataset.parse_all_graphs()
    labels = [get_graph_label(graph) for graph in graphs]

    return graphs, labels


def normalize_adj_matrix(adj_matrix):
    """
    Normalizes a adjacency matrix.

    :param adj_matrix: list of networkX graphs
    :return: normalized adj matrix
    """

    degrees = adj_matrix.sum(axis=1)
    root_inverse_degrees = np.divide(1, np.sqrt(degrees), out=np.zeros_like(degrees), where=degrees != 0)
    root_inverse_degree_matrix = np.zeros((adj_matrix.shape[0], adj_matrix.shape[1]))
    # fill the diagonal with normalized entries, obtainting root inverse D
    np.fill_diagonal(root_inverse_degree_matrix, root_inverse_degrees)
    adj_matrix_norm = np.matmul(root_inverse_degree_matrix, np.matmul(adj_matrix, root_inverse_degree_matrix))
    return adj_matrix_norm


def one_hot_features(graphs, features, labels):
    # map labels form [1,6] to [0,5]
    labels = np.array(labels)
    labels = labels - 1

    # features of ENZYMES are the node labels in a on-hot encoding concatenated with the attribute features
    attributes = get_padded_node_attributes(graphs)
    features = np.dstack((features, attributes))

    return features, labels


def get_normalized_data(name):
    """
    Loads the dataset and returns the normalized adj matrices.

    :param name: name of the dataset
    :return: list of normalized adj matrices, list of feature vectors, list of labels
    """

    graphs, labels = load_dataset(name)

    features = get_padded_node_labels(graphs)

    adj_matrices = get_padded_adjacency(graphs)

    adj_matrices_norm = [normalize_adj_matrix(adj_matrix) for adj_matrix in adj_matrices]

    if name == "ENZYMES":
        features, labels = one_hot_features(graphs, features, labels)

    return np.asarray(adj_matrices_norm), np.asarray(features), np.asarray(labels)
