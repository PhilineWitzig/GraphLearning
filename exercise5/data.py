"""
This module contains all functions for loading and preprocessing the challenge's
datasets.
"""
import numpy as np

from utils.data_utils import get_graph_label, get_node_labels
from utils.dataset_parser import Parser


def load_dataset_graph(name):
    """
    Loads the graph dataset from file given a dataset name in form of a string
    for graph classification.

    :param name:    string of dataset name
    :return:        loaded networkX graphs, graph labels
    """
    dataset = Parser(name)

    graphs = dataset.parse_all_graphs()
    labels = [get_graph_label(graph) for graph in graphs]

    return graphs, labels


def load_dataset_node(name):
    """
    Loads the graph dataset from file given a dataset name in form of a string
    for node classification.

    :param name:    string of dataset name
    :return:        loaded networkX graphs, node labels
    """

    dataset = Parser(name)

    graphs = dataset.parse_all_graphs()
    labels = [get_node_labels(graph) for graph in graphs]

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
