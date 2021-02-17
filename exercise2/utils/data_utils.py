#!/usr/bin/env python3
import networkx as nx
import numpy as np


def get_adjacency_matrix(G):
    """
    :param G: A networkx graph
    :return: The adjacency matrix as a dense numpy array
    """
    A = nx.to_numpy_array(G)
    return A


def get_graph_label(G):
    """
    :param G: A networkx graph
    :return: The label (class) of G
    """
    c = G.graph['label']
    return c


def get_node_labels(G):
    """
    :param G: A networkx graph G=(V,E)
    :return: A array of node labels of length |V|
    """
    labels = np.int32([node[1]["node_label"] for node in G.nodes(data=True)])
    return labels


def get_node_attributes(G):
    """
    :param G: A networkx graph G=(V,E)
    :return: A numpy array of shape (|V|, a), where a is the length of the node attribute vector
    """
    attributes = np.float32([node[1]["node_attributes"] for node in G.nodes(data=True)])
    return attributes


def get_padded_adjacency(graphs):
    """
    Computes a 3D Tensor A of shape (k,n,n) that stacks all adjacency matrices.
    Here, k = |graphs|, n = max(|V|) and A[i,:,:] is the padded adjacency matrix of the i-th graph.
    :param graphs: A list of networkx graphs
    :return: Numpy array A
    """
    max_size = np.max([g.order() for g in graphs])
    A_list = [get_adjacency_matrix(g) for g in graphs]
    A_padded = [np.pad(A, [0, max_size-A.shape[0]]) for A in A_list]

    return np.float32(A_padded)


def get_padded_node_labels(graphs):
    """
    Computes a 3D Tensor X with shape (k, n, l) that stacks the node labels of all graphs.
    Here, k = |graphs|, n = max(|V|) and l is the number of distinct node labels.
    Node labels are encoded as l-dimensional one-hot vectors.

    :param graphs: A list of networkx graphs
    :return: Numpy array X
    """
    node_labels = [get_node_labels(g) for g in graphs]
    all_labels = np.hstack(node_labels)
    max_label = np.max(all_labels)
    min_label = np.min(all_labels)
    label_count = max_label-min_label+1

    max_size = np.max([g.order() for g in graphs])
    n_samples = len(graphs)

    X = np.zeros((n_samples, max_size, label_count), dtype=np.float32)
    for i, g in enumerate(graphs):
        X[i, np.arange(len(g.nodes())), node_labels[i]-min_label] = 1.0

    return X


def get_padded_node_attributes(graphs):
    """
    Computes a 3D Tensor X with shape (k, n, a) that stacks the node attributes of all graphs.
    Here, k = |graphs|, n = max(|V|) and a is the length of the attribute vectors.

    :param graphs: A list of networkx graphs
    :return: Numpy array X
    """
    node_attributes = [get_node_attributes(g) for g in graphs]

    max_size = np.max([g.order() for g in graphs])
    padded = [np.vstack([x, np.zeros((max_size-x.shape[0], x.shape[1]), dtype=np.float32)])
              for x in node_attributes]
    stacked = np.stack(padded, axis=0)
    return stacked
