"""
This module contains the model (architecture) defintion for the CLIQUE dataset.
"""
import logging

from utils.data_utils import *
import networkx as nx
from data import *
import tqdm
import numpy as np
import random
import sys
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense


def get_clique_model(n_features):
    """
    Returns a model specifically tailored to classify the Clique dataset using
    node neighbor feature.

    :param n_features:  size of considered neighborhood in neighborhood node feature. E.g.
                        if neighborhood feature of a node is
                        of size 9x9 n_features would be 9.

    :return: Tensorflow Neural Network
    """

    node_features_input_layer = tf.keras.Input((n_features*n_features), dtype=tf.float32)
    x = Dense(units=n_features)(node_features_input_layer)
    x = Dense(units=n_features)(x)
    x = Dense(units=n_features)(x)
    x = Dense(units=2, activation=tf.keras.activations.softmax)(x)

    model = tf.keras.Model(inputs=node_features_input_layer, outputs=x)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def preprocess_clique_data(graph, labels):
    """
    Pre-processes the clique dataset. The parameters are the same for all
    pre-processing functions.
    The feature_type param defines which node features are to be extracted.

    :param graph:   networkx graph
    :param labels:  list of labels

    :return:        computed features and label
    """

    features = get_graph_information(graph, 6)
    features = features.reshape((features.shape[0], 6 * 6))

    labels = np.asarray(labels)
    labels = labels - 1
    return features, labels[0]


def get_neighbor_information(G, adjecency_matrix, node, max_size):
    """
    Computes neighborhood features on node level: The neighborhood feature of a node is the adjacency matrix of the subgraph
    induced by the neighbors of node, excluding node itself.
    We restrict the number of considered neighbors to be max_size, and if a node has
    less than max_size neighbors, the subgraph is padded with isolated dummy nodes. I.e. the last columns and rows contain 0
    values.

    :param G:                networkX graph
    :param adjecency_matrix: adjacency Matrix of G
    :param node:             node for which feature will be computed
    :param max_size:         Maximal size of considered neighborhood of node
                             (not including the node itself).
                             I.e. the feature will have size max_size times max_size

    :return: Feature of node
    """

    neighbours_of_node = [neighbor-1 for neighbor in G.neighbors(node)]
    # index in adj. matrix of a neighbor node is node index -1

    # cut adjecency matrix to the neighbors rows and columns
    neighbor_matrix = adjecency_matrix[neighbours_of_node, :][:, neighbours_of_node]

    # Pad matrix to size by using 0 columns and rows
    pad_size = max_size - np.shape(neighbor_matrix)[0]
    pad_row = np.tile([0]*np.shape(neighbor_matrix)[0], (pad_size, 1))
    pad_column = np.tile([[0]]*max_size, (1, pad_size))

    neighbor_matrix = np.append(neighbor_matrix, pad_row, axis=0)
    neighbor_matrix = np.append(neighbor_matrix, pad_column, axis=1)

    return neighbor_matrix


def get_graph_information(G, max_size):
    """
    Computes the graph neighborhood feature: The graph neighborhood feature is
    a concatenation of node neighborhood features.

    :param G:           NetworkX Graph
    :param max_size:    Maximal size of considered neighborhood of node (not
                        including the node itself).
                        I.e. the feature will have size max_size times max_size.

    :return:            Vector containing all node features for each node in graph G.
                        I.e. the size is number of nodes in G times (max_size * max_size)
    """

    adjecency_matrix = get_adjacency_matrix(G)

    neighbor_vectors = []
    # Compute neighborhood feaure for each nde in graph
    for node in tqdm.tqdm(G.nodes, desc="Getting features"):
        neighbor_vectors.append(get_neighbor_information(G, adjecency_matrix, node, max_size))

    # Stack all node neighborhood features for each node in the graph
    neighbor_vectors = np.stack(neighbor_vectors)

    return neighbor_vectors
