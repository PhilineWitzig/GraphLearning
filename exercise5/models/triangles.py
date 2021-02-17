"""
This module contains the model (architecture) defintion for the TRIANGLES dataset.
"""
import numpy as np

from data import normalize_adj_matrix
from models.models import get_gcn_basic
from utils.data_utils import *
import networkx as nx
from data import *
import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import random
import sys

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, AveragePooling1D, MaxPooling1D


def get_triangles_model(n_nodes, n_features):
    """
    Build and return neural network to classify TRIANGLES dataset using the
    neighborhood feature.

    :param n_nodes:     maximal number of nodes over all graphs in dataset
    :param n_features:  size of considered neighborhood in neighborhood node feature.
                        E.g. if neighborhood feature of a node is
                        of size 9x9 n_features would be 9.

    :return:            Tensorflow Neural Network
    """

    node_features_input_layer = tf.keras.Input((n_nodes*n_features*n_features), dtype=tf.float32)
    # reshape input for average pool
    x = Reshape((-1, n_nodes*n_features*n_features))(node_features_input_layer)
    # Avoid Overfitting by pooling and dropout
    x = AveragePooling1D(pool_size=6, padding="valid", data_format='channels_first')(x)
    x = Dense(units=60, activation='relu')(x)
    x = AveragePooling1D(pool_size=6, padding="valid", data_format='channels_first')(x)
    x = Dense(units=60, activation='relu')(x)
    x = AveragePooling1D(pool_size=6, padding="valid", data_format='channels_first')(x)
    x = Dense(units=32, activation='relu')(x)
    x = Dropout(0.1)(x)

    x = Dense(units=32, activation='relu')(x)
    #x = Dense(units= 32,activation='relu')(x)

    x = Reshape((32,))(x)  # reshape to get appropriate prediction shape
    x = Dense(units=26, activation='softmax')(x)

    model = tf.keras.Model(inputs=node_features_input_layer, outputs=x)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def preprocess_triangles_data(graphs, max_size=9):
    """
    Pre-processes the clique dataset. The parameters are the same for all pre-processing functions.
    The feature_type param defines which node features are to be extracted from the graphs
    (this is not necessary for this dataset).

    :param graphs:      list of networkx graphs
    :param max_size:    maximum neighborhood size

    :return:            number of features
    """

    features = get_dataset_information(graphs, 38, max_size)
    return features


def get_neighbor_information_graph_classification(G, adjecency_matrix, node, max_size):
    """
    Computes neighborhood features on node level: The neighborhood feature of a node is the adjacency matrix of the subgraph
    induced by the neighbors of node, excluding node itself.
    We restrict the number of considered neighbors to be max_size, and if a node has
    less than max_size neighbors, the subgraph is padded with isolated dummy nodes. I.e. the last columns and rows contain 0
    values.

    :param G:                   networkX graph
    :param adjecency_matrix:    adjacency Matrix of G
    :param node:                node for which feature will be computed
    :param max_size:            maximal size of considered neighborhood of node (not including the node itself).
                                I.e. the feature will have size max_size times max_size

    :return:                    feature of node
    """
    # Find neighbors of the node by getting the index of every '1' entry in adjacency matrix of G
    neighbours_of_node = [index for index in range(
        len(adjecency_matrix[node])) if adjecency_matrix[node][index] == 1][:max_size]

    # cut adjecency matrix to the neighbor rows and columns to obtain induced subgraph matrix of nodes' neighbors
    neighbor_matrix = adjecency_matrix[neighbours_of_node, :][:, neighbours_of_node]

    # Pad matrix to size
    pad_size = max_size - np.shape(neighbor_matrix)[0]
    # Create rows and columns of 0 atappropriate shape
    pad_row = np.tile([0]*np.shape(neighbor_matrix)[0], (pad_size, 1))
    pad_column = np.tile([[0]]*max_size, (1, pad_size))

    neighbor_matrix = np.append(neighbor_matrix, pad_row, axis=0)
    neighbor_matrix = np.append(neighbor_matrix, pad_column, axis=1)

    return neighbor_matrix


def get_graph_information_graph_classification(G, max_size):
    """
    Computes the graph neighborhood feature: The graph neighborhood feature is a
    concatenation of node neighborhood features.

    :param G:           networkX Graph
    :param max_size:    maximal size of considered neighborhood of node (not including the node itself).
                        I.e. the feature will have size max_size times max_size.

    :return:            Vector containing all node features for each node in graph G.
                        I.e. the size is number of nodes in G times (max_size * max_size)
    """
    adjecency_matrix = get_adjacency_matrix(G)
    neighbor_vectors = []

    # Get node features for each node in G and save them to the neighborhood vectors
    for node in range(len(adjecency_matrix)):
        neighbor_vectors.append(get_neighbor_information_graph_classification(
            G, adjecency_matrix, node, max_size))

    # Stack all node features
    neighbor_vectors = np.stack(neighbor_vectors)

    return neighbor_vectors


def get_dataset_information(graphs, max_nodes, max_size):
    '''
    Computes the graph neighborhood feature for each graph in the dataset
    specified by graphs.

    :param graphs:      list of NetworkX graphs
    :param max_nodes:   maximal number of node any NetworkX graph in graphs
                        can have, used for padding
    :param max_size:    maximal size of considered neighborhood for any node
                        when computing it's neighborhood feature

    :return:            feature vectors representing the dataset
    '''

    dataset_vectors = []

    # Initialize padding matrix for "missing" nodes in graph
    pad = np.asarray([[0]*max_size]*max_size)

    # Compute graph neighborhood feature for each graph in the dataset
    for G in tqdm.tqdm(graphs, desc="Preprocessing graphs"):
        graph_feature = get_graph_information_graph_classification(G, max_size)

        # Pad the graph feature whenever neccessary by 0 matrices: If graph G has less nodes than the maximal number of nodes
        # of any graph in the dataset, add padding matrices consisting of 0 to the graph feature
        while(len(graph_feature) < max_nodes):
            graph_feature = np.concatenate((graph_feature, [pad]), axis=0)

        # Save computed graph feature as flattened array
        dataset_vectors.append(graph_feature.flatten())

    # Stack all graph features
    dataset_vectors = np.stack(dataset_vectors)

    return dataset_vectors
