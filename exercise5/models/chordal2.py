"""
This module contains the model (architecture) defintion for the CHORDAL2 dataset.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


from data import normalize_adj_matrix
from layers import GCN, SumPool


def get_chordal_two_model(n_nodes, n_features, num_classes):
    """
    Returns a model specifically tailored to classify the chordal two dataset.

    :param n_nodes:     number of nodes
    :param n_features:  number of features
    :param num_classes: number of target classes

    :return:            tf model
    """
    node_features_input_layer = layers.Input(shape=(n_nodes, n_features),
                                             name='node_feature_input')
    adjacency_matrix_input_layer = layers.Input(shape=(n_nodes, n_nodes),
                                                name='adjacency_matrix_input')

    x = GCN(64, "relu")([node_features_input_layer, adjacency_matrix_input_layer])
    x = layers.Dropout(rate=0.2)(x)

    x = SumPool(64)(x)
    x = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=[node_features_input_layer, adjacency_matrix_input_layer],
                           outputs=[x])
    return model


def preprocess_chordal_two_data(graphs, matrices, labels, feature_type):
    """
    Pre-processes the clique dataset. The parameters are the same for all pre-processing functions.
    The feature_type param defines which node features are to be extracted.

    :param graphs:          list on networkx graphs
    :param matrices:        adjacency matrices
    :param labels:          graph labels
    :param feature_type:    type of feature used for the GNNs

    :return:                normalized adjacency matrix, computed features, labels       
    """
    matrices = [normalize_adj_matrix(matrix) for matrix in matrices]
    matrices = np.asarray(matrices)

    if feature_type == "Eye":
        features = [np.eye(matrix.shape[0]) for matrix in matrices]
    elif feature_type == "Color":
        raise ValueError("Color not supported due to memory issues.")
    else:
        raise ValueError

    features = np.asarray(features)
    labels = np.asarray(labels)
    labels = labels - 1

    return matrices, features, labels
