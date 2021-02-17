"""
This module contains the model (architecture) defintion for the CHORDAL1 dataset.
"""
import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from data import normalize_adj_matrix
from feature_extractor import color_features
from layers import GCN, SumPool


def get_chordal_one_model(n_nodes, n_features, num_classes):
    """
    Returns a model specifically tailored to classify the chordal one dataset.

    :param n_nodes:     number of nodes
    :param n_features:  number of features
    :param num_classes: number of target classes

    :return:            tf model
    """
    node_features_input_layer = layers.Input(shape=(n_nodes, n_features),
                                             name='node_feature_input')
    adjacency_matrix_input_layer = layers.Input(shape=(n_nodes, n_nodes),
                                                name='adjacency_matrix_input')

    x = GCN(16, "relu")([node_features_input_layer, adjacency_matrix_input_layer])
    # x = layers.Dropout(rate=0.3)(x)

    # sum-pooling and dense connection to output nodes
    x = SumPool(16)(x)
    x = layers.Dense(num_classes, activation="softmax",
                     activity_regularizer=tf.keras.regularizers.l1_l2(0.1))(x)

    model = tf.keras.Model(inputs=[node_features_input_layer, adjacency_matrix_input_layer],
                           outputs=[x])
    return model


def preprocess_chordal_one_data(graphs, matrices, labels, feature_type):
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
        logging.info("Computing graph coloring for node features.")
        features = color_features(graphs, matrices.shape[1])
        logging.info("Finished node feature computation.")
    else:
        raise ValueError
    features = np.asarray(features)

    labels = np.asarray(labels)

    return matrices, features, labels
