"""
A collection of general purpose models to be used for more than one dataset.
"""
import tensorflow as tf
from tensorflow.keras import layers

from layers import GCN, SumPool, Skip_GCN


def get_gcn_basic(n_nodes, n_features, num_classes):
    node_features_input_layer = layers.Input(shape=(n_nodes, n_features),
                                             name='node_feature_input')
    adjacency_matrix_input_layer = layers.Input(shape=(n_nodes, n_nodes),
                                                name='adjacency_matrix_input')

    # convolutional block 1
    x = GCN(32, "relu")([node_features_input_layer, adjacency_matrix_input_layer])
    # x = layers.Dropout(rate=0.3)(x)

    # sum-pooling and dense connection to output nodes
    x = SumPool(32)(x)
    x = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=[node_features_input_layer, adjacency_matrix_input_layer],
                           outputs=[x])
    return model
