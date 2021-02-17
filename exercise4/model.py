"""
This file contains the building blocks for models and the assembled models.

The underlying architecture are two blocks of convolutions which may be connected directly or with an advanced pooling
layer in between them.
"""

import tensorflow as tf
from tensorflow.keras import layers
from layers import GCN, SumPool, Skip_GCN, DiffPool


def model_baseline(n_nodes, n_features, num_classes, feature_numbers, depth):
    """
    This functions creates the baseline version of a simple model for component evaluation.
    It consists of two times 'depth' GCN layers with no pooling layer in between the convolution blocks.

    :param n_nodes:      number of nodes we have features for
    :param n_features:  feature vector size
    :param num_classes: number of distinct labels
    :param feature_numbers: input dimensions of layers
    :param depth: number of convolutional layers in each convolutional block

    :return:            instance of tf.keras.Model implementing the graph network
    """
    node_features_input_layer = layers.Input(shape=(n_nodes, n_features),
                                             name='node_feature_input')
    adjacency_matrix_input_layer = layers.Input(shape=(n_nodes, n_nodes),
                                                name='adjacency_matrix_input')

    # convolutional block 1
    x = GCN(feature_numbers[0], "relu")([node_features_input_layer, adjacency_matrix_input_layer])
    x = layers.Dropout(rate=0.2)(x)
    for i in range(depth - 1):
        x = GCN(feature_numbers[0], "relu")([x, adjacency_matrix_input_layer])
        x = layers.Dropout(rate=0.2)(x)

    # convolutional block 2
    for i in range(depth):
        x = GCN(feature_numbers[1], "relu")([x, adjacency_matrix_input_layer])
        x = layers.Dropout(rate=0.2)(x)

    # sum-pooling and dense connection to output nodes
    x = SumPool(feature_numbers[1])(x)
    x = layers.Dropout(rate=0.2)(x)
    x = layers.Dense(feature_numbers[1], activation="relu")(x)
    x = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=[node_features_input_layer, adjacency_matrix_input_layer],
                           outputs=[x])
    return model


def model_GCN_skip(n_nodes, n_features, num_classes, feature_numbers, depth):
    """
    This function creates a simple model with GCN skip layers, to evaluate those.
    It has the same architecture as the baseline model, with GCN_skip layers instead of regular GCN layers

    :param n_nodes:      number of nodes we have features for
    :param n_features:  feature vector size
    :param num_classes: number of distinct labels
    :param feature_numbers: input dimensions of layers
    :param depth: number of convolutional layers in each convolutional block

    :return:            instance of tf.keras.Model implementing the graph network
    """
    node_features_input_layer = layers.Input(shape=(n_nodes, n_features),
                                             name='node_feature_input')
    adjacency_matrix_input_layer = layers.Input(shape=(n_nodes, n_nodes),
                                                name='adjacency_matrix_input')

    # convolutional block 1
    x = Skip_GCN(feature_numbers[0])([node_features_input_layer, adjacency_matrix_input_layer])
    x = layers.Dropout(rate=0.2)(x)
    for i in range(depth - 1):
        x = Skip_GCN(feature_numbers[0])([x, adjacency_matrix_input_layer])
        x = layers.Dropout(rate=0.2)(x)

    # convolutional block 2
    for i in range(depth):
        x = Skip_GCN(feature_numbers[1])([x, adjacency_matrix_input_layer])
        x = layers.Dropout(rate=0.2)(x)

    # sum-pooling and dense connection to output nodes
    x = SumPool(feature_numbers[1])(x)
    x = layers.Dropout(rate=0.2)(x)
    x = layers.Dense(feature_numbers[1], activation="relu")(x)
    x = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=[node_features_input_layer, adjacency_matrix_input_layer],
                           outputs=[x])
    return model


def model_advanced_pool(n_nodes, n_features, num_classes, pool_type, feature_numbers, depth, adv_pool_k):
    """
    This function creates a simple model with different pooling methods in between the convolutions to evaluate those.

    :param n_nodes:      number of nodes we have features for
    :param n_features:  feature vector size
    :param num_classes: number of distinct labels
    :param pool_type: advanced Pool layer to be used for in-between pooling
    :param feature_numbers: input dimensions of layers
    :param depth: number of convolutional layers in each convolutional block
    :param adv_pool_k: hyperparameter k, determining how many nodes are making up the pooled graph

    :return:            instance of tf.keras.Model implementing the graph network
    """

    node_features_input_layer = layers.Input(shape=(n_nodes, n_features),
                                             name='node_feature_input')
    adjacency_matrix_input_layer = layers.Input(shape=(n_nodes, n_nodes),
                                                name='adjacency_matrix_input')

    # convolutional block 1
    x = GCN(feature_numbers[0], "relu")([node_features_input_layer, adjacency_matrix_input_layer])
    x = layers.Dropout(rate=0.2)(x)

    for i in range(depth - 1):
        x = GCN(feature_numbers[0], "relu")([x, adjacency_matrix_input_layer])
        x = layers.Dropout(rate=0.2)(x)

    # advanced pooling layer
    adv_pool = pool_type(adv_pool_k)([x, adjacency_matrix_input_layer])

    # convolutional block 2
    x = GCN(feature_numbers[1], "relu")(adv_pool)
    x = layers.Dropout(rate=0.2)(x)

    for i in range(depth - 1):
        x = GCN(feature_numbers[1], "relu")([x, adv_pool[1]])
        x = layers.Dropout(rate=0.2)(x)

    # sum-pooling and dense connection to output nodes
    x = SumPool(feature_numbers[1])(x)
    x = layers.Dropout(rate=0.2)(x)
    x = layers.Dense(feature_numbers[1], activation="relu")(x)
    x = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=[node_features_input_layer, adjacency_matrix_input_layer],
                           outputs=[x])
    return model



def model_opt_architecture_enzymes(dataset_name, n_nodes, n_features, n_classes):

    """
    Definition of the network performing best on the datasets only using GCN
    layers as well as DiffPool layers.

    :param dataset_name:name of loaded dataset as string
    :param n_nodes:     number of input nodes
    :param n_features:  number of node features (feature vec size)
    :param n_classes:   number of target classes

    :return:            keras model following the diff pool architecture

    """

    # input layers
    node_features_input_layer = layers.Input(shape=(n_nodes, n_features),
                                             name='node_feature_input')
    adjacency_matrix_input_layer = layers.Input(shape=(n_nodes, n_nodes),
                                                name='adjacency_matrix_input')

    # skip convolution and diff pool layers in turn
    x = GCN(64, "relu")([node_features_input_layer, adjacency_matrix_input_layer])
    x = layers.Dropout(rate=0.3)(x)
    x = GCN(64, "relu")([x, adjacency_matrix_input_layer])
    x = layers.Dropout(rate=0.3)(x)
    x = GCN(64, "relu")([x, adjacency_matrix_input_layer])
    x = layers.Dropout(rate=0.3)(x)

    x_pooled = DiffPool(16)([x, adjacency_matrix_input_layer])

    x = GCN(64, "relu")(x_pooled)
    x = layers.Dropout(rate=0.3)(x)

    # global sum pooling and output layer
    global_pool = SumPool(64)(x)
    output = layers.Dense(n_classes, activation="softmax")(global_pool)
    model = tf.keras.Model(inputs=[node_features_input_layer, adjacency_matrix_input_layer],
                           outputs=[output])

    return model


def model_opt_architecture(dataset_name, n_nodes, n_features, n_classes):

    """
    Definition of the network performing best on the datasets only using GCN
    layers as well as DiffPool layers.

    :param dataset_name:name of loaded dataset as string
    :param n_nodes:     number of input nodes
    :param n_features:  number of node features (feature vec size)
    :param n_classes:   number of target classes

    :return:            keras model following the diff pool architecture

    """

    # input layers
    node_features_input_layer = layers.Input(shape=(n_nodes, n_features),
                                             name='node_feature_input')
    adjacency_matrix_input_layer = layers.Input(shape=(n_nodes, n_nodes),
                                                name='adjacency_matrix_input')

    # skip convolution and diff pool layers in turn
    x = GCN(64, "relu")([node_features_input_layer, adjacency_matrix_input_layer])
    x = GCN(64, "relu")([x, adjacency_matrix_input_layer])
    x = GCN(64, "relu")([x, adjacency_matrix_input_layer])
    x = layers.Dropout(rate=0.3)(x)

    if dataset_name == "NCI1":
        x_pooled = DiffPool(15)([x, adjacency_matrix_input_layer])
    elif dataset_name == "PROTEINS":
        x_pooled = DiffPool(20)([x, adjacency_matrix_input_layer])
    else:
        print("Invalid dataset. Must be either NCI1 or PROTEINS")
        return

    x = GCN(64, "relu")(x_pooled)
    x = layers.Dropout(rate=0.3)(x)

    # global sum pooling and output layer
    global_pool = SumPool(64)(x)
    output = layers.Dense(n_classes, activation="softmax")(global_pool)
    model = tf.keras.Model(inputs=[node_features_input_layer, adjacency_matrix_input_layer],
                           outputs=[output])

    return model


def model_2diff_architecture(dataset_name, n_nodes, n_features, n_classes):

    """
    Definition of the network having two Diff Pool layers. The first layer
    has a reduction rate of 50% of the avergae number of input nodes. The second
    Diff Pool layer has again a reduction rate of 50%.


    :param dataset_name:name of loaded dataset as string
    :param n_nodes:     number of input nodes
    :param n_features:  number of node features (feature vec size)
    :param n_classes:   number of target classes

    :return:            keras model following the diff pool architecture

    """

    # input layers
    node_features_input_layer = layers.Input(shape=(n_nodes, n_features),
                                             name='node_feature_input')
    adjacency_matrix_input_layer = layers.Input(shape=(n_nodes, n_nodes),
                                                name='adjacency_matrix_input')

    # skip convolution and diff pool layers in turn
    x = GCN(64, "relu")([node_features_input_layer, adjacency_matrix_input_layer])
    x = GCN(64, "relu")([x, adjacency_matrix_input_layer])
    x = GCN(64, "relu")([x, adjacency_matrix_input_layer])
    x = layers.Dropout(rate=0.3)(x)

    if dataset_name == "NCI1":
        x_pooled = DiffPool(15)([x, adjacency_matrix_input_layer])
    elif dataset_name == "PROTEINS":
        x_pooled = DiffPool(20)([x, adjacency_matrix_input_layer])
    elif dataset_name == "ENZYMES":
        x_pooled = DiffPool(16)([x, adjacency_matrix_input_layer])
    else:
        print("Invalid dataset {}. Must be either NCI1, ENZYMES oder PROTEINS".format(dataset_name))
        return

    x = GCN(64, "relu")(x_pooled)
    x = GCN(64, "relu")([x, x_pooled[1]])
    x = GCN(64, "relu")([x, x_pooled[1]])
    x = layers.Dropout(rate=0.3)(x)

    if dataset_name == "NCI1":
        x_pooled = DiffPool(7)([x, x_pooled[1]])
    elif dataset_name == "PROTEINS":
        x_pooled = DiffPool(10)([x, x_pooled[1]])
    elif dataset_name == "ENZYMES":
        x_pooled = DiffPool(8)([x, x_pooled[1]])
    else:
        print("Invalid dataset {}. Must be either NCI1, ENZYMES oder PROTEINS".format(dataset_name))
        return

    x = GCN(64, "relu")(x_pooled)
    x = layers.Dropout(rate=0.3)(x)

    # global sum pooling and output layer
    global_pool = SumPool(64)(x)
    output = layers.Dense(n_classes, activation="softmax")(global_pool)
    model = tf.keras.Model(inputs=[node_features_input_layer, adjacency_matrix_input_layer],
                           outputs=[output])

    return model
