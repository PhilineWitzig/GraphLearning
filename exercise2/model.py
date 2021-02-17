"""
This module contains the implementation of the GNN architectures.
"""

import tensorflow as tf
from tensorflow.keras import layers


class GCN(layers.Layer):
    """
    Call function takes a list of two tensors as input.
    The first entry  is the last node embedding.
    The second is the normalized adjacency matrix.
    """

    def __init__(self, feature_num, nl):
        """
        :param feature_num: number of features this layers outputs.
        :param nl: non-linearity which is applied after convoluting.
        """
        super(GCN, self).__init__()
        self.feature_num = feature_num
        self.nl = nl

    def build(self, input_shape):
        w_init = tf.initializers.GlorotUniform()
        self.w = tf.Variable(initial_value=w_init(
            shape=(input_shape[0][-1], self.feature_num), dtype='float32'), trainable=True)

    def call(self, inputs, **kwargs):
        if type(inputs) is not list or not len(inputs) == 2:
            raise Exception('GCN must be called on a list of two tensors. Got: ' + str(inputs))
        X = inputs[0]
        A = inputs[1]
        x = tf.matmul(A, X)
        x = tf.matmul(x, self.w)
        if self.nl == "relu":
            x = tf.nn.relu(x)
        elif self.nl == "softmax":
            x = tf.nn.softmax(x)
        else:
            raise ValueError(
                f"GCN Layer only supports 'relu' and 'softmax' as non linearity, but {self.nl} was passed.")
        return x

    def get_config(self):
        # necessary function for serialization of layer.
        # As we do not need this, we implement the function to omit warnings.
        pass

    def compute_output_shape(self, input_shape):
        return input_shape[0][1], self.feature_num


class SumPool(layers.Layer):
    """
    Custom pooling layer which computes the sum over all feature vectors.
    Call function takes one tensor as input.
    """

    def __init__(self, num_outputs):
        """
        Initializes the custom sum pooling layer.

        :param num_outputs: number of units in the sum pooling layer
        """
        super(SumPool, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=[int(input_shape[-1]),
                                                       self.num_outputs], trainable=False)

    def call(self, node_features, **kwargs):
        return tf.math.reduce_sum(node_features, axis=1)

    def get_config(self):
        # necessary function for serialization of layer.
        # As we do not need this, we implement the function to omit warnings.
        pass


def model_GCN_node(num_vertices, num_node_features, num_classes):
    """
    Model architecture of GCN for node classification.
    The model takes two inputs, the first needs to be the node features.
    The second needs to be the preprocess adjacency matrix.
    It outputs the label for each node.

    :param num_vertices: number of vertices in graphs
    :param num_node_features: number of node features
    :param num_classes: nummer of classes to classify

    :return: instance of tf.keras.Model implementing the GCN node model architecture
    """
    # input layers, depending on size of node_features and number of vertices
    node_features_input_layer = layers.Input(
        shape=(num_vertices, 1), name='node_feature_input')
    adjacency_matrix_input_layer = layers.Input(
        shape=(num_vertices, num_vertices), name='adjacency_matrix_input')
    x = GCN(32, "relu")([node_features_input_layer, adjacency_matrix_input_layer])
    x = layers.Dropout(rate=0.2)(x)
    x = GCN(num_classes, "softmax")([x, adjacency_matrix_input_layer])

    model = tf.keras.Model(inputs=[node_features_input_layer, adjacency_matrix_input_layer], outputs=[x],
                           name='GCN_Graph')
    return model


def model_GCN_graph(n_nodes, n_features, num_classes):
    """
    Model architecture of GCN for graph classification.

    :param n_node:      number of nodes we have features for
    :param n_features:  feature vector size
    :param num_classes: number of distinct labels

    :return:            instance of tf.keras.Model implementing the graph network
    """
    node_features_input_layer = layers.Input(shape=(n_nodes, n_features),
                                             name='node_feature_input')
    adjacency_matrix_input_layer = layers.Input(shape=(n_nodes, n_nodes),
                                                name='adjacency_matrix_input')
    gcn_1 = GCN(64, "relu")([node_features_input_layer, adjacency_matrix_input_layer])
    gcn_1 = layers.Dropout(rate=0.2)(gcn_1)

    gcn_2 = GCN(64, "relu")([gcn_1, adjacency_matrix_input_layer])
    gcn_2 = layers.Dropout(rate=0.2)(gcn_2)

    gcn_3 = GCN(64, "relu")([gcn_2, adjacency_matrix_input_layer])
    gcn_3 = layers.Dropout(rate=0.2)(gcn_3)

    gcn_4 = GCN(64, "relu")([gcn_3, adjacency_matrix_input_layer])
    gcn_4 = layers.Dropout(rate=0.2)(gcn_4)

    gcn_5 = GCN(64, "relu")([gcn_4, adjacency_matrix_input_layer])
    gcn_5 = layers.Dropout(rate=0.2)(gcn_5)

    sum_pool = SumPool(64)(gcn_5)
    sum_pool = layers.Dropout(rate=0.2)(sum_pool)

    fc_1 = layers.Dense(64, activation="relu")(sum_pool)
    output = layers.Dense(num_classes, activation="softmax")(fc_1)

    model = tf.keras.Model(inputs=[node_features_input_layer, adjacency_matrix_input_layer],
                           outputs=[output])
    return model
