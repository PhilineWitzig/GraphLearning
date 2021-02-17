"""
This file contains the building blocks for models and the assembled models.
"""

import tensorflow as tf
import os
from tensorflow.keras import layers

# suppress system info and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class GCN(layers.Layer):
    """
    Call function takes a list of two tensors as input.
    The first entry is the last node embedding.
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
        node_embeddings = inputs[0]
        adj_matrix_norm = inputs[1]
        x = tf.matmul(adj_matrix_norm, node_embeddings)
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
        config = super(GCN, self).get_config()
        config.update({"feature_num": self.feature_num})
        return config

    def compute_output_shape(self, input_shape):
        return input_shape[0][1], self.feature_num


class Skip_GCN(layers.Layer):
    """
    Similar to GCN layer, but with additional skip connection.
    """

    def __init__(self, feature_num):
        """
        :param feature_num: number of features this layers outputs.
        """
        super(Skip_GCN, self).__init__()
        self.feature_num = feature_num

    def build(self, input_shape):
        w_init = tf.initializers.GlorotUniform()
        self.w_aggregated_embedd = tf.Variable(initial_value=w_init(
            shape=(input_shape[0][-1], self.feature_num), dtype='float32'), trainable=True)
        self.w_own_embedd = tf.Variable(initial_value=w_init(
            shape=(input_shape[0][-1], self.feature_num), dtype='float32'), trainable=True)

    def call(self, inputs, **kwargs):
        if type(inputs) is not list or not len(inputs) == 2:
            raise Exception('GCN must be called on a list of two tensors. Got: ' + str(inputs))
        node_embeddings = inputs[0]
        adj_matrix_norm = inputs[1]
        x_adj = tf.matmul(adj_matrix_norm, node_embeddings)
        x_adj = tf.matmul(x_adj, self.w_aggregated_embedd)
        x_own = tf.matmul(node_embeddings, self.w_own_embedd)
        x = x_adj + x_own
        x = tf.nn.relu(x)
        return x

    def get_config(self):
        config = super(Skip_GCN, self).get_config()
        config.update({"feature_num": self.feature_num})
        return config

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
        pass

    def call(self, node_features, **kwargs):
        return tf.math.reduce_sum(node_features, axis=1)

    def get_config(self):
        config = super(SumPool, self).get_config()
        config.update({"num_outputs": self.num_outputs})
        return config


class DiffPool(layers.Layer):
    """
    Custom pooling layer, reducing the size of the graph, by assigning each node (softly) to one of k clusters.
    These clusters are the nodes of the new graph.
    """

    def __init__(self, no_clusters):
        """
        :param no_clusters: hyperparameter k determining the number of clusters
        """
        super(DiffPool, self).__init__()
        self.no_clusters = no_clusters

    def build(self, input_shape):
        no_features = input_shape[0][2]
        w_init = tf.initializers.GlorotUniform()
        self.weight1 = tf.Variable(initial_value=w_init(
            shape=(no_features, no_features), dtype='float32'), trainable=True)
        self.weight2 = tf.Variable(initial_value=w_init(
            shape=(no_features, self.no_clusters), dtype='float32'), trainable=True)

    def call(self, inputs, **kwargs):
        if type(inputs) is not list or not len(inputs) == 2:
            raise Exception('DiffPool must be called on a list of two tensors. Got: ' + str(inputs))

        node_embed = inputs[0]
        adj_matrix = inputs[1]
        # matrix depicting the learned cluster assigning function
        assign_matr = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(node_embed, self.weight1)),
                                              self.weight2))
        # computation of adjacency matrix and node embeddings for graph build from computed clusters
        adj_matrix_pooled = tf.matmul(tf.matmul(tf.transpose(assign_matr, perm=[0, 2, 1]), adj_matrix), assign_matr)
        node_embed_pooled = tf.matmul(tf.transpose(assign_matr, perm=[0, 2, 1]), node_embed)

        return [node_embed_pooled, adj_matrix_pooled]

    def get_config(self):
        config = super(DiffPool, self).get_config()
        config.update({"no_clusters": self.no_clusters})
        return config


class TopKPool(layers.Layer):
    """
    Custom pooling layer, reducing the size of the graph, by deleting several nodes and edges. Only k nodes are kept.
    The decision function, which nodes are kept is learned.
    """
    def __init__(self, top_k):
        """
        :param top_k: Determines how many nodes are kept (and how many deleted) by the pooling layer
        """
        super(TopKPool, self).__init__()
        self.top_k = top_k

    def build(self, input_shape):
        no_features = input_shape[0][2]

        w_init = tf.initializers.GlorotUniform()
        self.weight_p = tf.Variable(initial_value=w_init(
            shape=(no_features, 1), dtype='float32'), trainable=True)

    def call(self, inputs, **kwargs):
        if type(inputs) is not list or not len(inputs) == 2:
            raise Exception('TopKPool must be called on a list of two tensors. Got: ' + str(inputs))

        node_embed = inputs[0]
        adj_matrix = inputs[1]

        # scores the importance of each node, preparing for deleting those deemed unimportant
        scored_nodes = tf.matmul(node_embed, self.weight_p)
        scored_nodes = tf.squeeze(scored_nodes, axis=2)

        # selecting the top k nodes
        top_k_values, top_k_indices = tf.math.top_k(scored_nodes, self.top_k)

        # converting the vector of top_k_indices to a matrix, which can be used for gather_nd
        # this is done by expanding the vector to a matrix, by multiplying by one-vectors from left and right and then
        # concatening those matrices
        # E.g: top_k_indices = [0 2]
        #      idx_1 = [0 0     idx_2 = [0 2    idx_comb = [0,0 0,2
        #               2 2]             0 2]               2,0 2,2]
        idx_1 = tf.matmul(tf.expand_dims(top_k_indices, axis=2),
                          tf.expand_dims(tf.ones([1, self.top_k], dtype=tf.dtypes.int32), axis=0))
        idx_2 = tf.matmul(tf.expand_dims(tf.ones([self.top_k, 1], dtype=tf.dtypes.int32), axis=0),
                          tf.expand_dims(top_k_indices, axis=1))
        idx_comb = tf.concat([tf.expand_dims(idx_1, axis=-1), tf.expand_dims(idx_2, axis=-1)], axis=-1)

        # adapting the adjacency matrix and node embeddings to the filtered top k nodes
        adj_matrix_pooled = tf.gather_nd(adj_matrix, idx_comb, batch_dims=1)
        node_embed_pooled = tf.math.multiply(tf.gather_nd(node_embed, tf.expand_dims(top_k_indices, axis=-1), batch_dims=1),
                                             tf.expand_dims(tf.math.tanh(top_k_values), axis=-1))

        return [node_embed_pooled, adj_matrix_pooled]


def get_config(self):
        config = super(TopKPool, self).get_config()
        config.update({"top_k": self.top_k})
        return config
