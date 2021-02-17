"""
Implements all models for Assignment 3.
"""

import tensorflow as tf
import config
from tensorflow.keras import layers


def get_model(input_layers, no_nodes):

    node_input = input_layers[0]
    rw_input = input_layers[1]
    neg_sample_input = input_layers[2]

    embedding = layers.Embedding(input_dim=no_nodes, output_dim=config.EMBED_DIM, name='embedding')

    node_embedding = embedding(node_input)
    rw_embedding = embedding(rw_input)
    neg_sample_embedding = embedding(neg_sample_input)

    x = tf.matmul(rw_embedding, tf.transpose(node_embedding, perm=[0, 2, 1]))
    x = tf.math.exp(x)

    # computing normalizing factor consisting of random walk and negative sample embeddings
    rw_and_neg_sample = tf.concat([rw_embedding, neg_sample_embedding], axis=1)
    normalizer = tf.matmul(rw_and_neg_sample, tf.transpose(node_embedding, perm=[0, 2, 1]))
    normalizer = tf.math.exp(normalizer)
    normalizer = tf.math.reduce_sum(normalizer)

    x = x / normalizer

    model = tf.keras.Model(inputs=[node_input, rw_input, neg_sample_input], outputs=[x])
    return model
