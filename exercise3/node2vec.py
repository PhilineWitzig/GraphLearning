"""
This module is used for training the NNs.
Run this script to start training.
"""
import logging
import tensorflow as tf
from data import load_dataset
from random_walk import compute_random_walks
from model import get_model

import numpy as np
import config

from tensorflow.keras import layers


def custom_loss(y_true, y_pred):
    """
    Computes the custom loss for the training of the embedding.
    """
    neg_log = - tf.math.log(y_pred)
    loss = tf.math.reduce_sum(neg_log)
    return loss


def train_node2vec(dataset_name, param_return, param_in_out, no_rw=5, length_rw=5):
    """
    Trains the node2vec embedding.
    :param dataset_name:    string for name of dataset
    :param param_return: Bias of going back to the old node (p)
    :param param_in_out: Bias of moving forward to a new node (q)
    :param no_rw: number of random walks, default is 5
    :param length_rw: length of random walks, default is 5

    :return: the trained embedding matrix
    """

    logging.info("Training the node2vec embedding.")
    dataset = load_dataset(dataset_name)

    # we assume one graph per dataset for this assignment
    graph = dataset[0]
    number_nodes = graph.number_of_nodes()

    # TODO test parameters
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)

    logging.info("Computing random walks.")
    walks, walks_neg = compute_random_walks(graph, param_return, param_in_out, no_rw)

    logging.info("Compiling model.")
    node_input = layers.Input(shape=1, name="input_node")
    rw_input = layers.Input(shape=length_rw, name="input_random_walk")
    rw_neg_input = layers.Input(shape=length_rw, name="input_negative_sample")

    embed_model = get_model([node_input, rw_input, rw_neg_input], number_nodes)
    embed_model.compile(optimizer=tf.keras.optimizers.Adam(lr=config.EMBED_LR), loss=custom_loss)

    logging.info("Training.")
    node_input = np.asarray(np.arange(number_nodes).tolist() * no_rw)
    dummy_labels = np.zeros([number_nodes * no_rw, length_rw, 1], dtype='float64')
    embed_model.fit(x=[node_input, walks.astype(np.float64), walks_neg.astype(np.float64)],
                    y=dummy_labels, batch_size=config.EMBED_BATCH_SIZE, epochs=config.EMBED_EPOCH_MAX, verbose=2,
                    callbacks=[early_stopping_callback])

    embed_layer = embed_model.get_layer('embedding')
    embed_weights = embed_layer.get_weights()[0]

    return embed_weights
