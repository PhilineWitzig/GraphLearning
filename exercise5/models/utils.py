"""
Functions for loading and saving models for different datasets.
"""
import os

import numpy as np
import tensorflow as tf
import config
import logging
from sklearn import svm, model_selection


def load_clique_model():
    model = tf.keras.models.load_model(os.path.join("models", "saved", "CLIQUE"))
    return model


def load_triangles_model():
    model = tf.keras.models.load_model(os.path.join("models", "saved", "TRIANGLES"))
    return model


def load_chordal1_model():
    model = tf.keras.models.load_model(os.path.join("models", "saved", "CHORDAL1"))
    return model


def load_chordal2_model():
    model = tf.keras.models.load_model(os.path.join("models", "saved", "CHORDAL2"))
    return model


def load_connect_model(features, train_labels):
    logging.info("Maximum number of iterations: {}. Refinement steps: {}"
                 .format(config.MAX_ITER, config.REFINEMENT_STEPS))
    clf = svm.SVC(kernel="precomputed", max_iter=config.MAX_ITER)
    clf.fit(np.dot(features, features.T), train_labels)
    return clf


def store_clique_model():
    return 42


def store_triangles_model():
    return 42


def store_chordal1_model():
    return 42
