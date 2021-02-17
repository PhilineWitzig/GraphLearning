"""
This module contains the model (architecture) defintion for the CONNECT dataset.
"""
import numpy as np

from data import normalize_adj_matrix
from models.models import get_gcn_basic


def get_connect_model(n_nodes, n_features, num_classes):
    """
    Returns a model specifically tailored to classify the chordal one dataset.
    (Since a non GCN method was better suited for classifying this only returns a general model)
    """
    model = get_gcn_basic(n_nodes, n_features, num_classes)

    return model


def preprocess_connect_data(graphs, matrices, labels, feature_type):
    """
    Pre-processes the connect dataset. The parameters are the same for all pre-processing functions.
    The feature_type param defines which node features are to be extracted from the graphs
    (this is not necessary for this dataset).
    """
    matrices = [normalize_adj_matrix(matrix) for matrix in matrices]
    features = [np.eye(matrix.shape[0]) for matrix in matrices]

    matrices = np.asarray(matrices)
    features = np.asarray(features)
    labels = np.asarray(labels)
    labels = labels - 1

    return matrices, features, labels
