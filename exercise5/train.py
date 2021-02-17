"""
This module contains the main training loops for the different datasets.
"""
import os
import logging
import tensorflow as tf
import numpy as np
import random
import sys
import config
from data import load_dataset_node, load_dataset_graph
from models.chordal1 import preprocess_chordal_one_data, get_chordal_one_model
from models.chordal2 import get_chordal_two_model, preprocess_chordal_two_data
from models.clique import get_clique_model, preprocess_clique_data
from models.connect import get_connect_model, preprocess_connect_data
from models.triangles import get_triangles_model, preprocess_triangles_data
from utils.data_utils import get_padded_adjacency
from kernels import weisfeiler_lehman_kernel, degree_kernel

from sklearn import svm, model_selection
from sklearn.model_selection import RepeatedKFold
# suppress svm convergence warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

@ignore_warnings(category=ConvergenceWarning)
def train(modeltype, dataset, data_path, feature_type=None):
    """
    General training function for training and evaluating on different datasets
    with different classification strategies.
    """
    logging.info(f"Train on {dataset} via {modeltype}.")

    # depending on dataset laods either node or graph labels
    if dataset == "CLIQUE":
        loader = load_dataset_node
    else:
        loader = load_dataset_graph

    train_graphs, train_labels = loader(data_path + "_Train")
    eval_graphs, eval_labels = loader(data_path + "_Eval")


    if dataset=="CLIQUE":
        """
        Trains Tensorflow neural network for CLIQUE dataset and saves it

        """

        G = train_graphs[0]

        train_features, train_labels = preprocess_clique_data(G, train_labels)
        train_features = train_features.reshape((train_features.shape[0], 6 * 6))
        _, epochs, _ = get_gcn_params(dataset)
        model = get_clique_model(6)

        fit = model.fit(train_features, train_labels, epochs=epochs)
        train_scores = fit.history['accuracy']
        logging.info(f"Train Accs: {train_scores}")
        logging.info(f"Avg: {np.average(train_scores):.4f}")
        logging.info(f"Std: {np.std(train_scores):.4f}")
        logging.info(f"Max: {np.max(train_scores):.4f}")

        model.save(os.path.join("models", "saved", "CLIQUE_test"))
        sys.exit()

    if dataset == "TRIANGLES":
        """
        Trains Tensorflow neural network for TRIANGLES dataset and saves it
        """
        ## The original dataset is sorted by class label. We shuffle the graphs and labels to stop the classifier
        # from fitting to any ordering.

        zipped_graphs_labels = list(zip(train_graphs, train_labels))#zip graphs and labels to obtain the same shuffeling of both
        random.shuffle(zipped_graphs_labels)
        train_graphs, train_labels = zip(*zipped_graphs_labels)#unzip again

        #Get neighborhood features for the training dataset
        train_features = preprocess_triangles_data(train_graphs) #maximal number of nodes is 38 in the dataset
        train_labels= np.asarray(train_labels)

        _, epochs, _ = get_gcn_params(dataset)
        #Get the model and fit it
        model = get_triangles_model(38,9)
        fit = model.fit(train_features, train_labels, epochs=epochs)
        train_scores = fit.history['accuracy']
        logging.info(f"Train Accs: {train_scores}")
        logging.info(f"Avg: {np.average(train_scores):.4f}")
        logging.info(f"Std: {np.std(train_scores):.4f}")
        logging.info(f"Max: {np.max(train_scores):.4f}")
        model.save(os.path.join("models", "saved", "TRIANGLES_test"))
        #os._exit(0)
        sys.exit()

    # classify depending on classification strategy
    if modeltype == 'GCN':
        # obtain preprocessed and split data
        train_matrices, train_features, train_labels, eval_matrices, eval_features, eval_labels = get_preprocessed_data(
            dataset, train_graphs, train_labels, eval_graphs, eval_labels, feature_type)

        # get dataset specific model and hyperparams
        num_nodes = train_matrices[0].shape[1]
        num_features = train_features[0].shape[1]
        num_classes = len(np.unique(train_labels))
        model = get_model(dataset, num_nodes, num_features, num_classes)
        lr, epochs, batchsize = get_gcn_params(dataset)

        # define callbacks
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3
        )

        checkpoint_dir = os.path.join('models', 'saved', dataset)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        # compile, train and evaluate model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        fit = model.fit(x=[train_features, train_matrices], y=train_labels, batch_size=batchsize,
                        epochs=epochs, verbose=0, shuffle=True,
                        validation_data=([eval_features, eval_matrices], eval_labels),
                        callbacks=[model_checkpoint_callback])

        train_scores = fit.history['accuracy']
        val_scores = fit.history['val_accuracy']

    elif modeltype == "WL":
        logging.info("Maximum number of iterations: {}. Refinement steps: {}"
                     .format(config.MAX_ITER, config.REFINEMENT_STEPS))
        wl_matrix = weisfeiler_lehman_kernel(train_graphs)
        clf = svm.SVC(kernel="precomputed", max_iter=config.MAX_ITER)
        scores = model_selection.cross_validate(clf, wl_matrix, train_labels,
                                                cv=RepeatedKFold(n_splits=10, n_repeats=10),
                                                return_train_score=True)
        train_scores = scores['train_score']
        val_scores = scores['test_score']

    elif modeltype == "DK": # degree kernel
        deg_matrix = degree_kernel(train_graphs)
        clf = svm.SVC(kernel="precomputed", max_iter=config.MAX_ITER)
        scores = model_selection.cross_validate(clf, deg_matrix, train_labels,
                                                cv=RepeatedKFold(n_splits=10, n_repeats=10),
                                                return_train_score=True)
        train_scores = scores['train_score']
        val_scores = scores['test_score']
    else:
        raise ValueError

    # Experiment Output
    logging.info(f"Train Accs: {train_scores}")
    logging.info(f"Avg: {np.average(train_scores):.4f}")
    logging.info(f"Std: {np.std(train_scores):.4f}")
    logging.info(f"Max: {np.max(train_scores):.4f}")

    logging.info(f"Eval Accs: {val_scores}")
    logging.info(f"Avg: {np.average(val_scores):.4f}")
    logging.info(f"Std: {np.std(val_scores):.4f}")
    logging.info(f"Max: {np.max(val_scores):.4f}")


def get_preprocessed_data(dataset, train_graphs, train_labels, eval_graphs, eval_labels, feature_type):
    """
    Pre-processes the datasets.

    :param dataset:         string of input dataset
    :param train_graphs:    list of networkx graphs for training
    :param train_labels:    list of graph labels for training
    :param eval_graphs:     list of networkx graphs for evalutation
    :param eval_labels:     list of graph labels for evaluation
    :param feature_type:    string of feature type for the node/graph features
                            required for the GNNs

    :return:                adjacency matrices, computed node/graph features, labels
                            for both training and evaluation
    """

    # dataset specific preprocessing
    if dataset == "CLIQUE":
        preprocess_data = preprocess_clique_data
    elif dataset == "CHORDAL1":
        preprocess_data = preprocess_chordal_one_data
    elif dataset == "CHORDAL2":
        preprocess_data = preprocess_chordal_two_data
    elif dataset == "CONNECT":
        preprocess_data = preprocess_connect_data
    elif dataset == "TRIANGLES":
        preprocess_data = preprocess_triangles_data
    else:
        raise ValueError

    # pad matrices here, so graph size differences in train and eval dataset do not break the classification
    padded_matrices = get_padded_adjacency(train_graphs + eval_graphs)

    # matrix normalization, node feature loading/generation and label mappings are handled
    matrices, features, labels = preprocess_data(train_graphs + eval_graphs,
                                                 padded_matrices,
                                                 train_labels + eval_labels,
                                                 feature_type)

    # split into train and eval set again
    train_matrices = matrices[:len(train_graphs)]
    eval_matrices = matrices[len(train_graphs):]
    train_features = features[:len(train_graphs)]
    eval_features = features[len(train_graphs):]
    train_labels = labels[:len(train_graphs)]
    eval_labels = labels[len(train_graphs):]

    return train_matrices, train_features, train_labels, eval_matrices, eval_features, eval_labels


def get_model(dataset, n_nodes, n_features, num_classes):
    """
    Gets the dataset specific model.

    :param dataset:     string of dataset to be used
    :param n_nodes:     network input dimension
    :param n_features:  number of features
    :param num_classes: number of target classes

    :param model:       loaded model
    """

    if dataset == "CLIQUE":
        model = get_clique_model(n_nodes, n_features, num_classes)
    elif dataset == "CHORDAL1":
        model = get_chordal_one_model(n_nodes, n_features, num_classes)
    elif dataset == "CHORDAL2":
        model = get_chordal_two_model(n_nodes, n_features, num_classes)
    elif dataset == "CONNECT":
        model = get_connect_model(n_nodes, n_features, num_classes)
    elif dataset == "TRIANGLES":
        model = get_triangles_model(n_nodes, n_features, num_classes)
    else:
        raise ValueError

    return model


def get_gcn_params(dataset):
    """
    Gets the dataset specific hyperparameters.

    :param dataset: string of input dataset

    :return:        learning rate, number of epochs and batchsize
    """

    if dataset == "CLIQUE":
        lr = config.CLIQUE_LR
        epochs = config.CLIQUE_EPOCHS
        batchsize = config.CLIQUE_BATCHSIZE
    elif dataset == "CHORDAL1":
        lr = config.CHORDAL1_LR
        epochs = config.CHORDAL1_EPOCHS
        batchsize = config.CHORDAL1_BATCHSIZE
    elif dataset == "CHORDAL2":
        lr = config.CHORDAL2_LR
        epochs = config.CHORDAL2_EPOCHS
        batchsize = config.CHORDAL2_BATCHSIZE
    elif dataset == "CONNECT":
        lr = config.CONNECT_LR
        epochs = config.CONNECT_EPOCHS
        batchsize = config.CONNECT_BATCHSIZE
    elif dataset == "TRIANGLES":
        lr = config.TRIANGLES_LR
        epochs = config.TRIANGLES_EPOCHS
        batchsize = config.TRIANGLES_BATCHSIZE
    else:
        raise ValueError

    return lr, epochs, batchsize


if __name__ == "__main__":

    # datasets = ["CLIQUE", "CHORDAL1", "CHORDAL2", "CONNECT", "TRIANGLES"]
    datasets = ["CONNECT"]
    modeltype = "WL"
    feature_types = ["Eye"] # irrelevant for kernel methods

    for dataset in datasets:
        for feature_type in feature_types:
            train(modeltype, dataset, os.path.join("data", dataset, dataset), feature_type)
