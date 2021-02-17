"""
Executing this script computes the results for the optimal architecture using
advanced pooling layers and GCN layers.
"""

import logging
import tensorflow as tf
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

import config
from layers import DiffPool, TopKPool
from model import model_opt_architecture, model_opt_architecture_enzymes
from data import get_normalized_data


def train_eval(dataset_name, matrices, features, labels, num_classes):
    """
    Train loop for evaluating the best performing model architecture.

    :param dataset_name:    name of loaded dataset as string
    :param matrices:        adjacency matrices for graphs of input dataset
    :param features:        node features for graphs of input dataset
    :param num_classes:     number of distinct classes for input dataset
    """
    accuracy = []
    val_accuracy = []
    np_epochs = []
    # perform 10 times k-fold with the same random seed, to ensure reproducibility and gain more precise insights.
    for train_index, test_index in tqdm(StratifiedKFold(10, shuffle=True, random_state=345369).split(matrices, labels),
                                        total=10):

        # split data into training and validation
        matrices_train, matrices_test = matrices[train_index], matrices[test_index]
        features_train, features_test = features[train_index], features[test_index]
        labels_train, labels_test = tf.gather(labels, train_index), tf.gather(labels, test_index)

        # get model
        num_units = features_train.shape[1]
        num_features = features_train.shape[2]
        if dataset_name == "ENZYMES":
            model = model_opt_architecture_enzymes(dataset_name, num_units, num_features, num_classes)
        elif dataset_name == "NCI1" or dataset_name == "PROTEINS":
            model = model_opt_architecture(dataset_name, num_units, num_features, num_classes)
        else:
            print("Invalid dataset name. Choose ENZYMES, PROTEINS or NCI1.")

        # compile, train and evalutate the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.LR),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        fit = model.fit(x=[features_train, matrices_train],
                        y=labels_train,
                        batch_size=config.BATCH_SIZE,
                        epochs=config.EPOCHS,
                        verbose=0,
                        validation_data=([features_test, matrices_test], labels_test),
                        shuffle=True)

        # store performance
        accuracy.append(max(fit.history['accuracy']))
        val_accuracy.append(max(fit.history['val_accuracy']))
        np_epochs.append(len(fit.history['accuracy']))
        del model

    logging.info("\t Train:")
    logging.info(f"\t \t Average max acc: {np.mean(accuracy):.4f}")
    logging.info(f"\t \t Std dev: {np.std(accuracy):.4f}")
    logging.info("\t Validation:")
    logging.info(f"\t \t Average max acc: {np.mean(val_accuracy):.4f}")
    logging.info(f"\t \t Std dev: {np.std(val_accuracy):.4f}")


def prepare_datset(dataset_name, split):
    """
    Returns a stratified split of the dataset into training and test data.

    :param dataset_name:    name of chosen dataset as sting
    :param split:           float value in (0, 1) indicating the train and test split

    :return:                adjaceny matrices, fetaures, labels all split into a training
                            and test dataset as well as the number of classes in the dataset
    """
    matrices, features, labels = get_normalized_data(dataset_name)

    if dataset_name == "PROTEINS":
        labels = labels - 1 # protein labels range from [1,2] but they should be in [0,2)

    features = tf.keras.utils.normalize(features, axis=-1, order=2)
    num_classes = len(np.unique(labels))

    matrices_features = list(zip(matrices, features))

    _, matrices_features_split, _, labels_split = train_test_split(matrices_features, labels, test_size=split,
                                                                   random_state=465238674, stratify=labels)

    matrices_split = [matrix for matrix, _ in matrices_features_split]
    features_split = [feature for _, feature in matrices_features_split]
    return np.asarray(matrices_split), np.asarray(features_split), np.asarray(labels_split), num_classes


if __name__ == "__main__":

    logging.info(f"Starting experiments for assignment 4. Batch size: {config.BATCH_SIZE} "
                 f"and No. of epochs: {config.EPOCHS} are fixed.")

    # reducing dataset to contain less graphs, to reduce computational effort
    matrices, features, labels, num_classes = prepare_datset(config.DATASET, 0.5)
    train_eval(config.DATASET, matrices, features, labels, num_classes)
