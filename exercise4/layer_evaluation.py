"""
Executing this script does all necessary experiments for evaluating the custom layers.
"""

import logging
import tensorflow as tf
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

import config
from layers import DiffPool, TopKPool
from model import model_baseline, model_GCN_skip, model_advanced_pool
from data import get_normalized_data


def train_eval(matrices, features, labels, num_classes, model_type, feature_numbers, depth, adv_pool_k):
    """
    Main train and evaluation loop, constructing the different models, training and evaluating them.
    The results are logged.

    :param matrices:            adjacency matrices of graphs used for training
    :param features:            node features of graphs used for training
    :param labels:              labels of graphs used for training
    :param num_classes:         number of classes in the dataset
    :param model_type:          model type which is to trained, i.e. 'baseline', 'skip', 'diff', 'topK'
    :param feature_numbers:     (feature) dimensions for different GCN/skip_GCN layers
    :param depth:               depth of model to be trained, i.e. 1 or 2
    :param adv_pool_l:          
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
        if model_type == "baseline":
            model = model_baseline(num_units, num_features, num_classes, feature_numbers, depth)
        elif model_type == "skip":
            model = model_GCN_skip(num_units, num_features, num_classes, feature_numbers, depth)
        elif model_type == "diff":
            model = model_advanced_pool(num_units, num_features, num_classes, DiffPool, feature_numbers, depth, adv_pool_k)
        elif model_type == "topK":
            model = model_advanced_pool(num_units, num_features, num_classes, TopKPool, feature_numbers, depth, adv_pool_k)
        else:
            logging.error("Invalid mode. Pooling type needs to be either baseline, skip, diff or topK")
            return

        # early stopping callback, if no improvement
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',
            baseline=None, restore_best_weights=False
        )

        # compile, train and evalutate the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.LE_LR),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        fit = model.fit(x=[features_train, matrices_train],
                        y=labels_train,
                        batch_size=config.LE_BATCH_SIZE,
                        epochs=config.LE_EPOCHS,
                        verbose=0,
                        validation_data=([features_test, matrices_test], labels_test),
                        shuffle=True,
                        callbacks=[early_stopping_callback])

        # store performance
        accuracy.append(max(fit.history['accuracy']))
        val_accuracy.append(max(fit.history['val_accuracy']))
        np_epochs.append(len(fit.history['accuracy']))
        del model

    logging.info(f"\t On average trained {np.mean(np_epochs):.1f}/{config.LE_EPOCHS} epochs.")
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
    matrices, features, labels = get_normalized_data(dataset_name)  # no normalization required??

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

    logging.info(f"Starting experiments for assignment 4. Batch size: {config.LE_BATCH_SIZE} "
                 f"and No. of epochs: {config.LE_EPOCHS} are fixed. We use early stopping.")

    model_types = ['baseline', 'skip', 'diff', 'topK']
    depths = config.LE_DEPTH

    # reducing dataset to contain less graphs, to reduce computational effort
    matrices, features, labels, num_classes = prepare_datset(config.LE_DATASET, 0.5)

    # run experiment with various different architectures and parameters,
    # to gain insight about effect of individual layers.
    for model_type in model_types:
        for feature_numbers in config.LE_FEATURE_NUMBERS_LIST:
            for depth in depths:
                if model_type == 'diff' or model_type == 'topK':
                    for k in config.LE_ADV_POOLING_K:
                        logging.info(f"Model type: {model_type}")
                        logging.info(f"Width: {feature_numbers}")
                        logging.info(f"Depth: {depth}")
                        logging.info(f"k: {k}")
                        train_eval(matrices, features, labels, num_classes, model_type, feature_numbers, depth, k)
                else:
                    logging.info(f"Model type: {model_type}")
                    logging.info(f"Width: {feature_numbers}")
                    logging.info(f"Depth: {depth}")
                    train_eval(matrices, features, labels, num_classes, model_type, feature_numbers, depth, None)
