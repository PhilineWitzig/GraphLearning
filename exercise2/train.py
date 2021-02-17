"""
This module is used for training the NNs.
Run this script to start training.
"""
import os
import config
from data import *
from model import *
from sklearn.model_selection import StratifiedKFold
from datetime import datetime

prefix_to_dataset_name = {
    'cs': 'CiteSeer',
    'co': 'Cora'
}


def train_eval(args):
    """
    This function manages the training and evaluation based on the configurations
    chosen by the user.
    :param args:    parsed input arguments
                    --datasets:      string of dataset classifier should be trained on
                    --classifier:   string of classifier which we will train
    """

    if args.classifier == "Node":
        names = []
        if "CiteSeer" in args.datasets:
            names.append('cs')
        if "Cora" in args.datasets:
            names.append('co')

        if len(names) == 0:
            print("No valid dataset for node classifier was chosen.")
            sys.exit()

        train_eval_node_classifier(names)

    elif args.classifier == "Graph":
        train_graph_classifier(args.datasets)
    else:
        print("Invalid classifier.")
        sys.exit()


def train_graph_classifier(dataset_name):
    """
    Function to train and evaluate the given dataset on graph classification.
    All hyperparamter configurations (no. epochs, learning rate, batch size)
    for the training loop can be found in the
    configuration file. The training loop performs 10 k-fold cross validation
    to split the dataset into train and test datasets.
    The chosen optimizer is the Adam optimizer.

    :param dataset_name:    Array of string of dataset graph classifier should be trained
                            on
    """
    # get data
    matrices, features, labels_ = get_preprocessed_graph_data(dataset_name[0])
    features = tf.keras.utils.normalize(features, axis=-1, order=2)
    num_classes = len(np.unique(labels_))
    labels = tf.one_hot(indices=labels_, depth=num_classes)

    # cross validation
    accuracy = []
    val_accuracy = []
    for i, split in enumerate(StratifiedKFold(10, shuffle = True, random_state = 345369).split(matrices, labels_)):
        train_index, test_index = split
        print("Training on data set ", dataset_name[0], " in Fold number ", i + 1)
        logdir = "logs/scalars/graph/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + str(i + 1)
        os.makedirs(logdir)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, profile_batch=0)

        # split data into training and validation
        matrices_train, matrices_test = matrices[train_index], matrices[test_index]
        features_train, features_test = features[train_index], features[test_index]
        labels_train, labels_test = tf.gather(labels, train_index), tf.gather(labels, test_index)

        # get model
        num_units = features_train.shape[1]
        num_features = features_train.shape[2]
        model = model_GCN_graph(num_units, num_features, num_classes)
        
        # compile, train and evalutate the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.LR_GRAPH),
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        fit = model.fit(x=[features_train, matrices_train],
                        y=labels_train,
                        batch_size=config.BATCH_SIZE,
                        epochs=config.NUM_EPOCHS_GRAPH,
                        verbose=0,  # use tensorboard instead
                        validation_data=([features_test, matrices_test], labels_test),
                        shuffle=True, callbacks=[tensorboard_callback])

        # store performance
        accuracy.append(max(fit.history['accuracy']))
        val_accuracy.append(max(fit.history['val_accuracy']))
        print(f"Train accuracy for graph classifier: {accuracy[-1]}")
        print(f"Validation accuracy for graph classifier: {val_accuracy[-1]}")
        del model

    print(f"Average max train acc: {np.mean(accuracy)}")
    print(f"Average max validation acc: {np.mean(val_accuracy)}\n")


def train_eval_node_classifier(dataset_prefixes):
    """
    Function to train and evaluate the given datasets using the GCN node model.
    Uses a batch size of one and empirically determined learning rate.
    The chosen optimizer is Adam.

    :param dataset_prefixes: list of prefixes of the datasets to be trained and evaluated
    """

    for dataset_prefix in dataset_prefixes:
        print(f"Training and evaluating {prefix_to_dataset_name[dataset_prefix]}.")

        # get preprocessed data for train and validation set
        train_matrices, train_features, train_labels = get_preprocessed_data(
            dataset_prefix + '_train')
        eval_matrices, eval_features, eval_labels = get_preprocessed_data(dataset_prefix + '_eval')

        # calculate number of vertices and features and class labels from inputs
        num_vertices = train_matrices.shape[0]
        num_features = train_features[0].shape[1]
        num_label_classes = len(np.unique(train_labels))

        # expand adjacency matrices, since tf expects batched inputs
        train_matrices = np.expand_dims(train_matrices, axis=0)
        eval_matrices = np.expand_dims(eval_matrices, axis=0)

        acc = []
        val_acc = []
        # run train and evaluation 10 times and average results
        for i in range(10):
            print(f"\tTraining Run {i + 1}.")
            model = model_GCN_node(num_vertices, num_features, num_label_classes)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.LR_NODE),
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
            fit = model.fit(x=[train_features, train_matrices], y=train_labels, batch_size=1,
                            epochs=config.NUM_EPOCHS_NODE, verbose=0,
                            validation_data=([eval_features, eval_matrices], eval_labels))
            # save maximal accuracy values from each train/evaluation run
            acc.append(max(fit.history['accuracy']))
            val_acc.append(max(fit.history['val_accuracy']))
            del model
        print(f"Average max train acc: {np.mean(acc)}")
        print(f"Average max validation acc: {np.mean(val_acc)}\n")
