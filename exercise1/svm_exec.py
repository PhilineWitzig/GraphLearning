#!/usr/bin/env python
# coding: utf-8
"""
Exercise 4
Code for training the SVM on the datasets DD, ENZYMES and NCI1.
For choosing a kernel, set the KERNEL_TYPE flag in the config.py module.
"""

import sys
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

import config
from data_utils import get_graph_label
from dataset_parser import Parser
from sklearn import svm, model_selection
from sklearn.model_selection import RepeatedKFold
from graphlet import compute_graphlets, load_graphlets
from closed_walk import compute_closed_walks
from wl_kernel import compute_wl
from sklearn.metrics.pairwise import linear_kernel


def load_datasets(names):
    """
    Loads the graph datasets DD, ENZYMES and NCI1 and its labels.
    :params:    list of dataset names to load

    :return:    list of dataset names, list of loaded graphs for all datasets,
                labels for loaded graphs for all datasets
    """

    # load datasets
    datasets = []
    if "dd" in names:
        datasets.append(Parser('datasets/DD'))
    if "enzymes" in names:
        datasets.append(Parser('datasets/ENZYMES'))
    if "nci1" in names:
        datasets.append(Parser('datasets/NCI1'))
    if "specific" in names:
    	datasets.append(Parser('datasets/specific_dataset'))
    # convert datasets into lists graphs, labels
    datasets = [dataset.parse_all_graphs() for dataset in datasets]

    # remove graphs with cardinality smaller 5, as they cannot be used for our graphlet kernel
    datasets = [[graph for graph in graphs if len(list(graph)) >= 5] for graphs in datasets]
    label_sets = [[get_graph_label(graph) for graph in graphs] for graphs in datasets]

    return names, datasets, label_sets


def compute_gram_matrix(x):
    """
    Kernel function calculating the inner product.

    :param x: list of feature vectors
    :return: gram matrix
    """
    return linear_kernel(x, x, dense_output=True)


@ignore_warnings(category=ConvergenceWarning)
def run_svm(names, vector_sets, label_sets, max_it):
    """
    Trains an SVM for the given kernel using 10-fold cross validation with 10 repetitions.
    The number of iterations of the SVM can be adjusted by adjusting the MAX_IT parameter. Naturally higher Values
    lead to higher computation times and higher accuracies.

    We justify the cap by arguing that unreasonable computation times outweigh the importance of perfect accuracy
    results. The accuracy will not proportionally increase with computation time, but will almost remain the same,
    making unlimited number of iterations until convergence unfeasible. Improving the runtime is certainly a work in
    progress. By scaling the data beforehand we tried to fix convergence issues, especially with the enzymes dataset,
    but different scaling did not lead to success.

    :param names: list of names of the datasets
    :param vector_sets: list of vectors for each dataset
    :param label_sets: list of labels for each dataset
    :param max_it: number of maximal iterations for solver in svm
    """

    # zipping names, feature vector list and label list together
    zips = zip(names, vector_sets, label_sets)

    # 10-fold cross-validating svm 10 times for each dataset
    for name, vectors, labels in zips:
        vectors = np.array(vectors)
        labels = np.array(labels)
        gram = np.array(compute_gram_matrix(vectors))

        print(f"Training for Dataset {name}.")
        clf = svm.SVC(kernel="precomputed", max_iter=max_it)
        scores = model_selection.cross_validate(clf, gram, labels,
                                                cv=RepeatedKFold(n_splits=10, n_repeats=10), return_train_score=True)

        print("The Test Accuracies per run were:", scores['test_score'], "\n")
        print("Thus the test average accuracy over all runs was", np.average(scores['test_score']), "\n")
        print("With a standard deviation of ", np.std(scores['test_score']))
        print("and highest achieved test accuracy of ", np.max(scores['test_score']), ".\n")

        print("The train Accuracies per run were:", scores['train_score'], "\n")
        print("Thus the train average accuracy over all runs was", np.average(scores['train_score']), "\n")
        print("With a standard deviation of ", np.std(scores['train_score']))
        print("and highest achieved train accuracy of ", np.max(scores['train_score']), ".\n\n")


def main():
    """
    Optionally enable the loading of the respective kernel vectors instead of recomputing them on the fly to decrease
    computation time

    Optionally manually adjust the MAX_IT parameter.

    """
    names, datasets, labelsets = load_datasets(['dd', 'enzymes', 'nci1'])

    if config.KERNEL_TYPE == "graphlet":
        # vectors = compute_graphlets(datasets, names)
        vectors = load_graphlets(names)
    elif config.KERNEL_TYPE == "wl":
        vectors = compute_wl(datasets, names)
        # vectors = load_wl(names)
    elif config.KERNEL_TYPE == "closed_walk":
        vectors = compute_closed_walks(datasets, names)
        # vectors = load_closed_walks(names)
    else:
        print("Invalid kernel type", sys.exc_info()[0])
        raise

    run_svm(names, vectors, labelsets)


if __name__ == "__main__":
    main()
