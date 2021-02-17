"""
Main script for evaluation of the trained models for assignment 5.
"""
import argparse
import os

import numpy as np
import tensorflow as tf
import random

import config
from data import load_dataset_graph, load_dataset_node
from models.chordal1 import preprocess_chordal_one_data
from models.chordal2 import preprocess_chordal_two_data
from models.triangles import preprocess_triangles_data
from models.clique import preprocess_clique_data, get_clique_model
from models.connect import preprocess_connect_data
from models.utils import load_triangles_model, load_clique_model, load_chordal1_model, load_chordal2_model, load_connect_model
from utils.data_utils import get_padded_adjacency
from feature_extractor import color_hist_features
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings


@ignore_warnings(category=ConvergenceWarning)
def main(args):

    # case distinction because we might require preprocessing depending on the dataset
    if args.dataset == 'CHORDAL1':
        graphs, labels = load_dataset_graph(args.data_path)
        padded_matrices = get_padded_adjacency(graphs)

        matrices, features, labels = preprocess_chordal_one_data(
            graphs, padded_matrices, labels, "Eye")
        model = load_chordal1_model()
        test_score = model.evaluate([features, matrices], labels)
        print(f"The model achieved a evaluation accuracy of {test_score[1]:.4f} on CHORDAL1.")

    elif args.dataset == 'CHORDAL2':
        graphs, labels = load_dataset_graph(args.data_path)
        padded_matrices = get_padded_adjacency(graphs)

        matrices, features, labels = preprocess_chordal_two_data(
            graphs, padded_matrices, labels, "Eye")
        model = load_chordal2_model()
        test_score = model.evaluate([features, matrices], labels)
        print(f"The model achieved a evaluation accuracy of {test_score[1]:.4f} on CHORDAL2.")

    elif args.dataset == 'CLIQUE':
        graphs, labels = load_dataset_node(args.data_path)
        graph = graphs[0]

        # load model from file and evaluation on eval dataset
        model = load_clique_model()
        features, labels = preprocess_clique_data(graph, labels)
        test_score = model.evaluate(features, labels)
        print(f"The model achieved a evaluation accuracy of {test_score[1]:.4f} on CLIQUE.")

    elif args.dataset == 'CONNECT':
        print("Training the WL kernel from scratch on {}.".format(args.dataset))
        test_graphs, test_labels = load_dataset_graph(args.data_path)  # eval data
        train_graphs, train_labels = load_dataset_graph(os.path.join(
            "data", "CONNECT", "CONNECT") + "_Train")  # train data
        # do refinement in on run for color comparability
        features = color_hist_features(train_graphs + test_graphs)
        train_features = features[0:len(train_graphs)]
        test_features = features[len(train_graphs):]

        # trains the WL kernel from scratch
        model = load_connect_model(train_features, train_labels)
        pred = model.predict(np.dot(test_features, train_features.T))
        acc = accuracy_score(test_labels, pred)
        print("The model achieved and evaluation accuracy of {} on CONNECT."
              .format(acc))

    elif args.dataset == 'TRIANGLES':
        graphs, labels = load_dataset_graph(args.data_path)
        labels = np.asarray(labels)

        # load model from file and evaluation on eval dataset
        model = load_triangles_model()
        features = preprocess_triangles_data(graphs)

        test_score = model.evaluate(features, labels)
        print(f"The model achieved a evaluation accuracy of {test_score[1]:.4f} on TRIANGLES.")

    else:
        print("Invalid dataset {}. Valid datasets are 'CHORDAL1', 'CHORDAL2', 'CLIQUE', 'CONNECT', 'TRIANGLES' "
              .format(args.dataset))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['CHORDAL1', 'CHORDAL2', 'CLIQUE', 'CONNECT', 'TRIANGLES'],
                        help="Dataset which will be used.")
    parser.add_argument('--data_path', type=str, required=True,
                        help="Relative path to evalutation data, e.g. data/TRIANGLES/TRIANGLES_EVAL")

    args = parser.parse_args()
    main(args)
