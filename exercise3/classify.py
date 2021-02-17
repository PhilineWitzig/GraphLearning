"""
This module is used for training the NNs.
Run this script to start training.
"""
import argparse
import numpy as np
import logging
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
from create_embeddings import load_embeddings
from data import load_link_graph, load_dataset_nodelabel
from link_predictor import train_eval_split_cc, get_neg_edges, comp_edge_embeddings
from node2vec import train_node2vec


def link_predictor(dataset_name, graph, iter=5):
    """
    Trains the logistic regressor for link prediction. To do so, the graph is split
    into training and validation data. Edge embeddings are used as input for the
    logistic regression. In case an edge is available in the input graph, we set
    its label to 1. Otherwise the edge label is set to 0.
    The logistic regression is evaluated using the ROC AUC score and simply the
    accuracy store.

    :param dataset_name:    string for dataset name
    :param graph:           input networkX graph
    :param iter:            number of iterations of how often we repeat the
                            experiment
    """

    acc_train = []
    acc_eval = []
    roc_auc_eval = []
    roc_auc_train = []

    for i in range(iter):
        # split data into training and test dataset
        logging.info("No. of experiments: [{}/{}]".format(i, iter))
        train_edges, eval_edges = train_eval_split_cc(graph)
        num_train = len(train_edges)
        num_eval = len(eval_edges)
        train_neg_edges, eval_neg_edges = get_neg_edges(graph, num_train, num_eval)
        node_embeddings = train_node2vec(dataset_name, 1, 1)

        # compute edge embeddings and labels
        logging.debug("Getting edge embeddings for training and evaluation.")
        train_edge_embeddings = comp_edge_embeddings(train_edges, node_embeddings)
        eval_edge_embeddings = comp_edge_embeddings(eval_edges, node_embeddings)
        train_edge_neg_embeddings = comp_edge_embeddings(train_neg_edges, node_embeddings)
        eval_edge_neg_embeddings = comp_edge_embeddings(eval_neg_edges, node_embeddings)

        logging.debug("Getting edge labels.")
        train_edge_labels = np.ones([len(train_edges)])
        train_edge_neg_labels = np.zeros([len(train_neg_edges)])
        eval_edge_labels = np.ones([len(eval_edges)])
        eval_edge_neg_labels = np.zeros([len(eval_neg_edges)])

        # shuffle data
        training_data = list(zip(np.concatenate([train_edge_embeddings,
                                                 train_edge_neg_embeddings], axis=0),
                                 np.concatenate([train_edge_labels,
                                                 train_edge_neg_labels], axis=0)))
        eval_data = list(zip(np.concatenate([eval_edge_embeddings,
                                             eval_edge_neg_embeddings], axis=0),
                             np.concatenate([eval_edge_labels,
                                             eval_edge_neg_labels], axis=0)))
        random.shuffle(training_data)
        random.shuffle(eval_data)
        train_edge_embeddings, train_labels = zip(*training_data)
        eval_edge_embeddings, eval_labels = zip(*eval_data)

        # training
        logging.info("Training link predictor.")
        clf = LogisticRegression()
        clf.fit(train_edge_embeddings, train_labels)

        # evaluation
        logging.info("Evaluating link predictor")
        # get scores for training
        pred_train = clf.predict(train_edge_embeddings)
        acc_train.append(accuracy_score(train_labels, pred_train))
        pred_train_probs = clf.predict_proba(train_edge_embeddings)[:,1]
        roc_auc_train.append(roc_auc_score(train_labels, pred_train_probs))

        # get scores for eval
        pred_eval = clf.predict(eval_edge_embeddings)
        acc_eval.append(accuracy_score(eval_labels, pred_eval))
        pred_eval_probs = clf.predict_proba(eval_edge_embeddings)[:,1]
        roc_auc_eval.append(roc_auc_score(eval_labels, pred_eval_probs))

    # Logging results
    logging.info("TRAIN: Accuracies after {} experiment trials: {}".format(iter, acc_train))
    logging.info("TRAIN: Mean accuracy: {}".format(sum(acc_train) / len(acc_train)))
    logging.info("TRAIN: Standard deviation in accuracy: {}".format(np.std(np.asarray(acc_train))))
    logging.info("TRAIN: ROC AUC scores after {} experiment trials: {}".format(iter, roc_auc_train))

    logging.info("EVAL: Accuracies after {} experiment trials: {}".format(iter, acc_eval))
    logging.info("EVAL: Mean accuracy: {}".format(sum(acc_eval) / len(acc_eval)))
    logging.info("EVAL: Standard deviation in accuracy: {}".format(np.std(np.asarray(acc_eval))))
    logging.info("EVAL: ROC AUC scores after {} experiment trials: {}".format(iter, roc_auc_eval))


def classify_nodes(embeddings, labels):
    """
    Implementation of the node classifier.

    :param embeddings:  node embeddings
    :param labels:      target node labels
    """
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1230948562)
    labels = np.asarray(labels[0])
    mean_accs = []
    for train_index, test_index in tqdm(kfold.split(embeddings, labels)):
        train_embeddings = embeddings[train_index]
        test_embeddings = embeddings[test_index]
        train_labels = labels[train_index]
        test_labels = labels[test_index]

        clf = LogisticRegression()
        clf.fit(train_embeddings, train_labels)
        mean_acc = clf.score(test_embeddings, test_labels)
        mean_accs.append(mean_acc)

    print(f"Average validation accuracy is {np.mean(mean_accs)}")
    print(f"Standard deviation of accuracy is {np.std(mean_accs)}")


def main(args):
    if args.mode == 'Node':
        assert(args.input_file is not None)
        embeddings = load_embeddings(args.input_file)
        _, _, labels = load_dataset_nodelabel(args.dataset)
        classify_nodes(embeddings, labels)
    else:
        assert(args.dataset is not None)
        graph = load_link_graph(args.dataset)
        link_predictor(args.dataset, graph)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['Node', 'Link'],
                        help="Whether to do node classification or link prediction.")
    parser.add_argument('--dataset', type=str, required=True, choices=['Cora', 'CiteSeer', 'Facebook', 'PPI'],
                        help="Which dataset to use.")
    parser.add_argument('--input_file', type=str,
                        help="File in which the embeddings for node classification are stored.")

    args = parser.parse_args()
    main(args)
