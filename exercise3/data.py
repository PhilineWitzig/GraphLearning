"""
This module loads data and performs the preprocessing steps.
"""
import os
from utils.data_utils import get_node_attributes, get_node_labels
from utils.dataset_parser import Parser


def load_dataset(name):
    """
    Loads and parses one dataset.
    :params:    name of dataset to load
    :return:    list of loaded graphs
    """

    dataset = Parser(os.path.join('datasets', name))
    graphs = dataset.parse_all_graphs()

    return graphs


def load_dataset_nodelabel(name):
    """
    Loads one of the graph datasets CiteSeet or Cora and its labels.
    :params:    name of dataset to load
    :return:    list of loaded graphs, list of node features, list of labels
    """

    dataset = Parser(os.path.join('datasets', name))
    graphs = dataset.parse_all_graphs()

    # convert datasets into lists graphs, labels
    features = [get_node_attributes(graph) for graph in graphs]
    labels = [get_node_labels(graph) for graph in graphs]

    return graphs, features, labels


def load_link_graph(name):
    """
    Loads the singular graph contained in the datasets Facebook and PPI.
    :params:    name of dataset to load

    :return:    singular graph
    """
    dataset = Parser(os.path.join('datasets', name))
    graphs = dataset.parse_all_graphs()

    return graphs[0]
