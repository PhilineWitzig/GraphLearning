#!/usr/bin/env python3
import networkx as nx
import numpy as np
import os


def get_graph_label(G):
    """
    :param G: A networkx graph
    :return: The label (class) of G
    """
    c = G.graph['label']
    return c


def get_node_labels(G):
    """
    :param G: A networkx graph G=(V,E)
    :return: A array of node labels of length |V|
    """
    labels = [(node[0], np.int32(node[1]["node_label"])) for node in G.nodes(data=True)]
    labels.sort(key=lambda x: x[0])
    labels = [a for n, a in labels]
    return labels


def get_node_attributes(G):
    """
    :param G: A networkx graph G=(V,E)
    :return: A numpy array of shape (|V|, a), where a is the length of the node attribute vector
    """
    attributes = [(node[0], np.float32(node[1]["node_attributes"])) for node in G.nodes(data=True)]
    attributes.sort(key=lambda x: x[0])
    attributes = [a for n, a in attributes]
    return attributes


def write_A(path, graphs):
    """
    Writes the edges and indicator lists
    :param path: Path to the exported dataset
    :param graphs: List of the NetworkX Graphs
    """
    edges, indicator = merge(graphs)

    with open(os.path.join(path, os.path.basename(path) + '_A.txt'), 'w') as f:
        for u, v in edges:
            f.write(f'{u},{v}\n')

    with open(os.path.join(path, os.path.basename(path) + '_graph_indicator.txt'), 'w') as f:
        for i in indicator:
            f.write(f'{i}\n')


def write_graph_labels(path, graphs):
    """
    Writes the graph labels
    :param path: Path to the exported dataset
    :param graphs: List of the NetworkX Graphs
    """
    with open(os.path.join(path, os.path.basename(path) + '_graph_labels.txt'), 'w') as f:
        for G in graphs:
            label = get_graph_label(G)
            f.write(f'{label}\n')


def write_node_labels(path, graphs, has_labels=False):
    """
    Writes the node labels
    :param path: Path to the exported dataset
    :param graphs: List of the NetworkX Graphs
    :param has_labels: If set to true, the node labels stored in the Graphs are exported.
                       If set to false, all node labels are set to 1.
    """
    with open(os.path.join(path, os.path.basename(path) + '_node_labels.txt'), 'w') as f:
        for G in graphs:
            if has_labels:
                labels = get_node_labels(G)
            else:
                labels = [1 for _ in range(G.order())]

            for l in labels:
                f.write(f'{l}\n')


def write_node_attr(path, graphs):
    """
    Writes the node attributes
    :param path: Path to the exported dataset
    :param graphs: List of the NetworkX Graphs
    """
    with open(os.path.join(path, os.path.basename(path) + '_node_attributes.txt'), 'w') as f:
        for G in graphs:
            for l in get_node_attributes(G):
                line = f'{float(l[0]):.6f}'
                for x in l[1:]:
                    line += f', {float(x):.6f}'
                line += '\n'
                f.write(line)


def merge(graphs):
    """
    Computes merged edge list and indicator list for the dataset
    :param graphs: List of the NetworkX Graphs
    """

    num_nodes = 0
    edges = []
    n_indicator = []

    for i, G in enumerate(graphs):
        node_shift = 1 - min(G.nodes())
        edges += [(u + num_nodes + node_shift, v + num_nodes + node_shift) for u, v in G.edges()]
        edges += [(v + num_nodes + node_shift, u + num_nodes + node_shift) for u, v in G.edges()]
        order = G.order()
        num_nodes += order
        n_indicator += [i + 1 for _ in range(order)]

    return edges, n_indicator


def export_dataset(path, graphs, has_g_labels=False, has_n_labels=False, has_n_attributes=False):
    """
    Exports a set of NetworkX graphs to the TUD Graph format
    :param path: Path to the folder in which the dataset will be stored, i.e 'datasets/my_dataset'

    :param graphs: The dataset as a list of NetworkX graphs.
                   The nodes in each graph should be integers.

    :param has_g_labels: A boolean value that indicates if the dataset has graph-level labels. If set to True, the labels will be exported.
                         The labels are expected to be integers ranging from 1 to n if there are n different labels.
                         The label l of each NetworkX graph g has to be stored as graph-level data under the keyword 'label', i.e:
                         g.graph['label'] = l

    :param has_n_labels: A boolean value that indicates if the dataset has node-level labels.
                         If set to True, the labels will be exported. If set to false, a node label of '1' will be exported for every node, since the importer requires node labels.
                         The labels are expected to be integers ranging from 1 to n if there are n different labels.
                         The label l of a node v in the NetworkX graph g has to be stored as node-level data under the keyword 'node_label', i.e:
                         nx.set_node_attributes(g, {v: l}, 'node_label')

    :param has_n_attributes: A boolean value that indicates if the dataset has node-level attributes.
                         If set to True, the attributes will be exported.
                         The attributes are expected to be lists of floats with equal length across all nodes.
                         The attribute list a of a node v in the NetworkX graph g has to be stored as node-level data under the keyword 'node_attributes', i.e:
                         nx.set_node_attributes(g, {v: a}, 'node_attributes')
    """

    if not os.path.exists(path):
        os.mkdir(path)

    write_A(path, graphs)

    write_node_labels(path, graphs, has_n_labels)

    if has_g_labels:
        write_graph_labels(path, graphs)

    if has_n_attributes:
        write_node_attr(path, graphs)

