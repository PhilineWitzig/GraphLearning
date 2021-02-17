import random
import logging
import networkx as nx
import numpy as np
import config
from tqdm import tqdm


def train_eval_split_cc(graph):
    """
    Computes the train and eval split for all connected components in the passed
    graph individually to ensure that the training data will be connected.

    :param graph:   (possibly unconnected) networkX graph

    :return:        list of train edges, list of evaluation edges
    """

    logging.info("Computing train eval split for link prediction.")

    edges_train = []
    edges_eval = []

    for cc in nx.connected_components(graph):
        graph_cc = nx.Graph(graph.subgraph(cc))
        edges_cc_train, edges_cc_eval = train_eval_split(graph_cc)
        edges_train.append(edges_cc_train)
        edges_eval.append(edges_cc_eval)

    edges_train_flattened = [edge for edges_cc_train in edges_train for edge in edges_cc_train]
    edges_eval_flattened = [edge for edges_cc_eval in edges_eval for edge in edges_cc_eval]

    return edges_train_flattened, edges_eval_flattened


def train_eval_split(graph, split=0.2):
    """
    splits graph into train and eval edges, while ensuring that the train edges are connected

    :param graph: graph to be split
    :param split: split ratio
    :return: tuple of train edges and eval edges
    """

    assert(nx.is_connected(graph))

    number_total_edges = graph.number_of_edges()
    number_eval_edges = int(number_total_edges * split)
    step_size = number_eval_edges // 100 + 1

    edges_eval = []
    pbar = tqdm(desc='Train-Eval-Splitting', total=number_eval_edges)
    while len(edges_eval) < number_eval_edges:
        # take random edges of the remaining graph
        edges = list(graph.edges())
        random_edges = random.sample(edges, step_size)
        # remove chosen edges from graph
        graph.remove_edges_from(random_edges)
        # check whether remaining graph is still connected
        if nx.is_connected(graph):
            # add graph to eval edges
            edges_eval += random_edges
            pbar.update(step_size)
        else:
            # undo removal of edges
            graph.add_edges_from(random_edges)
    pbar.close()
    return graph.edges, edges_eval


def get_neg_edges(graph, num_train, num_eval):
    """
    takes number of train and eval samples of negative edges

    :param graph: graph to take negative edges from
    :param num_train: number of train samples
    :param num_eval: number of eval samples
    :return: tuple of negative edges
    """
    adj_matrix = nx.to_numpy_matrix(graph)
    adj_matrix_neg_idx = np.where(adj_matrix == 0)
    # add 1 to each element since graph nodes start from 1 and numpy matrices from 0
    us = adj_matrix_neg_idx[0] + 1
    vs = adj_matrix_neg_idx[1] + 1
    neg_edges = list(zip(us, vs))

    neg_samples = random.sample(neg_edges, num_train + num_eval)

    return neg_samples[:num_train], neg_samples[num_train:]


def comp_edge_embeddings(edges, node_embeddings):
    """
    Computes all edge embeddings for all possible edges in the edge set.

    :param nodes:           list of edges
    :param node_embeddings: node_embeddings for passed edges

    :return:                computed edges embedding for all possible edges
    """

    logging.info("Computing edge embeddings.")

    edge_embeddings = np.zeros([len(edges), config.EMBED_DIM])

    for i in tqdm(range(len(edges))):
        cur_edge = edges[i]
        edge_embeddings[i] = np.multiply(node_embeddings[cur_edge[0] - 1], node_embeddings[cur_edge[1] - 1])

    return edge_embeddings
