import random

import numpy as np
import networkx as nx

from tqdm import tqdm


def compute_random_walks(graph, param_return, param_in_out, no_rw=5, length=5):
    """
    Returns the a fixed number of random walks of a specific length as well as the
    negative sampling for an input graph.

    :param graph: a networkX graph
    :param param_return: Bias of going back to the old node (p)
    :param param_in_out: Bias of moving forward to a new node (q)
    :param no_rw: number of random walks, default is 5
    :param length: length of random walks, default is 5
    :param save: Whether or not to save the random walk

    :return: list of random walks for all nodes in G as well as the
             negative sampling for all nodes in G
    """
    walks = []
    walks_neg = []

    for i in range(no_rw):
        for node in tqdm(list(graph.nodes), desc=f"Random walking {i+1}"):
            w = compute_random_walk(graph, node, param_return, param_in_out, length)
            w_neg = negative_sampling(graph, w)
            walks.append(w)
            walks_neg.append(w_neg)

    walks = np.asarray(walks)
    walks_neg = np.asarray(walks_neg)

    return walks, walks_neg


def compute_random_walk(graph, startnode, param_return, param_in_out, length):
    """
    Computes and returns random walks of size length. The parameters control
    how fast the walk explores or leaves the neighborhood of starting at the startnode.

    :param graph: NetworkX graph
    :param startnode: Integer, Initial node
    :param param_return: Bias of going back to the old node (p)
    :param param_in_out: Bias of moving forward to a new node (q)
    :param length: Integer length of desired random walk
    """

    def _lookup_weight(dist):
        if dist == 0:
            return 1 / param_return
        elif dist == 1:
            return 1
        elif dist == 2:
            return 1 / param_in_out
        else:
            raise ValueError("Distances greater than 2 are not supported.")

    walk = [startnode]
    for i in range(length):
        if i == 0:
            # special case for first node in walk
            neighbors = list(graph.neighbors(startnode))
            # Pick the first step uniformly at random from all neighbors (old_neighbors) of the startnode
            next_node = random.choice(neighbors)
            walk.append(next_node)
        else:
            neighbors = list(graph.neighbors(walk[-1]))
            dist_to_neighbors = []
            for neighbor in neighbors:
                dist_to_neighbors.append(nx.shortest_path_length(graph, walk[-2], neighbor))

            weights = [_lookup_weight(dist) for dist in dist_to_neighbors]
            next_node = random.choices(neighbors, weights=weights, k=1)[0]
            walk.append(next_node)

    # subtract 1 from each node, since the node names start with 1 and keras expects them to start at 0
    walk = [node - 1 for node in walk]

    # return walk without startnode
    return walk[1:]


def negative_sampling(graph, walk):
    """
    Function implementing negative sampling

    :param graph: NetworkX graph
    :param walk: Integer list indicating the walk over the given graph

    :return: List of integers indicating the negative sample
    """
    # Candidates are all nodes except nodes from the walk
    candidates = set(range(graph.number_of_nodes() - 1)) ^ set(walk)
    # randomly sample the same amount of nodes as the random walk
    negative_walk = random.sample(list(candidates), k=len(walk))
    return negative_walk
