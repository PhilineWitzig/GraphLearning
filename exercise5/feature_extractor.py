"""
Collection of feature extracting algorithms.
"""
import networkx as nx
import random
import numpy as np
import config
from tqdm import tqdm
from utils.data_utils import get_node_attributes


def color_to_hist(graphs, color_counts, no_colors):
    """
    Computes the histograms of the current graph coloring for a lits of input graphs.

    :param graphs:          dictionary holding the current coloring, indexed by vertices
    :param color_counts:    coloring of the different graphs
    :param no_colors:       number of potential colors in the graph

    :return:            np vector representing the histogram
    """

    hists = []
    for i in range(len(graphs)):
        cur_hist = np.zeros([no_colors])
        for color, count in color_counts[i].items():
            cur_hist[color - 1] += count
        hists.append(cur_hist)

    return hists


def get_neighbor_labels(graph, v, sort=True):
    """
    Return the labels the node's neighbors either sorted or not.

    :param graph:   networkx graph
    :param v:       node v in the graph for which we check the neighbors
    :param sort:    whether the labels should be returned in a sorted manner or not

    :return:        list of neighbor labels
    """
    neighbor_indices = [nv for nv in graph.neighbors(v)]
    neighbor_labels = []
    if sort:
        neighbor_labels = sorted(graph.nodes[nl]['node_label'] for nl in neighbor_indices)
    else:
        neighbor_labels = graph.nodes[neighbor_indices]['node_label']

    return neighbor_labels


def color_refinement(graphs):
    """
    Executes a finite number of color refinement steps given networkx graphs.

    :param graphs:  list of networkx graph

    :return:        list the refined graphs, a list of dictionaries containing
                    containig the colors of a graph and how often they occur
    """
    coloring = dict()
    refined_graphs = []
    color_counts = []

    for i in range(len(graphs)):
        color_count_dict = dict()
        graph = graphs[i]
        graph_refined = graph.copy()

        for v in graph.nodes:
            current_label = graph.nodes[v]['node_label']
            neighbor_labels = get_neighbor_labels(graph, v)

            # "zip" vertex label and labels of neighbors
            merged_labels = str((neighbor_labels, current_label))

            # generate a label dict based on merged labels
            if merged_labels in coloring.keys():
                color = coloring[merged_labels]
            else:
                color = len(coloring) + 1
                coloring[merged_labels] = color

            # relabel nodes
            graph_refined.nodes[v]['node_label'] = color

            # count how often each color occurs
            if color in color_count_dict:
                color_count_dict[color] += 1
            else:
                color_count_dict[color] = 1

        color_counts.append(color_count_dict)
        refined_graphs.append(graph_refined)

    return refined_graphs, color_counts


def load_node_features(graphs):
    return [get_node_attributes(graph) for graph in graphs]


def random_walk_features(graphs, param_return, param_in_out, length):
    """
    Performs a random walk of length l on each node in each graph and reaturn the random walks as node features.

    :param graphs: list of networkx graphs
    :param param_return: Bias of going back to the old node (p)
    :param param_in_out: Bias of moving forward to a new node (q)
    :param length: length of random walks
    """

    features = []

    for graph in tqdm(graphs):
        graph_features = []
        for node in list(graph.nodes):
            w = compute_random_walk(graph, node, param_return, param_in_out, length)
            graph_features.append(w)
        features.append(graph_features)
    return features


def compute_random_walk(graph, startnode, param_return, param_in_out, length):
    """
    Computes and returns random walks of size length. The parameters control
    how fast the walk explores or leaves the neighborhood of starting at the startnode.

    :param graph:           networkX graph
    :param startnode:       integer, Initial node
    :param param_return:    bias of going back to the old node (p)
    :param param_in_out:    bias of moving forward to a new node (q)
    :param length:          integer length of desired random walk

    :return:                walk without start node
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


def degree_features(graphs):
    """
    Computes the node degrees for all nodes in a graph and stores them as a feature vector.

    :param graphs:  list of networkx graphs

    :return:        feature_vectors containing the degree features
    """
    features = []
    max_len = 0

    for i in range(0, len(graphs)):
        degree_list = [degree for (node, degree) in list(graphs[i].degree())]
        features.append(degree_list)
        if len(degree_list) > max_len:
            max_len = len(degree_list)

    # Zero-Padding so that feature vectors have equal lengths
    feature_vectors = np.zeros((len(graphs), max_len))
    for i in range(0, len(graphs)):
        feature_vectors[i, 0:len(graphs[i])] = features[i]

    return feature_vectors


def color_features(graphs, max_number_nodes, steps=config.REFINEMENT_STEPS):
    """
    Computes node color features by running the color refinement algorithm and
    representing a node's color as a one-hot-vector.

    :param graphs:              list of networkx graphs
    :param max_number_nodes:    maximum number of nodes in all graphs
    :param steps:               number of refinement steps during color refinement

    :return:                    color feature vectors
    """

    node_features = []

    for i in range(steps):
        graphs, _ = color_refinement(graphs)

    for i in range(len(graphs)):
        node_feats_per_graph = []
        for n in graphs[i].nodes:
            node_feats_per_graph.append(graphs[i].nodes[n]['node_label'] - 1)
        node_features.append(node_feats_per_graph)

    max_label = get_max_node_color(node_features)

    node_features_one_hot = np.zeros([len(graphs), max_number_nodes, max_label + 1])

    for i, node_feats_per_graph in enumerate(tqdm(node_features, desc="One hot encoding color refinements")):
        node_feats_per_graph_np = np.array([node_feats_per_graph]).reshape(-1)
        node_feats_per_graph_one_hot = np.eye(max_label + 1)[node_feats_per_graph_np]

        node_features_one_hot[i, :node_feats_per_graph_one_hot.shape[0]] = node_feats_per_graph_one_hot

    return node_features_one_hot

def get_max_node_color(node_features):
    """
    Get the maximum node color in the current graph coloring.

    :param node_features:   node features containing the coloring

    :return:                max node color in features
    """
    max_color = node_features
    while isinstance(max_color, list):
        max_color = np.max(max_color)
    return max_color


def color_hist_features(graphs, steps=config.REFINEMENT_STEPS):
    """
    Computes the color histograms for each graph for a specified number of refinement steps.

    :param graphs:   a list of networksx graphs

    :return:         a list of the color histograms
    """

    hist_collec = []

    for i in range(steps):
        graphs, color_counts = color_refinement(graphs)
        no_colors = len(list(set(color for cur_coloring in color_counts for color in cur_coloring.keys())))
        # compute histograms from graph colors
        color_hists = color_to_hist(graphs, color_counts, no_colors)
        hist_collec.append(color_hists)

    accum_hists = np.zeros((len(graphs), no_colors))
    padding = np.zeros((no_colors))

    for i in range(len(hist_collec)):
        for j in range(len(hist_collec[i])):
            cur_hist_padded = padding.copy()
            cur_hist_padded[0:len(hist_collec[i][j])] = hist_collec[i][j]
            accum_hists[j] += cur_hist_padded

    return accum_hists
