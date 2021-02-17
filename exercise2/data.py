"""
This module loads data and performs the preprocessing steps required for
feeding the dataset to the GNN.
"""
import sys
from utils.data_utils import *
from utils.dataset_parser import Parser


def load_dataset_nodelabel(name):
    """
    Loads one of the graph datasets DD, ENZYMES and NCI1, CiteSeer_Eval,CiteSeer_Train,Cora_Eval,Cora_Train and its labels.
    :params:    name of dataset to load

    :return:    list of loaded graphs, list of node features, list of labels
    """

    if "cs_train" == name:
        dataset = Parser('datasets/CiteSeer_Train')
    elif "cs_eval" == name:
        dataset = Parser('datasets/CiteSeer_Eval')
    elif "co_train" == name:
        dataset = Parser('datasets/Cora_Train')
    elif "co_eval" == name:
        dataset = Parser('datasets/Cora_Eval')
    else:
        raise ValueError(f"Given dataset name {name} is not a valid node dataset.")
    dataset = dataset.parse_all_graphs()
    # convert datasets into lists graphs, labels
    features = [get_node_attributes(graph) for graph in dataset]
    labels = [get_node_labels(graph) for graph in dataset]

    return dataset, features, labels


def load_dataset_graph(name):
    """
    Loads the graph dataset from file given a dataset name in form of a string.

    :param name:    string of dataset name
    :return:        loaded networkX graphs, graph labels
    """
    if name == "Nci1":
        dataset = Parser("datasets/NCI1")
    elif name == "Enzymes":
        dataset = Parser("datasets/ENZYMES")
    else:
        raise ValueError(f"Given dataset name {name} is not a valid graph dataset.")

    dataset = dataset.parse_all_graphs()
    labels = [get_graph_label(graph) for graph in dataset]

    return dataset, labels


def f(x):
    """
    Auxilliary function implementing degree normalization

    :param x: Float Value
    :return: normalized value
    """

    return np.divide(1, np.sqrt(x), out=np.zeros_like(x), where=x != 0)



def normalize_adjacency_matrices(dataset):
    """
    Function implementing matrix normalization of one adjacency matrix for a
    given graph.
    
    :param dataset: list of a single networkX graphs
    :return: list of normalized adjacency matrices
    """

    #  get the adjecency matrix and add the identity matrix to it
    n = np.size(get_adjacency_matrix(dataset[0]), 0)  # dimension of adj.matrix
    # add self loops by identity matrix
    adj_matrix_self_loops = np.add(get_adjacency_matrix(dataset[0]), np.identity(n))
    # compute diagonal matrix of degrees D
    D_diagonal = adj_matrix_self_loops.sum(axis=1)  # get list of node degrees
    D_diagonal = np.array(list(map(f, D_diagonal)))  # normalization
    D = np.zeros((n, n))  # make nxn matrix filled with zeros
    # fill the diagonal with normalized entries, obtainting root inverse D
    np.fill_diagonal(D, D_diagonal)

    #  Symmetric normalization of adj.matrix
    A = np.matmul(adj_matrix_self_loops, D)
    A = np.matmul(D, A)

    return A


def normalize_adj_matrices_for_graphs(graphs):
    """
    Normalizes all adjacency matrices of graphs in the dataset.

    :param graphs:  list of networkX graphs
    :return:        np array of normalized adjacency matrices
    """

    adjacency_matrices = get_padded_adjacency(graphs)

    # add self_loops
    for i in range(adjacency_matrices.shape[0]):
        adjacency_matrices[i] = np.add(adjacency_matrices[i],
                                       np.identity(adjacency_matrices.shape[2]))

    D_diagonal = adjacency_matrices.sum(axis=1)
    D_diagonal = np.array(list(map(f, D_diagonal)))
    D = np.zeros(adjacency_matrices.shape)

    for i in range(adjacency_matrices.shape[0]):
        np.fill_diagonal(D[i], D_diagonal[i])

    A = np.matmul(adjacency_matrices, D)
    A = np.matmul(D, A)

    return A


def get_preprocessed_data(name):
    """
    loads and preprocesses the given dataset.

    :param name: name of dataset to load and preprocess. Possible are 'cs_eval', 'cs_train',
    'co_eval', 'co_train'.

    :return: list of normalized adjacency matrices, list of node features and tensor of one hot encoded node labels,
        number of distinct label classes
    """
    dataset, features, labels = load_dataset_nodelabel(name)
    matrices = normalize_adjacency_matrices(dataset)

    matrices = np.asarray(matrices)
    features = np.asarray(features)
    labels = np.asarray(labels)

    # map labels form [1,7] to [0,6]
    labels = labels - 1

    return matrices, features, labels


def get_preprocessed_graph_data(name):
    """
    This function loads a dataset for graph classification.
    In the case of the Nci1 datset, feature vectors are the padded node labels.
    In the case of the Enzymes dataset, feature vectors are the padded node labels
    concatenated with the padded node attributes.
    Any other dataset will lead to an error.

    :param name:    string of dataset which is to be loaded
    """

    dataset, labels = load_dataset_graph(name)
    labels = np.asarray(labels)

    if name == "Nci1":
        matrices = normalize_adj_matrices_for_graphs(dataset)
        # features of NCI1 are the node labels in a one-hot encoding
        features = get_padded_node_labels(dataset)
    elif name == "Enzymes":
        matrices = normalize_adj_matrices_for_graphs(dataset)
        # map labels form [1,6] to [0,5]
        labels = labels - 1
        node_labels_one_hot = get_padded_node_labels(dataset)
        attributes = get_padded_node_attributes(dataset)
        # features of ENZYMES are the node labels in a on-hot encoding concatenated with the attribute features
        features = np.dstack((node_labels_one_hot, attributes))
    else:
        print("Invalid dataset.")
        sys.exit()

    return matrices, features, labels
