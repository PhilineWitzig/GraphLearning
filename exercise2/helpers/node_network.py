#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras import layers
import sys
import numpy as np
import tensorflow as tf
from utils.data_utils import get_adjacency_matrix, get_node_attributes, get_node_labels
from utils.dataset_parser import Parser


def load_datasets_nodelabel(names):
    """
    Loads the graph datasets DD, ENZYMES and NCI1, CiteSeer_Eval,CiteSeer_Train,Cora_Eval,Cora_Train and its labels.
    :params:    list of dataset names to load

    :return:    list of dataset names, list of loaded graphs for all datasets,
                node attributes for loaded graphs for all datasets
    """

    # load datasets
    datasets = []
    if "cs_eval" in names:
        datasets.append(Parser('datasets/CiteSeer_Eval'))
    if "cs_train" in names:
        datasets.append(Parser('datasets/CiteSeer_Train'))
    if "co_eval" in names:
        datasets.append(Parser('datasets/Cora_Eval'))
    if "co_train" in names:
        datasets.append(Parser('datasets/Cora_Train'))

    # convert datasets into lists graphs, labels
    datasets = [dataset.parse_all_graphs() for dataset in datasets]
    attr_sets = [[get_node_attributes(graph) for graph in graphs] for graphs in datasets]
    labels = [[get_node_labels(graph) for graph in graphs] for graphs in datasets]
    # attr_sets is a list of length n, where n is the number of datasets. Then attr[0] contains a list of all node attributes
    # for dataset 0. Thus attr[0][0] contains the actual node attribute matrix (X^0) for the graph of fataset 0.
    return names, datasets, attr_sets, labels


def f(x):
    '''
    Auxilliary function implementing degree normalization

    :param x: Float Value
    :return: normalized value
    '''
    return np.divide(1, np.sqrt(x))


def preprocess_matrices(datasets):
    '''
    Function implementing matreix normalization of adjecency matrices for all given graphs

    :param datasets: Array of datasets, each being a list of a single networkX graph
    :return: Array of normalized adjecency matrices, index 0 holding the matrix for the first data set and so on.
    '''

    matrices = []
    for i, dataset in enumerate(datasets):
        # get the adjecency matrix and add the identity matrix to it
        n = np.size(get_adjacency_matrix(dataset[0]), 0)  # dimension of adj.matrix
        adj_matrix_self_loops = np.add(get_adjacency_matrix(
            dataset[0]), np.identity(n))  # add self loops by identity matrix
        # compute diagonal matrix of degrees D
        D_diagonal = adj_matrix_self_loops.sum(axis=1)  # get list of node degrees
        D_diagonal = np.array(list(map(f, D_diagonal)))  # normalization
        D = np.zeros((n, n))  # make nxn matrix filled with zeros
        # fill the diagonal with normalized entries, obtainting root inverse D
        np.fill_diagonal(D, D_diagonal)

        # Symmetric normalization of adj.matrix
        A = np.matmul(adj_matrix_self_loops, D)
        A = np.matmul(D, A)

        matrices.append(A)  # collect all computed matrices

    return matrices


# In[2]:


"""
This module contains the implementation of the GNN architectures.
"""


class GCN(layers.Layer):
    """
    Call function takes a list of two tensors as input.
    The first entry  is the last node embedding.
    The second is the normalized adjacency matrix.
    """

    def __init__(self, feature_num, nl):
        """
        :param feature_num: number of features this layers outputs.
        :param nl: non-linearity which is applied after convoluting.
        """
        super(GCN, self).__init__()
        self.feature_num = feature_num
        self.nl = nl

    def build(self, input_shape):
        w_init = tf.initializers.GlorotUniform()
        self.w = tf.Variable(initial_value=w_init(
            shape=(input_shape[0][-1], self.feature_num), dtype='float32'), trainable=True)

    def call(self, inputs, **kwargs):
        if type(inputs) is not list or not len(inputs) == 2:
            raise Exception('GCN must be called on a list of two tensors. Got: ' + str(inputs))
        X = inputs[0]
        A = inputs[1]
        x = tf.matmul(A, X)
        x = tf.matmul(x, self.w)
        if self.nl == "relu":
            x = tf.nn.relu(x)
        elif self.nl == "softmax":
            x = tf.nn.softmax(x)
        else:
            raise ValueError(
                f"GCN Layer only supports 'relu' and 'softmax' as non linearity, but {self.nl} was passed.")
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0][1], self.feature_num


class SumPool(layers.Layer):
    def __init__(self, units):
        super(GCN, self).__init__()
        self.no_units = units

    def build(self):
        # TODO: check if this operation is valid
        self.w = tf.Variable(trainable=False)

    def call(self, node_features):
        # TODO: check if this operation is valid or if we need to do it by hand
        # In this case we will reuqire the no_units attribute
        return tf.math.reduce_sum(node_features, axis=0)


def model_GCN_node(v, k0, num_classes):
    """
    Model architecture of GCN for node classification.

    :param v: number of vertices in graphs
    :param k0: number of node features
    :param num_classes: nummer of classes to classify

    :return: instance of tf.keras.Model implementing the GCN node model architecture
    """
    # input layers, depending on size of node_features and
    node_features_input_layer = layers.Input(shape=(v, k0), name='node_feature_input')
    adjacency_matrix_input_layer = layers.Input(shape=(v, v), name='adjacency_matrix_input')
    x = GCN(32, "relu")([node_features_input_layer, adjacency_matrix_input_layer])
    x = GCN(num_classes, "softmax")([x, adjacency_matrix_input_layer])

    model = tf.keras.Model(inputs=[node_features_input_layer, adjacency_matrix_input_layer], outputs=[x],
                           name='GCN_Graph')
    return model


def model_GCN_graph(v, k0, num_classes):
    """
    Model architecture of GCN for graph classification.

    :param v:           number of vertices in the graph
    :param k0:          number of node features
    :param num_classes: number of graph classes to classify
    """
    node_features_input_layer = layers.Input(shape=(v, k0),
                                             name='node_feature_input')
    adjacency_matrix_input_layer = layers.Input(shape=(v, v),
                                                name='adjacency_matrix_input')
    gcn_1 = GCN(64, "relu")(something)
    gcn_2 = GCN(64, "relu")([gcn_1, adjacency_matrix_input_layer])
    gcn_3 = GCN(64, "relu")([gcn_2, adjacency_matrix_input_layer])
    gcn_4 = GCN(64, "relu")([gcn_3, adjacency_matrix_input_layer])
    gcn_5 = GCN(64, "relu")([gcn_4, adjacency_matrix_input_layer])
    sum_pool = SumPool(64)([gcn_5])
    fc_1 = layers.Dense(input=sum_pool, units=64, activation="relu")
    fc_2 = layers.Dense(input=fc_1, units=num_classes)
    softmax = layers.softmax(fc_2)

    model = tf.keras.Model(inputs=[node_features_input_layer, adjacency_matrix_input_layer],
                           outpus=[softmax])
    return model


# In[3]:


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# supresses warnings regarding float64 to float32 conversion for better readability

# We use Cross Entropy for our loss
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def loss(model, x, y, training):
    '''
    Function returning the loss of the given model.
    Uses the predefined loss of loss_object (which is Categocial Cross Entropy)

    :param model: Tensorflow Neural Network
    :param x: Feature data
    :param y: True Labels for the given feature data
    '''
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training=training)
    return loss_object(y, y_)


def grad(model, x, y):
    '''
    Function calculating the gradient of the given model.
    Uses the predefined loss of loss_object (which is Categocial Cross Entropy)

    :param model: Tensorflow Neural Network
    :param x: Feature data
    :param y: True Labels for the given feature data

    :return: Loss and Gradient
    '''
    with tf.GradientTape() as tape:
        loss_value = loss(model, x, y, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


# Define Optimizer and learning rate
# TODO: Find optimal learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


def train_model(num_epochs, model, features, labels):
    '''
    Function to train the given model. Will print accuracy and loss every 50 epochs.
    Uses a batch size of one and predefined learning rate given by the predefined optimizer.
    The optimizer is chosen to be Adam.

    :param num_epochs: Number of epochs
    :param model: Tensorflow Neural Network which should be trained
    :param features: Input for the Neural Network, given as list of inputs. The first entry should be the node attributes
                    and the second entry should be the normalized adecency matrix. Both have to be tensors.
    :param labels: True node labels
    '''
# Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []

    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # Optimize the model
        loss_value, grads = grad(model, features, labels)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Track progress
        epoch_loss_avg.update_state(loss_value)  # Add current batch loss
        # Compare predicted label to actual label
        # training=True is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        epoch_accuracy.update_state(labels, model(features, training=True))

      # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epoch % 50 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_accuracy.result()))


# In[ ]:


NUM_EPOCHS = 200


# TODO: Perform preprocessing beforehand and hand this to run_node_label_training.
# Currently we preprocess every iteration (10 times)
def run_node_label_training(names):
    '''
    Function used to train models on specified datasets. The function will automatically train the node classifying model
    on the training data associated with the given name for a fixed amount of epochs, printing the accuracy and loss per 50
    epochs.

    :param names: List of Strings representing the data sets. Options: "CiteSeer" or "Cora". e.g. ["CiteSeer"].
    :return: Trained Model for the last given data set in names. If only one Name was passed,
             it will return the accoring model

    '''

    load_datasets_names = []
    if "CiteSeer" in names:
        load_datasets_names.append('cs_train')
    if "Cora" in names:
        load_datasets_names.append('co_train')

    if not load_datasets_names:
        print("Please specify a dataset. Options are CiteSeer and Cora")

    for name in load_datasets_names:
        _, data, attr, labels = load_datasets_nodelabel([name])
        print("Currently Peprocessing data")
        normalized_adj_matrix = preprocess_matrices(data)[0]
        print("Matrix normalization completed")

        print("Beginning training for dataset", name)

        number_of_features = len(attr[0][0][0])
        number_of_nodes = len(attr[0][0])
        number_of_labels = np.amax(labels)

        normalized_adj_matrix = tf.convert_to_tensor(preprocess_matrices(data)[0])  # A
        node_attributes = tf.convert_to_tensor(attr[0][0])
        model = model_GCN_node(number_of_nodes, number_of_features, (number_of_labels+1))
        # TODO: check if +1 is valid or just a stupid quick fix
        # by default the labels range [0,number_of_labels), since we range [1,number_of_labels] we add 1 instead.

        features = [node_attributes, normalized_adj_matrix]
        labels_ = labels[0][0]

        train_model(NUM_EPOCHS, model, features, labels_)
    return model

# EXAMPLE:
# run_node_label_training(["CiteSeer"])


def test_node_network(name):
    if name == "CiteSeer":
        _, data, attr, labels = load_datasets_nodelabel(["cs_eval"])
    elif name == "Cora":
        _, data, attr, labels = load_datasets_nodelabel(["co_eval"])
    else:
        print("Please specify a dataset. Options: CiteSeer, Cora. Please spellcheck.")

    print("Begin Testing")
    test_accuracy = tf.keras.metrics.Accuracy()
    normalized_adj_matrix = tf.convert_to_tensor(preprocess_matrices(data)[0])  # A
    node_attributes = tf.convert_to_tensor(attr[0][0])
    features = [node_attributes, normalized_adj_matrix]
    labels_ = labels[0][0]

    accuracies = []
    for i in range(10):
        print("Begin training iteration number ", i)
        model = run_node_label_training([name])
        # training=False is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        logits = model(features, training=False)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        accuracy = test_accuracy(prediction, labels_)
        accuracies.append(accuracy.numpy())
        print("Iteration {:02d} testing accuracy: {:.3%}".format(i, test_accuracy.result()))

    print("Average testing accuracy is", np.average(accuracies))


test_node_network("CiteSeer")


# In[ ]:
