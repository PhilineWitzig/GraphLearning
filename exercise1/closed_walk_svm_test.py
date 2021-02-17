import os
import numpy as np

from data_utils import get_adjacency_matrix
from numpy import linalg as LA
from progress.bar import Bar
from data_utils import get_graph_label
from dataset_parser import Parser

from sklearn import svm
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


CLOSED_WALK_VECTOR_SIZE = 3 #maximal length of closed walks
COMPUTE_KERNEL_MATRICES = False #Set true to compute all kernel matrixes from scratch, if false, the program will load previous results instead
SVM_MAX_ITERATIONS = 10 #Number of Iterations for SVM classification
MAX_IT = 10000 #Number of SVM iterations

def load_datasets():
    """
    Utility function returning the datasets as list of lists with respective names and true labels for each graph
    """
    names = ['dd', 'enzymes', 'nci1']
    #logging.info("[INFO] Loading datasets: {}, {}, {}."
                # .format(names[0], names[1], names[2]))

    dd = Parser('datasets/DD')
    enzymes = Parser('datasets/ENZYMES')
    nci1 = Parser('datasets/NCI1')

    # convert datasets into lists graphs, labels
    datasets = [dd.parse_all_graphs(),
                enzymes.parse_all_graphs(),
                nci1.parse_all_graphs()]

    
    label_sets = [[get_graph_label(graph) for graph in graphs] for graphs in datasets]

    return names, datasets, label_sets

def compute_closed_walks_matrix(datasets, names):
    """
    Computes the closed walk kernel matrix for all datasets, will save all computed results in the process in the Folder "data"
    
    :param datasets: List of datasets, where each dataset is a list of NetworkX graphs
    :param names: List of Strings, which are the names of the dataset at the respective index
    
    :return: Closed Walk Matrices for each dataset as a List of Matrices
    """
    #Make directory accordingly
    if not os.path.exists(os.path.join(os.getcwd(), "data/closed_walk")):
        os.makedirs(os.path.join(os.getcwd(), "data", "closed_walk"))

    # Initialize Datastructures
    closed_walk_sets = [] #holds the respective closed walk vector for each graph for all datasets
    matrix_set = [] 

    #compute all closed walk vectors for each dataset respectively
    for i, dataset in enumerate(datasets):
        closed_walks = np.zeros([len(dataset), CLOSED_WALK_VECTOR_SIZE],dtype=np.int32)
        bar = Bar("Processing " + names[i], max=len(dataset))
        for j, graph in enumerate(dataset):
            bar.next()
            closed_walks[j] = closed_walks_vector(graph, CLOSED_WALK_VECTOR_SIZE)
        bar.finish()
        
        #Compute the Kernel matrix once all graphs of a single dataset are processed
        kernel_matrix = np.dot(closed_walks,closed_walks.T)
        
        #Save the kernel:matrix and the computed vectors seperatly
        np.save(os.path.join(os.getcwd(),
                            "data", "closed_walk", "dataset_" + names[i] + ".npy"), closed_walks)
        np.save(os.path.join(os.getcwd(),
                            "data", "closed_walk", "dataset_" + names[i] + "_matrix"+".npy"), kernel_matrix)

        closed_walk_sets.append(closed_walks)
        matrix_set.append(kernel_matrix)
        
    #print(closed_walk_sets)
    return matrix_set


def load_closed_walks(names):
    """
    Utility Function loading and returning saved npy files for closed walk vectors. Will return all vectors of the dataset
    
    :param name: List of Names of datasets ("dd", "enzymes","nci1")
    :return: List of the respective lists of vectors
    
    """
    closed_walk_sets = []
    for name in names:
        closed_walks = np.load(os.path.join(os.getcwd(),
                                            "data", "closed_walk", "dataset_" + name + ".npy"))
        closed_walk_sets.append(closed_walks)
    #print(closed_walk_sets)    
    return closed_walk_sets



def load_kernel_matrix(name):
    """
    Utility Function to return the specific closed walk kernel matrix for the given dataset
    
    :param name: Name of the dataset ("dd", "enzymes","nci1")
    :return: Kernel Matrix 
    """
    kernel_matrix = np.load(os.path.join(os.getcwd(),
                                    "data", "closed_walk", "dataset_" + name+"_matrix" + ".npy"))
    return kernel_matrix


def closed_walks_vector(x, max_i):
    """
    Function calculating the vector of dimension max_i containing the number of closed walks of length 2 upto max_i
    We make use of the spectral theorem for the calculation of the closed path number by using the eigenvalues
    of the adjecency matrix
    
    :param x: A NetworkX Graph 
    
    :return: Vector of dimension max_i, index i contains the number of closed walks of length i+2
    
    """

    ##Initialization of used datastructures##
    walks_x = np.zeros(max_i, dtype = np.int32)  # Vector holding the number of closed walks
    eig_values, _ = LA.eig(get_adjacency_matrix(x))  # Vector holding the eigenvalues of the Adjecency Matrix

    for i in range(max_i):
        # Summation of the powered eigenvalues, we start with i+2
        # since closed walks start with length 2. walks_x[0] will thus hold the walks of length 2, not 0.
        walks_x[i] = int((np.sum([np.power(y, i + 2) for y in eig_values])))
        if walks_x[i] < 1:  # Handling of numerical nuisance
            walks_x[i] = 0
    return walks_x

@ignore_warnings(category=ConvergenceWarning)
def run_svm(kernel_matrix, labels):
    """
    Trains an SVM for the given kernel using 10-fold cross validation with 10 repititions.
    The number of iterations of the SVM can be adjusted by adjusting the MAX_IT parameter. Naturally higher Values
    lead to higher computation times and higher accuracies.
    
    We justify the cap by agruing that unreasonable computation
    times outweigh the importance of perfect accuracy results. The accuracy will not proportionally increase with
    computation time, but will almost remain the same, making unlimited number of iterations until convergence unfeasable.
    Improving the runtime is certainly a work in progress. By scaling the data beforehand we tried to fix convergence issues,
    especially with the enzymes dataset, but different scaling did not lead to success. 
    
    :param kernel_matrix: name of the kernel to be used for training, possible options: "graphlet","wl", "closed_walk"
    :param labels: True labels for the graphs in the given dataset
    
    """
    X = kernel_matrix
    
    #Scaling the kernel to values in range (0,1) to prevent long convergence times
    scaler = MinMaxScaler() #Switching between MinMax and StandardScaler. We didnt see improved results between the two.
    #scaler=StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    
    y = labels
    clf = svm.SVC(kernel="precomputed", max_iter = MAX_IT)
    scores = model_selection.cross_val_score(clf, kernel_matrix, y,
                                                       cv=RepeatedKFold(n_splits=10, n_repeats=10, random_state=345369))
    print("The Accuracies per run were:", scores, "\n")
    print("Thus the average accuracy over all runs was", np.average(scores), "\n")
    print("With a standard deviation of ", np.std(scores))
    print("and highest achieved Accuracy of ", np.max(scores), ".")
    

if __name__ == "__main__":
    """
    By default this will use precomputed gram matrices for the closed walk kernel and train an SVM via 10fold crossvalidation
    and 10 repeats. If the Flag COMPUTE_KERNEL_MATRICES is set, it will instead compute the Matrices from scratch, significantly inscreasing runtime
    """
    
    #Compute Kernel Matrixes from scratch if the Flag is set, by default disabled
    if(COMPUTE_KERNEL_MATRICES):
        matrix_set = compute_closed_walks_matrix(datasets,names)
        #print(matrix_set[0]) refers to Dataset "dd"
        #print(matrix_set[1]) refers to Dataset "enzymes"
        #print(matrix_set[2]) refers to Dataset "nci1"

        
         #for explicit computation of specific datasets please use 
            #compute_closed_walks_matrix([datasets[0]],["dd"]))
            #compute_closed_walks_matrix([datasets[1]],["enzymes"]))
            #compute_closed_walks_matrix([datasets[3]],["nci1"]))
    
    
    names, datasets, labelsets = load_datasets()
    
    #Train and run the SVM on each dataset
    for i, name in enumerate(names):
        print("Currently training for dataset", name)
        kernel_matrix = load_kernel_matrix(name)
        run_svm(kernel_matrix, labelsets[i])
