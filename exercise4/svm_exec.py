#!/usr/bin/env python
# coding: utf-8
"""
Exercise 4
Code for training the SVM on the datasets DD, ENZYMES and NCI1.
For choosing a kernel, set the KERNEL_TYPE flag in the config.py module.
"""

import sys
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
import config_svm
import pickle
from utils.data_utils import get_graph_label
from utils.dataset_parser import Parser
from sklearn import svm, model_selection
from sklearn.model_selection import RepeatedKFold
from graphlet import compute_graphlets, load_graphlets, graph_to_hist
#from closed_walk import compute_closed_walks
#from wl_kernel import compute_wl
from sklearn.metrics.pairwise import linear_kernel


def load_datasets(names):
	"""
	Loads the graph datasets DD, ENZYMES and NCI1 and its labels.
	:params:    list of dataset names to load

	:return:    list of dataset names, list of loaded graphs for all datasets,
				labels for loaded graphs for all datasets
	"""

	# load datasets
	datasets = []
	if "dd" in names:
		datasets.append(Parser('datasets/DD'))
	if "enzymes" in names:
		datasets.append(Parser('datasets/ENZYMES'))
	if "nci1" in names:
		datasets.append(Parser('datasets/NCI1'))
	if "Specific" in names:
		datasets.append(Parser('datasets/specific_dataset'))
	if "Specific_20" in names:
		datasets.append(Parser('datasets/specific_dataset_20'))
	if "Specific_bin" in names:
		datasets.append(Parser('datasets/specific_dataset_bin'))
	if "Specific_less_triangles" in names:
		datasets.append(Parser('datasets/specific_dataset_less_triangles_abgabe'))
	# convert datasets into lists graphs, labels
	datasets = [dataset.parse_all_graphs() for dataset in datasets]

	# remove graphs with cardinality smaller 5, as they cannot be used for our graphlet kernel
	datasets = [[graph for graph in graphs if len(list(graph)) >= 5] for graphs in datasets]
	label_sets = [[get_graph_label(graph) for graph in graphs] for graphs in datasets]

	return names, datasets, label_sets


def compute_gram_matrix(x):
	"""
	Kernel function calculating the inner product.

	:param x: list of feature vectors
	:return: gram matrix
	"""
	return linear_kernel(x, x, dense_output=True)


@ignore_warnings(category=ConvergenceWarning)
def run_svm(names, vector_sets, label_sets, max_it):
	"""
	Trains an SVM for the given kernel using 10-fold cross validation with 10 repetitions.
	The number of iterations of the SVM can be adjusted by adjusting the MAX_IT parameter. Naturally higher Values
	lead to higher computation times and higher accuracies.

	We justify the cap by arguing that unreasonable computation times outweigh the importance of perfect accuracy
	results. The accuracy will not proportionally increase with computation time, but will almost remain the same,
	making unlimited number of iterations until convergence unfeasible. Improving the runtime is certainly a work in
	progress. By scaling the data beforehand we tried to fix convergence issues, especially with the enzymes dataset,
	but different scaling did not lead to success.

	:param names: list of names of the datasets
	:param vector_sets: list of vectors for each dataset
	:param label_sets: list of labels for each dataset
	:param max_it: number of maximal iterations for solver in svm
	"""

	# zipping names, feature vector list and label list together
	zips = zip(names, vector_sets, label_sets)

	# 10-fold cross-validating svm 10 times for each dataset
	for name, vectors, labels in zips:
		vectors = np.array(vectors)
		labels = np.array(labels)
		gram = np.array(compute_gram_matrix(vectors))
		print(np.shape(gram))
		print(f"Training for Dataset {name}.")
		clf = svm.SVC(kernel="precomputed", max_iter=max_it)
		scores = model_selection.cross_validate(clf, gram, labels,
											   cv=RepeatedKFold(n_splits=5, n_repeats=5), return_train_score=True)
		print("The Test Accuracies per run were:", scores['test_score'], "\n")
		print("Thus the test average accuracy over all runs was", np.average(scores['test_score']), "\n")
		print("With a standard deviation of ", np.std(scores['test_score']))
		print("and highest achieved test accuracy of ", np.max(scores['test_score']), ".\n")

		print("The train Accuracies per run were:", scores['train_score'], "\n")
		print("Thus the train average accuracy over all runs was", np.average(scores['train_score']), "\n")
		print("With a standard deviation of ", np.std(scores['train_score']))
		print("and highest achieved train accuracy of ", np.max(scores['train_score']), ".\n\n")

		print("Dummy Classifier:")
		dummy = DummyClassifier(strategy = "constant", constant = 25)
		dummy_scores = model_selection.cross_validate(dummy, gram, labels,
											   cv=RepeatedKFold(n_splits=5, n_repeats=5), return_train_score=True)
		print("The Test Accuracies per run were:", dummy_scores['test_score'], "\n")
		print("Thus the test average accuracy over all runs was", np.average(dummy_scores['test_score']), "\n")
		print("With a standard deviation of ", np.std(dummy_scores['test_score']))
		print("and highest achieved test accuracy of ", np.max(dummy_scores['test_score']), ".\n")

		print("The train Accuracies per run were:", dummy_scores['train_score'], "\n")
		print("Thus the train average accuracy over all runs was", np.average(dummy_scores['train_score']), "\n")
		print("With a standard deviation of ", np.std(dummy_scores['train_score']))
		print("and highest achieved train accuracy of ", np.max(dummy_scores['train_score']), ".\n\n")

@ignore_warnings(category=ConvergenceWarning)
def proof_classification(names, vector_sets, label_sets, max_it):
	"""
	Trains an SVM for the given kernel and saves it to disk. To proof that the dataset is indeed classifiable the 
	function will also evaluate a baseline dummy classifier to compare it to our model.
	
	:param names: list of names of the datasets
	:param vector_sets: list of vectors for each dataset
	:param label_sets: list of labels for each dataset
	:param max_it: number of maximal iterations for solver in svm
	"""

	# zipping names, feature vector list and label list together
	zips = zip(names, vector_sets, label_sets)
	for name, vectors, labels in zips:
		vectors = np.array(vectors)
		labels = np.array(labels)

		print(f"Training for Dataset {name}.")
		clf = svm.SVC(kernel="precomputed", max_iter=max_it)

		X_train, X_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.33, random_state=42)
		vectors = np.concatenate((X_train,X_test))
		train_indices = np.shape(X_train)[0]
		gram = np.array(compute_gram_matrix(vectors))

		X_train = gram[:train_indices,:train_indices]
		X_test = gram[(train_indices):,:(train_indices)]

		clf.fit(X_train,y_train)
		print("Following test predictions were made:")
		print(clf.predict(X_test))
		print("Classifier has a training accuracy of ", clf.score(X_train,y_train))
		print("Classifier has a test accuracy of ", clf.score(X_test,y_test))
		print("Baseline: Dummy Classifier predicts the class 13 everytime")
		dummy = DummyClassifier(strategy = "constant", constant = 13)
		dummy.fit(X_train,y_train)
		print("Dummy has a training accuracy of ", dummy.score(X_train,y_train))
		print("Dummy has a test accuracy of ", dummy.score(X_train,y_train))

		pkl_filename = "pickle_model.pkl"
		with open(pkl_filename, 'wb') as file:
			pickle.dump(clf, file)

def make_prediction(graphs):
	''' 
	Loads precomputed SVM with graphlet kernel and prints the prediction on the given dataset
	:param graphs: List of graphs to be predicted
	'''

	
	#load graphlet vectors of dataset for which the svm was fitted
	vectors = np.array(load_graphlets(["Specific_less_triangles"]))[0]
	prediction_indices = len(graphs) #number of predictions we need to make
	old_indices = 3484-prediction_indices #number of samples during training time-new predictions to be made
	vectors = vectors[:old_indices] #use only number of samples-number of new prediction many graphs to compare our new vectors with

	#Compute the graphlet vectors for each graph we need to predict
	prediction_vectors = []
	for graph in graphs:
		graphlet_embed = graph_to_hist(graph)
		prediction_vectors.append(graphlet_embed)
	
	prediction_vectors=np.array(prediction_vectors)

	#Put new graphlet vectors after the vectors of our comparison
	vectors = np.concatenate((vectors,prediction_vectors))
	gram = np.array(compute_gram_matrix(vectors)) #compute gram matrix of old+new vectors

	#gram matrix of the graphs we want to predict
	prediction_gram = gram[old_indices:]
	
	#load model and predict 
	pkl_filename = "pickle_model.pkl"
	with open(pkl_filename, 'rb') as file:
		pickle_model = pickle.load(file)
	print(pickle_model.predict(prediction_gram))

def main():
	"""
	Optionally enable the loading of the respective kernel vectors instead of recomputing them on the fly to decrease
	computation time


	"""
	names, datasets, labelsets = load_datasets(['Specific_less_triangles'])
	if config_svm.KERNEL_TYPE == "graphlet":
		# vectors = compute_graphlets(datasets, names)
		vectors = load_graphlets(names)
	elif config_svm.KERNEL_TYPE == "wl":
		vectors = compute_wl(datasets, names)
		# vectors = load_wl(names)
	elif config_svm.KERNEL_TYPE == "closed_walk":
		vectors = compute_closed_walks(datasets, names)
		# vectors = load_closed_walks(names)
	else:
		print("Invalid kernel type", sys.exc_info()[0])
		raise
	
	#run_svm(names, vectors, labelsets,10000)
	proof_classification(names,vectors,labelsets,10000)

if __name__ == "__main__":
	main()
