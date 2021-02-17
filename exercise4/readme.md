# Information about Exercise 4

## How to use command line interface

In order to execute the series of experiments evaluating the individual layers, execute [layer_evaluation.py](layer_evaluation.py) in the root dir.
The results will be printed in the command line and stored in a [log file](logs/experiment.log).

## Results

### Layer evaluation experiments

#### Experimental Setup

We chose a subset of the NCI1 dataset (50% stratified sample), a learning rate of 0.0001 and a maximum of 100 epochs of training
(training was stopped early when the validation loss did not improve for 5 epochs), to be consistent throughout all experiments.
We introduced early stopping in the experimental setup to make training feasible.

We ran four different model architectures:
- a baseline version with standard GCN layers and without advanced pooling (baseline)
- a model similar to the baseline version, with GCN skip layers instead of standard GCN layers (skip)
- a model with standard GCN layers and a pooling layer performing diff pooling (diff)
- a model with standard GCN layers and a pooling layer performing topK pooling (topK)

Each model is build up from two block of convolutions with a variable number of GCN layers in each block (depth).
The number of features (width) of each GCN layer is fixed within each block, but variable for different blocks.
The advanced pooling layers are located in between the convolution blocks, if they are used. There hyper-parameter k is variable.

We choose the following values for the parameters to gain insight about the individual layer's impact over multiple runs:
- depth: 1 or 2
- width: block 1 128, block 2 64 or block 1&2 64
- k: 2, 10 or 50

This resulted in 32 distinct experiments, in which the dataset was trained and evaluated with a 10-fold stratified split.

#### Results

'Baseline':

| Width  | Depth | Train acc (Std. Dev) | Eval acc (Std. Dev) |
|--------|-------|----------------------|---------------------|
| 64,64  | 1     | 0.7188 (0.0122)      | 0.7348 (0.0398)     |
| 64,64  | 2     | 0.7284 (0.01)        | 0.7445 (0.0382)     |
| 128,64 | 1     | 0.7313 (0.0093)      | 0.7445 (0.0382)     |
| 128,64 | 2     | 0.7333 (0.0139)      | 0.745  (0.0352)     |

'Skip':

| Width  | Depth | Train acc (Std. Dev) | Eval acc (Std. Dev) |
|--------|-------|----------------------|---------------------|
| 64,64  | 1     | 0.751 (0.0285)       | 0.7581 (0.0485)     |
| 64,64  | 2     | 0.7074 (0.1018)      | 0.7193 (0.1151)     |
| 128,64 | 1     | 0.7114 (0.1066)      | 0.7075 (0.1173)     |
| 128,64 | 2     | 0.6822 (0.1173)      | 0.6822 (0.1223)     |

'Diff':

| Width  | Depth | k  | Train acc (Std. Dev) | Eval acc (Std. Dev) |
|--------|-------|----|----------------------|---------------------|
| 64,64  | 1     | 2  | 0.5064 (0.007)       | 0.5051 (0.0095)     |
| 64,64  | 1     | 10 | 0.6593 (0.0788)      | 0.6686 (0.0899)     |
| 64,64  | 1     | 50 | 0.7104 (0.011)       | 0.7143 (0.0355)     |
| 64,64  | 2     | 2  | 0.5024 (0.0028)      | 0.5002 (0.0017)     |
| 64,64  | 2     | 10 | 0.546 (0.0804)       | 0.5562 (0.0897)     |
| 64,64  | 2     | 50 | 0.719 (0.0215)       | 0.7324 (0.0462)     |
| 128,64 | 1     | 2  | 0.5033 (0.0036)      | 0.5007 (0.0027)     |
| 128,64 | 1     | 10 | 0.6065 (0.1013)      | 0.5995 (0.1011)     |
| 128,64 | 1     | 50 | 0.7142 (0.0179)      | 0.7265 (0.038)      |
| 128,64 | 2     | 2  | 0.5032 (0.0032)      | 0.5027 (0.0082)     |
| 128,64 | 2     | 10 | 0.5718 (0.1004)      | 0.5728 (0.1098)     |
| 128,64 | 2     | 50 | 0.7392 (0.0192)      | 0.7401 (0.0419)     |

'TopK':

| Width  | Depth | k  | Train acc (Std. Dev) | Eval acc (Std. Dev) |
|--------|-------|----|----------------------|---------------------|
| 64,64  | 1     | 2  | 0.5139 (0.0076)      | 0.5197 (0.0169)     |
| 64,64  | 1     | 10 | 0.6122 (0.0291)      | 0.6277 (0.052)      |
| 64,64  | 1     | 50 | 0.622 (0.0589)       | 0.6161 (0.0667)     |
| 64,64  | 2     | 2  | 0.5142 (0.0133)      | 0.5284 (0.0215)     |
| 64,64  | 2     | 10 | 0.5794 (0.0518)      | 0.5913 (0.0663)     |
| 64,64  | 2     | 50 | 0.5823 (0.074)       | 0.589  (0.0941)     |
| 128,64 | 1     | 2  | 0.5221 (0.013)       | 0.5328 (0.0249)     |
| 128,64 | 1     | 10 | 0.5987 (0.0599)      | 0.6169 (0.0675)     |
| 128,64 | 1     | 50 | 0.6031 (0.0647)      | 0.6014 (0.066)      |
| 128,64 | 2     | 2  | 0.5139 (0.0139)      | 0.5256 (0.0139)     |
| 128,64 | 2     | 10 | 0.5131 (0.0369)      | 0.5095 (0.0369)     |
| 128,64 | 2     | 50 | 0.5995 (0.0755)      | 0.6102 (0.0913)     |


### Optimal Model Architecture Experiments

Using the results from the layer experiments, we decided on creating an architecture consisting of normal GCN layers with DiffPooling
since we obtained the highest accuracies for these layers on NCI1.

To train the architecture, run [model_evaluation.py](model_evaluation.py). The dataset the model is trained on can be set in [config.py](config.py)
under "Training configurations for optimal architecture". Depending on the dataset, a slightly different architecture is chosen to achieve optimal
results (i.e. additional dropout layers of ENZYMES). Note that the number of epochs might also vary depending on the dataset as documented below.

Again, we chose 10 k-fold cross validation, a learning rate of 0.001 and a different number of epochs depending on the
dataset:
- ENZYMES: 500 (cf. assignment 2)
- NCI1: 50 (due to early convergence)
- PROTEINS: 50 (due to early convergence)
We did not use any early stopping in this case to avoid early termination. The batch size was set to 24, using Adam as an optimizer.

We chose a pooling ratio of approx. 50%, thus having different pooling layer dimensions depending on the dataset:
- Since NCI1 has on average 29.87 nodes, we chose the first pooling dimension to be 15.
- Since ENZYMES has on average 32.63 nodes, we chose the first pooling dimension to be 16.
- Since PROTEINS has on average 39.05 nodes, we chose the first pooling dimension to be 20.

In terms of the network design, we opted for the following network architecture using our insights from the above layer experiments and literature
(nc = number of classes, nn = average node number per graph)

| Type      | Param |
|-----------|-------|
| GCN       | 64    |
| GCN       | 64    |
| GCN       | 64    |
| Drop      | 0.2   |
| DiffPool  | 0.5*nn|
| GCN       | 64    |
| Drop      | 0.3   |
| SumPool   | 64    |
| FC        | nc    |

- We also tested a hierarchical GCN block with GCN(128), GCN(64), GCN(32) for the first GCN Block and keeping the feature dimensions at 32.
This lead to a minor decrease in accuracy of about 1%.
- Moreover, we implemented deeper architectures, i.e. with multiple GCN blocks followed by DiffPool layers. However, this lead to a decrease in accuracy of about 10% on NCI1.
- Furthermore, we tested to deploy one pooling layer after each GCN layer. However, this lead to a strong decrease in accuracy of about 10%. We assume
that the 3-layer GCN for assignment matrix generation in the DiffPool layer is necessary to encode graph features more adequately.
- Introducing dropout layers after each GCN layer was particularly important for training the architecture on ENZYMES. In this case we had to deal with
strong overfitting. We chose a dropout rate of 30% for enzymes. Higher drop put rates lead to a decrease in accuracy.
For NCI1 and PROTEINS, having dropout layers after the GCN blocks was sufficient. Otherwise the network underfitted slightly.
- Last but not least, we also manipulated the pooling rate. Having a pooling rate of 75% did not lead to an increase in accuracy for NCI1. In contrast,
we achieved a train and validation accuracy of 79% in the case of PROTEINS. For ENZYMES we noticed an increase in validation accuracy of only 1%.
 Pooling rates below 50% decreased the accuracy.


(Above avg. node values from: Madria, Sanjay, et al., eds. Big Data Analytics: 7th International Conference, BDA 2019, Ahmedabad, India, December 17–20, 2019, Proceedings. Vol. 11932. Springer Nature, 2019.)


#### Results

'NCI1':

Train:
- Average max acc: 0.7749
- Std dev: 0.0065

Validation:
- Average max acc: 0.7640
- Std dev: 0.0382



'PROTEINS':

Train:
- Average max acc: 0.7502
- Std dev: 0.1138

Validation:
- Average max acc: 0.7694
- Std dev: 0.1340


'ENZYMES':

Train:
- Average max acc: 0.6607
- Std dev: 0.0342

Validation:
- Average max acc: 0.5133
- Std dev: 0.0618



## Regarding exercise 4

The dataset is located in [datasets\specific_dataset_less_triangles_abgabe](datasets/specific_dataset_less_triangles_abgabe).

The script we used to make it is [make_dataset.py](make_dataset.py) which uses the function "make_specific_dataset()".
We made a dataset consisting of 5200 graphs, which have between 5 and 70 nodes. The task is to predict the number of triangles each graph has.

The dataset has 26 different classes, where the class is the number of triangles in the graph (i.e. between 0 and 25 triangles). Each class has a total of
200 representatives. There are no node attributes or labels. The jupyter notebook [data_set_ex4.ipynb](data_set_ex4.ipynb) guides through all steps of the computation of the dataset,
the classifier and making predictions.

To execute training and saving of a classifier execute [svm_exec.py](svm_exec.py) in root dir, or refer to the jupyter notebook [data_set_ex4.ipynb](data_set_ex4.ipynb).


The classifier we used is a SVM with the graphlet kernel of Sheet 1 with subgraph size being 5 nodes and vector size 34. The according gram matrix is precomputed.


Classifier has a training accuracy of  0.09184845005740529

Classifier has a test accuracy of  0.07925407925407925

Baseline: Dummy Classifier predicts the class 13 everytime

Dummy has a training accuracy of  0.03559127439724455

Dummy has a test accuracy of  0.03559127439724455


To make predictions on new graphs please refer to the last cell of [data_set_ex4.ipynb](data_set_ex4.ipynb), as there is currently no command line interface
to make predictions just yet.


The saved classifier is called [pickle_model.pkl](pickle_model.pkl) and needs the graphlet vectors found in data/grpahlet/dataset_Specific_less_triangles.npy.
