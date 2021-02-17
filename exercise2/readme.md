# Information about Exercise 2

## How to use command line interface
Execute [cmd_interface.py](cmd_interface.py) in root dir.
Necessary arguments are datasets and classifier:

'classifier' should be either 'Node' or 'Graph', determining which model is to be used.

'datasets' should be a comma separated list of datasets to be trained and evaluated by the network.
For the 'Node' classifier, 'CiteSeer' and 'Cora' are valid datset inputs and for the 'Graph' classifier 'ENZYMES' and 'NCI1'.
Please only use one dataset for the graph classifier.

Example Command:
```
group1\exercise2>python3 cmd_interface.py --datasets CiteSeer,Cora --classifier Node
```
The command line interface executes the train and evaluation loop 10 times and outputs the average maximal training and validation accuracy.

## Project Structure
The network's architecture definitions can be found in the [model.py](model.py) file. We implement custom layers for graph convolution, for the Sum Pool layer as well for fully connected layers.

Any function related to loading datasets and preprocessing the datasets can be found in [data.py](data.py).

The training loops for both, the graph and the node classifier, are located in [train.py](train.py).

We plot our results for the graph classifier using tensorboard. The respective log files are contained in the [logs folder](logs/scalars/graph). Confer the [this folder](logs/scalars/graph/enzymes_shuffeling) for the best results on ENZYMES.

Note that depending on the dataset and the classifier, you might have to adjust the hyperparameters in the [config.py](config.py) to obtain the results we report below.

## Results

### Node Classifier

Number of Epochs: 150

Learning rate: 0.001

Optimizer: Adam


The following table shows the results when using the model structure specified by the exercise sheet.
To combat overfitting we introduced a dropout layer in between the GCN layers with dropout rate 0.2.

Dataset | Avg max train acc | Avg max validation acc
--- | --- | ---
CiteSeer | 0.9092 | 0.743
Cora | 0.9114 | 0.8039

Adding additional GCN layers (k=64,32,16) led to no improvements or a slight decrease in accuracies (~0.1%).

Decreasing k from 32 to 16 led to a decrease in all accuracies (1 to 3%).
Increasing the width k of the GCN layer from 32 to 64,128,256,512 or 1024 led to a constant increase in accuracy scores:

CiteSeer:

k | Avg max train acc | Avg max validation acc
--- | --- | ---
32 | 0.909 | 0.7416
64 | 0.9263 | 0.7462
128 | 0.9344 | 0.746
256 | 0.9353 | 0.7474
512 | 0.9367 | 0.7477
1024 | 0.9366 | 0.7478

Cora:

k | Avg max train acc | Avg max validation acc
--- | --- | ---
32 | 0.9115 | 0.8045
64 | 0.9251 | 0.8118
128 | 0.9369 | 0.8182
256 | 0.9474 | 0.8253
512 | 0.9544 | 0.8280
1024 | 0.9587 | 0.8308


### Graph classifier
For training the graph classifier, we performed 10 k-fold cross validation to split the dataset into training and validation datasets.
The samples are shuffled for each run.
We used the following training configurations and hyperparamters:

#### Configurations and results for ENZYMES:

Number of Epochs: 750

Learning rate: 0.001

Optimizer: Adam

Dropout: we introduced a dropout layer after every convolutional layer and the Sum Pooling layer with a dropout rate of 20%.

Results: Validation accuracy of roughly 63,8% on average over all folds.
          Training accuracy of roughly 77,1% on average over all folds.
          The exact acuracies of each fold can be found in [results_enzymes_with_shuffeling.txt](results/results_enzymes_with_shuffeling.txt). The other .txt files contain results from previous runs where we experienced with different hyperparamters and training configurations. We obtained the main improvements with shuffeling during stratified sampling, increasing the number of epochs and introducing dropout layers. The provided tensorboards clearly show that without shuffeling, only some of the runs achieve state of the art accuracies. Thus, the performance highly depends on the split of the dataset.

#### Configurations NCI1:

Number of Epochs: 200

Learning rate: 0.001

Optimizer: Adam

Dropout: we introduced a dropout layer after every convolutional layer and the Sum Pooling layer with a dropout rate of 20%.

Results: Validation accuracy of roughly 80.7% on average over all folds.
         Training accuracy of roughly 79.6% on average over all folds.
         The exact acuracies of each fold can be found in [results_nci1_with_shuffeling.txt](results/results_nci1_with_shuffeling.txt). Based on our gained insights with the enzymes dataset, we only performed cross validation with shuffeling. Since the network started overfitting after about 200 epochs, we stopped training here.
