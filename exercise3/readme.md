# Information about Exercise 3

## How to use command line interface
Execute [create_embeddings.py](create_embeddings.py) in root dir to create embeddings of exercise 2.
Necessary arguments are datasets, output directory and the two parameters return and out. The arguments below are listed in order:

'dataset': should either be Cora, CiteSeer, Facebook or PPI

'output_dir': should specify a output directory

'param_return': refers to the variable p in the exercise sheet. It determines the probability of going back to a previous node during the random walk

'pram_in_out': refers to q in the exercise sheet. It determines the probability of visiting a new neighbor node of the current node during the random walk


Example Command:
```
group1\exercise3>python3 create_embeddings.py --dataset CiteSeer --output_dir "C:\Users\..."  --param_return 0.1 --param_in_out 1
```
The command will perform five random walks per node in the graph, and calculate the node2vec embedding according to task 1 and 2.

Execute [classify.py](classify.py) in root dir to classify and evaluate.

Depending on the first argument the command will either classify nodes or will act as a link predictor. The node classifier uses precomputed node embeddings, that need to be provided.
The neccessary arguments are in order:

'mode': should be either 'Node' for node classification or 'Link' for link prediction

'dataset': possible choices for a node classifier are CiteSeer and Cora, Link prediction offers either Facebook or PPI

'input_file': path to the embedding of the graph that should be used in classification, is only necessary for Node classification. For Link prediction an arbitrary path can be specified or argument can be ignored entirely.

Example command for node classification:
```
group1\exercise3>python3 classify.py --mode Node --dataset CiteSeer --input_file "C:\Users\...\group1\exercise3\results\embeddings\CiteSeer_p_1.0_q_0.1_embeddings.npy"

```

Example command for link prediction:

```
group1\exercise3>python3 classify.py --mode Link --dataset PPI

```

## Project Structure
Any function related to loading datasets and preprocessing the datasets can be found in [data.py](data.py).

We store our used embeddings for node classification in the [embeddings folder](results/embeddings). We did not include embeddings for the different batch sizes, but instead only for batch size 128 and optimizer Adam.

The function implementing random walks and negative sampling are in [random_walk.py](random_walk.py).

All functions related to the of the node2vec model are located in [node2vec.py](node2vec.py), while the related general model is implemented in [model.py](model.py).

The general functions related to the link predictor model are in [link_predictor.py](link_predictor.py).

Training of the node classifier and link predictor, as well as classification functions are handled in [classify.py](classify.py)

All functions related to loading, creating and storing embeddings are found in [create_embeddings.py](create_embeddings.py).

Note that depending on the dataset and the classifier, you might have to adjust the hyperparameters in the [config.py](config.py) to obtain the results we report below.

## Results

### Node Classifier

For the Node Classifier we tried lots of different parameters during the embedding process of node2vec. We experimented with different optimizers (Adam and RMSProp), additional values for p (influences probability of going back during a random walk) and q (influences probability of going forward to a new neighbor of the current node during random walking). We also tested different batch sizes and patience for the callback. The training of the embedding stops once the loss didn't improve for the specified number of epochs by the patience parameter. We only included all embeddings with the recommended batch size and Adam optimizer.

We obtained the following results:

First we fixed the batch size of 128 and optimizer Adam as per reccommendation and trained on different (p, q) values. The patience was fixed as 3.

Cora:

(p, q) | Avg. validation acc | standard deviation
--- | --- | ---
(1,1) | 75,3% | 1,48%
(1,0.1) | 72,4% | 1,30%
(0.1,1) | 80,5% | 1,85%
(0.5,0.5) | 74,2% | 2,39%
(0.3, 0.7) | 77,3% | 2,63%
(0.7,0.3) | 72,3%| 1,74%


CiteSeer:

(p, q) | Avg. validation acc | standard deviation
--- | --- | ---
(1,1) | 59,60% | 2,546%
(1,0.1) | 58,94% | 2,584%
(0.1,1) | 60,60% | 2,216%
(0.5,0.5) | 58,76% | 2,077%
(0.3, 0.7) | 60,44% | 2,507%
(0.7,0.3) | 58,99% | 3,082%
(0.01,1)  | 59,3% | 2,604%

We then looked at the best(p, q) combination for each dataset and looked at different batch sizes:

Cora, p = 0.1 and q = 1.0

Batch Size | Avg. validation acc | standard deviation
--- | --- | ---
32 | 82,64% | 2,079%
512 | 81,02% | 2,587%
2048 | 81,87% | 2,30%
4096 | 81,54% | 1,530%

CiteSeer, p = 0.1 and q = 1.0

Batch Size | Avg. validation acc | standard deviation
--- | --- | ---
32 | 60,90% | 2,785%
512 | 59,48% | 2,573%
2048 | 61,08% | 2,590%
4096 | 61,35% | 2,903%

We tried RMSProp with different batch sizes as well, but the results were not noteworthy. However, we noticed a slightly longer convergence time.


We noticed that the classifier slightly underperforms for CiteSeer. We did in fact look into this, but the node2vec embedding performs well for every other task. Thus we think this might be caused by bad random walks, or could be improved by tuning of p and q. As hyperparameter tuning was not required, we didn't investigate further than that.

### Link Predictor
For the link predictor, we chose to compute a 80:20 training/test split. Since the PPI dataset was not connected, we computed the splits on the connected components. The node embeddings were computed using p = q = 1. The edge embeddings were computed using the Hadamard product. There were no hyperparameters which could be tuned and we obtained the ROC AUC scores from the reference paper for, both, the Facebook and the PPI dataset.

Below, you can find the results of the link predictor for the Facebook graph as well as the PPI dataset.

#### Facebook:

- TRAIN: Accuracies after 5 experiment trials: [0.938, 0.938, 0.939, 0.94, 0.937]

- TRAIN: Mean accuracy: 0.938

- TRAIN: Standard deviation in accuracy: 0.0008

- TRAIN: ROC AUC scores after 5 experiment trials: [0.976, 0.976, 0.975, 0.976, 0.975]


- EVAL: Accuracies after 5 experiment trials: [0.939, 0.937, 0.938, 0.937, 0.939]

- EVAL: Mean accuracy: 0.938

- EVAL: Standard deviation in accuracy: 0.001

- EVAL: ROC AUC scores after 5 experiment trials: [0.977, 0.975, 0.975, 0.974, 0.975]


#### PPI:

- TRAIN: Accuracies after 5 experiment trials: [0.779, 0.786, 0.784, 0.791, 0.784]

- TRAIN: Mean accuracy: 0.785

- TRAIN: Standard deviation in accuracy: 0.004

- TRAIN: ROC AUC scores after 5 experiment trials: [0.829, 0.837, 0.834, 0.841, 0.828]


- EVAL: Accuracies after 5 experiment trials: [0.783, 0.783, 0.781, 0.79, 0.78]

 - EVAL: Mean accuracy: 0.783

- EVAL: Standard deviation in accuracy: 0.004

- EVAL: ROC AUC scores after 5 experiment trials: [0.833, 0.837, 0.827, 0.843, 0.826]
