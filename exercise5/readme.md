# Information about Exercise 5 (THE CHALLENGE)


## Details on Execution

To test the saved models, execute the [main](main.py) function with the required parameters 'dataset' and 'data_path'.
The models have been trained using the dataset specific models, which are located in the [models subfolder](models) and
the modular [train script](train.py), which allows for different datasets and parameter sets to be trained simultaneously.


## Results

In the following subsections we present our results from using different classifiers and parameters. 
Since we did some initial testing to explore how different classifiers perform on each dataset, we narrowed our optimization for each dataset early and 
did not test all variations possible on each dataset. 
We **highlight** the best performing solutions. These are the classifiers we saved and use when executing [main](main.py). Note that we did not save the WL kernel 
as discussed in the seminar session but instead train it from scratch as this should only take about a minute.

The results for the kernel methods represent the averaged accuracies for the different runs in cross validation. 
The results for the GCN's represent the maximum accuracies obtained during several training/evaluation cycles.

### Features

Since the GCN layer requires an initial node embedding matrix and only the CLIQUE dataset provided node features to be used for that,
we came up with several different ways to initialize the node embeddings:

The most basic approach is to use an *EYE* marix as the node embedding matrix. 
The eye matrix encodes a one-hot assignment of the position each node represents in the graph. 

Another approach was to compute a random walk of length l starting a each node, normalize the walk and create a node feature of length l in that way.
However, this node feature never led to any learning for any dataset and therefore we do not present more detailed results.

The third type of feature vector is derived from *COLOR*ing the nodes, by applying the coloring steps from the Weisfeiler-Lehmann Kernel and 
use a one-hot encoding of the colors as the node features.

In addition to those features we also considered a *Neighborhood* feature on both a node and graph level.
For a node v computes the subgraph of the input graph G induced by the neighboring nodes of v (excluding v itself). I.e. we restrict the adjacency 
Matrix of G to the rows and columns of the neighbors of v. 
For this to be used on graphs we stacked all obtained features for each graph in the dataset. 

### Kernels
We tested two types of kernels: The Weisfeiler Lehman kernel and a degree kernel. 
In the following we will reference the *Degree Kernel* by DK.

The degree kernel is a linear kernel composed of the graphs' node degrees, i.e. each graph is represented as a vector of length |V| where
each entry corresponds to the node's degree respectively.

For the WL kernel we manipulated the number of refinement steps.
For both kernel types, we used an iteration cap of 100 iterations since validation accuracy converged and
to avoid extreme overfitting.


### CHORDAL1

WL:

| steps=2     |             | steps=4     |             | steps=8    |             | steps=10    |             |
|-------------|-------------|-------------|-------------|------------|-------------|-------------|-------------|
| Train acc.  | Val acc.    | Train acc.  | Val acc.    | Train acc. | Val acc.    |Train acc.   | Val acc.    |
|53.24% (0.04)|49.21% (0.06)|59.84% (0.05)|50.99% (0.05)|64.3% (0.04)|50.63% (0.05)|64.45% (0.04)|50.5% (0.06) |


DK:

| 100k iter   |             | 10000k iter |             |
|-------------|-------------|-------------|-------------|
| Train acc.  | Val acc.    | Train acc.  | Val acc.    |
|51.31% (0.02)|50.85% (0.06)|65.83% (0.01)|50.55% (0.06)|

GCN:

(nc) stands for not converging, which indicated as random performance on the validation set

| Structure                                  |   | Eye Features |           | Color Features |           | Neighborhood Feature (4)   |           |
|--------------------------------------------|---|--------------|-----------|----------------|-----------|----------------|-----------|
|                                            |   | Train acc.   | Val acc.  | Train acc.     | Val acc.  | Train acc.     | Val acc.  |
| GCN(32) GCN(16) SumPool(16) Dense(softmax) |   | 0.985        | 0.57 (nc) | **0.9987**         | 0.51 (nc) |                |           |
| GCN(32) SumPool(32) Dense(softmax)         |   | 0.9975       | 0.58      | 0.9962         | 0.53 (nc) |                |           |
| Skip_GCN(32) SumPool(32) Dense(softmax)    |   | 0.5          | 0.5       | 0.9950         | 0.5       |                |           |
| GCN(64) GCN(64) SumPool(64) Dense(softmax) |   | 0.9837       | 0.54      | 0.9975         | 0.47 (nc) |                |           |
| **GCN(16) SumPool(16) Dense(softmax + l1l2 regularizer)** |   | 0.9937       | **0.64**      |          |  |                |           |
| AvgPool(6) Dense(64) AvgPool(6) 2xDense(32) Dense(softmax) |   |       |      |         |              |     0.7937     |   0.59    |

### CHORDAL2



WL:

| steps=2     |             | steps=4     |             | steps=8     |             | steps=10    |             |
|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| Train acc.  | Val acc.    | Train acc.  | Val acc.    | Train acc.  | Val acc.    |Train acc.   | Val acc.    |
|51.76% (0.07)|50.6%  (0.08)|51.94% (0.08)|49.8%  (0.09)|53.98% (0.08)|51.77% (0.08)|53.89% (0.08)|51.78% (0.08)|


DK:

| 100k iter   |             | 10000k iter |             |
|-------------|-------------|-------------|-------------|
| Train acc.  | Val acc.    | Train acc.  | Val acc.    |
|64.65% (0.12)|64.6% (0.12) |91.09% (0.01)|90.68% (0.02)|

GCN:

Color features could not be used, since the coloring produces ~180.000 colors. 
Hot encoding those per node for each of the 5000 graphs would require ~350GB memory.

| Structure                                  |   | Eye Features |          |
|--------------------------------------------|---|--------------|----------|
|                                            |   | Train acc.   | Val acc. |
| **GCN(32) GCN(16) SumPool(16) Dense(softmax)** |   | **0.9805**       | **0.977**    |
| GCN(32) SumPool(32) Dense(softmax)         |   | 0.9765       | 0.976    |
| GCN(64) GCN(64) SumPool(64) Dense(softmax) |   | 0.9635       | 0.972    |


### CLIQUE

GCN:

| Structure                          |   | Node Features |          | Color Features |          | Node + Color Features |          |Neighborhood Feature(6)|          |
|------------------------------------|---|---------------|----------|----------------|----------|-----------------------|----------|-----------------------|----------|
|                                    |   | Train acc.    | Val acc. | Train acc.     | Val acc. | Train acc.            | Val acc. |    Train acc.         | Val acc. |
| GCN(softmax)                       |   | 0.7578        | 0.6744   | 0.964          | 0.6793   | 0.9633                | 0.7107   |                       |          |
| GCN(32) GCN(softmax)               |   | 0.9681        | 0.6115   | 0.9633         | 0.6867   | 0.9781                | 0.7314   |                       |          |
| GCN(64) GCN(softmax)               |   | 0.9737        | 0.6248   | 0.9759         | 0.69     | 0.9774                | 0.7337   |                       |          |
| Skip_GCN(64) GCN(softmax)          |   | 0.9556        | 0.6607   | 0.9726         | 0.727    | 0.9789                | 0.7656   |                       |          |
| Skip_GCN(64) Dropout GCN(softmax)  |   |               |          |                |          | 0.9826                | 0.7585   |                       |          |
| Skip_GCN(128) GCN(softmax)         |   |               |          |                |          | 0.9819                | 0.7656   |                       |          |
| Skip_GCN(128) Dropout GCN(softmax) |   |               |          |                |          | 0.9822                | 0.7622   |                       |          |
| **3 Layer Dense(6)+Softmax**           |   |               |          |                |          |                       |          |**0.9867**             | **0.99** |


### CONNECT

WL:

| steps=2     |             | steps=4     |             | steps=8     |             | **steps=10**    |             |
|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| Train acc.  | Val acc.    | Train acc.  | Val acc.    | Train acc.  | Val acc.    |Train acc.   | Val acc.    |
|59.49% (0.08)|57.52% (0.08)|92.57% (0.05)|88.03% (0.05)|98.27% (0.02)|93.26% (0.02)|**99.03%** (0.01)|**94.02%** (0.02)|


DK:

| 100k iter   |             | 10000k iter |             |
|-------------|-------------|-------------|-------------|
| Train acc.  | Val acc.    | Train acc.  | Val acc.    |
|44.31% (0.01)|30.25% (0.03)|44.29% (0.01)|30.26% (0.02)|

GCN:

| Structure                          |   | Eye Features |          |
|------------------------------------|---|--------------|----------|
|                                    |   | Train acc.   | Val acc.  |
| GCN(64) Dropout SumPool(64) Dense(64) Dropout Dense(Softmax)|   |0.35 |0.36 |
| GCN(64) GCN(64) Dropout SumPool(64) Dense(64) Dropout Dense(Softmax)|   |0.16 |0.16 |
| GCN(64) GCN(64) GCN(64) Dropout SumPool(64) Dense(64) Dropout Dense(Softmax)|   |0.25 |0.25 |
| GCN(64) GCN(64) SumPool(64) Dense(64) Dense(Softmax)|   |0.4 |0.38 |
| GCN(128) Dropout SumPool(64) Dense(64) Dropout Dense(Softmax)|   |0.18 |0.19 |
| Skip_GCN(64) Dropout SumPool(64) Dense(64) Dropout Dense(Softmax)|   |0.18 |0.19 |
| Skip_GCN(64) Dropout SumPool(64) Dense(64) Dropout Dense(Softmax)|   |0.18 |0.19 |



### TRIANGLES

WL:

| steps=2     |             | steps=4     |             | steps=8     |             | steps=10    |             |
|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| Train acc.  | Val acc.    | Train acc.  | Val acc.    | Train acc.  | Val acc.    |Train acc.   | Val acc.    |
|66.43% (0.04)|17.27% (0.01)|83.64% (0.01)|19.1% (0.02) |83.56% (0.01)|19.01% (0.02)|83.55% (0.01)|18.68% (0.01)|


DK:

| 100k iter   |             | 10000k iter |             |
|-------------|-------------|-------------|-------------|
| Train acc.  | Val acc.    | Train acc.  | Val acc.    |
|27.51% (0.00)|17.42% (0.01)|27.58% (0.00)|17.71% (0.02)|


GCN:

| Structure                          |   | Eye Features |          |Neighborhood Feature(9)|          |
|------------------------------------|---|--------------|----------|-----------------------|----------|
|                                    |   | Train acc.   | Val acc.  |    Train acc.         | Val acc. |
| GCN(64) Dropout SumPool(64) Dense(64) Dropout Dense(Softmax)|   |0.11 |0.11 | | |
| GCN(64) GCN(64) Dropout SumPool(64) Dense(64) Dropout Dense(Softmax)|   |0.09 |0.09 | | |
| GCN(64) GCN(64) GCN(64) Dropout SumPool(64) Dense(64) Dropout Dense(Softmax)|   |0.09 |0.09 | | |
| GCN(64) GCN(64) SumPool(64) Dense(64) Dense(Softmax)|   |0.35 |0.13 | | |
| GCN(128) Dropout SumPool(64) Dense(64) Dropout Dense(Softmax)|   |0.04 |0.04 | | |
| Skip_GCN(64) Dropout SumPool(64) Dense(64) Dropout Dense(Softmax)|   |0.04 |0.04 | | |
| Skip_GCN(64) Dropout SumPool(64) Dense(64) Dropout Dense(Softmax)|   |0.04 |0.04 | | |
| **AveragePooling + Dense (x3) Dropout Dense(32) Dense(Softmax)**|   |               |          |**0.4967**             | **0.4885**|


