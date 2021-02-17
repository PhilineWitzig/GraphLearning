This directory contains graph datasets and utility functions for the first exercise.

The subdirectory "datasets" contains the three main datasets you will focus on for now: DD [1], NCI1 [2] and ENZYMES [3].
These sets are standard benchmarks for graph classification.
The python script `dataset_parser` contains a class that will help you load the datasets.
For example, the following code will load NCI1:

```
from dataset_parser import Parser
parser = Parser('datasets/NCI1')
graphs = parser.parse_all_graphs()
```

The graphs are returned as a list of NetworkX graph objects (https://networkx.github.io/).
The python script `data_utils.py` contains some useful functions for working with the graphs.
For example, you may use the following commands to extract the node labels of a the first graph:

```
import data_utils
G = graphs[0]
node_labels = data_utils.get_node_labels(G) 
```

Note that the graphs of the Enzymes dataset do not just have discrete node labels but also additional real-numbered vectors as node attributes. 
These are not relevant for the first exercise.

References:

[1] P. D. Dobson and A. J. Doig. "Distinguishing enzyme structures from non-enzymes without alignments."

[2] K. M. Borgwardt, C. S. Ong, S. Schoenauer, S. V. N. Vishwanathan, A. J. Smola, and H. P. Kriegel. "Protein function prediction via graph kernels"

[3] N. Wale and G. Karypis. "Comparison of descriptor spaces for chemical compound retrieval and classification"
