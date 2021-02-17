This file contains the PROTEINS dataset to be used for evaluation in Task 3.
ENZYMES and NCI1 can be reused from sheet-1.

Additionally, we provide the python script `dataset_exporter.py`, 
which will help you with exporting NetworkX graphs to our dataset format in Task 4.
For example, the following code will export a dataset of 100 random graphs:

```
from dataset_exporter import export_dataset
import networkx as nx

graphs = [nx.gnm_random_graph(10, 20) for _ in range(100)]
export_dataset('datasets/Random_Graphs', graphs)
```