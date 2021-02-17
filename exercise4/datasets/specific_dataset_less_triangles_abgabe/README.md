This file contains the new dataset for exercise 4.

It consists of 5200 graphs with 26 different graph lables, ranging from 0 to 25. Any graph has between 5 and 70 nodes.
The graph labels indicate the number of triangles (3-cliques) in the graph, and there are no node attributes or labels.

The graphs are produced by random generation, and then tested for the number of triangles they contain
using the predefined networkX function to find cliques.
The generation picks the number of nodes randomly between 5 and 70 and will connect any two nodes with a probability of 20%.
This is to keep the graphs smaller due to the NP-hard calculation we need to determine the number of 3-cliques.
We use brute force, as we "reroll" our random graph until it has the right amount of triangles in it. 
Sadly we do not check for isomorphic graphs in each class, as generation already does take really long. 



