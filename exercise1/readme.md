# How to use command line interface
Execute [cmd_interface.py](svm_exec.py) in root dir.
Necessary positional commands are kernel and datasets. An optional command is '-max_it', 
which specifies the number of maximal iterations per svm train. It defaults to 100000 and can be disabled, 
when assigning -1.

'kernel' should be one out of 'graphlet', 'wl' and 'random_walk' and specifies which kernel function is to be used in the svm training.
'datasets' should be a comma separated list of datasets to be trained and evaluated by the svm.

Example Command:
```
group1\exercise1>cmd_interface.py wl dd,enzymes,nci1 -max_it=100000
```

Since 10 times 10-fold cross-validation was a requirement, this is automatically performed when calling the command and the results are printed to the console.

# Implementation Structure of Exercise 1
In this document we explain the structure of our sourcecode and where to find the corresponding files for each exercise.

### Exercise 1
The [graphlet python file](graphlet.py) implements the conversion of a graph to a histogram vector.
The computation of the inner product of these histogram vectors is handled in the [execution file](svm_exec.py).

### Exercise 2
The module [wl_kernel.py](wl_kernel.py) contains the code for the Weisfeiler Lehman (WL) kernel and implements the color refinement algorithm.
All hyperparameters required for the WL kernel are contained in the [configuration file](config.py).
4 iterations of color refinement are executed and a weighted sum of the resulting histograms is computed (as proposed in the introduction video). 
We assume uniform weights to be set to 1.0 since no further details on setting the weights were given in the exercise.
The implementation of the color refinement algorithm is oriented on the paper "Color Refinement and its Applications".

### Exercise 3
Code for the closed walk kernel is available as a [python_file](closed_walk.py).
Since the numbers blow up quite significantly, the length of closed walks is capped to 3 at default, but can be changed by using the parameter CLOSED_WALK_VECTOR_SIZE.
The number of iterations for SVM training had to be capped to avoid long runtimes. Eventhough the data has been scaled, the process of fitting still takes too much time
to converge. Especially since the accuracy will not increase as drastically as the runtime does.


### Exercise 4
The execution of the SVM is available as a [python file](svm_exec.py) and can be achieved by using the command line interface as described above.
The results of the 10 times 10-fold evaluation of each dataset are located in the [results folder](results). We provide .tex document which contains 
tables for the results obtained when using a max number of iterations of 20k for performance reasons. All accuracy results from the 10 times 10-fold
evaluation are located in a [.txt document](results/results_200k.txt). It is worth mentioning that higher accuracies were achievable when the solver 
continued until convergence.

### Exercise 5
Will be send as a separate file.
