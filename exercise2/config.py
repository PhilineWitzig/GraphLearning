"""
This module contains all hyperparameters required for training and general
configurations.
"""


# ----------------------------------------
# Hyperparameters for the node classifier
# ----------------------------------------
NUM_EPOCHS_NODE = 150
LR_NODE = 0.001

# -----------------------------------------
# Hyperparameters for the graph classifier
# -----------------------------------------

## For Dataset Enzymes ##
LR_GRAPH = 0.001
BATCH_SIZE = 16
NUM_EPOCHS_GRAPH = 750
