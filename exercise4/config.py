"""
This file contains project wide parameters.
"""

import logging
import os
import tensorflow as tf

# General settings
LOG_LEVEL = logging.INFO

if not os.path.exists(os.path.join(os.getcwd(), "logs")):
    os.makedirs(os.path.join(os.getcwd(), "logs"))

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/experiment.log", 'w+'),
        logging.StreamHandler()
    ]
)

# Disable Tf Warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# Training configurations for optimal architecture
EPOCHS = 500
BATCH_SIZE = 24
LR = 0.001
DATASET = 'ENZYMES'


# Layer evaluation hyperparameters
LE_EPOCHS = 100
LE_BATCH_SIZE = 16
LE_LR = 0.001
LE_DATASET = 'NCI1'

# Lists of different parameters to be applied to evaluation
LE_FEATURE_NUMBERS_LIST = [[64, 64], [128, 64]]
LE_ADV_POOLING_K = [2, 10, 50]
LE_DEPTH = [1, 2]


# Hyperparameters for Pooling
no_clusters = 20    # this is k in Dense diff. Pooling
top_k = 30          # this is l in Top-k Pooling
