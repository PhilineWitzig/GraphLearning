"""
This file contains all configurations and hyperparameters for training.
"""
import logging
import os

# General settings
LOG_LEVEL = logging.DEBUG

if not os.path.exists(os.path.join(os.getcwd(), "logs")):
    os.makedirs(os.path.join(os.getcwd(), "logs"))
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/node2vec.log"),
        logging.StreamHandler()
    ]
)

# Settings for node2Vec
EMBED_DIM = 128
EMBED_LR = 1e-3
EMBED_BATCH_SIZE = 128
EMBED_EPOCH_MAX = 1000  # set arbitratily high, as we use early stopping

# Parameters for random walks
PARAM_RETURN = 4    # p
PARAM_IN_OUT = 1    # q
