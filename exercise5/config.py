"""
Configutations for assignment 5.
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
        logging.FileHandler("logs/experiment.log"),
        logging.StreamHandler()
    ]
)

# Disable Tf Warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# WL kernel params
MAX_ITER = 100
REFINEMENT_STEPS = 10


# GCN params depending on the dataset
CLIQUE_LR = 0.001
CLIQUE_EPOCHS = 90
CLIQUE_BATCHSIZE = 2700

CHORDAL1_LR = 0.01
CHORDAL1_EPOCHS = 100
CHORDAL1_BATCHSIZE = 32

CHORDAL2_LR = 0.01
CHORDAL2_EPOCHS = 100
CHORDAL2_BATCHSIZE = 16

CONNECT_LR = 0.01
CONNECT_EPOCHS = 50
CONNECT_BATCHSIZE = 16

TRIANGLES_LR = 0.001
TRIANGLES_EPOCHS = 130
TRIANGLES_BATCHSIZE = 5200
