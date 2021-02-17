#!/usr/bin/env python3
"""
This module contains all hyperparameters for exercise sheet 1.
"""

GRAPHLET_VECTOR_SIZE = 34
PREPROC_STORE = False  # boolean whether feature vectors should be stored during pre-processing for faster development
KERNEL_TYPE = "graphlet"  # set kernel used, options: "graphlet", "wl", "random_walk"
CLOSED_WALK_VECTOR_SIZE = 2
REFINEMENT_ITER = 4
MAX_IT = 100000
