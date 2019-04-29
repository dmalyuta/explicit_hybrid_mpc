"""
Global variables.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import os

N_PROC = 10 # How many parallel processes to run

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),os.pardir))
STATUS_FILE = PROJECT_DIR+'/status' # Overall program status (text file)
STATISTICS_FILE = PROJECT_DIR+'/data/statistics.pkl' # Overall statistics
ERR_FILE = PROJECT_DIR+'/data/err.pkl' # Absolute and relative error settings for optimization oracle
TREE_FILE = PROJECT_DIR+'/data/tree.pkl' # Overall tree
TOTAL_VOLUME_FILE = PROJECT_DIR+'/data/total_volume.pkl' # Volume of set in which to compute the partition
NODE_DIR = PROJECT_DIR+'/data/' # Directory of individual branches
MUTEX_FILE = PROJECT_DIR+'/mutex' # Mutex file for subprocess syncing