"""
Global variables.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import os

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),os.pardir))

N_PROC = int(open(PROJECT_DIR+'/n_proc.txt','r').readline()) # How many parallel processes to run

STATUS_FILE = PROJECT_DIR+'/status.txt' # Status file name
ERR_FILE = PROJECT_DIR+'/data/err.pkl'
TREE_FILE = PROJECT_DIR+'/data/tree.pkl'
NODE_DIR = PROJECT_DIR+'/data/'
MUTEX_FILE = PROJECT_DIR+'/mutex'