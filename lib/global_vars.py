"""
Global variables.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import os
import cvxpy as cvx

N_PROC = 10 # How many parallel processes to run

SOLVER_OPTIONS = dict(solver=cvx.MOSEK, verbose=False) # Optimization solver options
#SOLVER_OPTIONS = dict(solver=cvx.GUROBI, verbose=False, Threads=1)
#SOLVER_OPTIONS = dict(solver=cvx.ECOS_BB, verbose=False)

EXAMPLE = 'cwh_xy' # Which example to run. Options: {'cwh_z','cwh_xy','pendulum'}

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),os.pardir))
STATUS_FILE = PROJECT_DIR+'/status.txt' # Overall program status (text file)
STATISTICS_FILE = PROJECT_DIR+'/data/statistics.pkl' # Overall statistics
ERR_FILE = PROJECT_DIR+'/data/err.pkl' # Absolute and relative error settings for optimization oracle
TREE_FILE = PROJECT_DIR+'/data/tree.pkl' # Overall tree
TOTAL_VOLUME_FILE = PROJECT_DIR+'/data/total_volume.pkl' # Volume of set in which to compute the partition
NODE_DIR = PROJECT_DIR+'/data/' # Directory of individual branches
MUTEX_FILE = PROJECT_DIR+'/mutex' # Mutex file for subprocess syncing