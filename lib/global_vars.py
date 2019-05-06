"""
Global variables.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import os
import cvxpy as cvx

# MPC problem parameters (**set via command line arguments**)
EXAMPLE = None # Which example to run. Options: {'cwh_z','cwh_xy','cwh_xyz'}
MPC_HORIZON = None # MPC prediction horizon length
ABS_FRAC = None # Fraction of full set for computing partition absolute error tolerance
REL_ERR = None # Partition relative error tolerance
RUNTIME_DIR = None # Runtime directory (contains files to run the code and the data generated)
DATA_DIR = None # Directory for files generated at runtime
STATUS_FILE = None # Overall program status (text file)
STATISTICS_FILE = None # Overall statistics
TREE_FILE = None # Overall tree
ETA_RLS_FILE = None # Overall tree
BRANCHES_FILE = None # Tree branches, used for tree building
IDLE_COUNT_FILE = None # Idle process count

SOLVER_OPTIONS = dict(solver=cvx.MOSEK, verbose=False) # Optimization solver options
VERBOSE = False # Whether to print debug info to terminal
#SOLVER_OPTIONS = dict(solver=cvx.GUROBI, verbose=False, Threads=1)
#SOLVER_OPTIONS = dict(solver=cvx.ECOS_BB, verbose=False)
STATUS_TAG = 11 # MPI Isend tag for worker status
NEW_WORK_TAG = 22 # MPI Isend tag for new work for a worker process
NEW_BRANCH_TAG = 33 # MPI Isend tag for new branch root to put into queue
FINISHED_BRANCH_TAG = 44 # MPI Isend tag for finished branch
SCHEDULER_PROC = 0 # MPI rank of scheduler process
ERROR = '>>> ' # Error message prefix
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),os.pardir)) # Project root directory