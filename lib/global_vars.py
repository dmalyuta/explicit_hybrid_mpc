"""
Global variables.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import os
import cvxpy as cvx
from mpi4py import MPI

EXAMPLE = 'cwh_z' # Which example to run. Options: {'cwh_z','cwh_xy','pendulum'}
ABS_FRAC = 0.25 # Fraction of full set for computing partition absolute error tolerance
REL_ERR = 1.0 # Partition relative error tolerance
SOLVER_OPTIONS = dict(solver=cvx.MOSEK, verbose=False) # Optimization solver options
VERBOSE = True # Whether to print debug info to terminal
#SOLVER_OPTIONS = dict(solver=cvx.GUROBI, verbose=False, Threads=1)
#SOLVER_OPTIONS = dict(solver=cvx.ECOS_BB, verbose=False)
COMM = MPI.COMM_WORLD # MPI communicator among all processes
N_PROC = COMM.Get_size() # Total number of processes
STATUS_TAG = 11 # MPI Isend tag for worker status
NEW_WORK_TAG = 22 # MPI Isend tag for new work for a worker process
NEW_BRANCH_TAG = 33 # MPI Isend tag for new branch root to put into queue
FINISHED_BRANCH_TAG = 44 # MPI Isend tag for finished branch
IDLE_WORKER_COUNT_TAG = 55 # MPI Isend tag for number of idle workers
SCHEDULER_PROC = 0 # MPI rank of scheduler process
WORKER_PROCS = [i for i in range(N_PROC) if i!=SCHEDULER_PROC] # MPI ranks of worker processes
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),os.pardir))
STATUS_FILE = PROJECT_DIR+'/status.txt' # Overall program status (text file)
STATISTICS_FILE = PROJECT_DIR+'/data/statistics.pkl' # Overall statistics
TREE_FILE = PROJECT_DIR+'/data/tree.pkl' # Overall tree
BRANCHES_FILE = PROJECT_DIR+'/data/branches.pkl' # Tree branches, used for tree building
