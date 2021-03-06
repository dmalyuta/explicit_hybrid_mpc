"""
Global variables.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import os

# MPC problem parameters (**set via command line arguments**)
EXAMPLE = None # Which example to run. Options: {'cwh_z','cwh_xy','cwh_xyz','pendulum'}
MPC_HORIZON = None # MPC prediction horizon length
ABS_FRAC = None # Fraction of full set for computing partition absolute error tolerance
REL_ERR = None # Partition relative error tolerance
RUNTIME_DIR = None # Runtime directory (contains files to run the code and the data generated)
DATA_DIR = None # Directory for files generated at runtime
STATUS_FILE = None # Overall program status (text file)
STATISTICS_FILE = None # Overall statistics
TREE_FILE = None # Overall tree
ETA_RLS_FILE = None # Overall tree
IDLE_COUNT_FILE = None # Idle process count

SOLVER_OPTIONS = dict(solver='MOSEK', verbose=False) # Optimization solver options
VERBOSE = 2 # 0: no printing; 1: print error msgs; 2: print error+info msgs
STATUS_TAG = 11 # MPI Isend tag for worker status
NEW_WORK_TAG = 22 # MPI Isend tag for new work for a worker process
NEW_BRANCH_TAG = 33 # MPI Isend tag for new branch root to put into queue
FINISHED_BRANCH_TAG = 44 # MPI Isend tag for finished branch
SCHEDULER_PROC = 0 # MPI rank of scheduler process
ERROR = '>>> ' # Error message prefix
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(
    os.path.realpath(__file__)),os.pardir)) # Project root directory
MAX_RECURSION_LIMIT = 50 # Limit on how many times worker's ecc or lcss can
                         # recurse before submitting both left and right children to task queue

LOOPRATE_PROBE_WINDOW = 1. # [s] Duration of window for scheduler main loop frequency measurement
STATUS_WRITE_FREQUENCY = 1. # [Hz] Frequency for updating STATUS_FILE
STATISTICS_WRITE_FREQUENCY = 0.2 # [Hz] Frequency for updating STATISTICS_FILE
