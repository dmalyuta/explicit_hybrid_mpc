"""
Subtree partitioning slave process.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import sys
sys.path.append('./lib/')

import pickle

import global_vars
from examples import example
from partition import spinner,ecc

# Get process number
if len(sys.argv)<2:
    raise RuntimeError('No process number argument passed in')
proc_num = int(sys.argv[1])

# Load the optimization problem oracle
err = pickle.load(open(global_vars.ERR_FILE,'rb'))
oracle = example(abs_err=err['abs_err'],rel_err=err['rel_err'])[1]

# Start the slave loop
alg_call = lambda branch,location: ecc(oracle,branch,location)
spinner(proc_num,alg_call,wait_time=0.)