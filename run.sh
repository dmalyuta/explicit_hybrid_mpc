#!/bin/bash
#
# Run the partitioning process
# NB: Make sure that you have set the right Python virtualenv

NUM_WORKERS=20 # Number of worker processes

mpiexec -n $((NUM_WORKERS+1)) ipython main.py
