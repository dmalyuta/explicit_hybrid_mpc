#!/bin/bash
#
# Run the partitioning process
# NB: Make sure that you have set the right Python virtualenv

NUM_WORKERS=419 # Number of worker processes

mpirun -n $((NUM_WORKERS+1)) ipython main.py
