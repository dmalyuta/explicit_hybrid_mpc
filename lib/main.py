"""
Executes the scheduler and workers for the appropriate MPI process ranks.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import mpi4py
mpi4py.rc.recv_mprobe = False # resolve UnpicklingError (https://tinyurl.com/mpi4py-unpickling-issue)
from mpi4py import MPI

import global_vars
import prepare
import scheduler
import worker

def main():
    """Runs the scheduler for one process, and the worker for all other processes."""
    prepare.set_global_variables()
    if MPI.COMM_WORLD.Get_rank()==global_vars.SCHEDULER_PROC:
        args = prepare.parse_args()
        scheduler.main(args['ecc_tree'])
    else:
        worker.main()

if __name__=='__main__':
    main()
