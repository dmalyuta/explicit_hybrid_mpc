"""
Master file.

Executes approximate mixed-integer convex multiparametric programming [1].
In other words, builds an epsilon-suboptimal partition which can be used to do
explicit MPC control.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

from mpi4py import MPI

import scheduler
import worker

def main():
    """
    Runs the scheduler for the 0-th MPI process, and the worker for all other
    processes.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank==0:
        scheduler.main()
    else:
        worker.main()

if __name__=='__main__':
    main()
