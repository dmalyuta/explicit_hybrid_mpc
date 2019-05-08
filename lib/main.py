"""
Executes the scheduler and workers for the appropriate MPI process ranks.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import tools
import global_vars
import prepare
import scheduler
import worker

def main():
    """Runs the scheduler for one process, and the worker for all other processes."""
    prepare.set_global_variables()
    if tools.MPI.rank()==global_vars.SCHEDULER_PROC:
        scheduler.main()
    else:
        worker.main()

if __name__=='__main__':
    main()
