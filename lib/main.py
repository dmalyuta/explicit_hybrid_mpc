"""
Executes the scheduler and workers for the appropriate MPI process ranks.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import global_vars
import tools
import scheduler
import worker

def main():
    """Runs the scheduler for one process, and the worker for all other processes."""
    tools.set_global_variables(require_timestamp=True)
    if global_vars.COMM.Get_rank()==global_vars.SCHEDULER_PROC:
        args = tools.parse_args()
        scheduler.main(args['ecc_tree'])
    else:
        worker.main()

if __name__=='__main__':
    main()
