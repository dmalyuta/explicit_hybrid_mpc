"""
Master file which spawns parallel processes to build the partition.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import os
import glob
import pickle

from examples import example
import global_vars

# Mutex
open(global_vars.MUTEX_FILE,'a').close()

# Process status
# At index i:
#   0 - process is idle
#   1 - process is running
status_file = open(global_vars.STATUS_FILE,'w')
status_file.write(''.join('0' for _ in range(global_vars.N_PROC)))
status_file.close()

# Main tree and initial node to be explored
def save_leaves_into_queue(node,location=''):
    if node.is_leaf():
        pickle.dump(node,open(global_vars.NODE_DIR+'node_%s.pkl'%(location),'wb'))
    else:
        save_leaves_into_queue(node.left,location+'0')
        save_leaves_into_queue(node.right,location+'1')

partition,oracle = example(abs_frac=0.5,rel_err=2.0)
pickle.dump(dict(abs_err=oracle.eps_a,rel_err=oracle.eps_r),open(global_vars.ERR_FILE,'wb'))
pickle.dump(partition,open(global_vars.TREE_FILE,'wb'))
for file in glob.glob(global_vars.NODE_DIR+'node_*.pkl'):
    os.remove(file)
save_leaves_into_queue(partition)
