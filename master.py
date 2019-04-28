"""
Master file which spawns parallel processes to build the partition.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import os
import glob
import time
import pickle
import fcntl
import subprocess as sp

import global_vars
from examples import example
from tools import get_nodes_in_queue

def setup():
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
    for prefix in ['node','tree']:
        for file in glob.glob(global_vars.NODE_DIR+'%s_*.pkl'%(prefix)):
            os.remove(file)
    save_leaves_into_queue(partition)
    
def build_tree(cursor,location=''):
    if cursor.is_leaf():
        try:
            subtree_filename = global_vars.NODE_DIR+'tree_%s.pkl'%(location)
            subtree_file = open(subtree_filename,'rb')
            subtree = pickle.load(subtree_file)
            os.remove(subtree_filename) # cleanup
            if subtree.is_leaf():
                cursor.data = subtree.data
            else:
                cursor.left = subtree.left
                cursor.right = subtree.right
        except FileNotFoundError:
            pass # This is a leaf in the final tree
    if not cursor.is_leaf():
        build_tree(cursor.left,location+'0')
        build_tree(cursor.right,location+'1')

if __name__=='__main__':
    setup()
    # Start slave processes
    proc = [sp.Popen(['python','slave.py',str(i)]) for i in range(global_vars.N_PROC)]
    # Wait for termination criterion
    check_period = 5 # [s] How frequently to check termination criterion
    finished = False
    mutex = open(global_vars.MUTEX_FILE,'w')
    all_idle = ''.join('0' for _ in range(global_vars.N_PROC))
    while not finished:
        time.sleep(check_period)
        fcntl.lockf(mutex,fcntl.LOCK_EX)
        status = open(global_vars.STATUS_FILE,'r').readline()
        if status==all_idle and len(get_nodes_in_queue())==0:
            # No slave processes are working and no nodes in queue, so no
            # possibility of further nodes appearning in queue ==> terminate!
            for i in range(global_vars.N_PROC):
                proc[i].kill()
                finished = True
        fcntl.lockf(mutex,fcntl.LOCK_UN)
    # Build the full tree
    tree = pickle.load(open(global_vars.TREE_FILE,'rb'))
    build_tree(tree)
    pickle.dump(tree,open(global_vars.TREE_FILE,'wb'))
