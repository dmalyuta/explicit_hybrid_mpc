"""
Master file which spawns parallel processes to build the partition.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import sys
sys.path.append('./lib/')

import os
import glob
import time
import pickle
import subprocess as sp

import global_vars
from examples import example
from tools import get_nodes_in_queue,simplex_volume,Mutex

def setup():
    # Mutex
    open(global_vars.MUTEX_FILE,'a').close()

    # Process status
    # At index i:
    #   0 - process is idle
    #   1 - process is running
    status_file = open(global_vars.STATUS_FILE,'w')
    status_file.write('(nothing written yet)')
    status_file.close()
    # remove individual process status files
    for file in glob.glob(global_vars.PROJECT_DIR+'/data/status_proc_*.pkl'):
        os.remove(file)

    # Main tree and initial node to be explored
    partition,oracle = example(abs_frac=0.25,rel_err=1.0)
    with open(global_vars.ERR_FILE,'wb') as f:
        pickle.dump(dict(abs_err=oracle.eps_a,rel_err=oracle.eps_r),f)
    with open(global_vars.TREE_FILE,'wb') as f:
        pickle.dump(partition,f)
    with open(global_vars.TOTAL_VOLUME_FILE,'wb') as f:
        def total_volume(cursor):
            """Compute total volume of invariant set"""
            if cursor.is_leaf():
                return simplex_volume(cursor.data.vertices)
            else:
                return total_volume(cursor.left)+total_volume(cursor.right)
        pickle.dump(total_volume(partition),f)
    for prefix in ['node','tree']:
        for file in glob.glob(global_vars.NODE_DIR+'%s_*.pkl'%(prefix)):
            os.remove(file)
    save_leaves_into_queue(partition)

def save_leaves_into_queue(cursor,location=''):
    """
    Save tree leaves into queue.
    
    Parameters
    ----------
    cursor : Tree
        Root node handle.
    location : str, optional
        Cursor location relative to root. **Don't pass this in**.
    """
    if cursor.is_leaf():
        with open(global_vars.NODE_DIR+'node_%s.pkl'%(location),'wb') as f:
            pickle.dump(cursor,f)
    else:
        save_leaves_into_queue(cursor.left,location+'0')
        save_leaves_into_queue(cursor.right,location+'1')
            
def build_tree(cursor,location=''):
    """
    Build the tree from subtree files.
    **Caution**: modifies ``cursor`` (passed by reference).
    
    Parameters
    ----------
    cursor : Tree
        Root node handle.
    location : str, optional
        Cursor location relative to root. **Don't pass this in**.
    """
    if cursor.is_leaf():
        try:
            subtree_filename = global_vars.NODE_DIR+'tree_%s.pkl'%(location)
            with open(subtree_filename,'rb') as subtree_file:
                subtree = pickle.load(subtree_file)
            os.remove(subtree_filename) # cleanup
            cursor.data = subtree.data
            cursor.top = subtree.top
            if not subtree.is_leaf():
                cursor.left = subtree.left
                cursor.right = subtree.right
        except FileNotFoundError:
            pass # This is a leaf in the final tree
    if not cursor.is_leaf():
        build_tree(cursor.left,location+'0')
        build_tree(cursor.right,location+'1')
        
def save_tree():
    """
    Builds and saves the tree from subtree files.
    
    Returns
    -------
    tree : Tree
        Root of the built tree.
    """
    with open(global_vars.TREE_FILE,'rb') as f:
        tree = pickle.load(f)
    build_tree(tree)
    with open(global_vars.TREE_FILE,'wb') as f:
        pickle.dump(tree,f)
    return tree
        
def start_processes(which_alg):
    """
    Start the processes.
    
    Parameters
    ----------
    which_alg : {'ecc','lcss'}
        Which algorithm to execute.

    Returns
    -------
    proc : list
        List of processes.
    """
    proc = [sp.Popen(['python','slave.py',str(i),str(0 if which_alg=='ecc' else 1)])
            for i in range(global_vars.N_PROC)]
    return proc

class MainStatusWriter:
    def __init__(self):
        self.overall_status_list = []
        self.proc_status_list = []
        self.time_start = None
        
    def update(self,proc_status):
        """
        Write the main status file.
        
        Parameters
        ----------
        proc_status : list
            List of dicts from individual processes status files.
        overall_status_list : list
            List of dicts of overall status.
        """
        self.proc_status_list.append(proc_status)
        with open(global_vars.STATUS_FILE,'w') as status_file:
            # Overall status
            with open(global_vars.TOTAL_VOLUME_FILE,'rb') as f:
                total_volume = pickle.load(f)
            num_proc_active = sum([proc_status[i]['status']=='active' for i in range(global_vars.N_PROC)])
            num_proc_failed = sum([proc_status[i]['status']=='failed' for i in range(global_vars.N_PROC)])
            volume_filled_total = sum([proc_status[i]['volume_filled_total'] for i in range(global_vars.N_PROC)])
            volume_filled_frac = volume_filled_total/total_volume
            simplex_count_total = sum([proc_status[i]['simplex_count_total'] for i in range(global_vars.N_PROC)])
            time_active_total = sum([proc_status[i]['time_active_total'] for i in range(global_vars.N_PROC)])
            if self.time_start is None:
                self.time_start = time.time()
            time_elapsed = time.time()-self.time_start
            overall_status = dict(num_proc_active=num_proc_active,
                                  num_proc_failed=num_proc_failed,
                                  volume_filled_total=volume_filled_total,
                                  volume_filled_frac=volume_filled_frac,
                                  simplex_count_total=simplex_count_total,
                                  time_elapsed=time_elapsed,
                                  time_active_total=time_active_total)
            self.overall_status_list.append(overall_status)
            status_file.write('\n'.join([
                    '# overall',
                    'number of processes active: %d'%(num_proc_active),
                    'number of processes failed: %d'%(num_proc_failed),
                    'volume filled (total [%%]): %.4e'%(volume_filled_frac*100.),
                    'simplex_count: %d'%(simplex_count_total),
                    'time elapsed [s]: %d'%(time_elapsed),
                    'time active (total for all processes [s]): %.0f'%(time_active_total)
                    ])+'\n\n')
            # Individual processes' status
            for i in range(global_vars.N_PROC):
                data = proc_status[i]
                status_file.write('\n'.join([
                        '# proc %d'%(i),
                        'status: %s'%(data['status']),
                        'current branch: %s'%(data['current_branch']),
                        'volume filled (total [-]): %.4e'%(data['volume_filled_total']),
                        'volume filled (current [%%]): %.4e'%(data['volume_filled_current']*100.),
                        'simplex count (total [-]): %d'%(data['simplex_count_total']),
                        'simplex count (current [-]): %d'%(data['simplex_count_current']),
                        'time active (total [s]): %.0f'%(data['time_active_total']),
                        'time active (current [s]): %.0f'%(data['time_active_current']),
                        'time idle (total [s]): %.0f'%(data['time_idle'])
                        ])+'\n\n')
    
    def save_statistics(self):
        """
        Save the main statistics file.
        """
        with open(global_vars.STATISTICS_FILE,'wb') as f:
            pickle.dump(dict(overall=self.overall_status_list,
                             individual_process=self.proc_status_list),f)

def await_termination(proc,status_writer,check_period=5.):
    """
    Waits for partitioning process to end.
    
    Parameters
    ----------
    proc : list
        List of processes.
    check_period : float, optional
        How frequently (in seconds) to check termination criterion.
    """
    mutex = Mutex(-1)
    finished = False
    while not finished:
        time.sleep(check_period)
        # Read all process status files
        try:
            mutex.lock()
            nodes_in_queue = get_nodes_in_queue()
            proc_status = [None for i in range(global_vars.N_PROC)]
            for i in range(global_vars.N_PROC):
                with open(global_vars.PROJECT_DIR+'/data/status_proc_%d.pkl'%(i),'rb') as f:
                    proc_status[i] = pickle.load(f)
            mutex.unlock()
        except (FileNotFoundError,EOFError):
            # Too early - processes did not yet all create their status files
            mutex.unlock()
            continue
        # Update master status file (summary of all processes statuses)
        status_writer.update(proc_status)
        # Check terminator criterion
        all_idle = all([proc_status[i]['status']=='idle' for i in range(global_vars.N_PROC)])
        if all_idle and len(nodes_in_queue)==0:
            # No slave processes are working and no nodes in queue, so no
            # possibility of further nodes appearning in queue ==> terminate!
            for i in range(global_vars.N_PROC):
                proc[i].kill()
                finished = True

if __name__=='__main__':
    setup()
    status_writer = MainStatusWriter()
    for alg in ['ecc','lcss']:
        proc = start_processes(alg)
        await_termination(proc,status_writer,check_period=5.)
        tree = save_tree()
        if alg=='ecc':
            save_leaves_into_queue(tree)
    status_writer.save_statistics()
