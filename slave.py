"""
Subtree partitioning slave process.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import sys
sys.path.append('./lib/')

import time
import pickle

import global_vars
from examples import example
from partition import spinner,ecc,lcss
from tools import simplex_volume,Mutex

class StatusPublisher:
    def __init__(self,proc_num):
        """
        Parameters
        ----------
        proc_num : init
            Process number.
        """
        self.status_filename = global_vars.PROJECT_DIR+'/status_proc_%d'%(proc_num)
        try:
            with open(self.status_filename,'rb') as f:
                self.data = pickle.load(f)
            self.data['volume_filled_total'] = 0. # Reset volume filled
        except FileNotFoundError:
            self.data = dict(status='idle', # {'active','idle'}
                             current_branch='', # Location in tree of current subtree's root
                             volume_filled_total=0., # [-] Absolute units
                             volume_filled_current=0., # [%] of current root simplex's volume
                             simplex_count_total=0, # [-] How many simplices generated overall
                             simplex_count_current=0, # [-] How many simplices generated for current root simplex partition
                             time_active_total=0., # [s] Total time spent in 'active' status
                             time_active_current=0., # [s] Total time spent doing partitioning for the current simplex
                             time_idle=0. # [s] Total time spend in 'idle' status
                             )
        self.time_previous = time.time() # [s] Timestamp when time was last updated
        self.volume_current = None # [-] Volume of current root simplex
        self.__write() # Initial write
        
    def __write(self):
        """
        Write the data to file.
        """
        with open(self.status_filename,'wb') as f:
            pickle.dump(self.data,f)
    
    def set_new_root_simplex(self,R,location):
        """
        Set the new root simplex. Resets the ``*_current`` data fields.
        
        Parameters
        ----------
        R : np.array
            Matrix whose rows are the simplex vertices.
        location : string
            Location of this simplex in overall tree ('0' and '1' format where
            '0' means take left, '1' means take right starting from root node).
        """
        self.volume_current = simplex_volume(R)
        self.data['current_branch'] = location
        self.data['volume_filled_current'] = 0.
        self.data['simplex_count_current'] = 1
        self.data['time_active_current'] = 0.
        self.__write()
    
    def update(self,active=None,volume_filled_increment=None,simplex_count_increment=None):
        """
        Update the data.
        
        Parameters
        ----------
        active : bool, optional
            ``True`` if the process is in the 'active' state.
        volume_filled_increment : float, optional
            Volume of closed lead.
        simplex_count_increment : int, optional
            How many additional simplices added to partition.
        """
        # Update time counters
        time_now = time.time()
        dt = time_now-self.time_previous
        self.time_previous = time_now
        if self.data['status']=='active':
            self.data['time_active_total'] += dt
            self.data['time_active_current'] += dt
        else:
            self.data['time_idle'] += dt
        # Update status
        if active is not None:
            self.data['status'] = 'active' if active else 'idle'
        # Update volume counters
        if volume_filled_increment is not None:
            self.data['volume_filled_total'] += volume_filled_increment
            self.data['volume_filled_current'] += volume_filled_increment/self.volume_current
        # Update simplex counters
        if simplex_count_increment is not None:
            self.data['simplex_count_total'] += simplex_count_increment
            self.data['simplex_count_current'] += simplex_count_increment
        self.__write()

def parse_args():
    """
    Parse command-line arguments.
    
    Returns
    -------
    proc_num : int
        Process number.
    which_alg : str
        Which algorithm to run.
    """
    if len(sys.argv)<3:
        raise RuntimeError('No process number argument passed in')
    proc_num = int(sys.argv[1])
    which_alg = 'ecc' if int(sys.argv[2])==0 else 'lcss'
    return  proc_num, which_alg

def start_process(proc_num,which_alg):
    """
    Start the process.
    
    Parameters
    ----------
    Same as return values of ``parse_args()``.
    """
    # Load the optimization problem oracle
    with open(global_vars.ERR_FILE,'rb') as f:
        err = pickle.load(f)
    oracle = example(abs_err=err['abs_err'],rel_err=err['rel_err'])[1]
    
    # Create status writer for this process
    status_writer = StatusPublisher(proc_num)
    
    # Common mutex
    mutex = Mutex(proc_num,verbose=False)
    
    # Start the slave loop
    if which_alg=='ecc':
        alg_call = lambda branch,location: ecc(oracle,branch,location,status_writer,mutex)
    else:
        alg_call = lambda branch,location: lcss(oracle,branch,location,status_writer,mutex)
    spinner(proc_num,alg_call,status_writer,mutex,wait_time=5.)

if __name__=='__main__':
    proc_num,which_alg = parse_args()
    start_process(proc_num,which_alg)