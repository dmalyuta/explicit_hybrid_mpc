"""
Master file which spawns parallel processes to build the partition.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import os
import time
import glob
import pickle
import numpy as np

import global_vars
import tools
import prepare
from examples import example

class ETACalculator:
    """Calculates the ETA via recursive averaging with exponential forgetting."""
    def __init__(self,call_period,time_constant,store_history=False):
        """
        Parameters
        ----------
        call_period : float
            The period, in seconds, with which the ``update()`` method is going
            to be called.
        time_constant : float
            The exponential forgetting time constant. Measurements older than
            time_constant seconds are weighted with <=exp(-1)=0.368.
        store_history : bool, optional
            ``True`` to store RLS measurement history for debugging purposes.
        """
        self.lamda = call_period/time_constant # [1/s] 1/(time constant)
        self.state = 'init' # {'init','recurse'}
        self.sigma = 1. # Summer normalization term
        self.estimated_rate = None # [%/s] Volume filling rate estimate
        
        self.store_history = store_history
        if self.store_history:
            open(global_vars.ETA_RLS_FILE,'w').close() # clear the storage file
        
    def eta(self,volume_filled):
        """
        Get the current ETA.
        
        Parameters
        ----------
        volume_filled : float
            Fraction of total volume that is already partitioned.
            
        Returns
        -------
        eta : float
            The ETA estimate in seconds.
        """
        if self.estimated_rate is None or self.estimated_rate==0.:
            eta = None
        else:
            eta = (1-volume_filled)/self.estimated_rate
        return eta
        
    def reset(self):
        """Reset the estimator."""
        self.state = 'init'
        self.sigma = 1.
        self.estimated_rate = None
        
    def update(self,measured_rate):
        """
        Updates the current estimated volume filling rate.
        
        Parameters
        ----------
        measured_rate : float
            Volume filling rate [%/s] measurement.
        """
        if self.state=='init':
            self.estimated_rate = measured_rate
            self.state = 'recurse'
        else:
            self.sigma = 1+np.exp(-self.lamda)*self.sigma
            phi = 1/self.sigma
            self.estimated_rate = phi*measured_rate+(1-phi)*self.estimated_rate
        # debugging
        if self.store_history:
            with open(global_vars.ETA_RLS_FILE,'ab') as f:
                pickle.dump(dict(measurement=measured_rate,estimate=self.estimated_rate),f)

class MainStatusPublisher:
    def __init__(self,total_volume,call_period,status_write_period,statistics_save_period):
        """
        Parameters
        ----------
        total_volume: float
            Volume of the set that is being partitioned.
        call_period: float
            The period, in seconds, with which the ``update()`` method is going
            to be called.
        status_write_period: float
            The period, in seconds, with which to write into
            ``global_vars.STATUS_FILE``.
        statistics_save_period : float
            The period, in seconds, with which to save data into the statistics
            lists.
        """
        self.total_volume = total_volume
        self.time_previous,self.time_total = None,0.
        self.worker_procs = [i for i in range(tools.MPI.size()) if i!=global_vars.SCHEDULER_PROC]  # MPI ranks of worker processes
        
        # Create blank status and statistics files
        open(global_vars.STATUS_FILE,'w').close()
        open(global_vars.STATISTICS_FILE,'w').close()
        
        # ETA calculator
        eta_window_duration = int(10./call_period)*call_period # [s] Duration of window for finite-differencing to estimate volume filling rate
        eta_time_constant = 3*60. # [s] Time constant for ETA (more precisely - volume filling rate) estimation
        self.eta_last_measurement = None # Memory variable for volume filling rate measurement via finite-differencing
        self.eta_estimator = ETACalculator(call_period=eta_window_duration,time_constant=eta_time_constant,store_history=False)
        
        # Variables for controlling the writing frequency        
        self.eta_rls_update_counter = 0
        self.status_write_counter = 0
        self.statistics_save_counter = 0
        self.eta_estimate_freq = int(eta_window_duration/call_period)
        self.status_write_freq = int(status_write_period/call_period)
        self.statistics_save_freq = int(statistics_save_period/call_period)
        
    def update_time(self):
        """
        Updates ``self.time``.
        
        Parameters
        ----------
        algorithm: {'ecc','lcss'}, optional
            Which algorithm is being run.
        """
        if self.time_previous is None:
            self.time_previous = time.time()
        dt = time.time()-self.time_previous
        self.time_previous += dt
        self.time_total += dt
    
    def reset_eta(self):
        """Resets memory variables estimation."""
        self.eta_estimator.reset()
        self.eta_last_measurement = None
        self.eta_rls_update_counter = 0
    
    def update(self,proc_status,num_tasks_in_queue,force=False):
        """
        Write the main status file.
        
        Parameters
        ----------
        proc_status : list
            List of dicts of individual worker process statuses.
        num_tasks_in_queue : int
            Number of tasks in the queue.
        force : bool, optional
            ``True`` to force writing both the status and the statistics files.
        """
        # Determine if anything is to be done
        self.status_write_counter += 1
        self.statistics_save_counter += 1
        save_statistics = (self.statistics_save_counter%self.statistics_save_freq)==0
        if save_statistics:
            self.statistics_save_counter = 0 # reset
        write_status = (self.status_write_counter%self.status_write_freq)==0
        if write_status:
            self.status_write_counter = 0 # reset
        # Update the ETA estimate
        self.eta_rls_update_counter += 1
        update_eta = (self.eta_rls_update_counter%self.eta_estimate_freq)==0
        if force or save_statistics or write_status or update_eta:
            worker_idxs = list(range(len(self.worker_procs)))
            volume_filled_total = sum([proc_status[i]['volume_filled_total'] for i in worker_idxs if proc_status[i] is not None])
            volume_filled_frac = volume_filled_total/self.total_volume
        if update_eta:
            self.eta_rls_update_counter = 0 # reset
            # Measure the volume filling rate via finite-differencing
            first_call = self.eta_last_measurement is None
            self.update_time()
            new_measurement = dict(t=self.time_total,v=volume_filled_frac)
            if not first_call:
                rate_measurement = (new_measurement['v']-
                                    self.eta_last_measurement['v'])/(
                                        new_measurement['t']-
                                        self.eta_last_measurement['t'])
                self.eta_estimator.update(rate_measurement)
            self.eta_last_measurement = new_measurement
        # Do the updates, if any
        if force or save_statistics or write_status:
            # Compute overall status
            num_proc_active = sum([proc_status[i]['status']=='active' for i in
                                   worker_idxs if proc_status[i] is not None])
            num_proc_failed = sum([proc_status[i]['status']=='failed' for i in
                                   worker_idxs if proc_status[i] is not None])
            simplex_count_total = sum(
                [proc_status[i]['simplex_count_total'] for i in worker_idxs if
                 proc_status[i] is not None])
            time_active_total = sum([
                proc_status[i]['time_active_total'] for i in worker_idxs if
                proc_status[i] is not None])
            algorithms_list = [proc_status[i]['algorithm'] if proc_status[i] is
                               not None else None for i in worker_idxs]
            self.update_time()
            eta = self.eta_estimator.eta(volume_filled_frac)
            overall_status = dict(num_proc_active=num_proc_active,
                                  num_proc_failed=num_proc_failed,
                                  num_tasks_in_queue=num_tasks_in_queue,
                                  volume_filled_total=volume_filled_total,
                                  volume_filled_frac=volume_filled_frac,
                                  simplex_count_total=simplex_count_total,
                                  time_elapsed=self.time_total,
                                  algorithms_running=algorithms_list,
                                  time_active_total=time_active_total,
                                  eta=eta)
            # Save statistics
            if force or save_statistics:
                with open(global_vars.STATISTICS_FILE,'ab') as f:
                    pickle.dump(dict(overall=overall_status,process=proc_status),f)
            # Write the status file
            if force or write_status:
                with open(global_vars.STATUS_FILE,'w') as status_file:
                    status_file.write('\n'.join([
                            '# overall',
                            'number of processes active: %d'%(num_proc_active),
                            'number of processes failed: %d'%(num_proc_failed),
                            'number of tasks queue: %d'%(num_tasks_in_queue),
                            'volume filled (total [%%]): %.4e'%(volume_filled_frac*100.),
                            'simplex_count: %d'%(simplex_count_total),
                            'time elapsed [s]: %d'%(self.time_total),
                            'time active (total for all processes [s]): %.0f'%(time_active_total),
                            'processes: %d x ecc, %d x lcss'%(
                                sum([alg=='ecc' for alg in algorithms_list]),
                                sum([alg=='lcss' for alg in algorithms_list])),
                            'ETA [s]: %s'%(str(eta if eta is None else int(eta)))
                            ])+'\n\n')
                    for i in worker_idxs:
                        data = proc_status[i]
                        if data is None:
                            continue
                        status_file.write('\n'.join([
                                '# proc %d'%(self.worker_procs[i]),
                                'status: %s'%(data['status']),
                                'algorithm: %s'%(data['algorithm']),
                                'current branch:   %s'%(data['current_branch']),
                                'current location: %s'%(data['current_location']),
                                'volume filled (total [-]): %.4e'%(data['volume_filled_total']),
                                'volume filled (current [%%]): %.4e'%(data['volume_filled_current']*100.),
                                'simplex count (total [-]): %d'%(data['simplex_count_total']),
                                'simplex count (current [-]): %d'%(data['simplex_count_current']),
                                'time active (total [s]): %d'%(data['time_active_total']),
                                'time active (current [s]): %d'%(data['time_active_current']),
                                'time idle (total [s]): %d'%(data['time_idle']),
                                'time running ecc (total [s]): %d'%(data['time_ecc']),
                                'time running lcss (total [s]): %d'%(data['time_lcss'])
                                ])+'\n\n')

class Scheduler:
    def __init__(self):
        # Queue of nodes to be further partitioned
        self.task_queue = []
        
        self.call_period = 1./global_vars.SCHEDULER_RATE
        self.status_write_period = 1./global_vars.STATUS_WRITE_FREQUENCY
        self.statistics_save_period = 1./global_vars.STATISTICS_WRITE_FREQUENCY
        self.setup()
    
    def setup(self):
        """Initial setup. Prepares the task_queue to run the ECC initial
        feasible partition algorithm."""
        def total_volume(cursor):
            """Compute total volume of set. Cusors is triangulation tree's
            root."""
            if cursor.is_leaf():
                sx_vol = tools.simplex_volume(cursor.data.vertices)
                return 0. if cursor.data.is_epsilon_suboptimal else sx_vol
            else:
                return total_volume(cursor.left)+total_volume(cursor.right)

        # Initial tree setup
        _,tree,oracle = example(abs_frac=global_vars.ABS_FRAC,
                                rel_err=global_vars.REL_ERR)
        with open(global_vars.TREE_FILE,'wb') as f:
            # Save the delaunay triangulated version of the set to be
            # partitioned
            pickle.dump(tree,f)
        
        # Load pre-compute branches, if any
        args = prepare.parse_args()
        if args['use_branches']:
            tree = build_tree()
        else:
            # Clean up the file that'll hold the branches
            open(global_vars.BRANCHES_FILE,'w').close()

        # Create the status publisher
        self.status_publisher = MainStatusPublisher(
            total_volume=total_volume(tree),
            call_period=self.call_period,
            status_write_period=self.status_write_period,
            statistics_save_period=self.statistics_save_period)        

        # Populate the task queue
        self.populate_queue(tree)
        
        # Broadcast the suboptimality settings for worker oracles
        suboptimality_settings = dict(abs_err=oracle.eps_a,rel_err=oracle.eps_r)
        tools.MPI.broadcast(suboptimality_settings,
                            root=global_vars.SCHEDULER_PROC)

        # MPI communication requests setup
        self.worker_procs = [i for i in range(tools.MPI.size()) if
                             i!=global_vars.SCHEDULER_PROC] # Worker MPI ranks
        self.task_msg = [tools.NonblockingMPIMessageReceiver(
            source=worker_proc_num,tag=global_vars.NEW_BRANCH_TAG)
                         for worker_proc_num in self.worker_procs]
        self.completed_work_msg = [tools.NonblockingMPIMessageReceiver(
            source=worker_proc_num,tag=global_vars.FINISHED_BRANCH_TAG)
                                   for worker_proc_num in self.worker_procs]
        self.status_msg = [tools.NonblockingMPIMessageReceiver(
            source=worker_proc_num,tag=global_vars.STATUS_TAG)
                           for worker_proc_num in self.worker_procs]
        
        # Clean up the data directory
        for file in glob.glob(global_vars.DATA_DIR+'/branch_*.pkl'):
            os.remove(file)
        
        # Initialize idle worker count to all workers idle
        with open(global_vars.IDLE_COUNT_FILE,'wb') as f:
            pickle.dump(len(self.worker_procs),f)
        
        # Wait for all slaves to setup
        tools.MPI.global_sync() 
    
    def stop_workers(self):
        """Tell all worker processes to stop."""
        for worker_proc_num in self.worker_procs:
            tools.MPI.blocking_send(dict(action='stop'),dest=worker_proc_num,
                                    tag=global_vars.NEW_WORK_TAG)

    def spin(self):
        """
        Manages worker processes until the partitioning process is finished,
        then shuts the processes down and exits.
        """
        def publish_idle_count():
            """
            Communicate to worker processes how many more workers are idle than
            there are tasks in the queue. If there are more workers idle then
            there are tasks in the queue, we want currently active workers to
            offload some of their work to these "slacking" workers.
            """
            num_workers_idle = N_workers-sum(worker_active)
            num_workers_with_no_work = max(num_workers_idle-len(self.task_queue),0)
            tools.debug_print('idle worker count = %d'%(num_workers_idle))
            with open(global_vars.IDLE_COUNT_FILE,'wb') as f:
                pickle.dump(num_workers_with_no_work,f)
                
        N_workers = len(self.worker_procs)
        worker_active = [False]*N_workers
        worker_proc_status = [None]*N_workers
        worker_idxs = list(range(N_workers))
        worker2task = dict.fromkeys(worker_idxs)
        get_worker_proc_num = lambda i: self.worker_procs[i]
        while True:
            time.sleep(self.call_period)
            # Collect any new work from workers
            for i in worker_idxs:
                tasks = self.task_msg[i].receive('all')
                if tasks is not None:
                    tools.debug_print('received %d new tasks from worker %d'%
                                      (len(tasks),get_worker_proc_num(i)))
                    self.task_queue.extend(tasks)
            # Dispatch tasks to idle workers
            if len(self.task_queue)>0 and not all(worker_active):
                for i in worker_idxs:
                    if not worker_active[i]:
                        # Dispatch task to worker process worker_proc_num
                        task = self.task_queue.pop()
                        tools.debug_print(('dispatching task to worker %d (%d '
                                           'tasks left), data {}'%
                                           (get_worker_proc_num(i),
                                            len(self.task_queue))).format(task))
                        tools.MPI.blocking_send(task,
                                                dest=get_worker_proc_num(i),
                                                tag=global_vars.NEW_WORK_TAG)
                        worker2task[str(i)] = task
                        worker_active[i] = True
                    if len(self.task_queue)==0:
                        break # no more tasks to dispatch
                publish_idle_count()
            # Collect completed work from workers
            any_tasks_completed = False
            for i in worker_idxs:
                #NB: there's just one message that should ever be in the buffer
                finished_task = self.completed_work_msg[i].receive()
                if finished_task is not None:
                    tools.debug_print('received finished branch from worker %d'%
                                      (get_worker_proc_num(i)))
                    location = worker2task[str(i)]['location']
                    task_filename = global_vars.DATA_DIR+'/branch_%s.pkl'%(location)
                    with open(task_filename,'rb') as f:
                        finished_branch = pickle.load(f)
                        os.remove(task_filename)
                    with open(global_vars.BRANCHES_FILE,'ab') as f:
                        pickle.dump(finished_branch,f)
                    worker2task[str(i)] = None
                    worker_active[i] = False
                    any_tasks_completed = True
            if any_tasks_completed:
                publish_idle_count()
            # Update status file
            for i in worker_idxs:
                status = self.status_msg[i].receive('newest')
                if status is not None:
                    tools.debug_print('got status update from worker (%d)'%
                                      (get_worker_proc_num(i)))
                    worker_proc_status[i] = status
            self.status_publisher.update(worker_proc_status,len(self.task_queue))
            # Check termination criterion
            if not any(worker_active) and len(self.task_queue)==0:
                self.status_publisher.update(worker_proc_status,
                                             len(self.task_queue),force=True)
                self.status_publisher.reset_eta()
                return

    def populate_queue(self,cursor,__location=''):
        """
        Save tree leaves into a queue list. Only saves the leaves that are not
        already epsilon-suboptimal, and thus do not require further
        partitioning.

        Parameters
        ----------
        cursor : Tree
            Root of the tree whose leaves to save into queue.
        __location : string
            Current location of cursor. **Do not pass this in**.
        """
        if cursor.is_leaf():
            if not cursor.data.is_epsilon_suboptimal:
                algorithm = 'lcss' if hasattr(cursor.data,
                                              'commutation') else 'ecc'
                self.task_queue.append(dict(branch_root=cursor,
                                            location=__location,
                                            action=algorithm))
        else:
            self.populate_queue(cursor.left,__location+'0')
            self.populate_queue(cursor.right,__location+'1')

def build_tree():
    """
    Build the tree from saved branches and save it.
    
    Parameters
    ----------
    cursor : Tree
        Root node handle.
    location : str, optional
        Cursor location relative to root. **Don't pass this in**.
        
    Returns
    -------
    tree : Tree
        The built tree.
    """
    # Load the tree branches
    branches = dict()
    with open(global_vars.BRANCHES_FILE,'rb') as f:
        f.seek(0)
        while True:
            try:
                branch = pickle.load(f)
                branches.update({branch['location']:branch['branch_root']})
            except EOFError:
                break
    
    # Compile the tree
    def extend_tree_with_branches(cursor,__location=''):
        """
        Extend an existing tree using branches.
        
        Parameters
        ----------
        cursor : Tree
            Root of the tree to be extended.
        __location : str, optional
            Cursor location relative to root. **Don't pass this in**.
        """
        if cursor.is_leaf():
            if __location in branches:
                branch = branches[__location]
                branch.copy(cursor) # this may make cursor not a leaf anymore
        if not cursor.is_leaf():
            extend_tree_with_branches(cursor.left,__location+'0')
            extend_tree_with_branches(cursor.right,__location+'1')
    
    with open(global_vars.TREE_FILE,'rb') as f:
        tree = pickle.load(f)
    extend_tree_with_branches(tree)
    with open(global_vars.TREE_FILE,'wb') as f:
        pickle.dump(tree,f)
    return tree

def main(branches_filename=None):
    """
    Runs the scheduler process.
    
    Parameters
    ----------
    branches_filename : str, optional
        Absolute path to branches.pkl for pre-computed branches by a
        previous run of the algorithm. In this case, the branches get loaded
        and only the ones that are not yet terminated (i.e. whose leaves are not
        yet epsilon-suboptimal) get extended.
    """
    scheduler = Scheduler()
    scheduler.spin()
    build_tree()
    scheduler.stop_workers()
