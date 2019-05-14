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
import copy
import threading
import numpy as np

import global_vars
import tools
from examples import example

class ETACalculator:
    """Calculates the ETA via recursive averaging with exponential forgetting."""
    def __init__(self,call_period,time_constant,store_history=False):
        """
        Parameters
        ----------
        call_period, time_constant : see RLS.__init__ docstring
        store_history : bool, optional
            ``True`` to store RLS measurement history for debugging purposes.
        """
        self.rls = RLS(call_period,time_constant)
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
        if self.rls.estimate is None or self.rls.estimate==0.:
            eta = None
        else:
            eta = (1-volume_filled)/self.rls.estimate
        return eta
        
    def reset(self):
        """Reset the estimator."""
        self.rls.reset()
        
    def update(self,measured_rate):
        """
        Updates the current estimated volume filling rate.
        
        Parameters
        ----------
        measured_rate : float
            Volume filling rate [%/s] measurement.
        """
        self.rls.update(measured_rate)
        if self.store_history:
            with open(global_vars.ETA_RLS_FILE,'ab') as f:
                pickle.dump(dict(measurement=measurement,
                                 estimate=self.rls.estimate),f)

class RLS:
    """Recursive least squares with exponential forgetting for a scalar"""
    def __init__(self,call_period,time_constant):
        """
        Parameters
        ----------
        call_period : float
            The period, in seconds, with which the ``update()`` method is going
            to be called.
        time_constant : float
            The exponential forgetting time constant. Measurements older than
            time_constant seconds are weighted with <=exp(-1)=0.368.
        """
        self.lamda = call_period/time_constant # [1/s] 1/(time constant)
        self.state = 'init' # {'init','recurse'}
        self.sigma = 1. # Summer normalization term
        self.estimate = None # Estimate of the scalar value

    def reset(self):
        """Reset the estimator."""
        self.state = 'init'
        self.sigma = 1.
        self.estimate = None

    def update(self,measurement):
        """
        Updates the current estimate using the measurement.
        
        Parameters
        ----------
        measurement : float
            Measured value of the scalar quantity being estimated.
        """
        if self.state=='init':
            self.estimate = measurement
            self.state = 'recurse'
        else:
            self.sigma = 1+np.exp(-self.lamda)*self.sigma
            phi = 1/self.sigma
            self.estimate = phi*measurement+(1-phi)*self.estimate

class MainStatusPublisher:
    def __init__(self,total_volume,call_period,status_write_period,
                 statistics_save_period):
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
        # MPI ranks of worker processes
        self.worker_procs = [i for i in range(tools.MPI.size())
                             if i!=global_vars.SCHEDULER_PROC]
        
        # Create blank status and statistics files
        open(global_vars.STATUS_FILE,'w').close()
        open(global_vars.STATISTICS_FILE,'w').close()
        
        # ETA (i.e. time remaining) calculator
        # [s] duration of window for finite-differencing to estimate volume
        # filling rate
        eta_window_duration = 10.
        # [s] time constant for ETA (more precisely - volume filling rate)
        # estimation
        eta_time_constant = 3*60.
        # memory variable for volume filling rate measurement via
        # finite-differencing
        self.eta_last_measurement = None
        self.eta_estimator = ETACalculator(call_period=eta_window_duration,
                                           time_constant=eta_time_constant,
                                           store_history=False)

        # Scheduler main loop rate estimator
        self.looprate_estimator = [RLS(call_period=call_period,
                                       time_constant=5*call_period)
                                   for i in range(3)]
        
        # Variables for controlling the writing frequency
        self.write_time_prev = dict(eta=None,looprate=None,
                                    status=None,statistics=None)
        self.write_period = dict(eta=eta_window_duration,
                                 status=status_write_period,
                                 statistics=statistics_save_period)
        
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
    
    def reset_estimators(self):
        """Resets memory variables of estimators."""
        # ETA estimator
        self.eta_estimator.reset()
        self.eta_last_measurement = None
        self.write_time_prev['eta'] = None
        # Scheduler main loop frequency estimator
        for i in range(3):
            self.looprate_estimator[i].reset()
        self.write_time_prev['looprate'] = None
    
    def update(self,proc_status,num_tasks_in_queue,looprates,force=False):
        """
        Write the main status file.
        
        Parameters
        ----------
        proc_status : list
            List of dicts of individual worker process statuses.
        num_tasks_in_queue : int
            Number of tasks in the queue.
        looprates : list
            A list holding the [publisher,dispatcher,collector] looprates.
        force : bool, optional
            ``True`` to force writing both the status and the statistics files.
        """
        self.update_time()
        # Determine what updates to do
        save_statistics = (self.write_time_prev['statistics'] is None or
                           self.time_total-self.write_time_prev['statistics']>=
                           self.write_period['statistics'])
        write_status = (self.write_time_prev['status'] is None or
                        self.time_total-self.write_time_prev['status']>=
                        self.write_period['status'])
        update_eta = (self.write_time_prev['eta'] is None or
                      self.time_total-self.write_time_prev['eta']>=
                      self.write_period['eta'])
        # reset times
        if save_statistics:
            self.write_time_prev['statistics'] = self.time_total
        if write_status:
            self.write_time_prev['status'] = self.time_total
        if update_eta:
            self.write_time_prev['eta'] = self.time_total
        # Compute the fraction of the set already partitioned ("volume filled")
        if force or save_statistics or write_status or update_eta:
            worker_idxs = list(range(len(self.worker_procs)))
            volume_filled_total = sum([
                proc_status[i]['volume_filled_total'] for i in worker_idxs
                if proc_status[i] is not None])
            volume_filled_frac = volume_filled_total/self.total_volume
        # Update the ETA (i.e. remaining runtime) estimate
        if update_eta:
            # Measure the volume filling rate via finite-differencing
            first_call = self.eta_last_measurement is None
            new_measurement = dict(t=self.time_total,v=volume_filled_frac)
            if not first_call:
                rate_measurement = (new_measurement['v']-
                                    self.eta_last_measurement['v'])/(
                                        new_measurement['t']-
                                        self.eta_last_measurement['t'])
                self.eta_estimator.update(rate_measurement)
            self.eta_last_measurement = new_measurement
        # Update the schedular main loop frequency estimate estimate
        for i in range(3):
            self.looprate_estimator[i].update(looprates[i])
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
            time_idle_total = sum([
                proc_status[i]['time_idle'] for i in worker_idxs if
                proc_status[i] is not None])
            algorithms_list = [proc_status[i]['algorithm'] if proc_status[i] is
                               not None and proc_status[i]['status']=='active'
                               else None for i in worker_idxs]
            eta = self.eta_estimator.eta(volume_filled_frac)
            scheduler_looprates = dict(
                publisher=self.looprate_estimator[0].estimate,
                dispatcher=self.looprate_estimator[1].estimate,
                collector=self.looprate_estimator[2].estimate)
            overall_status = dict(num_proc_active=num_proc_active,
                                  num_proc_failed=num_proc_failed,
                                  num_tasks_in_queue=num_tasks_in_queue,
                                  volume_filled_total=volume_filled_total,
                                  volume_filled_frac=volume_filled_frac,
                                  simplex_count_total=simplex_count_total,
                                  time_elapsed=self.time_total,
                                  algorithms_running=algorithms_list,
                                  time_active_total=time_active_total,
                                  time_idle_total=time_idle_total,
                                  eta=eta,
                                  scheduler_looprate=scheduler_looprates)
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
                        'volume filled (total [%%]): %.4e'%(
                            volume_filled_frac*100.),
                        'simplex_count: %d'%(simplex_count_total),
                        'time elapsed [s]: %d'%(self.time_total),
                        'time active (total for all processes [s]): %.0f'%(
                            time_active_total),
                        'time idle (total for all processes [s]): %.0f'%(
                            time_idle_total),
                        'processes: %d x ecc, %d x lcss'%(
                            sum([alg=='ecc' for alg in algorithms_list]),
                            sum([alg=='lcss' for alg in algorithms_list])),
                        'ETA [s]: %s'%(str(eta if eta is None else
                                           int(np.round(eta)))),
                        'Scheduler loop frequencies [pub,disp,col] [Hz]: '
                        '[%s,%s,%s]'%(
                            str(scheduler_looprates['publisher']
                                if scheduler_looprates['publisher'] is None
                                else int(np.round(
                                        scheduler_looprates['publisher']))),
                            str(scheduler_looprates['dispatcher']
                                if scheduler_looprates['dispatcher'] is None
                                else int(np.round(
                                        scheduler_looprates['dispatcher']))),
                            str(scheduler_looprates['collector']
                                if scheduler_looprates['collector'] is None
                                else int(np.round(
                                        scheduler_looprates['collector']))))
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
                            'volume filled (total [-]): %.4e'%(
                                data['volume_filled_total']),
                            'volume filled (current [%%]): %.4e'%(
                                data['volume_filled_current']*100.),
                            'simplex count (total [-]): %d'%(
                                data['simplex_count_total']),
                            'simplex count (current [-]): %d'%(
                                data['simplex_count_current']),
                            'time active (total [s]): %d'%(
                                data['time_active_total']),
                            'time active (current [s]): %d'%(
                                data['time_active_current']),
                            'time idle (total [s]): %d'%(data['time_idle']),
                            'time running ecc (total [s]): %d'%(
                                data['time_ecc']),
                            'time running lcss (total [s]): %d'%(
                                data['time_lcss'])
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
        self.status_msg = [tools.NonblockingMPIMessageReceiver(
            source=worker_proc_num,tag=global_vars.STATUS_TAG)
                           for worker_proc_num in self.worker_procs]

        # Threading setup for main loop threads
        self.mutex = threading.Lock()
        self.stop_threads = False
        # shared quantities below
        self.get_worker_proc_num = lambda i: self.worker_procs[i]
        self.N_workers = len(self.worker_procs)
        self.idle_workers = []
        self.worker_proc_status = [None]*self.N_workers
        self.worker_idxs = list(range(self.N_workers))
        self.worker2task = dict.fromkeys(self.worker_idxs)
        self.publisher_looprate = 0
        self.dispatcher_looprate = 0
        self.collector_looprate = 0
        
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

    def __maintain_loop_rate(self,period,dt):
        """Maintain loop rate using the previous iteration's runtime dt."""
        sleep_time = period-dt
        if sleep_time>0:
            time.sleep(sleep_time)

    def __publish_idle_count(self):
        """
        Communicate to worker processes how many more workers are idle than
        there are tasks in the queue. If there are more workers idle then
        there are tasks in the queue, we want currently active workers to
        offload some of their work to these "slacking" workers.

        **NOT THREAD SAFE -- wrap with a mutex!**
        """
        num_tasks = len(self.task_queue)
        num_idle_workers = len(self.idle_workers)
        num_workers_with_no_work = max(num_idle_workers-num_tasks,0)
        tools.info_print('idle worker count = %d'%(num_idle_workers))
        with open(global_vars.IDLE_COUNT_FILE,'wb') as f:
            pickle.dump(num_workers_with_no_work,f)

    def status_publisher_thread(self):
        """Main loop thread which publishes the status."""
        self.mutex.acquire()
        period = self.call_period
        self.mutex.release()
        iteration_runtime = 0
        time_last = None
        while True:
            # Measure loop rate
            time_now = time.time()
            if time_last is not None:
                dt = time_now-time_last
                self.mutex.acquire()
                self.publisher_looprate = 1./dt
                self.mutex.release()
            time_last = time_now
            self.__maintain_loop_rate(period,iteration_runtime)
            iteration_tic = time.time()
            #-- Main code ---------------------------------------
            self.mutex.acquire()
            queue_length = len(self.task_queue)
            worker_status = copy.deepcopy(self.worker_proc_status)
            looprates = [self.publisher_looprate,self.dispatcher_looprate,
                         self.collector_looprate]
            self.mutex.release()
            self.status_publisher.update(worker_status,queue_length,looprates)
            #----------------------------------------------------
            self.mutex.acquire()
            stop = self.stop_threads
            if stop:
                self.status_publisher.update(self.worker_proc_status,
                                             len(self.task_queue),looprates,
                                             force=True)
                self.status_publisher.reset_estimators()
            self.mutex.release()
            if stop:
                tools.info_print('status publisher stopping')
                return
            iteration_toc = time.time()
            iteration_runtime = iteration_toc-iteration_tic

    def work_dispatcher_thread(self):
        """Main loop thread which dispatches tasks to workers."""
        self.mutex.acquire()
        period = self.call_period
        self.mutex.release()
        iteration_runtime = 0
        time_last = None
        while True:
            # Measure loop rate
            time_now = time.time()
            if time_last is not None:
                dt = time_now-time_last
                self.mutex.acquire()
                self.dispatcher_looprate = 1./dt
                self.mutex.release()
            time_last = time_now
            self.__maintain_loop_rate(period,iteration_runtime)
            iteration_tic = time.time()
            #-- Main code ---------------------------------------
            # Dispatch work to idle workers
            worker_count_changed = False
            while True:
                # Check exit condition
                self.mutex.acquire()
                num_tasks = len(self.task_queue)
                num_idle_workers = len(self.idle_workers)
                self.mutex.release()
                if num_tasks==0 or num_idle_workers==0:
                    break
                # Dispatch task to idle worker process
                self.mutex.acquire()
                task = self.task_queue.pop()
                idle_worker_idx = self.idle_workers.pop()
                self.mutex.release()
                tools.info_print(('dispatching task to worker (%d) (%d '
                                  'tasks left), data {}'%
                                  (self.get_worker_proc_num(idle_worker_idx),
                                   num_tasks-1)).format(task))
                tools.MPI.blocking_send(task,dest=self.get_worker_proc_num(
                    idle_worker_idx),tag=global_vars.NEW_WORK_TAG)
                self.mutex.acquire()
                self.worker2task[idle_worker_idx] = task
                self.mutex.release()
                worker_count_changed = True
            if worker_count_changed:
                self.mutex.acquire()
                self.__publish_idle_count()
                self.mutex.release()
            #----------------------------------------------------
            self.mutex.acquire()
            stop = self.stop_threads
            self.mutex.release()
            if stop:
                tools.info_print('work dispatcher stopping')
                return
            iteration_toc = time.time()
            iteration_runtime = iteration_toc-iteration_tic

    def work_collector_thread(self):
        """Main loop thread which collects new and finished tasks from
        workers. This thread also evaluates the stopping criterion."""
        self.mutex.acquire()
        period = self.call_period
        self.mutex.release()
        iteration_runtime = 0
        time_last = None
        while True:
            # Measure loop rate
            time_now = time.time()
            if time_last is not None:
                dt = time_now-time_last
                self.mutex.acquire()
                self.collector_looprate = 1./dt
                self.mutex.release()
            time_last = time_now
            self.__maintain_loop_rate(period,iteration_runtime)
            iteration_tic = time.time()
            #-- Main code ---------------------------------------
            # Capture finished workers that are now idle
            any_workers_became_idle = False
            for i in self.worker_idxs:
                status = self.status_msg[i].receive()
                if status is not None:
                    tools.info_print(('got status update from worker (%d), '
                                      'status {}'%(
                                          self.get_worker_proc_num(i))).format(
                                              status))
                    self.mutex.acquire()
                    self.worker_proc_status[i] = status
                    self.mutex.release()
                    if status['status']=='idle':
                        self.mutex.acquire()
                        self.idle_workers.append(i)
                        if self.worker2task[i] is not None:
                            self.worker2task[i] = None # reset
                        self.mutex.release()
                        any_workers_became_idle = True
            # Collect any new work from workers
            new_tasks_available = False
            for i in self.worker_idxs:
                task = self.task_msg[i].receive()
                if task is not None:
                    tools.info_print(('received new task from worker (%d), '
                                      'task {}'%(
                                          self.get_worker_proc_num(i))).format(
                                              task))
                    self.mutex.acquire()
                    self.task_queue.append(task)
                    self.mutex.release()
                    new_tasks_available = True
            if any_workers_became_idle or new_tasks_available:
                self.mutex.acquire()
                self.__publish_idle_count()
                self.mutex.release()
            #----------------------------------------------------
            self.mutex.acquire()
            self.stop_threads = (len(self.idle_workers)==self.N_workers and
                                 len(self.task_queue)==0)
            stop = self.stop_threads
            self.mutex.release()
            if stop:
                tools.info_print('work collector is stopping')
                return
            iteration_toc = time.time()
            iteration_runtime = iteration_toc-iteration_tic

    def spin(self):
        """
        Manages worker processes until the partitioning process is finished,
        then shuts the processes down and exits.
        """
        tools.info_print('creating the main loop threads')
        publisher = threading.Thread(target=self.status_publisher_thread)
        dispatcher = threading.Thread(target=self.work_dispatcher_thread)
        collector = threading.Thread(target=self.work_collector_thread)
        tools.info_print('starting the main loop threads...')
        collector.start()
        dispatcher.start()
        publisher.start()
        tools.info_print('waiting for the main loop threads to finish...')
        collector.join()
        dispatcher.join()
        publisher.join()
        tools.info_print('all main loop threads finished')

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
    def extend_tree_with_branches(cursor,__location=''):
        """
        Extend an existing tree using branch_*.pkl files in
        global_vars.DATA_DIR.
        
        Parameters
        ----------
        cursor : Tree
            Root of the tree to be extended.
        __location : str, optional
            Cursor location relative to root. **Don't pass this in**.
        """
        if cursor.is_leaf():
            try:
                branch_filename = (global_vars.DATA_DIR+
                                   '/branch_%s.pkl'%(__location))
                with open(branch_filename,'rb') as f:
                    branch = pickle.load(f)['branch_root']
                os.remove(branch_filename) # cleanup
                branch.copy(cursor) # this may make cursor not a leaf anymore
            except FileNotFoundError:
                pass # This is a leaf in the final tree
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
