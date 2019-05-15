"""
Subtree partitioning worker process.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import time
import pickle
import numpy as np

import global_vars
import tools
from tree import NodeData
from examples import example

class WorkerStatusPublisher:
    def __init__(self):
        self.data = dict(status='idle', # {'active','idle'}
                         current_branch='', # Location in tree of current subtree's root
                         current_location='', # Location in tree
                         algorithm='', # Which algorithm is being used (ecc or lcss)
                         volume_filled_total=0., # [-] Absolute units
                         volume_filled_current=0., # [%] of current root simplex's volume
                         simplex_count_total=0, # [-] How many simplices generated overall
                         simplex_count_current=0, # [-] How many simplices generated for current root simplex partition
                         time_active_total=0., # [s] Total time spent in 'active' status
                         time_active_current=0., # [s] Total time spent doing partitioning for the current simplex
                         time_idle=0., # [s] Total time spend in 'idle' status
                         time_ecc=0., # [s] Total time spent processing ECC algorithm tasks
                         time_lcss=0.) # [s] Total time spent processing L-CSS algorithm tasks
        self.time_previous = time.time() # [s] Timestamp when time was last updated
        self.volume_current = None # [-] Volume of current root simplex
        self.__write() # Initial write
    
    def __write(self):
        """Publish the data."""
        tools.MPI.nonblocking_send(self.data,dest=global_vars.SCHEDULER_PROC,
                                   tag=global_vars.STATUS_TAG)
    
    def set_new_root_simplex(self,R,location,algorithm):
        """
        Set the new root simplex. Resets the ``*_current`` data fields.
        
        Parameters
        ----------
        R : np.array
            Matrix whose rows are the simplex vertices.
        location : string
            Location of this simplex in overall tree ('0' and '1' format where
            '0' means take left, '1' means take right starting from root node).
        algorithm : {'ecc','lcss'}
            Which algorithm is to be run for the partitioning.
        """
        self.volume_current = tools.simplex_volume(R)
        self.data['current_branch'] = location
        self.data['volume_filled_current'] = 0.
        self.data['simplex_count_current'] = 1
        self.data['time_active_current'] = 0.
        self.data['algorithm'] = algorithm
        self.__write()
    
    def update(self,active=None,failed=False,volume_filled_increment=None,simplex_count_increment=None,location=None,algorithm=None):
        """
        Update the data.
        
        Parameters
        ----------
        active : bool, optional
            ``True`` if the process is in the 'active' state.
        failed : bool, optional
            ``True`` if algorithm failed (worker process shutdown)
        volume_filled_increment : float, optional
            Volume of closed lead.
        simplex_count_increment : int, optional
            How many additional simplices added to partition.
        location : str, optional
            Current location in the tree.
        algorithm : {'ecc','lcss'}
            Which algorithm is to be run for the partitioning.
        """
        # Update time counters
        dt = time.time()-self.time_previous
        self.time_previous += dt
        if self.data['status']=='active':
            self.data['time_active_total'] += dt
            self.data['time_active_current'] += dt
        else:
            self.data['time_idle'] += dt
        if self.data['algorithm']=='ecc':
            self.data['time_ecc'] += dt
        else:
            self.data['time_lcss'] += dt
        # Update status
        if active is not None:
            self.data['status'] = 'active' if active else 'idle'
            tools.info_print('sending status update, status = %s'%(self.data['status']))
        if failed is True:
            self.data['status'] = 'failed'
        # Update volume counters
        if volume_filled_increment is not None:
            self.data['volume_filled_total'] += volume_filled_increment
            self.data['volume_filled_current'] += volume_filled_increment/self.volume_current
        # Update simplex counters
        if simplex_count_increment is not None:
            self.data['simplex_count_total'] += simplex_count_increment
            self.data['simplex_count_current'] += simplex_count_increment
        # Update location
        if location is not None:
            self.data['current_location'] = location
        # Update algorithm
        if algorithm is not None:
            self.data['algorithm'] = algorithm
        self.__write()

class DifficultyChecker:
    def __init__(self,progress_rate_threshold=0.01):
        """
        Parameters
        ----------
        progress_rate_threshold : float
            Threshold progress rate [% volume filled/min] below which to judge
            the branch as "difficult".
        """
        self.threshold = progress_rate_threshold
        self.reset()

    def reset(self):
        """Reset memory variables"""
        self.time_last = None
        self.progress_last = None
        self.difficult = False

    def check(self,progress):
        """
        Parameters
        ----------
        progress : float
            The current progress.

        Returns
        -------
        self.difficult : bool
            If ``True``, the branch is difficult because the progress rate is
            below the threshold.
        """
        time_now = time.time()
        one_minute = 60 # [s]
        if self.time_last is None:
            self.time_last = time_now
            self.progress_last = progress
        else:
            if time_now-self.time_last>one_minute:
                progress_rate = progress-self.progress_last # [%/min]
                self.difficult = progress_rate<self.threshold
                if self.difficult:
                    tools.info_print('current branch is difficult')
        return self.difficult
        
class Worker:
    def __init__(self):
        self.setup()
        
    def setup(self):
        # Optimization problem oracle
        suboptimality_settings = tools.MPI.broadcast(None,root=global_vars.SCHEDULER_PROC)
        self.oracle = example(abs_err=suboptimality_settings['abs_err'],
                              rel_err=suboptimality_settings['rel_err'])[2]
        tools.info_print('made oracle')
        # Checker whether a branch is "difficult"
        self.difficulty = DifficultyChecker()
        # Status publisher
        self.status_publisher = WorkerStatusPublisher()
        tools.MPI.global_sync() # wait for all slaves to setup
        # Algorithm call selector
        def alg_call(which_alg,branch,location):
            if which_alg=='ecc':
                return self.ecc(branch,location)
            else:
                return self.lcss(branch,location)
        self.alg_call = alg_call
    
    def offload_child_computation(self,child,location,which_alg,
                                  prioritize_self=False,force='none'):
        """
        Offload partitioning for child to another worker process.
        
        Parameters
        ----------
        child : Tree
            The child for which computation is to be offloaded.
        location : string
            Location of the child in the overall tree.
        which_alg : {'ecc','lcss'}
            Which algorithm is to be run for the partitioning.
        prioritize_self : bool, optional
            If ``True``, ignore worker count but do respect recursion limit in
            terms of assigning work to self.
        force : {'none','offload','self'}, optional
            'offload' means choose to submit task to queue, no matter what;
            'self' means continue to work on task alone, no matter what; 'none'
            means no such forcing.
        """
        # --- Check worker count
        with open(global_vars.IDLE_COUNT_FILE,'rb') as f:
            try:
                idle_worker_count = pickle.load(f)
            except EOFError:
                # This may occur if the file is currently being written to by
                # the scheduler. In this case, conservatively assume that there
                # are no idle workers
                idle_worker_count = 0
        tools.info_print('idle worker count = %d'%(idle_worker_count))
        # --- Check recursion limit
        recursion_depth = (len(location)-
                           len(self.status_publisher.data['current_branch']))
        recurs_limit_reached = recursion_depth>global_vars.MAX_RECURSION_LIMIT
        if recurs_limit_reached:
            tools.error_print('recursion limit reached - submitting task'
                              ' to queue')
        # --- Check difficult of current branch
        progress = self.status_publisher.data['volume_filled_current']
        is_difficult = self.difficulty.check(progress)
        # --- Offloading logic
        if (force!='self' and
            (force=='offload' or recurs_limit_reached or is_difficult or
             (idle_worker_count>0 and not prioritize_self))):
            new_task = dict(branch_root=child,location=location,
                            action=which_alg)
            tools.MPI.nonblocking_send(new_task,dest=global_vars.SCHEDULER_PROC,
                                       tag=global_vars.NEW_BRANCH_TAG)
        else:
            self.status_publisher.update(algorithm=which_alg)
            self.alg_call(which_alg,child,location)
    
    def ecc(self,node,location):
        """
        Implementation of [1] Algorithm 2 lines 4-16. Pass a tree root
        node and this grows the tree until its leaves are feasible partition
        cells. **Caution**: modifies ``node`` (passed by reference).
        
        [1] D. Malyuta, B. Acikmese, M. Cacan, and D. S. Bayard,
        "Partition-based feasible integer solution pre-computation for hybrid
        model predictive control," in 2019 European Control Conference
        (accepted), IFAC, jun 2019.
        
        Parameters
        ----------
        node : Tree
            Tree root. Sufficient that it just holds node.data.vertices for the
            simplex vertices.
        location : string
            Location in tree of this node. String where '0' at index i means
            take left child, '1' at index i means take right child (at depth
            value i).
        """
        self.status_publisher.update(location=location)
        tools.info_print('ecc at location = %s'%(location))
        c_R = np.average(node.data.vertices,axis=0) # Simplex barycenter
        if not self.oracle.P_theta(theta=c_R,check_feasibility=True):
            raise RuntimeError('STOP, Theta contains infeasible regions')
        else:
            delta_hat,vx_inputs_and_costs = self.oracle.V_R(node.data.vertices)
            if delta_hat is None:
                S_1,S_2 = tools.split_along_longest_edge(node.data.vertices)[:2]
                child_left = NodeData(vertices=S_1)            
                child_right = NodeData(vertices=S_2)
                node.grow(child_left,child_right)
                self.status_publisher.update(simplex_count_increment=1)
                # Recursive call for each resulting simplex
                self.offload_child_computation(node.left,location+'0','ecc')
                self.offload_child_computation(node.right,location+'1','ecc',
                                               prioritize_self=True)
            else:
                # Assign feasible commutation to simplex
                Nvx = node.data.vertices.shape[0]
                vertex_costs = np.array([vx_inputs_and_costs[i][1]
                                         for i in range(Nvx)])
                vertex_inputs = np.array([vx_inputs_and_costs[i][0]
                                          for i in range(Nvx)])
                node.data = NodeData(vertices=node.data.vertices,
                                     commutation=delta_hat,
                                     vertex_costs=vertex_costs,
                                     vertex_inputs=vertex_inputs)
                self.offload_child_computation(node,location,'lcss',
                                               force='self')
    
    def lcss(self,node,location):
        """
        Implementation of [1] Algorithm 1 lines 4-20. Pass a tree root
        node and this grows the tree until its leaves are associated with an
        epsilon-suboptimal commutation*.
        **Caution**: modifies ``node`` (passed by reference).
        
        * Assuming that no leaf is closed due to a bad condition number.
        
        [1] D. Malyuta and B. Acikmese, "Approximate Mixed-integer
        Convex Multiparametric Programming," in Control Systems Letters (in
        review), IEEE.
        
        Parameters
        ----------
        node : Tree
            Tree root. Sufficient that it just holds node.data.vertices for the
            simplex vertices.
        location : string
            Location in tree of this node. String where '0' at index i means take
            left child, '1' at index i means take right child (at depth value i).
        """
        def add(child_left,child_right):
            """
            Make the node a parent of child_left and child_right.
            
            Parameters
            ----------
            child_left : NodeData
                Data for the "left" child in the binary tree.
            child_right : NodeData
                Data for the "right" child in the binary tree.
            """
            node.grow(child_left,child_right)
            self.status_publisher.update(simplex_count_increment=1)
            self.offload_child_computation(node.left,location+'0','lcss')
            self.offload_child_computation(node.right,location+'1','lcss',
                                           prioritize_self=True)
    
        def update_vertex_costs(v_mid,v_combo_idx,delta,old_vertex_inputs,
                                old_vertex_costs):
            """
            Compute a new set of optimal costs at the simplex vertices.
            
            Parameters
            ----------
            v_mid : np.array
                New vertex at which the current simplex is to be split into two.
            v_combo_idx : tuple
                Tuple of two elements corresponding to row index of the two
                vertices constituting the longest edge. The first vertex is
                removed from S_1 and the second vertex is removed from S_2,
                substituted for v_mid.
            delta : np.array
                The commutation that is to be associated with the two new
                simplices.
            old_vertex_costs : np.array
                Array of existing pre-computed vertex inputs.
            old_vertex_costs : np.array
                Array of existing pre-computed vertex costs.
            """
            u_opt_v_mid, V_delta_v_mid = self.oracle.P_theta_delta(
                theta=v_mid,delta=delta)[:2]
            vertex_inputs_S_1,vertex_inputs_S_2 = (old_vertex_inputs.copy(),
                                                   old_vertex_inputs.copy())
            vertex_costs_S_1,vertex_costs_S_2 = (old_vertex_costs.copy(),
                                                 old_vertex_costs.copy())
            vertex_inputs_S_1[v_combo_idx[0]] = u_opt_v_mid
            vertex_inputs_S_2[v_combo_idx[1]] = u_opt_v_mid
            vertex_costs_S_1[v_combo_idx[0]] = V_delta_v_mid
            vertex_costs_S_2[v_combo_idx[1]] = V_delta_v_mid
            return (vertex_inputs_S_1,vertex_inputs_S_2,
                    vertex_costs_S_1,vertex_costs_S_2)
    
        self.status_publisher.update(location=location)
        delta_epsilon_suboptimal = self.oracle.bar_E_delta_R(
            R=node.data.vertices,V_delta_R=node.data.vertex_costs)
        if delta_epsilon_suboptimal:
            # Close leaf
            node.data.is_epsilon_suboptimal = True
            node.data.timestamp = time.time()
            volume_closed = tools.simplex_volume(node.data.vertices)
            self.status_publisher.update(volume_filled_increment=volume_closed)
        else:
            delta_star,theta_star,new_vx_inputs_costs,cost_varies_little = (
                self.oracle.bar_D_delta_R(R=node.data.vertices,
                                          V_delta_R=node.data.vertex_costs,
                                          delta_ref=node.data.commutation))
            bar_D_delta_R_feasible = delta_star is not None
            if not bar_D_delta_R_feasible:
                # delta does not change in this case, so the same vertex inputs
                # and costs
                delta_star = node.data.commutation
                new_vertex_costs = node.data.vertex_costs
                new_vertex_inputs = node.data.vertex_inputs
            else:
                # extract the vertex inputs and costs associated with the better
                # commutation choice
                Nvx = node.data.vertices.shape[0]
                new_vertex_costs = np.array([new_vx_inputs_costs[i][1]
                                             for i in range(Nvx)])
                new_vertex_inputs = np.array([new_vx_inputs_costs[i][0]
                                              for i in range(Nvx)])
            if bar_D_delta_R_feasible and cost_varies_little:
                node.data.commutation = delta_star
                node.data.vertex_costs = new_vertex_costs
                node.data.vertex_inputs = new_vertex_inputs
                self.offload_child_computation(node,location,'lcss',
                                               force='self')
            else:
                S_1,S_2,v_idx = tools.split_along_longest_edge(
                    node.data.vertices)
                # Re-compute vertex costs
                v_mid = S_1[v_idx[0]]
                (vertex_inputs_S_1,vertex_inputs_S_2,
                 vertex_costs_S_1,vertex_costs_S_2) = update_vertex_costs(
                     v_mid,v_idx,delta_star,new_vertex_inputs,new_vertex_costs)
                # Make children
                child_left = NodeData(vertices=S_1,commutation=delta_star,
                                      vertex_costs=vertex_costs_S_1,
                                      vertex_inputs=vertex_inputs_S_1)
                child_right = NodeData(vertices=S_2,commutation=delta_star,
                                       vertex_costs=vertex_costs_S_2,
                                       vertex_inputs=vertex_inputs_S_2)
                add(child_left,child_right)

    def spin(self):
        """
        A loop which waits (by passive blocking) for the roots of a new branch
        to partition to be received from the scheduler process. When received,
        the branch is partitioned. When done, the partitioned branch ("grown
        tree") is sent back to the scheduler. The scheduler is responsible for
        aborting this loop.
        """
        while True:
            # Block until new data is received from scheduler
            tools.info_print('waiting for data')
            data = tools.MPI.blocking_receive(source=global_vars.SCHEDULER_PROC,
                                              tag=global_vars.NEW_WORK_TAG)
            tools.info_print('received data {}'.format(data))
            if data['action']=='stop':
                # Request from scheduler to stop
                return
            else:
                tools.info_print('got branch at location = %s'%
                                  (data['location']))
                self.status_publisher.update(active=True)
                # Get data about the branch to be worked on
                branch = data['branch_root']
                branch_location = data['location']
                self.status_publisher.set_new_root_simplex(branch.data.vertices,
                                                           branch_location,
                                                           data['action'])
                # Do work on this branch (i.e. partition this simplex)
                try:
                    tools.info_print('calling algorithm')
                    self.difficulty.reset() # Reset difficult checking
                    self.alg_call(data['action'],branch,branch_location)
                except:
                    self.status_publisher.update(failed=True)
                    raise
                # Save completed branch and notify scheduler that it is
                # available
                with open(global_vars.DATA_DIR+'/branch_%s.pkl'%
                          (data['location']),'wb') as f:
                    pickle.dump(data,f)
                tools.info_print('completed task at location = %s, '
                                 'notifying scheduler'%(data['location']))
                self.status_publisher.update(active=False)

def main():
    """Runs the worker process."""
    worker = Worker()
    worker.spin()
