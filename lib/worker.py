"""
Subtree partitioning worker process.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import time
import pickle
import numpy as np

import mpi4py
mpi4py.rc.recv_mprobe = False # resolve UnpicklingError (https://tinyurl.com/mpi4py-unpickling-issue)
from mpi4py import MPI

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
        self.req = None # Non-blocking MPI Request object (once the first request is made)
        self.__write() # Initial write
    
    def __write(self):
        """
        Write the data to file.
        """        
        self.req = MPI.COMM_WORLD.isend(self.data,dest=global_vars.SCHEDULER_PROC,tag=global_vars.STATUS_TAG)
        
    def reset_volume_filled_total(self):
        """Resets the ``volume_filled_total``."""
        self.data['volume_filled_total'] = 0.
    
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
    
    def update(self,active=None,failed=False,volume_filled_increment=None,simplex_count_increment=None,location=None):
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
            tools.debug_print('sending status update, status = %s'%(self.data['status']))
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
        self.__write()
        
class Worker:
    def __init__(self):
        self.setup()
        
    def setup(self):
        # Optimization problem oracle
        suboptimality_settings = MPI.COMM_WORLD.bcast(None,root=global_vars.SCHEDULER_PROC)
        self.oracle = example(abs_err=suboptimality_settings['abs_err'],
                              rel_err=suboptimality_settings['rel_err'])[2]
        tools.debug_print('made oracle')
        # Status publisher
        self.status_publisher = WorkerStatusPublisher()
        MPI.COMM_WORLD.Barrier() # wait for all slaves to setup
        # Algorithm call selector
        def alg_call(which_alg,branch,location):
            if which_alg=='ecc':
                return self.ecc(branch,location)
            else:
                return self.lcss(branch,location)
        self.alg_call = alg_call
    
    def offload_child_computation(self,child,location,which_alg):
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
        """
        with open(global_vars.IDLE_COUNT_FILE,'rb') as f:
            try:
                idle_worker_count = pickle.load(f)
            except EOFError:
                # This may occur if the file is currently being written to by
                # the scheduler. In this case, conservatively assume that there
                # are no idle workers
                idle_worker_count = 0
        tools.debug_print('idle worker count = %d'%(idle_worker_count))
        if idle_worker_count>0:
            new_task = dict(branch_root=child,location=location,action=which_alg)
            MPI.COMM_WORLD.isend(new_task,dest=global_vars.SCHEDULER_PROC,tag=global_vars.NEW_BRANCH_TAG)
        else:
            self.alg_call(which_alg,child,location)
    
    def ecc(self,node,location):
        """
        Implementation of [Malyuta2019a] Algorithm 2 lines 4-16. Pass a tree root
        node and this grows the tree until its leaves are feasible partition cells.
        **Caution**: modifies ``node`` (passed by reference).
        
        [Malyuta2019a] D. Malyuta, B. Acikmese, M. Cacan, and D. S. Bayard,
        "Partition-based feasible integer solution pre-computation for hybrid model
        predictive control," in 2019 European Control Conference (in review), IFAC,
        jun 2019.
        
        Parameters
        ----------
        node : Tree
            Tree root. Sufficient that it just holds node.data.vertices for the
            simplex vertices.
        location : string
            Location in tree of this node. String where '0' at index i means take
            left child, '1' at index i means take right child (at depth value i).
        """
        self.status_publisher.update(location=location)
        tools.debug_print('ecc at location = %s'%(location))
        c_R = np.average(node.data.vertices,axis=0) # Simplex barycenter
        if not self.oracle.P_theta(theta=c_R,check_feasibility=True):
            raise RuntimeError('STOP, Theta contains infeasible regions')
        else:
            delta_hat = self.oracle.V_R(node.data.vertices)
            if delta_hat is None:
                S_1,S_2 = tools.split_along_longest_edge(node.data.vertices)[:2]
                child_left = NodeData(vertices=S_1)            
                child_right = NodeData(vertices=S_2)
                node.grow(child_left,child_right)
                self.status_publisher.update(simplex_count_increment=1)
                # Recursive call for each resulting simplex
                self.offload_child_computation(node.left,location+'0','ecc')
                self.ecc(node.right,location+'1')
            else:
                # Assign feasible commutation to simplex
                vertex_inputs_and_costs = [self.oracle.P_theta_delta(theta=vertex,delta=delta_hat)
                                           for vertex in node.data.vertices]
                Nvx = node.data.vertices.shape[0]
                vertex_costs = np.array([vertex_inputs_and_costs[i][1] for i in range(Nvx)])
                vertex_inputs = np.array([vertex_inputs_and_costs[i][0] for i in range(Nvx)])
                node.data = NodeData(vertices=node.data.vertices,
                                     commutation=delta_hat,
                                     vertex_costs=vertex_costs,
                                     vertex_inputs=vertex_inputs)
                volume_closed = tools.simplex_volume(node.data.vertices)
                self.status_publisher.update(volume_filled_increment=volume_closed)
    
    def lcss(self,node,location):
        """
        Implementation of [Malyuta2019b] Algorithm 1 lines 4-20. Pass a tree root
        node and this grows the tree until its leaves are associated with an
        epsilon-suboptimal commutation*.
        **Caution**: modifies ``node`` (passed by reference).
        
        * Assuming that no leaf is closed due to a bad condition number.
        
        [Malyuta2019b] D. Malyuta, B. Acikmese, M. Cacan, and D. S. Bayard,
        "Partition-based feasible integer solution pre-computation for hybrid
        model predictive control," in Control Systems Letters (to be submitted),
        IEEE.
        
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
            self.lcss(node.right,location+'1')
    
        def update_vertex_costs(v_mid,v_combo_idx,delta,old_vertex_inputs,old_vertex_costs):
            """
            Compute a new set of optimal costs at the simplex vertices.
            
            Parameters
            ----------
            v_mid : np.array
                New vertex at which the current simplex is to be split into two.
            v_combo_idx : tuple
                Tuple of two elements corresponding to row index of the two vertices
                constituting the longest edge. The first vertex is removed from S_1 and
                the second vertex is removed from S_2, substituted for v_mid.
            delta : np.array
                The commutation that is to be associated with the two new simplices.
            old_vertex_costs : np.array
                Array of existing pre-computed vertex inputs.
            old_vertex_costs : np.array
                Array of existing pre-computed vertex costs.
            """
            u_opt_v_mid, V_delta_v_mid = self.oracle.P_theta_delta(theta=v_mid,delta=delta)[:2]
            vertex_inputs_S_1,vertex_inputs_S_2 = old_vertex_inputs.copy(),old_vertex_inputs.copy()
            vertex_costs_S_1,vertex_costs_S_2 = old_vertex_costs.copy(),old_vertex_costs.copy()
            vertex_inputs_S_1[v_combo_idx[0]] = u_opt_v_mid
            vertex_inputs_S_2[v_combo_idx[1]] = u_opt_v_mid
            vertex_costs_S_1[v_combo_idx[0]] = V_delta_v_mid
            vertex_costs_S_2[v_combo_idx[1]] = V_delta_v_mid
            return vertex_inputs_S_1,vertex_inputs_S_2,vertex_costs_S_1,vertex_costs_S_2
    
        self.status_publisher.update(location=location)
        delta_star,new_vertex_inputs_and_costs = self.oracle.D_delta_R(R=node.data.vertices,
                                                                       V_delta_R=node.data.vertex_costs,
                                                                       delta_ref=node.data.commutation)
        D_delta_R_infeasible = delta_star is None
        if D_delta_R_infeasible:
            # Close leaf
            node.data.is_epsilon_suboptimal = True
            node.data.timestamp = time.time()
            volume_closed = tools.simplex_volume(node.data.vertices)
            self.status_publisher.update(volume_filled_increment=volume_closed)
        else:
            Nvx = node.data.vertices.shape[0]
            new_vertex_costs = np.array([new_vertex_inputs_and_costs[i][1] for i
                                         in range(Nvx)])
            new_vertex_inputs = np.array([new_vertex_inputs_and_costs[i][0] for i
                                          in range(Nvx)])
            if self.oracle.in_variability_ball(R=node.data.vertices,
                                               V_delta_R=node.data.vertex_costs,
                                               delta_ref=node.data.commutation):
                node.data.commutation = delta_star
                node.data.vertex_costs = new_vertex_costs
                node.data.vertex_inputs = new_vertex_inputs
                self.lcss(node,location)
            else:
                S_1,S_2,v_idx = tools.split_along_longest_edge(node.data.vertices)
                # Re-compute vertex costs
                v_mid = S_1[v_idx[0]]
                vertex_inputs_S_1,vertex_inputs_S_2,vertex_costs_S_1,vertex_costs_S_2 = update_vertex_costs(
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
        A loop which waits (by passive blocking) for the roots of a new branch to
        partition to be received from the scheduler process. When received, the
        branch is partitioned. When done, the partitioned branch ("grown tree") is
        sent back to the scheduler. The scheduler is responsible for aborting this
        loop.
        """
        while True:
            # Block until new data is received from scheduler
            tools.debug_print('waiting for data')
            data = MPI.COMM_WORLD.recv(source=global_vars.SCHEDULER_PROC,tag=global_vars.NEW_WORK_TAG)
            tools.debug_print('received data {}'.format(data))
            if data['action']=='stop':
                # Request from scheduler to stop
                return
            elif data['action']=='reset_volume':
                # Reset total volume filled statistic
                self.status_publisher.reset_volume_filled_total()
            else:
                tools.debug_print('got branch at location = %s'%(data['location']))
                # Get data about the branch to be worked on
                branch = data['branch_root']
                branch_location = data['location']
                self.status_publisher.set_new_root_simplex(branch.data.vertices,branch_location,data['action'])
                self.status_publisher.update(active=True)
                # Do work on this branch (i.e. partition this simplex)
                try:
                    tools.debug_print('calling algorithm')
                    self.alg_call(data['action'],branch,branch_location)
                except:
                    self.status_publisher.update(failed=True)
                    raise
                # Save completed branch and notify scheduler that it is available
                with open(global_vars.DATA_DIR+'/branch_%s.pkl'%(data['location']),'wb') as f:
                    pickle.dump(data,f)
                MPI.COMM_WORLD.send(1,dest=global_vars.SCHEDULER_PROC,tag=global_vars.FINISHED_BRANCH_TAG)
                self.status_publisher.update(active=False)

def main():
    """Runs the worker process."""
    worker = Worker()
    worker.spin()
