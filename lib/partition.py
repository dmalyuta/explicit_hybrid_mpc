"""
Create a binary tree partition of the task space.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import sys
sys.path.append('../')

import os
import glob
import time
import fcntl
import pickle
import numpy as np

import global_vars
from general import warning
from tree import NodeData, Tree
from tools import split_along_longest_edge, simplex_volume, simplex_condition_number, get_nodes_in_queue
from polytope import Polytope

def ecc(oracle,node,location,animator=None,progressbar=None):
    """
    Implementation of [Malyuta2019a] Algorithm 2 lines 4-16. Pass a tree root
    node and this grows the tree until its leaves are feasible partition cells.
    **Caution**: modifies ``node`` (passed by reference).
    
    [Malyuta2019a] D. Malyuta, B. Açıkmeşe, M. Cacan, and D. S. Bayard,
    "Partition-based feasible integer solution pre-computation for hybrid model
    predictive control," in 2019 European Control Conference (in review), IFAC,
    jun 2019.
    
    Parameters
    ----------
    oracle : Oracle
        Container of optimization problems used by the partitioning process.
    node : Tree
        Tree root. Sufficient that it just holds node.data.vertices for the
        simplex vertices.
    location : string
        Location in tree of this node. String where '0' at index i means take
        left child, '1' at index i means take right child (at depth value i).
    animator : Animator, optional
        Pass an Animator object to animate the simplicial partitioning.
    progressbar : Progress, optional
        Pass a Progress object to print partitioning progress.
    """
    if animator is not None:
        # Animation preferences
        anim_open_leaf = dict(edgecolor='black',facecolor='none',linewidth=0.1)
        # Use "safe" rounding to int (so that conversion to int does not floor 0.99999999 down to 0)
        anim_closed_leaf = lambda delta: dict(facecolor=animator.get_color((delta>0.5).astype(int)))
    
    c_R = np.average(node.data.vertices,axis=0) # Simplex barycenter
    if not oracle.P_theta(theta=c_R,check_feasibility=True):
        raise RuntimeError('STOP, Theta contains infeasible regions')
    else:
        delta_hat = oracle.V_R(node.data.vertices)
        if delta_hat is None:
            S_1,S_2 = split_along_longest_edge(node.data.vertices)[:2]
            if animator is not None:
                animator.update(Polytope(V=S_1,A=False),**anim_open_leaf)
                animator.update(Polytope(V=S_2,A=False),**anim_open_leaf)
            child_left = NodeData(vertices=S_1)            
            child_right = NodeData(vertices=S_2)
            node.grow(child_left,child_right)
            if progressbar is not None:
                progressbar.update(count_increment=1)
            # Recursive call for each resulting simplex
            if len(get_nodes_in_queue())<global_vars.N_PROC:
                # Save the right child to a separate file, to be partitioned by
                # another thread
                pickle.dump(node.left,open(global_vars.NODE_DIR+('node_%s.pkl'%(location+'0')),'wb'))
            else:
                ecc(oracle,node.left,location+'0',animator,progressbar)
            ecc(oracle,node.right,location+'1',animator,progressbar)
        else:
            # Assign feasible commutation to simplex
            vertex_inputs_and_costs = [oracle.P_theta_delta(theta=vertex,delta=delta_hat)
                                       for vertex in node.data.vertices]
            Nvx = node.data.vertices.shape[0]
            vertex_costs = np.array([vertex_inputs_and_costs[i][1] for i in range(Nvx)])
            vertex_inputs = np.array([vertex_inputs_and_costs[i][0] for i in range(Nvx)])
            node.data = NodeData(vertices=node.data.vertices,
                                 commutation=delta_hat,
                                 vertex_costs=vertex_costs,
                                 vertex_inputs=vertex_inputs)
            if animator is not None:
                animator.update(Polytope(V=node.data.vertices,A=False),**anim_closed_leaf(delta_hat))
            if progressbar is not None:
                volume_closed = simplex_volume(node.data.vertices)
                progressbar.update(volume_increment=volume_closed)
                
def lcss(oracle,node,animator=None,progressbar=None):
    """
    Implementation of [Malyuta2019b] Algorithm 1 lines 4-20. Pass a tree root
    node and this grows the tree until its leaves are associated with an
    epsilon-suboptimal commutation*.
    **Caution**: modifies ``node`` (passed by reference).
    
    * Assuming that no leaf is closed due to a bad condition number.
    
    [Malyuta2019b] D. Malyuta, B. Açıkmeşe, M. Cacan, and D. S. Bayard,
    "Partition-based feasible integer solution pre-computation for hybrid
    model predictive control," in Control Systems Letters (to be submitted),
    IEEE.
    
    Parameters
    ----------
    oracle : Oracle
        Container of optimization problems used by the partitioning process.
    node : Tree
        Tree root. Sufficient that it just holds node.data.vertices for the
        simplex vertices.
    animator : Animator
        Pass an Animator object to animate the simplicial partitioning.
    progressbar : Progress
        Pass a Progress object to print partitioning progress.
    """
    if animator is not None:
        # Animation preferences
        anim_open_leaf = dict(edgecolor='black',facecolor='none',linewidth=0.1)
        # Use "safe" rounding to int (so that conversion to int does not floor 0.99999999 down to 0)
        anim_closed_leaf = lambda delta: dict(facecolor=animator.get_color((delta>0.5).astype(int)))
        
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
        if animator is not None:
            animator.update(Polytope(V=child_left.vertices,A=False),**anim_open_leaf)
            animator.update(Polytope(V=child_right.vertices,A=False),**anim_open_leaf)
        if progressbar is not None:
            progressbar.update(count_increment=1)
        for child in [node.left,node.right]:
            # Add open leaf
            lcss(oracle,child,animator,progressbar)

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
        u_opt_v_mid, V_delta_v_mid = oracle.P_theta_delta(theta=v_mid,delta=delta)[:2]
        vertex_inputs_S_1,vertex_inputs_S_2 = old_vertex_inputs.copy(),old_vertex_inputs.copy()
        vertex_costs_S_1,vertex_costs_S_2 = old_vertex_costs.copy(),old_vertex_costs.copy()
        vertex_inputs_S_1[v_combo_idx[0]] = u_opt_v_mid
        vertex_inputs_S_2[v_combo_idx[1]] = u_opt_v_mid
        vertex_costs_S_1[v_combo_idx[0]] = V_delta_v_mid
        vertex_costs_S_2[v_combo_idx[1]] = V_delta_v_mid
        return vertex_inputs_S_1,vertex_inputs_S_2,vertex_costs_S_1,vertex_costs_S_2

    bar_e_a_R, bar_e_r_R = oracle.bar_E_ar_R(R=node.data.vertices,
                                             V_delta_R=node.data.vertex_costs,
                                             delta_ref=node.data.commutation)
    infeasible = np.isinf(bar_e_a_R)
    eps_suboptimal = bar_e_a_R<=oracle.eps_a or bar_e_r_R<=oracle.eps_r
    if infeasible or eps_suboptimal:
        # Close leaf
        node.data.timestamp = time.time()
        if animator is not None:
            animator.update(Polytope(V=node.data.vertices,A=False),
                            **anim_closed_leaf(node.data.commutation))
        if progressbar is not None:
            volume_closed = simplex_volume(node.data.vertices)
            progressbar.update(volume_increment=volume_closed)
    else:
        delta_star = oracle.D_delta_R(R=node.data.vertices,
                                      V_delta_R=node.data.vertex_costs,
                                      delta_ref=node.data.commutation)
        D_delta_R_feasible = delta_star is not None
        if D_delta_R_feasible:
            new_vertex_inputs_and_costs = [oracle.P_theta_delta(theta=vertex,delta=delta_star)
                                           for vertex in node.data.vertices]
            Nvx = node.data.vertices.shape[0]
            new_vertex_costs = np.array([new_vertex_inputs_and_costs[i][1] for i
                                         in range(Nvx)])
            new_vertex_inputs = np.array([new_vertex_inputs_and_costs[i][0] for i
                                          in range(Nvx)])
            if oracle.in_variability_ball(R=node.data.vertices,
                                          V_delta_R=node.data.vertex_costs,
                                          delta_ref=node.data.commutation):
                node.data.commutation = delta_star
                node.data.vertex_costs = new_vertex_costs
                node.data.vertex_inputs = new_vertex_inputs
                lcss(oracle,node,animator,progressbar)
            else:
                S_1,S_2,v_idx = split_along_longest_edge(node.data.vertices)
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
        else:
            S_1,S_2,v_idx = split_along_longest_edge(node.data.vertices)
            # Re-compute vertex costs
            v_mid = S_1[v_idx[0]]
            vertex_inputs_S_1,vertex_inputs_S_2,vertex_costs_S_1,vertex_costs_S_2 = update_vertex_costs(
                        v_mid,v_idx,node.data.commutation,
                        node.data.vertex_inputs,node.data.vertex_costs)
            # Make children
            child_left = NodeData(vertices=S_1,
                                  commutation=node.data.commutation,
                                  vertex_costs=vertex_costs_S_1,
                                  vertex_inputs=vertex_inputs_S_1)
            child_right = NodeData(vertices=S_2,
                                   commutation=node.data.commutation,
                                   vertex_costs=vertex_costs_S_2,
                                   vertex_inputs=vertex_inputs_S_2)
            add(child_left,child_right)

def algorithm_call(algorithm,oracle,root,location,**kwargs):
    """
    Grow an initial binary tree given by root using algorithm() (either
    ``ecc()`` or ``lcss()``).
    
    Parameters
    ----------
    algorithm : callable
        Either ``ecc()`` or ``lcss()``.
    oracle : Oracle
        Container of optimization problems used by the partitioning process.
    node : Tree
        Tree root. Sufficient that it just holds node.data.vertices for the
        simplex vertices.
    **kwargs : dict
        Passed on to algorithm().
    """
    if root.is_leaf():
        algorithm(oracle,root,location,**kwargs)
    else:
        algorithm_call(algorithm,oracle,root.left,location+'0',**kwargs)
        algorithm_call(algorithm,oracle,root.right,location+'1',**kwargs)
            
def spinner(proc_num,algorithm_call,wait_time=5.):
    """
    Parameters
    ----------
    proc_num : int
        Process number.
    algorithm_call : callable
        Callable algorithm with signature algorithm_call(root,location) where
        root (Tree) is the subtree root and location (string) is the subtree
        root's location in the main tree.
    wait_time : float, optional
        How many seconds to wait before checking the tree node queue.
    """
    def set_status(active):
        """
        Set process status.
        
        Parameters
        ----------
        active : int
            Process status (1: active, 0: idle).
        """
        status = open('status.txt','r').readline()
        status = status[:proc_num]+str(active)+status[proc_num+1:]
        open('status.txt','w').write(status)
    
    def append_to_tree(tree,branch,location):
        """
        Appends branch to tree at location.
        **Caution**: modifies ``tree`` (passed by reference).
        
        Parameters
        ----------
        tree : Tree
            The tree to append to.
        branch : Tree
            The branch to be appended.
        location : string
            The location at which the branch is to be appended, specified as a
            string of '0' and '1' where '0' at index i means take left child,
            '1' at index i means take right child. Index i corresponds to depth
            level i.
        """
        # Browse to location in tree
        cursor = tree
        for i in range(len(location)):
            cursor = cursor.left if location[i]=='0' else cursor.right
        # At this point, cursor.data=branch.data (same starting node)
        if not branch.is_leaf():
            cursor.left = branch.left
            cursor.right = branch.right
        else:
            cursor.data = branch.data
    
    mutex = open('mutex','w')
    while True:
        time.sleep(wait_time)
        nodes_in_queue = get_nodes_in_queue()
        #print(nodes_in_queue)
        #print(os.listdir('./data/'))
        if len(nodes_in_queue)>0:
            # Try to get a node from the queue
            busy = False
            try:
                fcntl.lockf(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except IOError:
                # Another process is currently reading the queue. Check back later
                busy = True
            if busy:
                continue
            #print('Reading %s'%(nodes_in_queue[0]))
            subtree_file = open(nodes_in_queue[0],'rb')
            subtree = pickle.load(subtree_file)
            subtree_location = nodes_in_queue[0][len(global_vars.NODE_DIR+'node_'):-4]
            os.remove(nodes_in_queue[0])
            set_status(1)
            fcntl.lockf(mutex,fcntl.LOCK_UN)
            # Partition this node
            algorithm_call(subtree,subtree_location)
            # Update the full tree
            #print('Done... ')
            fcntl.lockf(mutex,fcntl.LOCK_EX)
            #print('saving... ')
            maintree = pickle.load(open(global_vars.TREE_FILE,'rb'))
            append_to_tree(maintree,subtree,subtree_location)
            pickle.dump(maintree,open(global_vars.TREE_FILE,'wb'))
            set_status(0)
            fcntl.lockf(mutex,fcntl.LOCK_UN)
            #print('saved')
    mutex.close()