"""
Create a binary tree partition of the task space.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import numpy as np
import numpy.linalg as la
import itertools

from oracle import Oracle
from partition import NodeData, Tree

# Parameters
eps_a = 1. # Absolute error tolerance
eps_r = 0.1 # Relative error tolerance

def ecc(oracle,node):
    """
    Implementation of [Malyuta2019a] Algorithm 2 lines 4-16. Pass a tree root
    node and this grows the tree until its leaves are feasible partition cells.
    **Caution**: modifies ``node`` (passed by reference).
    
    Parameters
    ----------
    oracle : Oracle
        Container of optimization problems used by the partitioning process.
    node : Tree
        Tree root. Sufficient that it just holds node.data.vertices for the
        simplex vertices.
    """
    c_R = np.average(node.data.vertices,axis=0) # Simplex barycenter
    if not oracle.P_theta(theta=c_R,check_feasibility=True):
        raise RuntimeError('STOP, Theta contains infeasible regions')
    else:
        delta_hat = oracle.V_R(node.data.vertices)
        if delta_hat is None:
            # Find midpoint along longest edge of simplex
            N_vertices = node.data.vertices.shape[0]
            vertex_combinations = list(itertools.combinations(range(N_vertices),2))
            longest_edge_idx = np.argmax([la.norm(node.data.vertices[vx_combo[0]]-
                                                  node.data.vertices[vx_combo[1]])
                                          for vx_combo in vertex_combinations])
            longest_edge_combo = vertex_combinations[longest_edge_idx]
            bar_v = node.data.vertices[longest_edge_combo[0]]
            bar_v_prime = node.data.vertices[longest_edge_combo[1]]
            v_mid = (bar_v+bar_v_prime)/2.
            # Split simplex in half at the midpoint
            vertex_set_1 = node.data.vertices.copy()
            vertex_set_1[longest_edge_combo[0]] = v_mid
            vertex_set_2 = node.data.vertices.copy()
            vertex_set_2[longest_edge_combo[1]] = v_mid
            child_left = NodeData(vertices=vertex_set_1)            
            child_right = NodeData(vertices=vertex_set_2)
            node.grow(child_left,child_right)
            # Recursive call for each resulting simplex
            ecc(oracle,node.left)
            ecc(oracle,node.right)
        else:
            # Assign feasible commutation to simplex
            vertex_costs = np.array([oracle.P_theta_delta(theta=vertex,delta=delta_hat)
                                     for vertex in node.data.vertices])
            node.data = NodeData(vertices=node.data.vertices,
                                 commutation=delta_hat,
                                 vertex_costs=vertex_costs)
            
#def cdc(oracle,tree):
#    """
#    
#    """

#side = 0.05
#center = np.array([0.1,0.1])
#Theta = np.array([center+np.array([-side,-side]),
#                  center+np.array([side,0.]),
#                  center+np.array([0.,side])])
Theta = np.array([[0.0,0.0],[0.05,0.0],[0.0,0.04]])

root = Tree(data=NodeData(vertices=Theta))
oracle = Oracle(eps_a=eps_a,eps_r=eps_r)

ecc(oracle,root)