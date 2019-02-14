"""
Create a binary tree partition of the task space.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import numpy as np

from oracle import Oracle
from partition import NodeData, Tree

# Parameters
eps_a = 1. # Absolute error tolerance
eps_r = 0.1 # Relative error tolerance

def ecc(oracle,R):
    c_R = np.average(R.data.vertices,axis=0) # barycenter
    if oracle.P_theta(theta=c_R,check_feasibility=True):
        raise RuntimeError('STOP, Theta contains infeasible regions')
    else:
        delta_hat = oracle.V_R(R)
        #TODO L10 onwards of ECC

side = 0.05
center = np.array([0.1,0.1])
Theta = np.array([center+np.array([-side,-side]),
                  center+np.array([side,0.]),
                  center+np.array([0.,side])])

root = Tree(data=NodeData(vertices=Theta))
oracle = Oracle(eps_a=eps_a,eps_r=eps_r)
