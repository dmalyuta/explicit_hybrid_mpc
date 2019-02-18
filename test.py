"""
Algorithm test script.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import numpy as np
import numpy.linalg as la

import sys
sys.path.append('lib/')
from mpc_library import SatelliteZ, RandomSystem
from oracle import Oracle
from polytope import Polytope
from tools import Progress, Animator, delaunay
from partition import algorithm_call, ecc, lcss

test = 'random' # satellite, random

if test=='satellite':
    # MPC law
    sat = SatelliteZ()
    
    # Parameters
    absolute_error = sat.N*la.norm(la.inv(sat.D_u)*sat.pars['delta_v_min'])**2 # Absolute error tolerance
    relative_error = 0.1 # Relative error tolerance
    cond_max = 1000. # Maximum simplex condition number for epsilon-suboptimal partition
    
    # Oracle problems
    oracle = Oracle(sat, eps_a=absolute_error, eps_r=relative_error)
    
    # Animation
    animator = None#Animator(1,sat)
    
    # The set to partition
    Theta = Polytope(R=[(-sat.pars['pos_err_max'],sat.pars['pos_err_max']),
                        (-sat.pars['vel_err_max'],sat.pars['vel_err_max'])])
    #animator.update(Theta,facecolor='none',edgecolor='black',linewidth=0.5,linestyle='--')    
    Theta = np.row_stack(Theta.V)
    
    # Initial triangulation
    partition, number_init_simplices, vol = delaunay(Theta)
    
    # Progress bar
    progressbar = Progress(vol,number_init_simplices)
    
    # Feasible partition
    algorithm_call(ecc,oracle,partition,animator=animator,progressbar=progressbar)
    
    # Reset progressbar
    progressbar.volume_closed = 0.
    progressbar.open_count = progressbar.closed_count
    progressbar.closed_count = 0
    
    # Epsilon-suboptimal partition
    algorithm_call(lcss,oracle,partition,
                   eps_a=absolute_error,eps_r=relative_error,rho_max=cond_max,
                   animator=animator,progressbar=progressbar)
elif test=='random':
    # MPC law
    sys = RandomSystem(n_x=2)
    
    # Parameters
    absolute_error = 1e-3 # Absolute error tolerance
    relative_error = 10e-2 # Relative error tolerance
    cond_max = 1000. # Maximum simplex condition number for epsilon-suboptimal partition
    
    # Fix absolute error and generate oracle
    oracle = Oracle(sys, eps_a=absolute_error, eps_r=relative_error)
    origin_neighborhood_frac = 0.01
    absolute_error = np.max([oracle.P_theta(theta=vx)[2] for vx in
                      [origin_neighborhood_frac*vx for vx in sys.rpi.V]])
    oracle = Oracle(sys, eps_a=absolute_error, eps_r=relative_error)
    
    # Animation
    animator = Animator(1,sys)
    
    # The set to partition
    Theta = sys.rpi
    animator.update(Theta,facecolor='none',edgecolor='black',linewidth=0.5,linestyle='--')    
    Theta = np.row_stack(Theta.V)
    
    # Initial triangulation
    partition, number_init_simplices, vol = delaunay(Theta)
    
    # Progress bar
    progressbar = Progress(vol,number_init_simplices)
    
    # Feasible partition
    algorithm_call(ecc,oracle,partition,animator=animator,progressbar=progressbar)
    
    # Reset progressbar
    progressbar.volume_closed = 0.
    progressbar.open_count = progressbar.closed_count
    progressbar.closed_count = 0
    
    # Epsilon-suboptimal partition
    algorithm_call(lcss,oracle,partition,
                   eps_a=absolute_error,eps_r=relative_error,rho_max=cond_max,
                   animator=animator,progressbar=progressbar)
else:
    raise ValueError('Unknown test value (%s)'%(test))