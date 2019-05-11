"""
Numerical examples that the explicit MPC algorithm is demonstrated on.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import numpy as np

import global_vars
import mpc_library
from oracle import Oracle
from polytope import Polytope
from tools import delaunay

def satellite_z_example(abs_frac=0.5,abs_err=None,rel_err=2.0):
    """
    Initializes the partition and oracle for the SatelliteZ example.
    
    Parameters
    ----------
    abs_frac : float, optional
        Fraction (0,1) away from the origin of the full size of the invariant
        set where to compute the absolute error.
    abs_err : float, optional
        Absolute error value. If provided, takes precedence over abs_frac.
    rel_err : float
        Relative error value.
        
    Returns
    -------
    full_set : Polytope
        The set to be partitioned.
    partition : Tree
        The initial invariant set, pre-partitioned into simplices via Delaunay
        triangulation.
    oracle : Oracle
        The optimization problem oracle.
    """
    # Plant
    sat = mpc_library.SatelliteZ()
    # The set to partition
    full_set = Polytope(R=[(-sat.pars['pos_err_max'],sat.pars['pos_err_max']),
                           (-sat.pars['vel_err_max'],sat.pars['vel_err_max'])])
    Theta = np.row_stack(full_set.V)
    # Create the optimization problem oracle
    if abs_err is None:
        oracle = Oracle(sat,eps_a=1.,eps_r=1.)
        abs_err = np.max([oracle.P_theta(theta=vx)[2]
                          for vx in [abs_frac*vx for vx in Theta]])
    oracle = Oracle(sat,eps_a=abs_err,eps_r=rel_err)
    # Initial triangulation
    partition, number_init_simplices, vol = delaunay(Theta)
    return full_set, partition, oracle

def satellite_xy_example(abs_frac=0.5,abs_err=None,rel_err=2.0):
    """
    Initializes the partition and oracle for the SatelliteXY example.
    
    Parameters
    ----------
    abs_frac : float, optional
        Fraction (0,1) away from the origin of the full size of the invariant
        set where to compute the absolute error.
    abs_err : float, optional
        Absolute error value. If provided, takes precedence over abs_frac.
    rel_err : float
        Relative error value.
        
    Returns
    -------
    full_set : Polytope
        The set to be partitioned.
    partition : Tree
        The initial invariant set, pre-partitioned into simplices via Delaunay
        triangulation.
    oracle : Oracle
        The optimization problem oracle.
    """
    # Plant
    sat = mpc_library.SatelliteXY()
    # The set to partition
    full_set = Polytope(R=[(-sat.pars['pos_err_max'],sat.pars['pos_err_max']),
                           (-sat.pars['pos_err_max'],sat.pars['pos_err_max']),
                           (-sat.pars['vel_err_max'],sat.pars['vel_err_max']),
                           (-sat.pars['vel_err_max'],sat.pars['vel_err_max'])])
    Theta = np.row_stack(full_set.V)
    # Create the optimization problem oracle
    if abs_err is None:
        oracle = Oracle(sat,eps_a=1.,eps_r=1.)
        abs_err = np.max([oracle.P_theta(theta=vx)[2]
                          for vx in [abs_frac*vx for vx in Theta]])
    oracle = Oracle(sat,eps_a=abs_err,eps_r=rel_err)
    # Initial triangulation
    partition, number_init_simplices, vol = delaunay(Theta)
    return full_set, partition, oracle
    
def satellite_xyz_example(abs_frac=0.5,abs_err=None,rel_err=2.0):
    """
    Initializes the partition and oracle for the SatelliteXYZ example.
    
    Parameters
    ----------
    abs_frac : float, optional
        Fraction (0,1) away from the origin of the full size of the invariant
        set where to compute the absolute error.
    abs_err : float, optional
        Absolute error value. If provided, takes precedence over abs_frac.
    rel_err : float
        Relative error value.
        
    Returns
    -------
    full_set : Polytope
        The set to be partitioned.
    partition : Tree
        The initial invariant set, pre-partitioned into simplices via Delaunay
        triangulation.
    oracle : Oracle
        The optimization problem oracle.
    """
    # Plant
    sat = mpc_library.SatelliteXYZ()
    # The set to partition
    full_set = Polytope(R=[(-sat.pars['pos_err_max'],sat.pars['pos_err_max']),
                           (-sat.pars['pos_err_max'],sat.pars['pos_err_max']),
                           (-sat.pars['pos_err_max'],sat.pars['pos_err_max']),
                           (-sat.pars['vel_err_max'],sat.pars['vel_err_max']),
                           (-sat.pars['vel_err_max'],sat.pars['vel_err_max']),
                           (-sat.pars['vel_err_max'],sat.pars['vel_err_max'])])
    Theta = np.row_stack(full_set.V)
    # Create the optimization problem oracle
    if abs_err is None:
        oracle = Oracle(sat,eps_a=1.,eps_r=1.)
        abs_err = np.max([oracle.P_theta(theta=vx)[2]
                          for vx in [abs_frac*vx for vx in Theta]])
    oracle = Oracle(sat,eps_a=abs_err,eps_r=rel_err)
    # Initial triangulation
    partition, number_init_simplices, vol = delaunay(Theta)
    return full_set, partition, oracle

def pendulum_example(abs_frac=0.5,abs_err=None,rel_err=2.0):
    """
    Initializes the partition and oracle for the InvertedPendulumOnCart example.
    
    Parameters
    ----------
    abs_frac : float, optional
        Fraction (0,1) away from the origin of the full size of the invariant
        set where to compute the absolute error.
    abs_err : float, optional
        Absolute error value. If provided, takes precedence over abs_frac.
    rel_err : float
        Relative error value.
        
    Returns
    -------
    full_set : Polytope
        The set to be partitioned.
    partition : Tree
        The initial invariant set, pre-partitioned into simplices via Delaunay
        triangulation.
    oracle : Oracle
        The optimization problem oracle.
    """
    # Plant
    pendulum = mpc_library.InvertedPendulumOnCart()
    # The set to partition
    full_set = pendulum.Theta
    Theta = np.row_stack(full_set.V)
    # Create the optimization problem oracle
    if abs_err is None:
        oracle = Oracle(pendulum,eps_a=1.,eps_r=1.)
        abs_err = np.max([oracle.P_theta(theta=vx)[2]
                          for vx in [abs_frac*vx for vx in Theta]])
    oracle = Oracle(pendulum,eps_a=abs_err,eps_r=rel_err)
    # Initial triangulation
    partition, number_init_simplices, vol = delaunay(Theta)
    return full_set, partition, oracle

def example(*args,**kwargs):
    """
    Wrapper which allows to globally set which example is to be used. Accepts
    and returns values documented in the respective function above that is
    called.
    """
    if global_vars.EXAMPLE=='cwh_z':
        return satellite_z_example(*args,**kwargs)
    elif global_vars.EXAMPLE=='cwh_xy':
        return satellite_xy_example(*args,**kwargs)
    elif global_vars.EXAMPLE=='cwh_xyz':
        return satellite_xyz_example(*args,**kwargs)
    elif global_vars.EXAMPLE=='pendulum':
        return pendulum_example(*args,**kwargs)
    else:
        raise ValueError('Unknown example (%s)'%(global_vars.EXAMPLE))

import cvxpy as cvx

global_vars.MPC_HORIZON = 10
#global_vars.SOLVER_OPTIONS = dict(solver=cvx.ECOS, verbose=False)
full_set, partition, oracle = pendulum_example()

#print(oracle.P_theta(np.array([0.1,0,0,0])))
#import sys
#sys.exit()

from simulator import Simulator

plant = mpc_library.InvertedPendulumOnCart()
def mpc(x):
    u,_,_,t = oracle.P_theta(x)
    print(x,u)
    return u,t
x_init = np.array([0.1,0,0,0])
T = 1 # [s] Simulation duration
simulator = Simulator(mpc,plant,T)
simout = simulator.run(x_init,label='pendulum')

#%%

import numpy.linalg as la
import matplotlib.pyplot as plt

fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot(111)
ax.plot(simout.t,la.norm(simout.u,axis=0),
        color='orange',linestyle='none',marker='.',markersize=10)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Input norm')
ax.set_xlim([0,simout.t[-1]])
plt.tight_layout()
plt.show(block=False)

fig = plt.figure(2)
plt.clf()
ax = fig.add_subplot(111)
ax.plot(simout.t,simout.x[0])
ax.plot(simout.t,simout.x[1])
ax.plot(simout.t,simout.x[2])