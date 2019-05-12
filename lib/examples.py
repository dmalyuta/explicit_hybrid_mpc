"""
Numerical examples that the explicit MPC algorithm is demonstrated on.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import numpy as np

import global_vars
import mpc_library
import tools
from oracle import Oracle
from polytope import Polytope

def create_oracle(mpc,set_vrep,abs_frac,abs_err,rel_err):
    """
    Creates the optimization problem oracle.
    
    Parameters
    ----------
    mpc : MPC
        The control law to be used.
    set_vrep : np.array
        Vertex representation of the set to be partitioned (every row is a
        vertex).
    abs_frac : float, optional
        Fraction (0,1) away from the origin of the full size of the invariant
        set where to compute the absolute error.
    abs_err : float, optional
        Absolute error value. If provided, takes precedence over abs_frac.
    rel_err : float
        Relative error value.

    Returns
    -------
    oracle : Oracle
        Optimization problem oracle.
    """
    if abs_err is None:
        oracle = Oracle(mpc,eps_a=1.,eps_r=1.)
        abs_err = np.max([oracle.P_theta(theta=vx)[2]
                          for vx in [abs_frac*vx for vx in set_vrep]])
    oracle = Oracle(mpc,eps_a=abs_err,eps_r=rel_err)
    return oracle

"""
Common parameters/returns for each *_example function below:

Initializes the partition and oracle for the <NAME> example.

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

def satellite_z_example(abs_frac=0.5,abs_err=None,rel_err=2.0):
    # Plant
    sat = mpc_library.SatelliteZ()
    # The set to partition
    full_set = Polytope(R=[(-sat.pars['pos_err_max'],sat.pars['pos_err_max']),
                           (-sat.pars['vel_err_max'],sat.pars['vel_err_max'])])
    full_set_vrep = np.row_stack(full_set.V)
    # Create the optimization problem oracle
    oracle = create_oracle(sat,full_set_vrep,abs_frac,abs_err,rel_err)
    # Initial triangulation
    partition, number_init_simplices, vol = tools.delaunay(full_set_vrep)
    return full_set, partition, oracle

def satellite_xy_example(abs_frac=0.5,abs_err=None,rel_err=2.0):
    # Plant
    sat = mpc_library.SatelliteXY()
    # The set to partition
    full_set = Polytope(R=[(-sat.pars['pos_err_max'],sat.pars['pos_err_max']),
                           (-sat.pars['pos_err_max'],sat.pars['pos_err_max']),
                           (-sat.pars['vel_err_max'],sat.pars['vel_err_max']),
                           (-sat.pars['vel_err_max'],sat.pars['vel_err_max'])])
    full_set_vrep = np.row_stack(full_set.V)
    # Create the optimization problem oracle
    oracle = create_oracle(sat,full_set_vrep,abs_frac,abs_err,rel_err)
    # Initial triangulation
    partition, number_init_simplices, vol = tools.delaunay(full_set_vrep)
    return full_set, partition, oracle
    
def satellite_xyz_example(abs_frac=0.5,abs_err=None,rel_err=2.0):
    # Plant
    sat = mpc_library.SatelliteXYZ()
    # The set to partition
    full_set = Polytope(R=[(-sat.pars['pos_err_max'],sat.pars['pos_err_max']),
                           (-sat.pars['pos_err_max'],sat.pars['pos_err_max']),
                           (-sat.pars['pos_err_max'],sat.pars['pos_err_max']),
                           (-sat.pars['vel_err_max'],sat.pars['vel_err_max']),
                           (-sat.pars['vel_err_max'],sat.pars['vel_err_max']),
                           (-sat.pars['vel_err_max'],sat.pars['vel_err_max'])])
    full_set_vrep = np.row_stack(full_set.V)
    # Create the optimization problem oracle
    oracle = create_oracle(sat,full_set_vrep,abs_frac,abs_err,rel_err)
    # Initial triangulation
    partition, number_init_simplices, vol = tools.delaunay(full_set_vrep)
    return full_set, partition, oracle

def pendulum_example(abs_frac=0.5,abs_err=None,rel_err=2.0):
    # Plant
    pendulum = mpc_library.InvertedPendulumOnCart()
    # Create the set to be partitioned
    # --------------------------------
    # Problem: overlap=0 for ECC feasibility algorithm if we use the vanilla
    # full set. Reason: when dxdt<v_eps and dxdt>v_eps at simplex vertices,
    # the commutation is forced to have the first two elements zero for the
    # first case and the last three elements zero for the seconds case. Hence,
    # ECC will not converge.
    # Solution: manually divide the set into three sections:
    #   1) dxdt>=v_eps
    #   2) dxdt<=-v_eps
    #   3) -v_eps<=dxdt<=v_eps
    full_set = Polytope(R=[(-pendulum.p_err,pendulum.p_err),
                           (-pendulum.ang_err,pendulum.ang_err),
                           (-pendulum.v_err,pendulum.v_err),
                           (-pendulum.rate_err,pendulum.rate_err)])
    set_1 = Polytope(R=[(-pendulum.p_err,pendulum.p_err),
                        (-pendulum.ang_err,pendulum.ang_err),
                        (pendulum.v_eps,pendulum.v_err),
                        (-pendulum.rate_err,pendulum.rate_err)])
    set_2 = Polytope(R=[(-pendulum.p_err,pendulum.p_err),
                        (-pendulum.ang_err,pendulum.ang_err),
                        (-pendulum.v_err,-pendulum.v_eps),
                        (-pendulum.rate_err,pendulum.rate_err)])
    set_3 = Polytope(R=[(-pendulum.p_err,pendulum.p_err),
                        (-pendulum.ang_err,pendulum.ang_err),
                        (-pendulum.v_eps,pendulum.v_eps),
                        (-pendulum.rate_err,pendulum.rate_err)])
    full_set_vrep = np.row_stack(full_set.V)
    set_1_vrep = np.row_stack(set_1.V)
    set_2_vrep = np.row_stack(set_2.V)
    set_3_vrep = np.row_stack(set_3.V)
    # Create the optimization problem oracle
    oracle = create_oracle(pendulum,full_set_vrep,abs_frac,abs_err,rel_err)
    # Initial triangulation
    partition_set_1,_,_ = tools.delaunay(set_1_vrep)
    partition_set_2,_,_ = tools.delaunay(set_2_vrep)
    partition_set_3,_,_ = tools.delaunay(set_3_vrep)
    partition = partition_set_1
    tools.join_triangulation(partition_set_2,partition_set_3)
    tools.join_triangulation(partition_set_1,partition_set_2)
    return full_set, partition_set_1, oracle

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
