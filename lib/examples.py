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
        abs_err = np.max([oracle.P_theta(theta=vx)[2] for vx in [abs_frac*vx for vx in Theta]])
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
        abs_err = np.max([oracle.P_theta(theta=vx)[2] for vx in [abs_frac*vx for vx in Theta]])
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
        abs_err = np.max([oracle.P_theta(theta=vx)[2] for vx in [abs_frac*vx for vx in Theta]])
    oracle = Oracle(sat,eps_a=abs_err,eps_r=rel_err)
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
    else:
        raise ValueError('Unknown example (%s)'%(global_vars.EXAMPLE))
