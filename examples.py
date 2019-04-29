"""
Numerical examples that the explicit MPC algorithm is demonstrated on.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import sys
sys.path.append('./lib/')

import numpy as np

from mpc_library import SatelliteZ,SatelliteXY
from oracle import Oracle
from polytope import Polytope
from tools import delaunay

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
    partition : Tree
        The initial invariant set, pre-partitioned into simplices via Delaunay
        triangulation.
    oracle : Oracle
        The optimization problem oracle.
    """
    # Plant
    sat = SatelliteXY()
    # The set to partition
    Theta = Polytope(R=[(-sat.pars['pos_err_max'],sat.pars['pos_err_max']),
                        (-sat.pars['pos_err_max'],sat.pars['pos_err_max']),
                        (-sat.pars['vel_err_max'],sat.pars['vel_err_max']),
                        (-sat.pars['vel_err_max'],sat.pars['vel_err_max'])])
    Theta = np.row_stack(Theta.V)
    # Create the optimization problem oracle
    if abs_err is None:
        oracle = Oracle(sat,eps_a=1.,eps_r=1.,kind='explicit')
        abs_err = np.max([oracle.P_theta(theta=vx)[2] for vx in [abs_frac*vx for vx in Theta]])
    oracle = Oracle(sat,eps_a=abs_err,eps_r=rel_err,kind='explicit')
    # Initial triangulation
    partition, number_init_simplices, vol = delaunay(Theta)
    return partition, oracle

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
    partition : Tree
        The initial invariant set, pre-partitioned into simplices via Delaunay
        triangulation.
    oracle : Oracle
        The optimization problem oracle.
    """
    # Plant
    sat = SatelliteZ()
    # The set to partition
    Theta = Polytope(R=[(-sat.pars['pos_err_max'],sat.pars['pos_err_max']),
                        (-sat.pars['vel_err_max'],sat.pars['vel_err_max'])])
    Theta = np.row_stack(Theta.V)
    # Create the optimization problem oracle
    if abs_err is None:
        oracle = Oracle(sat,eps_a=1.,eps_r=1.,kind='explicit')
        abs_err = np.max([oracle.P_theta(theta=vx)[2] for vx in [abs_frac*vx for vx in Theta]])
    oracle = Oracle(sat,eps_a=abs_err,eps_r=rel_err,kind='explicit')
    # Initial triangulation
    partition, number_init_simplices, vol = delaunay(Theta)
    return partition, oracle
    
def example(*args,**kwargs):
    """
    Wrapper which allows to globally set which example is to be used. Accepts
    and returns values documented in the respective function above that is
    called.
    """
#    return satellite_z_example(*args,**kwargs)
    return satellite_xy_example(*args,**kwargs)