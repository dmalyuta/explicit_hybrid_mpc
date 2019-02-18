"""
Generate data for L-CSS 2019 paper example.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import pickle
import time
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

import sys
sys.path.append('lib/')
from mpc_library import SatelliteZ
from oracle import Oracle
from polytope import Polytope
from tools import Progress, delaunay
from partition import algorithm_call, ecc, lcss
from simulator import Simulator

existing_data = 'data/all_partitions.pkl' # None or filepath

# MPC law
sat = SatelliteZ()

if existing_data is None:
    # Parameters
    Nres = 5 # How many "resolutions" to test
    rho_max = 1000 # Maximum condition number
    origin_neighborhood_fracs = np.linspace(0.1,0.5,Nres)
    relative_errors = np.linspace(0.1,2.0,Nres)
    
    # The set to partition
    Theta = Polytope(R=[(-sat.pars['pos_err_max'],sat.pars['pos_err_max']),
                        (-sat.pars['vel_err_max'],sat.pars['vel_err_max'])])
    Theta = np.row_stack(Theta.V)
    
    # Generate oracles
    absolute_errors = np.empty(Nres) # Filled in based on origin_neighborhood_fracs
    oracles = dict(explicit=[],semiexplicit=[])
    for i in range(Nres):
        for kind in ['explicit','semiexplicit']:
            # Make an oracle with a temporary absolute error
            oracle = Oracle(sat,eps_a=1.,eps_r=origin_neighborhood_fracs[i],kind=kind)
            absolute_errors[i] = np.max([oracle.P_theta(theta=vx)[2] for vx in
                                        [origin_neighborhood_fracs[i]*vx for vx in Theta]])
            oracles[kind].append(Oracle(sat,eps_a=absolute_errors[i],eps_r=relative_errors[i],kind=kind))
    
    # Generate partition for each error tolerance
    partitions = dict(explicit=[],semiexplicit=[])
    runtimes = dict(explicit=[],semiexplicit=[])
    for i in range(Nres):
        for kind in ['explicit','semiexplicit']:
            # Initial triangulation
            partition, number_init_simplices, vol = delaunay(Theta)
            
            # Progress bar
            progressbar = Progress(vol,number_init_simplices)
            
            # Feasible partition
            algorithm_call(ecc,oracles[kind][i],partition,progressbar=progressbar)
            
            # Reset progressbar
            progressbar.volume_closed = 0.
            progressbar.open_count = progressbar.closed_count
            progressbar.closed_count = 0
            
            # Epsilon-suboptimal partition
            t_start = time.time()
            algorithm_call(lcss,oracles[kind][i],partition,rho_max=rho_max,progressbar=progressbar)
            t_elapsed = time.time()-t_start
            runtimes[kind].append(t_elapsed)
            
            # Save the partition to a list and to a file
            partitions[kind].append(partition)
            pickle.dump(dict(absolute_error=absolute_errors[i],
                             relative_error=relative_errors[i],
                             origin_neighborhood_frac=origin_neighborhood_fracs[i],
                             partition=partition,
                             runtime=t_elapsed),open('data/partition_%s_%d.pkl'%(kind,i),'wb'))
    
    # Save the data
    pickle.dump(dict(absolute_errors=absolute_errors,
                     relative_errors=relative_errors,
                     origin_neighborhood_fracs=origin_neighborhood_fracs,
                     partitions=partitions),open('data/all_partitions.pkl','wb'))
else:
    data = pickle.load(open(existing_data,'rb'))
    absolute_errors = data['absolute_errors']
    relative_errors = data['relative_errors']
    origin_neighborhood_fracs = data['origin_neighborhood_fracs']
    partitions = data['partitions']
    Nres = len(absolute_errors)
    
    # Re-make the oracles
    sat = SatelliteZ()
    oracles = dict(explicit=[],semiexplicit=[])
    for i in range(Nres):
        for kind in ['explicit','semiexplicit']:
            oracles[kind].append(Oracle(sat,eps_a=absolute_errors[i],eps_r=relative_errors[i],kind=kind))
    
#%% Post-process
    
# Generate halfspace representation for each "left" child
def gen_hrep(root):
    """
    Generate halfspace representation of the "left" children in the binary
    tree (the right children do not need one thanks to mutual exclusivity).
    
    Parameters
    ----------
    root : Tree
        Tree root.
    """
    def hrep(V):
        """Convert vertex to halspace representation"""
        poly = Polytope(V=V)
        A,b = poly.A,poly.b
        return A,b
        
    if root.top:
        root.data.hrep_A,root.data.hrep_b = hrep(root.data.vertices)
        
    if not root.is_leaf():
        root.left.data.hrep_A,root.left.data.hrep_b = hrep(root.left.data.vertices)
        gen_hrep(root.left)
        gen_hrep(root.right)

for i in range(Nres):
    for kind in ['explicit','semiexplicit']:
        gen_hrep(partitions[kind][i])    

#%% Maximum tree depth plot

# TODO recursive function
    
#%% Tree depth vs. distance from origin of simplex center
    
# TODO recursive function
    
#%% Simulate implicit vs. semi-explicit MPC
# - for each accuracy level
# - ...
    
def f_delta_epsilon(root,theta):
    """
    Map theta to epsilon-suboptimal commutation and input.
    
    Parameters
    ----------
    root : Tree
        Tree root.
    theta : np.array
        Parameter value.
        
    Returns
    -------
    delta : np.array
        Epsilon-suboptimal commutation at theta.
    vertices : np.array
        Vertices of the simplex that theta is contained in.
    inputs : np.array
        Optimal inputs associated with delta at the vertices of the simplex.
    """
    def check_containment(A,b):
        """Check if theta \in {x : A*x<=b}"""
        return np.all(A.dot(theta)<=b)
    
    # Check that parameter has not exited the partitioned set
    if root.top and not check_containment(root.data.hrep_A,root.data.hrep_b):
        raise RuntimeError('Parameter out of domain')
    
    # Browse down the tree
    if not root.is_leaf():
        if check_containment(root.left.data.hrep_A,root.left.data.hrep_b):
            return f_delta_epsilon(root.left,theta)
        else:
            return f_delta_epsilon(root.right,theta)
    
    # If we get here, it's a leaf which theta is guaranteed to be in
    delta = root.data.commutation
    vertices = root.data.vertices
    inputs = root.data.vertex_inputs
    return delta, vertices, inputs

def mpc_implicit(oracle,theta):
    """
    Implicit MPC implementation.
    
    Parameters
    ----------
    oracle : Oracle
        Container of optimization problems related to the MPC algorithm.
    theta : np.array
        Parameter value (current state).
        
    Returns
    -------
    u_opt : np.aray
        Optimal input.
    """
    u_opt = oracle.P_theta(theta)[0]
    return u_opt

def mpc_semiexplicit(oracle,root,theta):
    """
    Semi-explicit MPC implementation.
    
    Parameters
    ----------
    oracle : Oracle
        Container of optimization problems related to the MPC algorithm.
    root : Tree
        Tree root.
    theta : np.array
        Parameter value (current state).
        
        
    Returns
    -------
    u_eps_subopt : np.aray
        Epsilon-suboptimal input.
    """
    delta_eps_subopt = f_delta_epsilon(root,theta)[0]
    u_eps_subopt = oracle.P_theta_delta(theta,delta_eps_subopt)[0]
    return u_eps_subopt

def mpc_explicit(oracle,root,theta):
    """
    Explicit MPC implementation.
    
    Parameters
    ----------
    oracle : Oracle
        Container of optimization problems related to the MPC algorithm.
    root : Tree
        Tree root.
    theta : np.array
        Parameter value (current state).
        
        
    Returns
    -------
    u_eps_subopt : np.aray
        Epsilon-suboptimal input.
    """
    vertices, inputs = f_delta_epsilon(root,theta)[1:]
    # Get mixing coefficients (i.e. theta = sum_i alpha_i*vertex_i)
    M = np.column_stack([vx-vertices[0] for vx in vertices[1:]])
    alpha = la.inv(M).dot(theta-vertices[0])
    alpha = np.concatenate([[1.-sum(alpha)],alpha])
    # Directly get the epsilon-suboptimal input
    u_eps_subopt = inputs.T.dot(alpha)
    return u_eps_subopt

# TODO simulation goes here using mpc_implicit, mpc_semiexplicit and mpc_explicit
simulator = Simulator(lambda x: mpc_implicit(oracles['semiexplicit'][0],x),sat,1*3600.)
sim_out_implicit = simulator.run(np.array([sat.pars['pos_err_max'],sat.pars['vel_err_max']]),label='implicit')

simulator = Simulator(lambda x: mpc_semiexplicit(oracles['semiexplicit'][0],partitions['semiexplicit'][0],x),sat,1*3600.)
sim_out_semiexplicit = simulator.run(np.array([sat.pars['pos_err_max'],sat.pars['vel_err_max']]),label='semiexplicit')

simulator = Simulator(lambda x: mpc_explicit(oracles['explicit'][0],partitions['explicit'][0],x),sat,1*3600.)
sim_out_explicit = simulator.run(np.array([sat.pars['pos_err_max'],sat.pars['vel_err_max']]),label='explicit')

#%%
# Plots
fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot(111)
ax.plot(sim_out_implicit.t,sim_out_implicit.x[0],label=sim_out_implicit.label)
ax.plot(sim_out_semiexplicit.t,sim_out_semiexplicit.x[0],label=sim_out_semiexplicit.label)
ax.plot(sim_out_explicit.t,sim_out_explicit.x[0],label=sim_out_explicit.label)
#ax.plot(sim_out.t,sim_out.x[1])

fig = plt.figure(2)
plt.clf()
ax = fig.add_subplot(111)
ax.semilogy(sim_out_implicit.t,sim_out_implicit.t_call,label=sim_out_implicit.label)
ax.semilogy(sim_out_semiexplicit.t,sim_out_semiexplicit.t_call,label=sim_out_semiexplicit.label)
ax.semilogy(sim_out_explicit.t,sim_out_explicit.t_call,label=sim_out_explicit.label)

#%% 

