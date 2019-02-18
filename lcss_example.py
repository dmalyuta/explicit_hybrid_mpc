"""
Generate data for L-CSS 2019 paper example.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import pickle
import time
import random
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import rc

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
    Nres = 2 # How many "resolutions" to test
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
            
            print('\n\n %s %d \n\n'%(kind,i))
            
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
    
    print('\n\n DONE \n\n')
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

# Run simulation with seed resetting (so that every simulation has the same
# noise realization)
N_orbits = 3 # Number of orbits to simulate
T_per_orbit = (2.*np.pi/sat.pars['wo']) # [s] Time for one orbit
T = N_orbits*T_per_orbit # Simulation final time
x_0 = np.array([sat.pars['pos_err_max'],sat.pars['vel_err_max']]) # Initial state

def as_si(x, ndp):
    """
    Convert number of scientific notation for label.
    
    Parameters
    ----------
    x : float
        The number.
    ndp : int
        Number of decimal places to show.
        
    Returns
    -------
    : str
        Number in scientific notation (decimal)*10^(exponent).
    """
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'{m:s}\cdot 10^{{{e:d}}}'.format(m=m, e=int(e))

# Implicit MPC
oracle = Oracle(sat,eps_a=1.,eps_r=1.) # Any eps_a,eps_r - just need P_theta()
simulator = Simulator(lambda x: mpc_implicit(oracle,x),sat,T)
random.seed(1)
sim_implicit = simulator.run(np.array([sat.pars['pos_err_max'],sat.pars['vel_err_max']]),label='implicit')

# Semi-explicit and explicit MPC
sim_offline = dict(semiexplicit=[],explicit=[])
for i in range(Nres):
    for kind in ['explicit','semiexplicit']:
        random.seed(1)
        mpc_call = mpc_explicit if kind=='explicit' else mpc_semiexplicit
        simulator = Simulator(lambda x: mpc_call(oracles[kind][i],partitions[kind][i],x),sat,T)
        sim_offline[kind].append(simulator.run(x_0,label='$\epsilon_{\mathrm{a}}=%s$, $\epsilon_{\mathrm{r}}=%s$'%
                                               (as_si(absolute_errors[i],2),as_si(relative_errors[i],2))))

#%%
# Plots

rc('text', usetex=True)
rc('font', family='serif')

# Position and velocity response comparison
pos_min = min(np.min(sim_implicit.x[0]),
              np.min([np.min(sim_offline['semiexplicit'][i].x[0]) for i in range(Nres)]),
              np.min([np.min(sim_offline['explicit'][i].x[0]) for i in range(Nres)]))
pos_max = min(np.max(sim_implicit.x[0]),
              np.max([np.max(sim_offline['semiexplicit'][i].x[0]) for i in range(Nres)]),
              np.max([np.max(sim_offline['explicit'][i].x[0]) for i in range(Nres)]))
vel_min = -0.2e-3#min(np.min(sim_implicit.x[1]),
              #np.min([np.min(sim_offline['semiexplicit'][i].x[1]) for i in range(Nres)]),
              #np.min([np.min(sim_offline['explicit'][i].x[1]) for i in range(Nres)]))
vel_max = 0.2e-3#min(np.max(sim_implicit.x[1]),
              #np.max([np.max(sim_offline['semiexplicit'][i].x[1]) for i in range(Nres)]),
              #np.max([np.max(sim_offline['explicit'][i].x[1]) for i in range(Nres)]))
fig = plt.figure(1,figsize=(7.,5.6))
plt.clf()
lines = []
labels = []
axs = []
for k,j,ylabel,kind in zip([1,3,2,4],
                           [0,1]*2,
                           ['Position [mm]','Velocity [mm/s]']*2,
                           ['semiexplicit','semiexplicit','explicit','explicit']):
    ax = fig.add_subplot(3,2,k)
    axs.append(ax)
    ax.grid(color='lightgray')
    line, = ax.plot(sim_implicit.t/T_per_orbit,sim_implicit.x[j]*1e3,color='gray',linewidth=2)
    if k==1:
        lines.append(line)
        labels.append(sim_implicit.label)
    for i in range(Nres):
        line, = ax.plot(sim_offline[kind][i].t/T_per_orbit,sim_offline[kind][i].x[j]*1e3,
                        linewidth=1)
        if k==1:
            lines.append(line)
            labels.append(sim_offline[kind][i].label)
    if k==1 or k==3:
        ax.set_ylabel(ylabel)
    else:
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.tick_params(axis='y',which='both',left=False,right=False)
    plt.autoscale(tight=True)
    if j==0:
        ax.set_ylim([pos_min*1e3,pos_max*1e3])
    else:
        ax.set_ylim([vel_min*1e3,vel_max*1e3])
    plt.setp(ax.get_xticklabels(), visible=False)
for k,kind in zip([5,6],['semiexplicit','explicit']):
    ax = fig.add_subplot(3,2,k)
    axs.append(ax)
    ax.grid(color='lightgray')
    ax.plot(sim_implicit.t/T_per_orbit,sim_implicit.u[0]*1e6,color='gray',linewidth=2,label=sim_implicit.label)
    for i in range(Nres):
        ax.plot(sim_offline[kind][i].t/T_per_orbit,sim_offline[kind][i].u[0]*1e6,
                label=sim_offline[kind][i].label,linewidth=0.5)
    ax.set_xlabel('Number of orbits')
    if k==5:
        ax.set_ylabel('Input [$\mu$m/s]')
    else:
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.tick_params(axis='y',which='both',left=False,right=False)
    plt.autoscale(tight=True)
    ax.set_ylim([-40,40])
plt.tight_layout(rect=[0,0.03,1,1])
plt.subplots_adjust(hspace=.0,wspace=.0)
fig.align_ylabels([axs[0],axs[1],axs[4]])
plt.figlegend(lines, labels, loc = 'lower center', ncol=5, labelspacing=0. , prop={'size':9})

yticklabels = ['']+[item.get_text() for item in axs[1].get_yticklabels()[1:]]
axs[1].set_yticklabels(yticklabels)
xticklabels = ['']+[item.get_text() for item in axs[-1].get_xticklabels()[1:]]
axs[-1].set_xticklabels(xticklabels)

# MPC call time statistics
lines, labels = [], []
t_call_max = max(np.max(sim_implicit.t_call),
                 np.max([np.max(sim_offline['semiexplicit'][i].t_call) for i in range(Nres)]))
t_call_min = np.min([np.min(sim_offline['explicit'][i].t_call) for i in range(Nres)])
fig = plt.figure(2,figsize=(6.6,2.4))
plt.clf()
ax = fig.add_subplot(131)
ax.grid()
line, = ax.semilogy(sim_implicit.t/T_per_orbit,sim_implicit.t_call*1e3,
            color='gray',linewidth=1)
lines.append(line)
labels.append(sim_implicit.label)
plt.autoscale(tight=True)
ax.set_ylim([t_call_min*1e3,t_call_max*1e3])
ax.set_ylabel('Evaluation time [ms]')
ax.set_xlabel('Number of orbits')
ax2 = fig.add_subplot(132)
ax2.grid()
for i in range(Nres):
    line, = ax2.semilogy(sim_offline['semiexplicit'][i].t/T_per_orbit,
                sim_offline['semiexplicit'][i].t_call*1e3,
                linewidth=1)
    lines.append(line)
    labels.append(sim_offline['semiexplicit'][i].label)
plt.autoscale(tight=True)
ax2.set_ylim([t_call_min*1e3,t_call_max*1e3])
plt.setp(ax2.get_yticklabels(), visible=False)
ax2.set_xlabel('Number of orbits')
ax3 = fig.add_subplot(133)
ax3.grid()
for i in range(Nres):
    ax3.semilogy(sim_offline['explicit'][i].t/T_per_orbit,
                sim_offline['explicit'][i].t_call*1e3,
                label=sim_offline['explicit'][i].label,
                linewidth=1)
plt.autoscale(tight=True)
ax3.set_ylim([t_call_min*1e3,t_call_max*1e3])
ax3.set_xlabel('Number of orbits')
plt.setp(ax3.get_yticklabels(), visible=False)
plt.tight_layout(rect=[0,0.07,1,1])
plt.subplots_adjust(hspace=.0,wspace=.0)
plt.figlegend(lines, labels, loc = 'lower center', ncol=5, labelspacing=0. , prop={'size':9})

ax2.set_xticklabels(['']+[item.get_text() for item in ax2.get_xticklabels()[1:]])
ax3.set_xticklabels(['']+[item.get_text() for item in ax3.get_xticklabels()[1:]])

# =============================================================================
# fig = plt.figure(1)
# plt.clf()
# ax = fig.add_subplot(111)
# ax.plot(sim_out_implicit.t,sim_out_implicit.x[0],label=sim_out_implicit.label)
# ax.plot(sim_out_semiexplicit.t,sim_out_semiexplicit.x[0],label=sim_out_semiexplicit.label)
# ax.plot(sim_out_explicit.t,sim_out_explicit.x[0],label=sim_out_explicit.label)
# #ax.plot(sim_out.t,sim_out.x[1])
# 
# fig = plt.figure(2)
# plt.clf()
# ax = fig.add_subplot(111)
# ax.semilogy(sim_out_implicit.t,sim_out_implicit.t_call,label=sim_out_implicit.label)
# ax.semilogy(sim_out_semiexplicit.t,sim_out_semiexplicit.t_call,label=sim_out_semiexplicit.label)
# ax.semilogy(sim_out_explicit.t,sim_out_explicit.t_call,label=sim_out_explicit.label)
# =============================================================================

#%% 

