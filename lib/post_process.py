"""
Data post-processing.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import os
import pickle
import numpy as np
import numpy.linalg as la
import matplotlib
import matplotlib.pyplot as plt
from progressbar import progressbar

import global_vars
import prepare
from simulator import Simulator
import mpc_library
from examples import example

matplotlib.rc('font',**{'family':'serif','size':12})
matplotlib.rc('text', usetex=True)

class PostProcessor:
    def __init__(self):
        prepare.set_global_variables()
        self.load_data()

        self.setup_mpc_called = False # Switch to run MPC setup

    def setup_mpc(self):
        """Sets up the MPC controllers."""
        if self.setup_mpc_called:
            return
        self.setup_mpc_called = True
        # Implicit MPC oracle
        self.invariant_set,_,oracle = example()
        self.implicit_mpc = mpc_library.ImplicitMPC(oracle)
        # Explicit MPC oracle
        self.explicit_mpc = mpc_library.ExplicitMPC(self.tree,oracle)

    def load_data(self):
        """
        Load data files into working memory.
    
        Returns
        -------
        statistics : dict
            Dictionary of algorithm progress statistics.
        tree : Tree
            The created partition.
        """
        # Load overall progress statistics
        self.statistics = dict()
        with open(global_vars.STATISTICS_FILE,'rb') as f:
            while True:
                try:
                    data = pickle.load(f)
                    for key in data['overall'].keys():
                        value = data['overall'][key]
                        if key in self.statistics.keys():
                            self.statistics[key].append(value)
                        else:
                            self.statistics[key] = [value]
                except EOFError:
                    break
        # Load tree
        with open(global_vars.TREE_FILE,'rb') as f:
            self.tree = pickle.load(f)

    def tree_stats(self):
        """
        Print the tree statistics: depth and leaf count.
        """
        def get_depth(cursor):
            """Compute tree depth by recursing through it. Do not count the
            depth levels of the initial Delaunay triangulation."""
            if cursor.is_leaf():
                return 1
            else:
                one = 1 if cursor.right.data is not None else 0
                return one+max(get_depth(cursor.left),get_depth(cursor.right))

        def count_leaves(cursor):
            """Count the number of leaves of the tree."""
            if cursor.is_leaf():
                return 1
            else:
                return count_leaves(cursor.left)+count_leaves(cursor.right)

        def get_size():
            """Get tree file size in MB."""
            return os.path.getsize(global_vars.TREE_FILE)/2**20

        # Optimized storage requirement
        # theory (assumes perfect binary tree)
        mu_f = 64//8 # [B] Float size
        # Vector dimensions
        if global_vars.EXAMPLE=='pendulum':
            p,n_hat = 4,1
        elif global_vars.EXAMPLE=='cwh_z':
            p,n_hat = 2,1
        elif global_vars.EXAMPLE=='cwh_xy':
            p,n_hat = 4,2
        elif global_vars.EXAMPLE=='cwh_xyz':
            p,n_hat = 6,3
        else:
            raise ValueError('Unknown example')
        leaves = count_leaves(self.tree)
        B2MB = 1./(2**20) # [B] -> [MB]
        opt_storage_theory = 3./2.*leaves*mu_f*p*(p+1)+leaves*(p+1)*n_hat*mu_f
        opt_storage_theory *= B2MB # B -> MB
        # actual tree
        def get_opt_memreq(cursor):
            """Get the optimized storage memory requirement [MB] in which the
            mixing matrix is stored directly."""
            if cursor.is_leaf():
                containment = mu_f*p*(p+1)
                inputs = mu_f*n_hat*(p+1)
                return (inputs+containment)*B2MB
            else:
                containment = 0 if cursor.left.is_leaf() else mu_f*p*(p+1)*B2MB
                return (containment+get_opt_memreq(cursor.left)+
                        get_opt_memreq(cursor.right))

        print('Partitioning runtime CPU [hr]           : %.2f'%(
            self.statistics['time_active_total'][-1]/3600))
        print('Partitioning runtime wall [hr]          : %.2f'%(
            self.statistics['time_elapsed'][-1]/3600))
        print('Max cores active                        : %d'%(
            np.max(self.statistics['num_proc_active'])))
        print('Average cores active                    : %d'%(
            np.round(np.average(self.statistics['num_proc_active']))))
        print('Tree depth                              : %d'%(get_depth(self.tree)))
        print('Tree leaf count                         : %d'%(leaves))
        print('Tree file size [MB]                     : %d'%(get_size()))
        print('Tree optimized file size (theory) [MB]  : %d'%(opt_storage_theory))
        print('Tree optimized file size [MB]           : %d'%(get_opt_memreq(self.tree)))

    def progress(self):
        """
        Visualize algorithm progress in terms of:
           - Volume filled
           - Number of simplices
        """
        simplex_count = self.statistics['simplex_count_total'][-1]
        fig = new_figure(figsize=(5.5,3))
        ax1 = fig.add_subplot(111)
        ax1.plot(self.statistics['time_elapsed'],
                 self.statistics['volume_filled_frac'],
                 color='black',linewidth=3,label='Volume filled')
        ax1.plot(self.statistics['time_elapsed'],
                 [count/simplex_count for count in
                  self.statistics['simplex_count_total']],
                 color='blue',linewidth=3,label='Simplex count')
        ax1.set_xlabel('Time elapsed [s]')
        ax1.set_ylabel('Fraction of final value')
        ax1.set_xlim([self.statistics['time_elapsed'][0],
                      self.statistics['time_elapsed'][-1]])
        ax1.set_ylim([0,1])
        ax1.legend(loc='lower right')
        ax2 = ax1.twinx()  # second axes that shares the same x-axis
        ax1.set_zorder(ax2.get_zorder()+1) # put ax1 in front
        ax1.patch.set_visible(False) # hide the 'canvas' 
        ax2.fill_between(self.statistics['time_elapsed'],0,
                         self.statistics['num_proc_active'],
                         color='orange',linewidth=0,alpha=0.4)
        ax2.set_ylabel('Active process count')
        ax2.set_ylim([0,max(self.statistics['num_proc_active'])])
        plt.tight_layout()
        plt.show(block=False)

    def simulate_and_plot(self,x_init,T,plot_funcs):
        """
        Simulate the system controlled via explicit and implicit MPC, then call
        the passed plotting functions.

        Parameters
        ----------
        x_init : np.float
            Initial condition.
        T : float
            Simulation duration.
        plot_funcs : list
            List of callable plot functions. Should have the call signature
            plot_func(sim_explicit,sim_implicit) where the two arguments are
            the output of Simulator:run().
        """
        self.setup_mpc()
        # Simulate explicit MPC
        simulator = Simulator(self.explicit_mpc,T)
        sim_explicit = simulator.run(x_init,label='explicit')
        # Simulate implicit MPC
        simulator = Simulator(self.implicit_mpc,0.5)#T)
        sim_implicit = simulator.run(x_init,label='implicit')
        # Called plotting functions
        for plot_func in plot_funcs:
            plot_func(sim_explicit,sim_implicit)

    def evaluation_time(self,N):
        """
        Compute statistics for how long it takes the implicit and explicit MPC
        implementation to compute the control input. This is done via Monte
        Carlo by calling the two algorithms for states uniformly randomly
        sampled from the partitioned invariant set.

        Parameters
        ----------
        N : int
            How many Monte Carlo trials to perform (for each implementation).
        """
        self.setup_mpc()
        parameter_samples = self.invariant_set.randomPoint(N=N)
        implicit_eval_times = np.empty(N)
        explicit_eval_times = np.empty(N)
        for i in progressbar(range(N)):
            _,implicit_eval_times[i] = self.implicit_mpc(parameter_samples[i])
            _,explicit_eval_times[i] = self.explicit_mpc(parameter_samples[i])
        implicit_average = np.average(implicit_eval_times)
        implicit_min = np.min(implicit_eval_times)
        implicit_max = np.max(implicit_eval_times)
        explicit_average = np.average(explicit_eval_times)
        explicit_min = np.min(explicit_eval_times)
        explicit_max = np.max(explicit_eval_times)
        # Printout evaluation time statistics
        s2ms = 1e3
        print('Implicit evaluation time (av. [min,max], ms): %.2f [%.2f,%.2f]'%
              (implicit_average*s2ms,implicit_min*s2ms,implicit_max*s2ms))
        print('Explicit evaluation time (av. [min,max], ms): %.2f [%.2f,%.2f]'%
              (explicit_average*s2ms,explicit_min*s2ms,explicit_max*s2ms))

def new_figure(figsize=None):
    """Create a new empty figure. Returns the figure object."""
    fig = plt.figure(figsize=figsize)
    plt.clf()
    return fig

def total_delta_v_usage(sim_explicit,sim_implicit,t_scale,u_scale):
    """Plot the input 2-norm history and usage statistics."""
    u_scale=[1e6,1e3] # 0: plot in micro meters/sec, 1: statistics in mm/s
    # Plot input norm history
    fig = new_figure(figsize=(5.5,3))
    ax = fig.add_subplot(111)
    ax.plot(sim_implicit.t*t_scale,la.norm(sim_implicit.u,axis=0)*u_scale[0],
            color='orange',linestyle='none',marker='.',markersize=10,
            label='implicit')
    ax.plot(sim_explicit.t*t_scale,la.norm(sim_explicit.u,axis=0)*u_scale[0],
            color='black',linestyle='none',marker='x',markersize=5,
            label='explicit')
    ax.set_xlabel('Number of orbits')
    ax.set_ylabel('$\Delta v$ usage [$\mu$m/s]')
    ax.legend(loc='lower right')
    ax.set_xlim([0,sim_explicit.t[-1]*t_scale])
    plt.tight_layout()
    plt.show(block=False)
    # Print out cumulative input 2-norm usage
    implicit_u_total = sum(la.norm(sim_implicit.u,axis=0)*u_scale[1])
    explicit_u_total = sum(la.norm(sim_explicit.u,axis=0)*u_scale[1])
    print('Summed input 2-norm (implicit): %.2f'%(implicit_u_total))
    print('Summed input 2-norm (explicit): %.2f'%(explicit_u_total))
    print('Overconsumption by explicit: %.2f %%'%
          ((explicit_u_total-implicit_u_total)/implicit_u_total*100))

def input_history(sim_explicit,sim_implicit):
    """Plot the input history."""
    # Plot input norm history
    fig = new_figure(figsize=(5.5,3))
    ax = fig.add_subplot(111)
    ax.plot(sim_implicit.t,sim_implicit.u[0],color='orange',
            linestyle='none',marker='.',markersize=10,label='implicit')
    ax.plot(sim_explicit.t,sim_explicit.u[0],color='black',
            linestyle='none',marker='x',markersize=5,label='explicit')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Force [N]')
    ax.legend(loc='lower right')
    ax.set_xlim([0,sim_explicit.t[-1]])
    plt.tight_layout()
    plt.show(block=False)

def state_history(sim_explicit,sim_implicit):
    """Plot the state history."""
    # Plot input norm history
    fig = new_figure(figsize=(6,8))
    ax = fig.add_subplot(411)
    ax.plot(sim_implicit.t,sim_implicit.x[0],color='orange',label='implicit')
    ax.plot(sim_explicit.t,sim_explicit.x[0],color='black',label='explicit')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position [m]')
    ax.legend()
    ax = fig.add_subplot(412)
    ax.plot(sim_implicit.t,sim_implicit.x[2],color='orange',label='implicit')
    ax.plot(sim_explicit.t,sim_explicit.x[2],color='black',label='explicit')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Velocity [m/s]')
    ax.legend()
    ax = fig.add_subplot(413)
    ax.plot(sim_implicit.t,np.rad2deg(sim_implicit.x[1]),color='orange',label='implicit')
    ax.plot(sim_explicit.t,np.rad2deg(sim_explicit.x[1]),color='black',label='explicit')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Angle [deg]')
    ax.legend()
    ax = fig.add_subplot(414)
    ax.plot(sim_implicit.t,np.rad2deg(sim_implicit.x[3]),color='orange',label='implicit')
    ax.plot(sim_explicit.t,np.rad2deg(sim_explicit.x[3]),color='black',label='explicit')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Angular rate [deg/s]')
    ax.legend()
    ax.legend(loc='lower right')
    ax.set_xlim([0,sim_explicit.t[-1]])
    plt.tight_layout()
    plt.show(block=False)

def main():
    """Run post-processing for data specified via command-line."""
    post_processor = PostProcessor()
    # Tree statistics
    post_processor.tree_stats()
    # Algorithm progress plot
    post_processor.progress()
    # Simulation comparison (implicit vs. explicit)
    if 'cwh' in global_vars.EXAMPLE:
        # CWH satellite example
        orbit_count = 3 # How many orbits to simulate for
        wo = mpc_library.satellite_parameters()['wo'] # [rad/s] Orbital rate
        T_per_orbit = (2.*np.pi/wo) # [s] Time for one orbit
        T = T_per_orbit*orbit_count # [s] Simulation duration
        x_init = np.zeros(post_processor.implicit_mpc.plant.n) # Initial condition
        post_processor.simulate_and_plot(x_init,T,[
            lambda exp,imp: total_delta_v_usage(exp,imp,t_scale=1/T_per_orbit)])
    else:
        # Inverted pendulum example
        T = 3 # [s] Simulation duration
        #x_init = np.array([0,np.deg2rad(0.1),0,0]) # Initial condition
        x_init = np.array([0,np.deg2rad(2),0.1,0]) # Initial condition
        post_processor.simulate_and_plot(x_init,T,[input_history,state_history])
    # Evaluation time statistics
    # post_processor.evaluation_time(N=1000)

if __name__=='__main__':
    main()
    
