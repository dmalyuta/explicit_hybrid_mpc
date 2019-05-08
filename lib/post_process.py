"""
Data post-processing.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

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
        self.setup()

    def setup(self):
        """Sets up the post-processor."""
        prepare.set_global_variables()
        self.fig_num = 1
        self.load_data()
        # Explicit MPC oracle
        self.explicit_mpc = mpc_library.ExplicitMPC(self.tree)
        # Implicit MPC oracle
        self.invariant_set,_,__implicit_oracle = example()
        def implicit_mpc(x):
            """
            Returns optimal control input for the given parameter x.

            Parameters
            ----------
            x : np.array
                Current state (parameter theta in the paper [1]).

            Returns
            -------
            u : np.array
                Epsilon-suboptimal input.
            t : float
                Evaluation time.
            """
            u,_,_,t = __implicit_oracle.P_theta(x)
            return u,t
        
        self.implicit_mpc = implicit_mpc

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

    def __new_figure(self,figsize=None):
        """Create a new empty figure. Returns the figure object."""
        fig = plt.figure(self.fig_num,figsize=figsize)
        self.fig_num += 1
        plt.clf()
        return fig

    def progress(self):
        """
        Visualize algorithm progress in terms of:
           - Volume filled
           - Number of simplices
        """
        simplex_count = self.statistics['simplex_count_total'][-1]
        fig = self.__new_figure(figsize=(5.5,3))
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
        
    def total_input_usage(self,T,t_scale=1,u_scale=[1,1],
                          x_label='Time [s]',y_label='Input norm'):
        """
        Compare the summed 2-norm of the input over a simulation of duration T,
        between implicit and explicit MPC.

        Parameters
        ----------
        T : float
            Simulation duration.
        t_scale : float, optional
            Coefficient by which to multiply time for plotting.
        u_scale : list, optional
            Coefficients by which to multiply input 2-norm for (first element)
            plotting and (second element) total usage printout.
        x_label : str, optional
            x-axis label.
        y_label : str, optional
            y-axis label.
        """
        # Initial condition and plant
        if global_vars.EXAMPLE=='cwh_z':
            plant = mpc_library.SatelliteZ()
        elif global_vars.EXAMPLE=='cwh_xy':
            plant = mpc_library.SatelliteXY()
        else:
            plant = mpc_library.SatelliteXYZ()
        x_init = np.zeros(plant.n_x)
        # Simulate explicit MPC
        simulator = Simulator(self.explicit_mpc,plant,T)
        sim_explicit = simulator.run(x_init,label='explicit')
        # Simulate implicit MPC
        simulator = Simulator(self.implicit_mpc,plant,T)
        sim_implicit = simulator.run(x_init,label='implicit')
        # Plot input norm history
        fig = self.__new_figure(figsize=(5.5,3))
        ax = fig.add_subplot(111)
        ax.plot(sim_implicit.t*t_scale,la.norm(sim_implicit.u,axis=0)*u_scale[0],
                color='orange',linestyle='none',marker='.',markersize=10,
                label='implicit')
        ax.plot(sim_explicit.t*t_scale,la.norm(sim_explicit.u,axis=0)*u_scale[0],
                color='black',linestyle='none',marker='x',markersize=5,
                label='explicit')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
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
    
def main():
    """Run post-processing for data specified via command-line."""
    post_processor = PostProcessor()
    # Progress plot
    #post_processor.progress()
    # Total input usage comparison (implicit vs. explicit)
    orbit_count = 3 # How many orbits to simulate for
    wo = mpc_library.satellite_parameters()['wo'] # [rad/s] Orbital rate
    T_per_orbit = (2.*np.pi/wo) # [s] Time for one orbit
    T = T_per_orbit*orbit_count # [s] Simulation duration
    post_processor.total_input_usage(T=T,t_scale=1/T_per_orbit,
                                     u_scale=[1e6,1e3],
                                     x_label='Number of orbits',
                                     y_label='$\Delta v$ usage [$\mu$m/s]')
    # Evaluation time statistics
    post_processor.evaluation_time(N=100)

if __name__=='__main__':
    main()
    
