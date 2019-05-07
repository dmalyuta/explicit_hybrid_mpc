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
        
        global_vars.CMD_LINE_ARGS = dict()
        global_vars.CMD_LINE_ARGS['nodes'] = 3
        global_vars.CMD_LINE_ARGS['tasks_per_node'] = 28
                          
        global_vars.EXAMPLE = 'cwh_z'
        global_vars.MPC_HORIZON = 4
        global_vars.ABS_FRAC = 0.1
        global_vars.REL_ERR = 0.1
        # Filenames
        global_vars.RUNTIME_DIR = global_vars.PROJECT_DIR+'/runtime/'+'post_process_test'
        global_vars.DATA_DIR = global_vars.RUNTIME_DIR+'/data'
        global_vars.STATUS_FILE = global_vars.DATA_DIR+'/status.txt' # Overall program status (text file)
        global_vars.STATISTICS_FILE = global_vars.DATA_DIR+'/statistics.pkl' # Overall statistics
        global_vars.TREE_FILE = global_vars.DATA_DIR+'/tree.pkl' # Overall tree
        global_vars.ETA_RLS_FILE = global_vars.DATA_DIR+'/rls.pkl' # Overall tree
        global_vars.BRANCHES_FILE = global_vars.DATA_DIR+'/branches.pkl' # Tree branches, used for tree building
        global_vars.IDLE_COUNT_FILE = global_vars.DATA_DIR+'/idle_count.pkl' # Idle process count
        
        self.fig_num = 1
        self.load_data()
        # Explicit MPC oracle
        self.explicit_mpc = mpc_library.ExplicitMPC(self.tree)
        # Implicit MPC oracle
        __implicit_oracle = example()[1]
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
        # Load statistics
        self.num_procs = (global_vars.CMD_LINE_ARGS['nodes']*
                          global_vars.CMD_LINE_ARGS['tasks_per_node']-1)
        self.statistics = dict(overall=dict(),
                               process=[dict() for _ in range(self.num_procs)])
        with open(global_vars.STATISTICS_FILE,'rb') as f:
            while True:
                try:
                    data = pickle.load(f)
                    # Overall
                    for key in data['overall'].keys():
                        value = data['overall'][key]
                        if key in self.statistics['overall'].keys():
                            self.statistics['overall'][key].append(value)
                        else:
                            self.statistics['overall'][key] = [value]
                    # Individual process
                    for i in range(len(data['process'])):
                        data_proc_i = data['process'][i]
                        if data_proc_i is None:
                            continue
                        for key in data_proc_i.keys():
                            value = data_proc_i[key]
                            if key in self.statistics['process'][i].keys():
                                self.statistics['process'][i][key].append(value)
                            else:
                                self.statistics['process'][i][key] = [value]
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
        simplex_count = self.statistics['overall']['simplex_count_total'][-1]
        fig = self.__new_figure(figsize=(5.5,3))
        ax1 = fig.add_subplot(111)
        ax1.plot(self.statistics['overall']['time_elapsed'],
                 self.statistics['overall']['volume_filled_frac'],
                 color='black',linewidth=3,label='Volume filled')
        ax1.plot(self.statistics['overall']['time_elapsed'],
                 [count/simplex_count for count in
                  self.statistics['overall']['simplex_count_total']],
                 color='blue',linewidth=3,label='Simplex count')
        ax1.set_xlabel('Time elapsed [s]')
        ax1.set_ylabel('Fraction of final value')
        ax1.set_xlim([self.statistics['overall']['time_elapsed'][0],
                      self.statistics['overall']['time_elapsed'][-1]])
        ax1.set_ylim([0,1])
        ax1.legend()
        ax2 = ax1.twinx()  # second axes that shares the same x-axis
        ax1.set_zorder(ax2.get_zorder()+1) # put ax1 in front
        ax1.patch.set_visible(False) # hide the 'canvas' 
        ax2.fill_between(self.statistics['overall']['time_elapsed'],0,
                         self.statistics['overall']['num_proc_active'],
                         color='orange',linewidth=0,alpha=0.4)
        ax2.set_ylabel('Active process count')
        ax2.set_ylim([0,self.num_procs])
        plt.tight_layout()
        plt.show(block=False)
        
    def total_input_usage(self,T,t_scale=1,u_scale=1,
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
        u_scale : float, optional
            Coefficient by which to multiply input 2-norm for plotting.
        x_label : str, optional
            x-axis label.
        y_label : str, optional
            y-axis label.
        """
#        # Initial condition and plant
#        if global_vars.EXAMPLE=='cwh_z':
#            plant = mpc_library.SatelliteZ()
#        elif global_vars.EXAMPLE=='cwh_xy':
#            plant = mpc_library.SatelliteXY()
#        else:
#            plant = mpc_library.SatelliteXYZ()
#        x_init = np.zeros(plant.n_x)
#        # Simulate explicit MPC
#        simulator = Simulator(self.explicit_mpc,plant,T)
#        self.sim_explicit = simulator.run(x_init,label='explicit')
#        # Simulate implicit MPC
#        simulator = Simulator(self.implicit_mpc,plant,T)
#        self.sim_implicit = simulator.run(x_init,label='implicit')
        # Plot input norm history
        fig = self.__new_figure(figsize=(5.5,3))
        ax = fig.add_subplot(111)
        ax.plot(self.sim_implicit.t*t_scale,la.norm(self.sim_implicit.u,axis=0)*u_scale,
                color='orange',linestyle='none',marker='.',markersize=10,
                label='implicit')
        ax.plot(self.sim_explicit.t*t_scale,la.norm(self.sim_explicit.u,axis=0)*u_scale,
                color='black',linestyle='none',marker='x',markersize=10,
                label='implicit')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend()
        plt.show(block=False)
    
if __name__=='__main__':
    """Run post-processing for data specified via command-line."""
#    post_processor = PostProcessor()
#    # Progress plot
#    post_processor.progress()
#    # Total input usage comparison (implicit vs. explicit)
#    orbit_count = 4 # How many orbits to simulate for
#    wo = mpc_library.satellite_parameters()['wo'] # [rad/s] Orbital rate
#    T_per_orbit = (2.*np.pi/wo) # [s] Time for one orbit
#    T = T_per_orbit*orbit_count # [s] Simulation duration
    post_processor.total_input_usage(T=T,t_scale=1/T_per_orbit,
                                     u_scale=1e6,
                                     x_label='Number of orbits',
                                     y_label='$\Delta v$ usage [$\mu$m/s]')
    
