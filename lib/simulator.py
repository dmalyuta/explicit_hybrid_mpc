"""
Closed-loop system simulator class.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import time
import numpy as np
import numpy.linalg as la
import progressbar

def array2dict(array,name):
    """
    Convert an 2D array to a dictionary where each entry is one
    column of the array.
    
    Parameters
    ----------
    array : array
        The array.
    name : str
        Prefix to column number.
        
    Returns
    -------
    : dict
        The dictionary.
    """
    return dict(pair for d in
                [{'%s%d'%(name,k):v if len(v)>1 else v[0] for k,v in
                zip([i],[array[:,i]])} for i in range(array.shape[1])]
                for pair in d.items())

class SimulationOutput:    
    def __init__(self):
        """
        Initialize the data.
        """
        # Raw simulation data
        self.t = []      # Simulation times
        self.t_call = [] # Execution time of one controller call
        self.x = [] # States
        self.u = [] # Inputs
        self.w = [] # Process disturbances
        self.v = [] # State estimate errors
        self.e = [] # Input errors
    
    def add(self,t,t_call,x,u,w,v,e):
        self.t.append(t)
        self.t_call.append(t_call)
        self.x.append(x)
        self.u.append(u)
        self.w.append(w)
        self.v.append(v)
        self.e.append(e)

    def compile2array(self):
        """
        Convert data lists into arrays.
        **NB**: cannot call add() afterwards!
        """
        self.t = np.array(self.t)
        self.t_call = np.array(self.t_call)
        self.x = np.column_stack(self.x)
        self.u = np.column_stack(self.u)
        self.w = np.column_stack(self.w)
        self.v = np.column_stack(self.v)
        self.e = np.column_stack(self.e)

class Simulator:
    def __init__(self, mpc, T):
        """
        Parameters
        ----------
        K : callable
            The controller.
        mpc : MPC
            The MPC law.
        T: float
            Simulation final time.
        """
        self.K = mpc.__call__ # controller
        self.G = mpc.plant # plant
        self.uncertain_system = hasattr(mpc,'specs')
        if self.uncertain_system:
            self.specs = mpc.specs
        # Sampling times: K for controller, G for plant
        # **NB: assumes that the call frequencies are integer**
        freq_K = int(1/mpc.T_s)
        freq_G = int(1/self.G.T_d)
        self.h = dict(K=1./freq_K,G=1./freq_G)
        if freq_G%freq_K!=0:
            raise ValueError('Plant and controller call times must be '
                             'divisible')
        self.T_f = T
        
        # Noise sampling functions
        if self.uncertain_system:
            def sampleNoise(noise_type, dim):
                """
                Noise sampling function.
                
                Parameters
                ----------
                noise_type : str
                    Where noise enters the system, should be 'process', 'state' or
                    'input'.
                dim : int
                    Dimension of the noise vector.
                """
                ntype = self.specs.P.type
                sampler = self.specs.P.sample_sim
                return lambda t,x,u: sum([sampler[i](t,x,u) for i in range(len(ntype)) if
                                          ntype[i]==noise_type])+np.zeros(dim)
            self.process_noise = sampleNoise('process',self.G.d)
            self.state_estimation_noise = sampleNoise('state',self.G.n)
            self.input_noise = sampleNoise('input',self.G.m)
        else:
            self.process_noise = lambda t,x,u: np.zeros(self.G.d)
            self.state_estimation_noise = lambda t,x,u: np.zeros(self.G.n)
            self.input_noise = lambda t,x,u: np.zeros(self.G.m)
        
        # Default simulation output data descriptions
        self.sim_history = SimulationOutput()

    def run(self, x_0, label=None):
        """
        Run the simulation from start to finish. Note that simulation begins
        with **measured state==actual state** which is important because the
        controller guarantees invariance following the **measured state** being
        contained in the invariant set rather than the **actual state**, while
        the initial condition is sampled for the actual state.
        
        Parameters
        ----------
        x_0 : np.array
            The initial state.
        label : str, optional
            If provided, set a label for the simulation output data that will
            be used for e.g. post-processor plots.
        
        Returns
        -------
        : DataFrame
            Simulation output data.
        """
        if label is not None:
            self.sim_history.label = label

        widgets=['<%s> '%(label),progressbar.Percentage(),' ',
                 progressbar.Bar(),' (',progressbar.ETA(),')']

        # Initialization
        self.G.init(x_0)
        x = self.G.x_0
        u = np.zeros(self.G.m)
        v = np.zeros(self.G.n)
        times = np.linspace(0,self.T_f,int(self.T_f/self.h['G']+1))
        t_last_call_K = None

        # Simulation
        eps_mach = np.finfo('float').eps
        for t in progressbar.progressbar(times,widgets=widgets):
            x_prev = np.copy(x)
            # Compute input
            if (t_last_call_K is None or
                t-t_last_call_K>=self.h['K']-eps_mach):
                t_last_call_K = t
                v = self.state_estimation_noise(t,x,u)
                z = x+v if t>0 else x # "Measured" state
                u,t_call = self.K(z)
            else:
                t_call = 0.
            e = self.input_noise(t,x,u)
            if la.norm(u)==0:
                # When input is off, no "rogue input" noise
                e = np.zeros(e.shape)
            # Compute process noise
            w = self.process_noise(t,x,u)
            # Update state
            x = self.G(u+e,w)
            # Save this iteration's data
            self.sim_history.add(t,t_call,x_prev,u,w,v,e)

        # Compile and return results
        self.sim_history.compile2array()
        return self.sim_history
