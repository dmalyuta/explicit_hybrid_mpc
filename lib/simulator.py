"""
Closed-loop system simulator class.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import time
import numpy as np
import numpy.linalg as la
from progressbar import progressbar

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
    def __init__(self, K, mpc, T):
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
        self.K = K
        self.G = mpc.plant
        self.uncertain_system = hasattr(mpc,'specs')
        if self.uncertain_system:
            self.specs = mpc.specs
        self.T_d = self.G.T_d
        self.T = T
        
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
        
        self.G.init(x_0)
        
        x = self.G.x_0
        u = np.zeros(self.G.m)
        times = np.linspace(0,self.T,int(self.T/self.T_d+1))
        for t in progressbar(times):
            x_prev = np.copy(x)
            v = self.state_estimation_noise(t,x,u)
            z = x+v if t>0 else x # "Measured" state
            u,t_call = self.K(z)            
            e = self.input_noise(t,x,u)
            if la.norm(u)==0:
                e = np.zeros(e.shape)
            w = self.process_noise(t,x,u)
            x = self.G(u+e,w)
            self.sim_history.add(t,t_call,x_prev,u,w,v,e)
        self.sim_history.compile2array()
        return self.sim_history
