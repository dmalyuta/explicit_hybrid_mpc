"""
Plant (dynamical system) class.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import numpy as np

class Plant:
    """
    Discrete-time plant.
    """
    def __init__(self,T_d,n,m,d):
        """
        Parameters
        ----------
        T_d : float
            Discretization time step.
        n,m,d : int
            State, input and discturbance dimensions.
        """
        self.T_d = T_d
        self.n = n
        self.m = m
        self.d = d
        self.x = np.zeros(self.n) # State, at zero by default
    
    def init(self, x_0):
        """
        Initialize plant state.
        
        Parameters
        ----------
        x_0 : np.array
            Set state to this value.
        """
        self.x_0 = x_0.copy()
        self.x = x_0.copy()
    
    def __call__(self, u, w):
        """
        Propagates dynamics by one time step, given current control input u and
        uncertainty w.
        
        Parameters
        ----------
        u : array
            Control input.
        w : array
            Uncertainty component.
            
        Returns
        -------
        x : array
            The next state.
        """
        self.x = self.state_update(self.x,u,w)
        return self.x.copy()

    def state_update(self,x,u,w):
        """State update - see above docstring for __call__."""
        raise NotImplementedError('state update method not implemented')

class LinearPlant(Plant):
    """
    Discrete-time plant.
    """
    def __init__(self, T_d, A, B, E):
        """
        Parameters
        ----------
        T_d : float
            Discretization time step.
        A,B,E : np.array
            Dynamic update equation (Ax+Bu+Ew) matrices.
        """
        super().__init__(T_d,A.shape[0],B.shape[1],E.shape[1])
        self.A = A
        self.B = B
        self.E = E
    
    def state_update(self, x, u, w):
        """State update."""
        x = self.A.dot(x)+self.B.dot(u)+self.E.dot(w)    
        return x
    
    def D(self, specs):
        """
        Create the disturbance gain matrix D for the concatenated disturbance
        p=(w,e,v) given the specifications specs.
        
        Parameters
        ----------
        specs : Specifications
            The specifications.
        
        Returns
        -------
        D : array
            The matrix D such that x+ = A*x+B*u+D*p where p=(w,e,v).
        """
        D = []
        # For state uncertainty, our model is z=x+v where z is estimated state,
        # x is actual state and v is the state estimation error. Therefore
        # x=z-v so Ax=A(z-v)=Az-Av therefore the multiplier matrix for the
        # state uncertainty is -A, not A.
        M = {'state':-self.A,'input':self.B,'process':self.E}
        for uncertainty_type in specs.P.type:
            D += [M[uncertainty_type]]
        D = np.hstack(D) if D!=[] else None
        return D

class NonlinearPlant(Plant):
    """Nonlinear plant. User must define their own custom __call__()."""
    def __init__(self,T_d,n,m,d):
        super().__init__(T_d,n,m,d)
