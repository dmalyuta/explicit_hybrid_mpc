"""
Classes for uncertainty sets, which allow the generation of unccertain
variables.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2018 University of Washington. All rights reserved.
"""

import itertools
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import cvxpy as cvx

import general
from polytope import Polytope
import set_synthesis as ss

class Set:
    def __init__(self, *args, **kwargs):
        self.setup(*args, **kwargs)
        
    def __call__(self, *args, **kwargs):
        return self.generateRandomPoint(*args, **kwargs)
        
    def setup(self):
        """
        Setup of the specific uncertainty set goes in here. A particular set
        may accept further arguments, which are passed down from __init__.
        """
        raise NotImplementedError(general.ERROR+"You must define the "
                                  "uncertainty set setup() method") 
        
    def generateRandomPoint(self):
        """
        Generates a random point in the uncertainty set. A particular set may
        accept further arguments, which are passed down from __call__.
        
        Returns
        -------
        : array
            The random point.
        """
        raise NotImplementedError(general.ERROR+"You must define the "
                                  "uncertainty set generateRandomPoint() "
                                  "method") 

class DependencyDescription:
    """
    Information container for the dependent noise model:
        
    ||q||_pq <= phi(||F_x*x||_px,||F_u*u||_pu).
    """
    def __init__(self,phi,pq,px=None,pu=None,Fx=None,Fu=None):
        """
        Parameters
        ----------
        phi : callable
            The upper bound function of state and input dependent
            uncertainty. Must be convex and nondecreasing. Call signature
            phi(nx,nu) where nx==cvx.norm(x,px) and nu==cvx.norm(u,pu), or
            any subset like only nx/only nu/nothing **but in that order**,
            and must return a valid CVXPY expression.
        pq : {1,2,np.inf}
            Norm on the dependent noise term.
        px : {1,2,np.inf}, optional
            State norm. Not specifying means that phi does not have a state
            norm component.
        pu : {1,2,np.inf}, optional
            Input norm. Not specifying means that phi does not have an
            input norm component.
        Fx : array, optional
            State linear transformation matrix.
        Fu : array, optional
            Input linear transformation matrix.
        """
        self.pq = pq
        self.px = px
        self.pu = pu
        self.Fx = Fx
        self.Fu = Fu
        self.phi = phi
 
    def phi_direct(self,x,u):
        """
        Uncertainty norm bound that is dependent on state and/or input norms.
        Allows passing directly just the state and input, abstracting away the
        internal mechanics.
        
        Parameters
        ----------
        x : array
            State.
        u : array
            Input.
        """
        if self.px is not None and self.pu is not None:
            return self.phi(cvx.norm(self.Fx*x if self.Fx is not None else x,
                                     self.px),
                            cvx.norm(self.Fu*u if self.Fu is not None else u,
                                     self.pu))
        elif self.px is not None and self.pu is None:
            return self.phi(cvx.norm(self.Fx*x if self.Fx is not None else x,
                                     self.px))
        elif self.pu is not None and self.px is None:
            return self.phi(cvx.norm(self.Fu*u if self.Fu is not None else u,
                                     self.pu))
        else:
            return self.phi()

class UncertaintySet(Set):
    """
    Implementation of input and state dependent uncertainty model of [1].
    
    [1] Malyuta, D., Acikmese, B. and Cacan, M., "Min-max Model Predictive
    Control for Constrained Linear Systems with State and Input Dependent
    Uncertainties," in American Control Conference, 2019. To be published.
    """
    def setup(self):
        # Set up the uncertainty set definition elements
        self.W = np.eye(0)
        self.L = []
        self.R = np.eye(0)
        self.r = np.array([])
        self.dependency = []
        self.mtx_map = []
        
        # Other helpful variables
        self.type = [] # Each uncertainty type: 'input', 'state' or 'process'
        self.is_dependent = [] # If this component is dependent or independent
        
        # Lists of sampling functions
        self.sample_w = [] # Functions which take arguments (t)
        self.sample_q = [] # Functions which take arguments (t,x,u)
        self.sample_sim = [] # Functions for sampling during simulation which take arguments (t,x,u)
        
        # List of individual sets corresponding to each added noise component
        self.sets = []
        
    def addIndependentTerm(self,uc_type,lb,ub,M=None,excitation=None):
        """
        Add to the w term a component sampled from a hyperrectangle.
        
        Parameters
        ----------
        uc_type : str
            Uncertainy type. Must be either of "input", "state" or "process".
        lb: array
            Lower bound.
        ub: array
            Upper bound.
        M : array, optional
            Map from potentially lower to a higher dimension. Useful e.g. if
            expressing uncertainty on a subset of the state.
        excitation : dict, optional
            Specifies the parameters of an optional Hyperrectangle sampling
            type. If specified, should have a field 'type' taking values
            'constant' or 'sinusoidal' and a field 'parameters' containing the
            corresponding excitation parameters.
        """
        lb = lb if type(lb) is np.ndarray else np.array([lb])
        ub = ub if type(ub) is np.ndarray else np.array([ub])
        dim = lb.size
        H = Hyperrectangle(lb,ub)
        if excitation is not None:
            param_is_list = type(excitation["parameters"])==list
            if excitation["type"]=="constant":
                if param_is_list:
                    H.constantExcitation(*excitation["parameters"])
                else:
                    H.constantExcitation(**excitation["parameters"])
            elif excitation["type"]=="sinusoidal":
                if param_is_list:
                    H.sinusoidalExcitation(*excitation["parameters"])
                else:
                    H.sinusoidalExcitation(**excitation["parameters"])
            else:
                raise ValueError(general.ERROR+"Unknown excitation type")
        P,p = H.convertToPolytope()[:2]
        sampling_function = lambda t: H(t)
        M = M if M is not None else np.eye(dim)
        self.mtx_map.append(M)
        self.sample_w.append(sampling_function)
        self.sample_sim.append(lambda t,x,u: M.dot(sampling_function(t)))
        self.sets.append(H)
        self.R = sla.block_diag(self.R,P)
        self.r = np.concatenate([self.r,p])
        self.W = sla.block_diag(self.W,M)
        self.type.append(uc_type)
        self.is_dependent.append(False)
        # Extend the other matrices
        for i in range(len(self.L)):
            self.L[i] = np.vstack((self.L[i],np.zeros((M.shape[0],self.L[i].shape[1]))))
            
    def addDependentTerm(self,uc_type,dep_model,dim=None,L=None):
        """
        Add a new q term, i.e. a state and/or input dependent term.
        
        Parameters
        ----------
        uc_type : str
            Uncertainy type. Must be either of "input", "state" or "process".
        dep_model : DependencyDescription
            Description of the state and input dependency (see class above).
        dim : int, optional
            Dimension of this q term. Obligatory if S is not provided.
        L : array, optional
            Optional specification of mapping matrix for set,
            {L*q : ||q||<=phi(||x||,||u||)}. Obligatory if dim is not provided.
        """
        if L is None:
            L = np.eye(dim)
        else:
            dim = L.shape[1]
        self.mtx_map.append(L)
        self.L.append(np.vstack((np.zeros((self.W.shape[0],dim)),L)))
        self.dependency.append(dep_model)
        ball = NormBall(dim,dep_model.pq)
        self.sets.append(ball)
        sampler = lambda x,u: ball(dep_model.phi_direct(cvx.Constant(x),cvx.Constant(u)).value)
        self.sample_q.append(sampler)
        self.sample_sim.append(lambda t,x,u: L.dot(sampler(x,u)))
        self.type.append(uc_type)
        self.is_dependent.append(True)
        # Extend the other matrices
        self.W = np.vstack((self.W,np.zeros((L.shape[0],self.W.shape[1]))))
        for i in range(len(self.L)-1):
            self.L[i] = np.vstack((self.L[i],np.zeros((L.shape[0],self.L[i].shape[1]))))
            
    def generateRandomPoint(self,t,x,u):
        """
        Uniformly sample the uncertainty set.
        
        Parameters
        ----------
        t : float
            Time.
        x : array
            Current state.
        u : array
            Current input.
        """
        w = np.concatenate([self.sample_w[i](t) for i in
                            range(len(self.sample_w))])
        q = [self.sample_q[i](x,u) for i in range(len(self.sample_q))]
        return self.W.dot(w)+sum([self.L[i].dot(q[i]) for i in
                                  range(len(self.sample_q))])

class ExcitationTypes:
    """
    Excitation types. Currently implemented only for the Hyperrectangle set.
    """
    DEFAULT = 0
    CONSTANT = 1
    SINUSOIDAL = 2

class Hyperrectangle(Set):
    def setup(self, lb, ub):
        """
        Parameters
        ----------
        lb: array
            Lower bound.
        ub: array
            Upper bound.
        """
        self.lb = lb
        self.ub = ub
            
        self.excitation_types = ExcitationTypes()
        self.excitation = self.excitation_types.DEFAULT
        
    def generateRandomPoint(self,t=None):
        """
        Samples the hyperrectangle. Default behaviour is to uniformly randomly
        sample the hyperrectangle.
        
        Parameters
        ----------
        t : float, optional
            Time.
        """
        if self.excitation == self.excitation_types.CONSTANT:
            return self.excitation_value
        if self.excitation == self.excitation_types.SINUSOIDAL:
            if t is None:
                raise ValueError(general.ERROR+
                                 "t must be specified for sinusoidal disturbance")
            return self.excitation_value(t)
        elif self.excitation == self.excitation_types.DEFAULT:
            return np.random.uniform(low=self.lb, high=self.ub)
        
    def constantExcitation(self, value):
        """
        Make the sampling function output a constant value inside the
        hyperrectangle. Useful for testing a constant persistent excitation.
        
        Parameters
        ----------
            value : array
                The value to be output.
        """
        if np.any(value>self.ub) or np.any(value<self.lb):
            raise ValueError(general.ERROR+"Value cannot be outside the hyperrectangle")
        
        self.excitation = self.excitation_types.CONSTANT
        self.excitation_value = value
        
    def sinusoidalExcitation(self, amplitude, period, offset=0.):
        """
        Make the sampling function output a sinusoid sinde the hyperrectangle:
            
            disturbance(t) = amplitude*sin(2*pi*t/period)+offset
        
        Parameters
        ----------
            amplitude : float
                Sinusoid amplitude.
            period : float
                Sinusoid period.
            offset : float, optional
                Sinusoid offset from zero.
        """
        if np.any(offset+amplitude>self.ub) or np.any(offset-amplitude<self.lb):
            raise ValueError(general.ERROR+
                             "Sinusoid cannot go outside the hyperrectangle")

        self.excitation = self.excitation_types.SINUSOIDAL
        self.excitation_value = lambda t: amplitude*np.sin(2*np.pi*t/period)+offset

    def convertToPolytope(self):
        R = []
        for lb,ub in zip(self.lb,self.ub):
            R.append((lb,ub))
        poly = Polytope(R=R)
        return poly.A, poly.b, poly

class Ellipsoid(Set):
    """
    An ellipsoid set of the form {x : x^T*P*x <= t} where P is fixed and t is
    defined at runtime.
    """
    def setup(self, P):
        """
        Parameters
        ----------
            P: array
                Symmetric positive definite matrix defining the ellipsoid
                shape.
        """
        if not np.allclose(P, P.T):
            raise ValueError(general.ERROR+"P must be symmetric")
        if not np.all(la.eigvals(P) > 0):
            raise ValueError(general.ERROR+"P must be positive definite")
        if la.cond(P) > 1e5:
            raise ValueError(general.ERROR+"P is very badly conditioned (condition number "
                            "%.4e)"%(la.cond(P)))
            
        self.P = P # {x : x^T*P*x <= t}
        self.S = la.inv(P) # "Covariance" matrix, {x : x^T*S^-1*x <= t}
        self.L = la.cholesky(self.S) # Cholesky factorization of S => S=L*L^T
        
    def generateRandomPoint(self,t):
        """
        Uniformly samples the ellipsoid {x : x^T*P*x <= t}. Implements [1].
        
        [1] Dezert, J. and Musso, C. "An Efficient Method for Generating Points
            Uniformly Distributed in Hyperellipsoids", Proceedings of the
            Workshop on Estimation, Tracking and Fusion: A Tribute to Yaakov
            Bar-Shalom, Naval Postgraduate School, May 2001.
        
        Parameters
        ----------
        t : float
            Right hand side of the inequality ellipsoid definition, which
            scales the ellipsoid.
            
        Returns
        -------
        : array
            The random point.
        """
        return ss.sampleUniformEllipsoid(L=self.L,t=t)
    
    def convertToHyperrectangle(self, t):
        """
        Find a minimum-volume hyperrectangle that outer bounds the ellipsoid
        {x : x^T*P*x <= t}
        
        Parameters
        ----------
        t : float
            Right hand side of the inequality ellipsoid definition, which
            scales the ellipsoid.
        
        Returns
        -------
        H : Hyperrectangle
            Smallest hyperrectangle that contains the ellipsoid.
        """
        S = self.S
        m = S.shape[0]
        # Find upper bound
        u = cvx.Variable(m)
        y = cvx.Variable(m)
        cost = cvx.Minimize(sum(u))
        constraints  = [cvx.inv_pos(4*y[i])*S[i,i]+y[i]*t<=u[i] for i in range(m)]
        constraints += [y >= 0]
        problem = cvx.Problem(cost, constraints)
        problem.solve(solver=cvx.ECOS)
        u = np.array(u.value.T).flatten()
        # Find lower bound
        l = cvx.Variable(m)
        cost = cvx.Maximize(sum(l))
        constraints  = [-cvx.inv_pos(4*y[i])*S[i,i]-y[i]*t>=l[i] for i in range(m)]
        constraints += [y >= 0]
        problem = cvx.Problem(cost, constraints)
        problem.solve(solver=cvx.ECOS)
        l = np.array(l.value.T).flatten()
        # Make the hyperrectangle
        H = Hyperrectangle(l,u)
        return H

class NormBall(Set):
    """
    A p-norm ball {A*x : ||x||_p <= b} \subset R^n where b can be variable.
    """
    def setup(self, dim, p, A=None):
        """
        Parameters
        ----------
        dim : int
            Dimension.
        p : 1,2 or np.inf
            What p-norm to use.
        A : array, optional
            Direct mapping matrix, must be square and invertible. Identity by
            default.
        """
        A = np.eye(dim) if A is None else A
        self.p = p
        if self.p==1:
            # Polytope
            # Generate the matrix of facet normals
            P = np.array(list(itertools.product([-1,1], repeat=dim)))
            P = P.dot(la.inv(A))
            # Make the set with a default facet distance
            p = np.ones(dim*2)
            self.set = Polytope(P,p)
        elif self.p==2:
            # Ellipsoid
            self.set = Ellipsoid(la.matrix_power(la.inv(A),2))
            #sampling_function = lambda x,u: E(phi(x,u).value**2)
        else:
            # Polytope
            # Generate the matrix of facet normals
            P = Polytope(R=[(-1.,1.) for i in range(dim)]).A
            P = P.dot(la.inv(A))
            # Make the set with a default facet distance
            p = np.ones(dim*2)
            self.set = Polytope(P,p)

    def generateRandomPoint(self, r):
        """
        Generates a random point inside the norm ball.
        
        Parameters
        ----------
        r : float
            Ball radius.
            
        Returns
        -------
        random_point : array
            A random point in the norm ball.
        """
        if self.p==2:
            random_point = self.set(r**2)
        else:
            # 1- or infinity-norm
            p_default = np.copy(self.set.b)
            self.set.b *= r
            random_point = self.set.randomPoint()
            self.set.b = p_default
        return random_point
        
    def convertToPolytope(self, r):
        if self.p==2:
            H = self.set.convertToHyperrectangle(r**2)
            A,b,poly = H.convertToPolytope()
        else:
            # 1- or infinity-norm
            A,b = self.set.A, self.set.b*r
            poly = Polytope(A,b)
        return A,b,poly

class AffineSet:
    def __init__(self,x0,E,v=None):
        """
        Parameters
        ----------
        x0 : array
            Affine set offset from origin.
        E : array
            Affine set basis.
        v : array, optional
            Custom values where element 0 is associated with x0 and for i>0 the
            i-th element is associated with the point x0+E_i where E_i is the
            i-th column of E.
        """
        self.x0 = x0
        self.E = E
        self.v = v
