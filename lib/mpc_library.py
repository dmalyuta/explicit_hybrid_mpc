"""
Library of MPC problems.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import time
import numpy as np
import numpy.linalg as la
from numpy.linalg import matrix_power as mpow
import scipy.linalg as sla
import cvxpy as cvx

import global_vars
import tools
from polytope import Polytope,subtractHyperrectangles
import uncertainty_sets as uc
from specifications import Specifications
from plant import Plant

"""
All MPC classes provide access to the following variables:
    - N : prediction horizon length
    - Nu : commutation dimension per time step
    - n_x : state dimension
    - n_u : control dimension
    - x : list of optimized state variables
    - u : list of optimized input variables
    - delta : optimized commutation variable
    - x0 : the parameter variable
    - V : optimized cost expression
    - cost : CVXPY cost
    - make_constraints : callable function to make basic MPC constraints
"""

class MPC:
    def __init__(self):
        def make_constraints(theta,x,u,delta,delta_sum_constraint=True):
            """
            Reusable problem constraints for commutation as a variable or as a
            parameter.
            
            Parameters
            ----------
            delta: cvx.Variable or cvx.Parameter
                Commutation.
                
            Returns
            -------
            constraints : list
                The optimization problem constraints.
            """
            raise NotImplementedError('make_constraints() not implemented')
        self.make_constraints = make_constraints
        
    def setup_RMPC(self,Q_coeff):
        """
        Setup robust MPC optimization problem.
        
        Parameters
        ----------
        Q_coeff : float
            Coefficient to multiply identity state weight by in the cost.
        """
        # Setup optimization problem matrices and other values
        W = self.specs.P.W
        R = self.specs.P.R
        r = self.specs.P.r
        U_partitions = subtractHyperrectangles(self.specs.U_int,self.specs.U_ext)
        H = [Up.A for Up in U_partitions]
        h = [Up.b for Up in U_partitions]
        self.Nu = len(H) # Number of convex sets whose union makes the control set (**excluding** the {0} set)
        qq = [(1./(1-1./dep.pq) if dep.pq!=1 else np.inf) for dep in
              self.specs.P.dependency]
        D = self.plant.D(self.specs)
        n_q = len(self.specs.P.L)
        # scaling
        D_w = Polytope(self.specs.P.R,self.specs.P.r).computeScalingMatrix()
        self.D_x = self.specs.X.computeScalingMatrix()
        self.p_u,self.D_u = [None for _ in range(self.Nu)],[None for _ in range(self.Nu)]
        for i in range(self.Nu):
            # Scale to unity and re-center with respect to set {u_i: H[i]*u_i<=h[i]}
            self.p_u[i] = np.mean(Polytope(A=H[i],b=h[i]).V,axis=0) # barycenter
            self.D_u[i] = np.empty(H[i].shape)
            for j in range(H[i].shape[0]):
                alpha = cvx.Variable()
                center = self.p_u[i]
                ray = H[i][j]
                cost = cvx.Maximize(alpha)
                constraints = [H[i]*(center+alpha*ray)<=h[i]]
                problem = cvx.Problem(cost,constraints)
                problem.solve(**global_vars.SOLVER_OPTIONS)
                self.D_u[i][j] = alpha.value*ray
            self.D_u[i] = self.D_u[i].T
        self.D_u_box = self.specs.U_ext.computeScalingMatrix()
        
        # Robust term for polytopic noise computation
        G,n_g = self.specs.X.A, self.specs.X.b.size
        def robust_term(i,j,k):
            """
            Computes \max_{R*w<=r} G_j^T*A^{k-1-i}*D*W*w which is the effect of
            the worst case independent disturbance at time step i when looking
            at its propagated effect at time step k, along the j-th facet of
            the invariant set.
            
            Parameters
            ----------
            i : int
                Time step in the k-step horizon.
            j : int
                Invariant polytope facet index.
            k : int
                Horizon length, k.
                
            Returns
            -------
            : float
                \max_{R*w<=r} G_j^T*A^{k-1-i}*D*W*w.
            """
            w = cvx.Variable(R.shape[1])
            cost = cvx.Maximize(G[j].dot(mpow(self.plant.A,k-1-i)).dot(D).dot(W).dot(D_w)*w)
            constraints = [R.dot(D_w)*w <= r]
            problem = cvx.Problem(cost,constraints)
            return problem.solve(**global_vars.SOLVER_OPTIONS)
        sum_sigma = []
        for k in tools.fullrange(self.N):
            sum_sigma_facet = []
            for j in range(n_g):
                sum_sigma_facet.append(sum([robust_term(i,j,k) for i in range(k)]))
            sum_sigma += sum_sigma_facet
        sum_sigma = np.array(sum_sigma)
        
        # Setup MPC optimization problem
        def make_ux():
            """
            Make input and state optimization variables.
            
            Returns
            -------
            variables : list
                Dictionary of variable lists. 'x' amd 'u' are dimensional
                (unscaled) states and inputs; 'xhat' and 'uhat' are
                dimensionless (scaled) states and inputs.
            """
            uhat = [[cvx.Variable(self.D_u[i].shape[1]) for k in range(self.N)] for i in range(self.Nu)]
            u = [[self.p_u[i]+self.D_u[i]*uhat[i][k] for k in range(self.N)] for i in range(self.Nu)]
            xhat = [cvx.Variable(self.n_x) for k in range(self.N+1)]
            x = [self.D_x*xhat[k] for k in range(self.N+1)]
            variables = dict(u=u,x=x,uhat=uhat,xhat=xhat)
            return variables
        
        self.make_ux = make_ux
            
        self.x0 = cvx.Parameter(self.n_x)
        self.delta = cvx.Variable(self.Nu*self.N, boolean=True)
        xu = self.make_ux()
        self.u = xu['u']
        self.x = xu['x']
        xhat = xu['xhat']
        
        def get_u0_value():
            """
            Get the optimal value of the first input in the MPC horizon.
            
            Returns
            -------
            u0_opt : array
                Optimal value of the first input in the MPC horizon.
            """
            return sum([self.u[__i][0].value for __i in range(self.Nu)])
        
        self.get_u0_value = get_u0_value
        
        Q = Q_coeff*np.eye(self.n_x)
        R = np.eye(self.n_u)
        self.V = (sum([cvx.quad_form(la.inv(self.D_u_box)*sum([self.u[i][k] for i in range(self.Nu)]),R) for k in range(self.N)])+
                  sum([cvx.quad_form(xhat[k],Q) for k in tools.fullrange(1,self.N)]))
        self.cost = cvx.Minimize(self.V)
        
        def make_constraints(theta,x,u,delta,delta_sum_constraint=True):
            constraints = []
            # Nominal dynamics
            sum_u = lambda k: sum([u[__i][k] for __i in range(self.Nu)])
            constraints += [x[k+1] == self.plant.A*x[k]+self.plant.B*sum_u(k) for k in range(self.N)]
            constraints += [x[0] == theta]
            # Robustness constraint tightening
            G,g,n_g = self.specs.X.A, self.specs.X.b, self.specs.X.b.size
            constraints += [G*x[k]+
                            sum_sigma[(k-1)*n_g:k*n_g]+
                            sum([sum([
                            np.array([
                            la.norm(G[j].dot(mpow(self.plant.A,k-1-i)).dot(D).dot(self.specs.P.L[l]),ord=qq[l])
                            for j in range(n_g)])*
                            self.specs.P.dependency[l].phi_direct(x[i],sum_u(i))
                            for l in range(n_q)])
                            for i in range(k)])
                            <=g
                            for k in tools.fullrange(self.N)]
            # Input constraint
            for i in range(self.Nu):
                constraints += [H[i]*u[i][k] <= h[i]*delta[self.Nu*k+i] for k in range(self.N)]
            # input is in at least one of the convex subsets
            if delta_sum_constraint:
                constraints += [sum([delta[self.Nu*k+i] for i in range(self.Nu)]) <= 1 for k in range(self.N)]
            return constraints
        
        self.make_constraints = make_constraints

class SatelliteZ(MPC):
    """
    CWH z-dynamics with non-convex input constraint (off or on and
    lower-bounded) and constraint tightening for noise robustness.
    """
    def __init__(self):
        super().__init__()
        # Parameters
        self.N = global_vars.MPC_HORIZON # Prediction horizon length
        self.pars = satellite_parameters()
        
        # Specifications
        X = Polytope(R=[(-self.pars['pos_err_max'],self.pars['pos_err_max']),
                        (-self.pars['vel_err_max'],self.pars['vel_err_max'])])
        U_ext = Polytope(R=[(-self.pars['delta_v_max'],self.pars['delta_v_max'])])
        U_int = Polytope(R=[(-self.pars['delta_v_min'],self.pars['delta_v_min'])])
        # uncertainty set
        P = uc.UncertaintySet() # Left empty for chosen_spec=='nominal'
        n = 1 # Position dimension
        one,I,O = np.ones(n),np.eye(n),np.zeros((n,n))
        P.addIndependentTerm('process',lb=-self.pars['w_max']*one,ub=self.pars['w_max']*one)
        P.addIndependentTerm('state',lb=-np.concatenate([self.pars['p_max']*one,
                                                         self.pars['v_max']*one]),
                                     ub=np.concatenate([self.pars['p_max']*one,
                                                        self.pars['v_max']*one]))
        P.addDependentTerm('input',uc.DependencyDescription(
                           lambda: cvx.Constant(self.pars['sigma_fix']),pq=2),dim=n)
        P.addDependentTerm('state',uc.DependencyDescription(
                           lambda nfx: self.pars['sigma_pos']*nfx,pq=np.inf,px=2,
                           Fx=np.hstack((I,O))),dim=n,L=np.vstack((I,O)))
        P.addDependentTerm('state',uc.DependencyDescription(
                           lambda nfx: self.pars['sigma_vel']*nfx,pq=np.inf,px=2,
                           Fx=np.hstack((O,I))),dim=n,L=np.vstack((O,I)))
        P.addDependentTerm('input',uc.DependencyDescription(
                           lambda nfu: self.pars['sigma_rcs']*nfu,pq=2,pu=2),dim=n)
        self.specs = Specifications(X,(U_int,U_ext),P)
        
        # Make plant
        # continuous-time
        self.n_x, self.n_u = 2,1
        A_c = np.array([[0.,1.],[-self.pars['wo']**2,0.]])
        B_c = np.array([[0.],[1.]])
        E_c = B_c.copy()
        # discrete-time
        A = sla.expm(A_c*self.pars['T_s'])
        B = A.dot(B_c)
        M = np.block([[A_c,E_c],[np.zeros((self.n_u,self.n_x+self.n_u))]])
        E = sla.expm(M)[:self.n_x,self.n_x:]
        self.plant = Plant(self.pars['T_s'],A,B,E)

        # Setup the RMPC problem        
        self.setup_RMPC(Q_coeff=1e-2)

class SatelliteXY(MPC):
    """
    CWH (x,y)-dynamics with non-convex input constraint (off or on and
    lower-bounded) and constraint tightening for noise robustness.
    """
    def __init__(self):
        super().__init__()
        # Parameters
        self.N = global_vars.MPC_HORIZON # Prediction horizon length
        self.pars = satellite_parameters()
        
        # Specifications
        X = Polytope(R=[(-self.pars['pos_err_max'],self.pars['pos_err_max']),
                        (-self.pars['pos_err_max'],self.pars['pos_err_max']),
                        (-self.pars['vel_err_max'],self.pars['vel_err_max']),
                        (-self.pars['vel_err_max'],self.pars['vel_err_max'])])
        U_ext = Polytope(R=[(-self.pars['delta_v_max'],self.pars['delta_v_max']),
                            (-self.pars['delta_v_max'],self.pars['delta_v_max'])])
        U_int = Polytope(R=[(-self.pars['delta_v_min'],self.pars['delta_v_min']),
                            (-self.pars['delta_v_min'],self.pars['delta_v_min'])])
        # uncertainty set
        P = uc.UncertaintySet() # Left empty for chosen_spec=='nominal'
        n = 2 # Position dimension
        one,I,O = np.ones(n),np.eye(n),np.zeros((n,n))
        P.addIndependentTerm('process',lb=-self.pars['w_max']*one,ub=self.pars['w_max']*one)
        P.addIndependentTerm('state',lb=-np.concatenate([self.pars['p_max']*one,
                                                         self.pars['v_max']*one]),
                                     ub=np.concatenate([self.pars['p_max']*one,
                                                        self.pars['v_max']*one]))
        P.addDependentTerm('input',uc.DependencyDescription(
                           lambda: cvx.Constant(self.pars['sigma_fix']),pq=2),dim=n)
        P.addDependentTerm('state',uc.DependencyDescription(
                           lambda nfx: self.pars['sigma_pos']*nfx,pq=np.inf,px=2,
                           Fx=np.hstack((I,O))),dim=n,L=np.vstack((I,O)))
        P.addDependentTerm('state',uc.DependencyDescription(
                           lambda nfx: self.pars['sigma_vel']*nfx,pq=np.inf,px=2,
                           Fx=np.hstack((O,I))),dim=n,L=np.vstack((O,I)))
        P.addDependentTerm('input',uc.DependencyDescription(
                           lambda nfu: self.pars['sigma_rcs']*nfu,pq=2,pu=2),dim=n)
        self.specs = Specifications(X,(U_int,U_ext),P)
        
        # Make plant
        # continuous-time
        self.n_x, self.n_u = 4,2
        A_c = np.array([[0.,0.,1.,0.],
                        [0.,0.,0.,1.],
                        [3.*self.pars['wo']**2,0.,0.,2.*self.pars['wo']],
                        [0.,0.,-2.*self.pars['wo'],0.]])
        B_c = np.array([[0.,0.],
                        [0.,0.],
                        [1.,0.],
                        [0.,1.]])
        E_c = B_c.copy()
        # discrete-time
        A = sla.expm(A_c*self.pars['T_s'])
        B = A.dot(B_c)
        M = np.block([[A_c,E_c],[np.zeros((self.n_u,self.n_x+self.n_u))]])
        E = sla.expm(M)[:self.n_x,self.n_x:]
        self.plant = Plant(self.pars['T_s'],A,B,E)

        # Setup the RMPC problem        
        self.setup_RMPC(Q_coeff=1e-2)

class SatelliteXYZ(MPC):
    """
    CWH (x,y,z)-dynamics with non-convex input constraint (off or on and
    lower-bounded) and constraint tightening for noise robustness.
    """
    def __init__(self):
        super().__init__()
        # Parameters
        self.N = global_vars.MPC_HORIZON # Prediction horizon length
        self.pars = satellite_parameters()
        
        # Specifications
        X = Polytope(R=[(-self.pars['pos_err_max'],self.pars['pos_err_max']),
                        (-self.pars['pos_err_max'],self.pars['pos_err_max']),
                        (-self.pars['pos_err_max'],self.pars['pos_err_max']),
                        (-self.pars['vel_err_max'],self.pars['vel_err_max']),
                        (-self.pars['vel_err_max'],self.pars['vel_err_max']),
                        (-self.pars['vel_err_max'],self.pars['vel_err_max'])])
        U_ext = Polytope(R=[(-self.pars['delta_v_max'],self.pars['delta_v_max']),
                            (-self.pars['delta_v_max'],self.pars['delta_v_max']),
                            (-self.pars['delta_v_max'],self.pars['delta_v_max'])])
        U_int = Polytope(R=[(-self.pars['delta_v_min'],self.pars['delta_v_min']),
                            (-self.pars['delta_v_min'],self.pars['delta_v_min']),
                            (-self.pars['delta_v_min'],self.pars['delta_v_min'])])
        # uncertainty set
        P = uc.UncertaintySet() # Left empty for chosen_spec=='nominal'
        n = 3 # Position dimension
        one,I,O = np.ones(n),np.eye(n),np.zeros((n,n))
        P.addIndependentTerm('process',lb=-self.pars['w_max']*one,ub=self.pars['w_max']*one)
        P.addIndependentTerm('state',lb=-np.concatenate([self.pars['p_max']*one,
                                                         self.pars['v_max']*one]),
                                     ub=np.concatenate([self.pars['p_max']*one,
                                                        self.pars['v_max']*one]))
        P.addDependentTerm('input',uc.DependencyDescription(
                           lambda: cvx.Constant(self.pars['sigma_fix']),pq=2),dim=n)
        P.addDependentTerm('state',uc.DependencyDescription(
                           lambda nfx: self.pars['sigma_pos']*nfx,pq=np.inf,px=2,
                           Fx=np.hstack((I,O))),dim=n,L=np.vstack((I,O)))
        P.addDependentTerm('state',uc.DependencyDescription(
                           lambda nfx: self.pars['sigma_vel']*nfx,pq=np.inf,px=2,
                           Fx=np.hstack((O,I))),dim=n,L=np.vstack((O,I)))
        P.addDependentTerm('input',uc.DependencyDescription(
                           lambda nfu: self.pars['sigma_rcs']*nfu,pq=2,pu=2),dim=n)
        self.specs = Specifications(X,(U_int,U_ext),P)
        
        # Make plant
        # continuous-time
        self.n_x, self.n_u = 6,3
        A_c = np.array([[0.,0.,0.,1.,0.,0.],
                        [0.,0.,0.,0.,1.,0.],
                        [0.,0.,0.,0.,0.,1.],
                        [3.*self.pars['wo']**2,0.,0.,0.,2.*self.pars['wo'],0.],
                        [0.,0.,0.,-2.*self.pars['wo'],0.,0.],
                        [0.,0.,-self.pars['wo']**2,0.,0.,0.]])
        B_c = np.array([[0.,0.,0.],
                        [0.,0.,0.],
                        [0.,0.,0.],
                        [1.,0.,0.],
                        [0.,1.,0.],
                        [0.,0.,1.]])
        E_c = B_c.copy()
        # discrete-time
        A = sla.expm(A_c*self.pars['T_s'])
        B = A.dot(B_c)
        M = np.block([[A_c,E_c],[np.zeros((self.n_u,self.n_x+self.n_u))]])
        E = sla.expm(M)[:self.n_x,self.n_x:]
        self.plant = Plant(self.pars['T_s'],A,B,E)

        # Setup the RMPC problem        
        self.setup_RMPC(Q_coeff=1e-2)

class ExplicitMPC:
    def __init__(self,tree):
        """
        Parameters
        ----------
        tree : Tree
            Root of the explicit MPC partition tree.
        """
        self.tree = tree
        self.setup()

    def setup(self):
        """Readies the explicit MPC implementation for use."""
        self.compute_simplex_basis_inverse()
        self.eps = np.finfo(np.float64).eps # machine epsilon precision

    def compute_simplex_basis_inverse(self):
        """
        Give a simplex S=co{v_0,...,v_n}\in R^n, let M=[v_1-v_0 ... v_n-v_0]\in
        R^{n x n} and c=v_0. We can check if x\in S by verifying that every
        element of inv(S)*(x-c) is \in [0,1].

        This function computes S for each "left child" simplex of the
        partition. Since the partition is a binary tree, no need to compute for
        the "right child" since, if not in left child ==> in right child.
        """
        def compute_Minv_for_each_left_child(cursor):
            """
            See the above description.
            **NB**: modifies cursor (adds new member variable Minv to it).
            
            Parameters
            ----------
            cursor : Tree
                Tree root.
            """
            Minv = lambda v: la.inv(np.column_stack([_v-v[0] for _v in v[1:]]))
            if cursor.is_leaf():
                cursor.data.Minv = Minv(cursor.data.vertices)
            else:
                cursor.left.data.Minv = Minv(cursor.left.data.vertices)
                compute_Minv_for_each_left_child(cursor.left)
                compute_Minv_for_each_left_child(cursor.right)
        compute_Minv_for_each_left_child(self.tree)

    def check_containment(self,x,cell):
        """
        Checks if x \in simplex cell.

        Parameters
        ----------
        x : np.array
            Vector to be checked if it is contained in the simplex.
        cell : NodeData
            Cell data.

        Returns
        -------
        is_contained : bool
            ``True`` if x \in cell.
        """
        c = cell.vertices[0]
        Minv = cell.Minv
        alpha = list(Minv.dot(x-c))
        alpha.append(1-sum(alpha)) # alpha[0], but store last since O(1)
        is_contained = np.all([a>=-self.eps and a<=1+self.eps for a in alpha])
        return is_contained

    def get_containing_cell(self,x):
        """
        Get the data of the cell which contains parameter x.

        Parameters
        ----------
        x : np.array
            Parameter in the multiparameteric MPC (i.e. current state).

        Returns
        -------
        : NodeData
            Data of the cell which contains x.
        """
        def browse_tree(cursor):
            """
            Searches the tree of the cell which contains x.

            Parameters
            ----------
            cursor : Tree
                Root of the tree.
            """
            if cursor.is_leaf():
                return cursor.data
            else:
                if self.check_containment(x,cursor.left.data):
                    return browse_tree(cursor.left)
                else:
                    return browse_tree(cursor.right)
        return browse_tree(self.tree)
    
    def __call__(self,x):
        """
        Returns epsilon-suboptimal control input for the given parameter x.

        [1] D. Malyuta, B. Acikmese, M. Cacan, and D. S. Bayard, "Approximate
        Mixed-integer Convex Multiparametric Programming," in Control Systems
        Letters (in review), IEEE.

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
        tic = time.time()
        R = self.get_containing_cell(x)
        alpha = R.Minv.dot(x-R.vertices[0])
        alpha0 = 1-sum(alpha)
        u = alpha0*R.vertex_inputs[0]+R.vertex_inputs[1:].T.dot(alpha)
        toc = time.time()
        t = toc-tic
        return u,t

def satellite_parameters():
    """
    Compile CWH MPC common parameters.

    Returns
    -------
    pars : dict
        Dictionary of parameters.
    """
    # Raw values
    pars = {'mu': 3.986004418e14,  # [m^3*s^-2] Standard gravitational parameter
            'R_E': 6378137.,       # [m] Earth mean radius
            'h_E': 415e3,          # [m] Orbit height above Earth sea level
            'T_s': 100,            # [s] Silent time
            'pos_err_max': 10e-2,  # [m] Maximum position error
            'vel_err_max': 1e-3,   # [m/s] Maximum velocity error
            'delta_v_max': 2e-3,   # [m/s] Maximum input delta-v
            'w_max': 50e-9,        # [m/s^2] Maximum exogenous acceleration (atmospheric drag)
            'sigma_fix': 1e-6,     # [m/s] Fixed input error
            'sigma_pos':2e-2,      # Position estimate error growth slope with position magnitude
            'sigma_vel':1e-3,      # Velocity estimate error growth slope with velocity magnitude
            'input_ang_err': 2.,   # [deg] Cone opening angle for input error
            'p_max': 0.4e-2,       # [m] Maximum position estimate error
            'v_max': 4e-6}         # [m/s] Maximum velocity estimate error
    # Derived values
    pars.update({'a': pars['h_E']+pars['R_E']})             # [m] Orbit radius
    pars.update({'wo': np.sqrt(pars['mu']/pars['a']**3)})   # [rad/s] Orbital rate
    pars.update({'delta_v_min': pars['delta_v_max']*0.01}) # [m/s] Min non-zero delta-v input (in each axis)
    pars.update({'sigma_rcs': np.tan(np.deg2rad(pars['input_ang_err'])/2.)}) # Input error growth slope with input magnitude
    return pars
