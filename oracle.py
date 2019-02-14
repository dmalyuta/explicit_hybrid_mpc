"""
Library of optimization problems used by the partitioning process.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import numpy as np
import numpy.linalg as la
import cvxpy as cvx

class Oracle:
    """
    "Oracle" problems, i.e. the optimization programs that need to be called
    by the partitioning algorithm.
    """
    solver_options = dict(solver=cvx.GUROBI, verbose=False)
    
    def __init__(self,eps_a,eps_r):
        """
        Pre-parses the oracle problems.
        
        Parameters
        ----------
        eps_a : float
            Absolute error tolerance.
        eps_r : float
            Relative error tolerance (>0 where 0 is fully tight, 1 is 100%
            relative error, etc.).
        """
        self.eps_a = eps_a
        self.eps_r = eps_r
        
        # Parameters
        m = 1. # [kg] Cart mass
        h = 1./20. # [s] Time step
        self.N = 10 # Prediction horizon length
        
        # Discretized dynamics Ax+Bu
        self.n_x,n_u = 2,1
        A = np.array([[1.,h],[0.,1.]])
        B = np.array([[h**2/2.],[h]])/m
        
        # Control constraints
        self.Nu = 3 # Number of control convex subsets, whose union makes the control set
        lb, ub = 0.05, 1.
        P = [np.array([[1.],[-1.]]),np.array([[-1.],[1.]]),np.array([[1.],[-1.]])]
        p = [np.array([ub*m,-lb*m]),np.array([ub*m,-lb*m]),np.zeros((2))]
        M = np.array([ub*m,ub*m])*10
        
        # Control objectives
        e_p_max = 0.1 # [m] Max position error
        e_v_max = 0.2 # [m/s] Max velocity error
        u_max = 10.*m # [N] Max control input
        
        # Cost
        D_x = np.diag([e_p_max,e_v_max])
        D_u = np.diag([u_max])
        Q = 100.*la.inv(D_x).dot(np.eye(self.n_x)).dot(la.inv(D_x))
        R = la.inv(D_u).dot(np.eye(n_u)).dot(la.inv(D_u))
        
        # MPC optimization problem
        x = [cvx.Variable(self.n_x) for k in range(self.N+1)]
        self.u = [cvx.Variable(n_u) for k in range(self.N)]
        self.delta = cvx.Variable(self.Nu*self.N, boolean=True)
        self.x0 = cvx.Parameter(self.n_x)
        
        V = sum([cvx.quad_form(x[k],Q)+
                 cvx.quad_form(self.u[k],R) for k in range(self.N)])
        cost = cvx.Minimize(V)
        
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
            constraints = []
            constraints += [x[k+1] == A*x[k]+B*u[k] for k in range(self.N)]
            constraints += [x[0] == theta]
            constraints += [x[-1] == np.zeros(self.n_x)]
            for i in range(self.Nu):
                constraints += [P[i]*u[k] <= p[i]+M*delta[self.Nu*k+i] for k in range(self.N)]
            if delta_sum_constraint:
                constraints += [sum([delta[self.Nu*k+i] for i in range(self.Nu)]) <= self.Nu-1 for k in range(self.N)]
            return constraints
        
        # Make P_theta, the original MINLP
        constraints = make_constraints(self.x0,x,self.u,self.delta)
        self.minlp = cvx.Problem(cost,constraints)
        self.minlp_feasibility = cvx.Problem(cvx.Minimize(0),constraints) # just to check feasibility
        
        # Make P_theta_delta, the fixed-commutation NLP
        self.delta_fixed = cvx.Parameter(self.Nu*self.N)
        constraints = make_constraints(self.x0,x,self.u,self.delta_fixed)
        self.nlp = cvx.Problem(cost,constraints)
        
        # Make V^R, the find-feasible-commutation-in-simplex MINLP
        x_theta = [[cvx.Variable(self.n_x) for k in range(self.N+1)] for _ in range(self.n_x+1)]
        u_theta = [[cvx.Variable(n_u) for k in range(self.N+1)] for _ in range(self.n_x+1)]
        self.vertices = [cvx.Parameter(self.n_x) for _ in range(self.n_x+1)]
        feasibility_constraints = []
        for i in range(self.n_x+1):
            # Add a set of constraints for each vertex
            feasibility_constraints += make_constraints(self.vertices[i],x_theta[i],u_theta[i],self.delta,
                                                        delta_sum_constraint=True if i==0 else False)
        self.feasibility_in_simplex = cvx.Problem(cost,feasibility_constraints)
        
        # Make bar_E_a^R, the convex absolute error over-approximator
        self.vertex_costs = cvx.Parameter(self.n_x+1)
        alpha = cvx.Variable(self.n_x+1,nonneg=True)
        self.theta_in_simplex = sum([alpha[i]*self.vertices[i] for i in range(self.n_x+1)])
        constraints = make_constraints(self.theta_in_simplex,x,self.u,self.delta)
        # theta stays in simplex
        extra_constraints = []
        in_simplex = [sum(alpha)==1]
        extra_constraints += in_simplex
        # commutation is not equal to the reference commutation
        delta_offset = cvx.Variable(self.Nu*self.N, boolean=True)
        extra_constraints += [self.delta[self.Nu*k+i] == self.delta_fixed[self.Nu*k+i]+
                              (1-2*self.delta_fixed[self.Nu*k+i])*delta_offset[self.Nu*k+i]
                              for k in range(self.N) for i in range(self.Nu)]
        extra_constraints += [sum([delta_offset[self.Nu*k+i] for k in range(self.N)
                                   for i in range(self.Nu)]) >= 1]
        constraints += extra_constraints
        # cost using affine over-approximator for reference commutation
        bar_V = sum([alpha[i]*self.vertex_costs[i] for i in range(self.n_x+1)])
        cost_abs_err = cvx.Maximize(bar_V-V)
        self.abs_err_overapprox = cvx.Problem(cost_abs_err,constraints)
        
        # Make denominator of bar_E_r^R, which is P_theta restricted to
        # commutation not being equal to a reference commutation and minimum
        # found over a full simplex
        self.rel_err_denom = cvx.Problem(cost,constraints)
        
        # Make problems that allow verifying simplex-is-in-variability-ball
        constraints = make_constraints(self.theta_in_simplex,x,self.u,self.delta_fixed)+in_simplex
        self.min_over_simplex_for_this_delta = cvx.Problem(cost,constraints)
        constraints = make_constraints(self.theta_in_simplex,x,self.u,self.delta)+in_simplex
        self.min_over_simplex_for_any_delta = cvx.Problem(cost,constraints)
        
        # Make D_delta^R, the MINLP that searches for a more optimal commutation
        # First make problem fo finding min over simplex, *not* using a reference commutation
        constraints = make_constraints(self.theta_in_simplex,x,self.u,self.delta)+extra_constraints
        self.min_over_simplex_for_any_delta_except_ref = cvx.Problem(cost,constraints)
        # Now made D_delta^R itself
        epsilon = cvx.Variable()
        self.min_V_except_delta = cvx.Parameter() # Value = self.min_over_simplex_for_any_delta_except_ref.solve()
        constraints = make_constraints(self.theta_in_simplex,x,self.u,self.delta)+extra_constraints
        constraints += [bar_V-epsilon >= V,
                        epsilon >= self.eps_a,
                        epsilon >= self.eps_r*self.min_V_except_delta]
        # constraint that delta has to be feasible everywhere in R
        constraints += feasibility_constraints
        self.find_more_optimal_commutation = cvx.Problem(cvx.Maximize(epsilon),constraints)

    def P_theta(self,theta,check_feasibility=False):
        """
        Multiparametric conic MINLP.
        
        Parameters
        ----------
        theta : np.array
            Parameter value.
        check_feasibility : bool (optional)
            Only check if the problem is feasible at theta.
        
        Returns
        -------
        : bool
            ``True`` if the problem is feasible.
        --OR--
        u_opt : np.array
            Optimal input value.
        delta_opt : np.array
            Optimal commutation.
        """
        self.x0.value = theta
        if check_feasibility:
            self.minlp_feasibility.solve(**self.solver_options)
            return (self.minlp_feasibility.status == cvx.OPTIMAL)
        else:
            self.minlp.solve(**self.solver_options)
            u_opt = self.u[0].value
            delta_opt = self.delta.value
            return u_opt, delta_opt

    def P_theta_delta(self,theta,delta):
        """
        Fixed-commutation multiparametrix conic NLP.
        
        Parameters
        ----------
        theta : np.array
            Parameter value.
        delta : np.array
            Commutation value.
            
        Returns
        -------
        u_opt : np.array
            Optimal input value.
        J_opt : float
            Optimal cost.
        """
        self.x0.value = theta
        self.delta_fixed.value = delta
        self.nlp.solve(**self.solver_options)
        J_opt = self.nlp.value
        u_opt = self.u[0].value
        return u_opt, J_opt
    
    def V_R(self,R):
        """
        Feasibility MINLP that finds a feasible commutation in simplex R.
        
        Parameters
        ----------
        R : np.array
            2D array whose rows are simplex vertices (length self.n_x+1).
            
        Returns
        -------
        delta_feas : np.array
            Feasible commutation in R.
        """
        for k in range(self.n_x+1):
            self.vertices[k].value = R[k]
        self.feasibility_in_simplex.solve(**self.solver_options)
        delta_feas = self.delta.value
        return delta_feas
    
    def bar_E_ar_R(self,R,V_delta_R,delta_ref):
        """
        Absolute and relative error over-approximations using affine
        over-approximator of cost V_delta^*.
        
        Parameters
        ----------
        R : np.array
            2D array whose rows are simplex vertices (length self.n_x+1).
        V_delta_R : np.array
            Array of cost at corresponding vertex in R.
        delta_ref : np.array
            Reference "baseline" commutation against which suboptimality is to
            be checked (i.e. the one whose cost is being over-approximated).
            
        Returns
        -------
        bar_e_a_R : float
            Over-approximated absolute error.
        bar_e_r_R : float
            Over-approximated relative error.
        """
        self.delta_fixed.value = delta_ref
        for k in range(self.n_x+1):
            self.vertices[k].value = R[k]
        self.vertex_costs.value = V_delta_R
        self.abs_err_overapprox.solve(**self.solver_options)
        self.rel_err_denom.solve(**self.solver_options)
        bar_e_a_R = self.abs_err_overapprox.value
        bar_e_r_R = bar_e_a_R/self.rel_err_denom.value
        return bar_e_a_R,bar_e_r_R
    
    def in_variability_ball(self,R,V_delta_R,delta_ref):
        """
        Check if variation of V_delta^* over simplex R is small enough (within
        epsilon-suboptimality tolerance).
        
        Parameters
        ----------
        R : np.array
            2D array whose rows are simplex vertices (length self.n_x+1).
        V_delta_R : np.array
            Array of cost at corresponding vertex in R.
        delta_ref : np.array
            Reference "baseline" commutation against which suboptimality is to
            be checked (i.e. the one whose cost is being over-approximated).
            
        Returns
        -------
        : bool
            ``True`` if R is in the variability ball (for some offset).
        """
        self.delta_fixed.value = delta_ref
        for k in range(self.n_x+1):
            self.vertices[k].value = R[k]
        max_lhs = np.max(V_delta_R)
        min_lhs = self.min_over_simplex_for_this_delta.solve(**self.solver_options)
        rhs = max(self.eps_a,self.eps_r*self.min_over_simplex_for_any_delta.solve(**self.solver_options))
        #print(max_lhs,min_lhs,max_lhs-min_lhs,rhs)
        return max_lhs-min_lhs<rhs
    
    def D_delta_R(self,R,V_delta_R,delta_ref):
        """
        Try to find a more optimal commutation in R than delta_ref.
        
        Parameters
        ----------
        R : np.array
            2D array whose rows are simplex vertices (length self.n_x+1).
        V_delta_R : np.array
            Array of cost at corresponding vertex in R.
        delta_ref : np.array
            Reference "baseline" commutation against which suboptimality is to
            be checked (i.e. the one whose cost is being over-approximated).
            
        Returns
        -------
        better_delta : np.array
            More optimal commutation.
        """
        self.delta_fixed.value = delta_ref
        for k in range(self.n_x+1):
            self.vertices[k].value = R[k]
        self.vertex_costs.value = V_delta_R
        self.min_V_except_delta.value = self.min_over_simplex_for_any_delta_except_ref.solve(**self.solver_options)
        self.find_more_optimal_commutation.solve(**self.solver_options)
        better_delta = self.delta.value
        #print(self.find_more_optimal_commutation.value,self.eps_r*self.min_V_except_delta.value)
        #self.min_over_simplex_for_any_delta.solve(**self.solver_options)
        #print(self.min_V_except_delta.value,self.min_over_simplex_for_any_delta.value)
        return better_delta

eps_a,eps_r = 1., 0.1
oracle = Oracle(eps_a=eps_a,eps_r=eps_r)

# =============================================================================
# # Test baseline MINLP
# u,delta = oracle.P_theta(np.array([0.05,0.04]))
# print(u)
# 
# # Test fixed-commutation NLP
# u2,J = oracle.P_theta_delta(np.array([0.0,0.0]),delta)
# print(u2)
# 
# # Test find-feasible-commutation-in-simplex MINLP
# #side=0.1
# R = [np.array([0.0,0.0]),np.array([0.05,0.0]),np.array([0.0,0.04])]
# delta_feas = oracle.V_R(R)
# print(delta_feas)
# =============================================================================

# =============================================================================
# # Test absolute error over-approximation
# delta_ref = delta
# V_delta_R = [oracle.P_theta_delta(vertex,delta_ref)[1] for vertex in R]
# e_abs_max = oracle.bar_E_ar_R(R,V_delta_R,delta_ref)
# print(e_abs_max)
# 
# # Test if R is inside the variability ball
# x_o = np.array([0.5,0.2])
# side = 0.05
# R = [x_o,x_o+np.array([side,0.0]),x_o+np.array([0.0,side])]
# delta_ref = delta
# V_delta_R = [oracle.P_theta_delta(vertex,delta_ref)[1] for vertex in R]
# in_variability_ball = oracle.in_variability_ball(R,V_delta_R,delta_ref)
# print(in_variability_ball)
# 
# # Test finding a more optimal commutation in R
# x_o = np.array([0.5,0.2])
# side = 0.2
# R = [x_o,x_o+np.array([side,0.0]),x_o+np.array([0.0,side])]
# delta_ref = delta
# V_delta_R = [oracle.P_theta_delta(vertex,delta_ref)[1] for vertex in R]
# better_delta = oracle.D_delta_R(R,V_delta_R,delta_ref)
# print(better_delta)
# =============================================================================

