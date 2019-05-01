"""
Library of optimization problems used by the partitioning process.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import numpy as np
import cvxpy as cvx

import global_vars

class Oracle:
    """
    "Oracle" problems, i.e. the optimization programs that need to be called
    by the partitioning algorithm.
    """
    def __init__(self,mpc,eps_a,eps_r,kind='semiexplicit'):
        """
        Pre-parses the oracle problems.
        
        Parameters
        ----------
        mpc : object
            One of the controller objects from ``mpc_library.py``.
        eps_a : float
            Absolute error tolerance.
        eps_r : float
            Relative error tolerance (>0 where 0 is fully tight, 1 is 100%
            relative error, etc.).
        kind : {'semiexplicit','explicit'}
            Supports partitioning for semi-explicit or explicit MPC
            implementations.
        """
        self.mpc = mpc
        self.eps_a = eps_a
        self.eps_r = eps_r
        self.kind = kind
        
        # Make P_theta, the original MINLP
        constraints = self.mpc.make_constraints(self.mpc.x0,self.mpc.x,self.mpc.u,self.mpc.delta)
        self.minlp = cvx.Problem(self.mpc.cost,constraints)
        self.minlp_feasibility = cvx.Problem(cvx.Minimize(0),constraints) # just to check feasibility
        
        # Make P_theta_delta, the fixed-commutation NLP
        self.delta_fixed = cvx.Parameter(self.mpc.Nu*self.mpc.N)
        constraints = self.mpc.make_constraints(self.mpc.x0,self.mpc.x,self.mpc.u,self.delta_fixed)
        self.nlp = cvx.Problem(self.mpc.cost,constraints)
        self.nlp_feasibility = cvx.Problem(cvx.Minimize(0),constraints) # just to check feasibility
        
        # Make V^R, the find-feasible-commutation-in-simplex MINLP
        x_theta = [[self.mpc.D_x*cvx.Variable(self.mpc.n_x) for k in range(self.mpc.N+1)] for _ in range(self.mpc.n_x+1)]
        u_theta = [[self.mpc.D_u*cvx.Variable(self.mpc.n_u) for k in range(self.mpc.N)] for _ in range(self.mpc.n_x+1)]
        self.vertices = [cvx.Parameter(self.mpc.n_x) for _ in range(self.mpc.n_x+1)]
        feasibility_constraints = []
        for i in range(self.mpc.n_x+1):
            # Add a set of constraints for each vertex
            feasibility_constraints += self.mpc.make_constraints(self.vertices[i],x_theta[i],u_theta[i],self.mpc.delta,
                                                                 delta_sum_constraint=True if i==0 else False)
        self.feasibility_in_simplex = cvx.Problem(self.mpc.cost,feasibility_constraints)
        
        # Make bar_E_a^R, the convex absolute error over-approximator
        self.vertex_costs = cvx.Parameter(self.mpc.n_x+1)
        self.alpha = cvx.Variable(self.mpc.n_x+1,nonneg=True)
        self.theta_in_simplex = sum([self.alpha[i]*self.vertices[i] for i in range(self.mpc.n_x+1)])
        constraints = self.mpc.make_constraints(self.theta_in_simplex,self.mpc.x,self.mpc.u,self.mpc.delta)
        # theta stays in simplex
        in_simplex = [sum(self.alpha)==1]
        constraints += in_simplex
        # cost using affine over-approximator for reference commutation
        self.bar_V = sum([self.alpha[i]*self.vertex_costs[i] for i in range(self.mpc.n_x+1)])
        cost_abs_err = cvx.Minimize(self.mpc.V-self.bar_V)
        self.abs_err_overapprox = cvx.Problem(cost_abs_err,constraints)
        
        # Make denominator of bar_E_r^R, which is P_theta restricted to
        # commutation not being equal to a reference commutation and minimum
        # found over a full simplex
        self.rel_err_denom = cvx.Problem(self.mpc.cost,constraints)
        
        # Make problems that allow verifying simplex-is-in-variability-ball
        constraints = self.mpc.make_constraints(self.theta_in_simplex,self.mpc.x,self.mpc.u,self.delta_fixed)+in_simplex
        self.min_over_simplex_for_this_delta = cvx.Problem(self.mpc.cost,constraints)
        constraints = self.mpc.make_constraints(self.theta_in_simplex,self.mpc.x,self.mpc.u,self.mpc.delta)+in_simplex
        self.min_over_simplex_for_any_delta = cvx.Problem(self.mpc.cost,constraints)
        
        # Make D_delta^R, the MINLP that searches for a more optimal commutation
        self.epsilon = cvx.Parameter()
        self.D_delta_R_constraints = self.mpc.make_constraints(self.theta_in_simplex,self.mpc.x,self.mpc.u,self.mpc.delta)+in_simplex
        self.D_delta_R_constraints += [self.bar_V-self.epsilon >= self.mpc.V]
        self.D_delta_R_constraints += feasibility_constraints # commutation has to be feasible everywhere in R

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
        J_opt : float
            Optimal cost.
        t_solve : float
            Solver time.
        """
        self.mpc.x0.value = theta
        if check_feasibility:
            self.minlp_feasibility.solve(**global_vars.SOLVER_OPTIONS)
            return (self.minlp_feasibility.status == cvx.OPTIMAL)
        else:
            self.minlp.solve(**global_vars.SOLVER_OPTIONS)
            u_opt = self.mpc.u[0].value
            delta_opt = self.mpc.delta.value
            J_opt = self.minlp.value
            t_solve = self.minlp.solver_stats.solve_time
            return u_opt, delta_opt, J_opt, t_solve

    def P_theta_delta(self,theta,delta,check_feasibility=False):
        """
        Fixed-commutation multiparametrix conic NLP.
        
        Parameters
        ----------
        theta : np.array
            Parameter value.
        delta : np.array
            Commutation value.
        check_feasibility : bool (optional)
            Only check if the problem is feasible at theta.
            
        Returns
        -------
        u_opt : np.array
            Optimal input value.
        J_opt : float
            Optimal cost.
        t_solve : float
            Solver time.
        """
        self.mpc.x0.value = theta
        self.delta_fixed.value = delta
        if check_feasibility:
            self.nlp_feasibility.solve(**global_vars.SOLVER_OPTIONS)
            return (self.nlp_feasibility.status == cvx.OPTIMAL)
        else:
            self.nlp.solve(**global_vars.SOLVER_OPTIONS)
            J_opt = self.nlp.value
            u_opt = self.mpc.u[0].value
            t_solve = self.nlp.solver_stats.solve_time
            return u_opt, J_opt, t_solve
    
    def V_R(self,R):
        """
        Feasibility MINLP that finds a feasible commutation in simplex R.
        
        Parameters
        ----------
        R : np.array
            2D array whose rows are simplex vertices (length self.mpc.n_x+1).
            
        Returns
        -------
        delta_feas : np.array
            Feasible commutation in R.
        """
        for k in range(self.mpc.n_x+1):
            self.vertices[k].value = R[k]
        self.feasibility_in_simplex.solve(**global_vars.SOLVER_OPTIONS)
        delta_feas = self.mpc.delta.value
        return delta_feas
    
    def bar_E_ar_R(self,R,V_delta_R,delta_ref):
        """
        Absolute and relative error over-approximations using affine
        over-approximator of cost V_delta^*.
        
        Parameters
        ----------
        R : np.array
            2D array whose rows are simplex vertices (length self.mpc.n_x+1).
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
        for k in range(self.mpc.n_x+1):
            self.vertices[k].value = R[k]
        self.vertex_costs.value = V_delta_R
        self.abs_err_overapprox.solve(**global_vars.SOLVER_OPTIONS)
        bar_e_a_R = -self.abs_err_overapprox.value
        if np.isinf(bar_e_a_R):
            bar_e_r_R = np.inf
        else:
            self.rel_err_denom.solve(**global_vars.SOLVER_OPTIONS)
            bar_e_r_R = bar_e_a_R/self.rel_err_denom.value if self.rel_err_denom.value>0 else np.inf
        return bar_e_a_R, bar_e_r_R

    def in_variability_ball(self,R,V_delta_R,delta_ref):
        """
        Check if variation of V_delta^* over simplex R is small enough (within
        epsilon-suboptimality tolerance).
        
        Parameters
        ----------
        R : np.array
            2D array whose rows are simplex vertices (length self.mpc.n_x+1).
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
        for k in range(self.mpc.n_x+1):
            self.vertices[k].value = R[k]
        max_lhs = np.max(V_delta_R)
        min_lhs = self.min_over_simplex_for_this_delta.solve(**global_vars.SOLVER_OPTIONS)
        rhs = max(self.eps_a,self.eps_r*self.min_over_simplex_for_any_delta.solve(**global_vars.SOLVER_OPTIONS))
        #print(max_lhs,min_lhs,max_lhs-min_lhs,rhs)
        return max_lhs-min_lhs<rhs
    
    def D_delta_R(self,R,V_delta_R,delta_ref):
        """
        Try to find a more optimal commutation in R than delta_ref.
        
        Parameters
        ----------
        R : np.array
            2D array whose rows are simplex vertices (length self.mpc.n_x+1).
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
        for k in range(self.mpc.n_x+1):
            self.vertices[k].value = R[k]
        self.vertex_costs.value = V_delta_R
        self.min_V = self.min_over_simplex_for_any_delta.solve(**global_vars.SOLVER_OPTIONS)
        self.epsilon.value = max(self.eps_a,self.eps_r*self.min_V)
        # Solve D_delta^R problem
        # **Secret sauce**: procedurally add new delta_ref's to list of
        # commutations that delta should not equal, in case D_delta^R comes up
        # with deltas for which P_theta_delta is infeasible at the vertices of
        # R (due to numerical issues - mathematically this should not happen)
        delta_neq_refs = []
        while True:
            delta_offset = cvx.Variable(self.mpc.Nu*self.mpc.N, boolean=True)
            delta_neq_refs += [self.mpc.delta[self.mpc.Nu*k+i] == delta_ref[self.mpc.Nu*k+i]+
                               (1-2*delta_ref[self.mpc.Nu*k+i])*delta_offset[self.mpc.Nu*k+i]
                               for k in range(self.mpc.N) for i in range(self.mpc.Nu)]
            delta_neq_refs += [sum([delta_offset[self.mpc.Nu*k+i] for k in range(self.mpc.N)
                               for i in range(self.mpc.Nu)]) >= 1]
            find_more_optimal_commutation = cvx.Problem(cvx.Minimize(0),self.D_delta_R_constraints+delta_neq_refs)
            find_more_optimal_commutation.solve(**global_vars.SOLVER_OPTIONS)
            better_delta = self.mpc.delta.value
            if better_delta is None:
                # If keep on adding delta_ref's that delta cannot equal, will
                # eventually get into this case
                new_vertex_inputs_and_costs = None
                return better_delta,new_vertex_inputs_and_costs # D_delta^R infeasible
            # Check if P_theta_delta is feasible at each vertex
            try:
                new_vertex_inputs_and_costs = [self.P_theta_delta(vertex,better_delta) for vertex in R]
                return better_delta,new_vertex_inputs_and_costs
            except:
                # Not feasible at some vertex -> solver must have returned an infeasible solution
                # for find_more_optimal_commutation (numerical troubles)
                delta_ref = better_delta
