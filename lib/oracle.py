"""
Library of optimization problems used by the partitioning process.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import copy
import numpy as np
import cvxpy as cvx
import mosek

import global_vars
import tools

class Oracle:
    """
    "Oracle" problems, i.e. the optimization programs that need to be called
    by the partitioning algorithm.
    """
    def __init__(self,mpc,eps_a,eps_r):
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
        """
        self.mpc = mpc
        self.eps_a = eps_a
        self.eps_r = eps_r
        
        # Make P_theta, the original MINLP
        constraints = self.mpc.make_constraints(self.mpc.x0,self.mpc.x,
                                                self.mpc.u,self.mpc.delta)
        self.minlp = cvx.Problem(self.mpc.cost,constraints)
        # just to check feasibility
        self.minlp_feasibility = cvx.Problem(cvx.Minimize(0),constraints)
        
        # Make P_theta_delta, the fixed-commutation NLP
        self.delta_fixed = cvx.Parameter(self.mpc.delta_size*self.mpc.N)
        constraints = self.mpc.make_constraints(self.mpc.x0,self.mpc.x,
                                                self.mpc.u,self.delta_fixed)
        self.nlp = cvx.Problem(self.mpc.cost,constraints)
        # just to check feasibility
        self.nlp_feasibility = cvx.Problem(cvx.Minimize(0),constraints)
        
        # Make V^R, the find-feasible-commutation-in-simplex MINLP
        x_theta = [self.mpc.make_ux()['x'] for _ in range(self.mpc.n_x+1)]
        u_theta = [self.mpc.make_ux()['u'] for _ in range(self.mpc.n_x+1)]
        self.vertices = [cvx.Parameter(self.mpc.n_x)
                         for _ in range(self.mpc.n_x+1)]
        self.feasibility_constraints = []
        for i in range(self.mpc.n_x+1):
            # Add a set of constraints for each vertex
            self.feasibility_constraints += self.mpc.make_constraints(
                self.vertices[i],x_theta[i],u_theta[i],self.mpc.delta,
                delta_sum_constraint=True if i==0 else False)
        
        # Make problems that allow verifying simplex-is-in-variability-ball
        # variables
        self.alpha = cvx.Variable(self.mpc.n_x+1,nonneg=True)
        self.theta_in_simplex = sum([self.alpha[i]*self.vertices[i]
                                     for i in range(self.mpc.n_x+1)])
        # problems
        in_simplex = [sum(self.alpha)==1]
        constraints = self.mpc.make_constraints(self.theta_in_simplex,
                                                self.mpc.x,self.mpc.u,
                                                self.delta_fixed)+in_simplex
        self.min_over_simplex_for_this_delta = cvx.Problem(self.mpc.cost,
                                                           constraints)
        constraints = self.mpc.make_constraints(self.theta_in_simplex,
                                                self.mpc.x,
                                                self.mpc.u,
                                                self.mpc.delta)+in_simplex
        self.min_over_simplex_for_any_delta = cvx.Problem(self.mpc.cost,
                                                          constraints)

        # Make \bar E^R, the MINLP that verifies if the commutation is
        # epsilon-suboptimal
        self.vertex_costs = cvx.Parameter(self.mpc.n_x+1)
        self.bar_V = sum([self.alpha[i]*self.vertex_costs[i]
                          for i in range(self.mpc.n_x+1)])
        bar_E_delta_R_constraints = self.mpc.make_constraints(
            self.theta_in_simplex,self.mpc.x,self.mpc.u,
            self.mpc.delta)+in_simplex
        bar_E_delta_R_constraints += [self.bar_V-self.mpc.V>=self.eps_a,
                                self.bar_V-self.mpc.V>=self.eps_r*self.mpc.V]
        self.bar_E_delta_R_problem = cvx.Problem(cvx.Minimize(0),bar_E_delta_R_constraints)
        
        # Make D_delta^R, the MINLP that searches for a more optimal commutation
        # variables
        self.bar_D_delta_R_constraints = (bar_E_delta_R_constraints+
                                          self.feasibility_constraints)

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
            u_opt = self.mpc.get_u0_value()
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
            u_opt = self.mpc.get_u0_value()
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
        vx_inputs_and_costs : list
            List of vertex optimal inputs and costs using delta_feas.
        """
        for k in range(self.mpc.n_x+1):
            self.vertices[k].value = R[k]
        # Solve V_R problem
        # **Secret sauce**: procedurally add new delta_ref's to list of
        # commutations that delta should not equal, in case V_R comes up
        # with deltas for which P_theta_delta is infeasible at the vertices of
        # R (due to numerical issues - mathematically this should not happen)
        delta_neq_other_deltas = []
        while True:
            # Solve V_R
            V_R = cvx.Problem(cvx.Minimize(0),self.feasibility_constraints+
                              delta_neq_other_deltas)
            V_R.solve(**global_vars.SOLVER_OPTIONS)
            delta_feas = self.mpc.delta.value
            if delta_feas is None:
                vx_inputs_and_costs = None
                return delta_feas,vx_inputs_and_costs
            
            # Check if P_theta_delta is feasible at each vertex
            try:
                vx_inputs_and_costs = self.__compute_vx_inputs_and_costs(
                    R,delta_feas)
                return delta_feas,vx_inputs_and_costs
            except:
                # Not feasible at some vertex -> solver must have returned an
                # infeasible solution for bar_D_delta_R (numerical troubles)
                tools.error_print('V_R output failed for R={}'.format(R.tolist()))
                delta_neq_other_deltas += self.__delta_neq_constraint(delta_feas)
    
    def in_variability_ball(self,R,V_delta_R,delta_ref,delta_star,theta_star):
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
        delta_star : np.array
            More optimal commutation found by bar_D_delta_R.
        theta_star : np.array
            Parameter value at which commutation found by bar_D_delta_R is more
            optimal.
            
        Returns
        -------
        : bool
            ``True`` if R is in the variability ball (for some offset).
        """
        self.delta_fixed.value = delta_ref
        for k in range(self.mpc.n_x+1):
            self.vertices[k].value = R[k]
        max_lhs = np.max(V_delta_R)
        if global_vars.SOLVER_OPTIONS['solver']==cvx.MOSEK:
            # (Numerical) failure mode fix
            #
            # Commit: 304d1a58c94a15e4bdf9f23bd84937db0ac90e6c
            #
            # Description: running
            # ```
            # main.py -e pendulum -N 5 -a 0.1 -r 0.1 --runtime-dir=<RUNTIME_DIR>
            # ```
            # at location 1111111111111111111111111111111111111111111111
            # 111111111100100000000011011 the call to
            # self.min_over_simplex_for_this_delta fails because MOSEK's
            # heuristics incorrectly choose to solve the dual instead of the
            # primal. This returns status UNKNOWN and fails the optimization.
            # See https://docs.mosek.com/9.0/toolbox/presolver.html#index-4 for
            # more info.
            #
            # Fix: as suggested in https://docs.mosek.com/9.0/pythonapi/
            # debugging-numerical.html#further-suggestions, I force MOSEK to
            # solve the primal problem.
            force_primal = {mosek.iparam.intpnt_solve_form:
                            mosek.solveform.primal}
            solver_options_bk = copy.deepcopy(global_vars.SOLVER_OPTIONS)
            if 'mosek_params' in global_vars.SOLVER_OPTIONS.keys():
                global_vars.SOLVER_OPTIONS['mosek_params'].update(force_primal)
            else:
                global_vars.SOLVER_OPTIONS['mosek_params'] = force_primal
        min_lhs = self.min_over_simplex_for_this_delta.solve(
            **global_vars.SOLVER_OPTIONS)
        if global_vars.SOLVER_OPTIONS['solver']==cvx.MOSEK:
            # Restore original solver options
            global_vars.SOLVER_OPTIONS = solver_options_bk
        V_delta_theta = self.P_theta_delta(theta=theta_star,delta=delta_star)[1]
        rhs = max(self.eps_a,self.eps_r*V_delta_theta)
        return max_lhs-min_lhs<rhs

    def bar_E_delta_R(self,R,V_delta_R):
        """
        Verify if commutation, whose affine cost over-approximator is V_delta_R,
        is epsilon-suboptimal in R.

        Parameters
        ----------
        R : np.array
            2D array whose rows are simplex vertices (length self.mpc.n_x+1).
        V_delta_R : np.array
            Array of cost at corresponding vertex in R.

        Returns
        -------
        eps_suboptimal : bool
            True if delta_ref is epsilon-suboptimal.
        """
        for k in range(self.mpc.n_x+1):
            self.vertices[k].value = R[k]
        self.vertex_costs.value = V_delta_R
        self.bar_E_delta_R_problem.solve(**global_vars.SOLVER_OPTIONS)
        # If could not find a satisfactory delta_star, it means epsilon
        # suboptimality holds
        eps_suboptimal = self.mpc.delta.value is None
        return eps_suboptimal        
    
    def bar_D_delta_R(self,R,V_delta_R,delta_ref):
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
        delta_star : np.array
            More optimal commutation.
        theta_star : np.array
            Parameter value at which delta_start is more optimal.
        vx_inputs_and_costs : list
            List of vertex optimal inputs and costs using delta_star.
        var_small : boolean
            If ``True``, variation of the optimal cost using the commutation
            delta_ref is small enough over the simplex R.
        """
        for k in range(self.mpc.n_x+1):
            self.vertices[k].value = R[k]
        self.vertex_costs.value = V_delta_R
        # **Secret sauce**: procedurally add new delta_ref's to list of
        # commutations that delta should not equal, in case D_delta^R
        # comes up with deltas for which P_theta_delta is infeasible at
        # the vertices of R (due to numerical issues - mathematically
        # this should not happen)
        delta_blacklist = []#self.__delta_neq_constraint(delta_ref)
        while True:
            bar_D_delta_R = cvx.Problem(cvx.Minimize(0),
                                        self.bar_D_delta_R_constraints+
                                        delta_blacklist)
            bar_D_delta_R.solve(**global_vars.SOLVER_OPTIONS)
            if (self.mpc.delta.value is None and
                global_vars.SOLVER_OPTIONS['solver']==cvx.MOSEK):
                # (Numerical) failure mode fix
                #
                # Commit: d2cd83cd0cc02038332a3e9415c1ddb9b50d2a67
                #
                # Description: running
                # ```
                # main.py -e pendulum -N 5 -a 0.1 -r 0.1 --runtime-dir=<RUNTIME_DIR>
                # ```
                # at location 11111111111111111111111111111111111111111111111111
                # 1111111111111111111010101001000000... MOSEK keeps failing
                # because of some numerical troubles (Gurobi appears to work).
                #
                # Fix: as sugested in https://docs.mosek.com/9.0/pythonapi/
                # debugging-numerical.html#further-suggestions, turning off
                # presolve and forcing primal solves the problem.
                mosek_numerics = {mosek.iparam.intpnt_solve_form:
                                  mosek.solveform.primal,
                                  mosek.iparam.presolve_use:
                                  mosek.presolvemode.off}
                solver_options_bk = copy.deepcopy(global_vars.SOLVER_OPTIONS)
                if 'mosek_params' in global_vars.SOLVER_OPTIONS.keys():
                    global_vars.SOLVER_OPTIONS['mosek_params'].update(
                        mosek_numerics)
                else:
                    global_vars.SOLVER_OPTIONS['mosek_params'] = mosek_numerics
                bar_D_delta_R.solve(**global_vars.SOLVER_OPTIONS)
                if global_vars.SOLVER_OPTIONS['solver']==cvx.MOSEK:
                    # Restore original solver options
                    global_vars.SOLVER_OPTIONS = solver_options_bk
            delta_star = self.mpc.delta.value
            theta_star = self.theta_in_simplex.value
            if delta_star is None or np.array_equal(delta_star.astype(int),
                                                    delta_ref.astype(int)):
                # Treat finding delta_star==delta_ref the same way as the
                # problem being infeasible --> but this seems to be a
                # numerically better way to go about it (MOSEK doesn't seem to
                # fail in this case...)
                delta_star = None
                theta_star = None
                vx_inputs_and_costs = None
                var_small = None
                return delta_star,theta_star,vx_inputs_and_costs,var_small
            # Check that subsequent optimization problems are feasible,
            # which makes sure that the MICP did not choose delta_star "too
            # ambitiously" (i.e. by relaxing the constraints somewhat)
            try:
                # 1) Check if P_theta_delta is feasible at each vertex
                vx_inputs_and_costs = self.__compute_vx_inputs_and_costs(
                    R,delta_star)
                # 2) Check if in_variability_ball is feasible
                var_small = self.in_variability_ball(R,V_delta_R,delta_ref,
                                                     delta_star,theta_star)
                return delta_star,theta_star,vx_inputs_and_costs,var_small
            except:
                # Not feasible at some vertex -> solver must have returned
                # an infeasible solution for bar_D_delta_R (numerical
                # troubles)
                tools.error_print('bar_D_delta_R output failed for R={}, '
                                  'V_delta_R={}, delta_ref={}'.format(
                                      R.tolist(),V_delta_R.tolist(),
                                      delta_ref.tolist()))
                delta_blacklist += self.__delta_neq_constraint(delta_star)

    def __compute_vx_inputs_and_costs(self,R,delta):
        """
        Computes the list of optimal inputs and costs at vertices of simplex R,
        using delta commutation.

        Parameters
        ----------
        R : list
            Simplex in vertex-representation at whose vertices to compute the
            vertex inputs and costs.
        delta : np.array
            Commutation to use of the calculation.

        Returns
        -------
        vx_inputs_and_costs : list
            The list of optimal inputs and costs at the vertices of simplex R.
        """
        Nvx = len(R)
        vx_inputs_and_costs = [None for _ in range(Nvx)]
        for i in range(Nvx):
            vertex = R[i]
            vx_inputs_and_costs[i] = self.P_theta_delta(theta=vertex,
                                                        delta=delta)
            status = self.nlp.status
            if status!=cvx.OPTIMAL and status!=cvx.OPTIMAL_INACCURATE:
                raise cvx.SolverError('problem infeasible')
        return vx_inputs_and_costs

    def __delta_neq_constraint(self,delta_ref):
        """
        Create a set of constraints that self.mpc.delta is not equal to
        delta_ref.

        Parameters
        ----------
        delta_ref : np.array
            Commutation which self.mpc.delta must not equal.

        Returns
        -------
        delta_neq_delta_ref : list
            List of constraints ensuring that self.mpc.delta is not equal to
            delta_ref. You can append this to the list of constraints of your
            optimization problem.
        """
        delta_offset = cvx.Variable(self.mpc.delta_size*self.mpc.N,
                                    boolean=True)
        delta_neq_delta_ref = []
        delta_neq_delta_ref += [
            self.mpc.delta[self.mpc.delta_size*k+i]==delta_ref[
                self.mpc.delta_size*k+i]+
            (1-2*delta_ref[self.mpc.delta_size*k+i])*
            delta_offset[self.mpc.delta_size*k+i] for k in range(self.mpc.N)
            for i in range(self.mpc.delta_size)]
        delta_neq_delta_ref += [sum([delta_offset[self.mpc.delta_size*k+i]
                                     for k in range(self.mpc.N)
                                     for i in range(self.mpc.delta_size)])>=1]
        return delta_neq_delta_ref
