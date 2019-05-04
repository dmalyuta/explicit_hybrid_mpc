"""
Computational geometry classes : polytope, simplex, vertex, etc.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2018 University of Washington. All rights reserved.
"""

import math
import numpy as np
import numpy.linalg as la
import scipy.spatial as ss
import cvxpy as cvx
import cdd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import global_vars

class PolytopeError(Exception):
    """
    Custom exception of this code.
    
    Parameters
    ----------
    message : str
        Descriptive test.
    """
    def __init__(self,message,*args,**kwargs):
        Exception.__init__(self,global_vars.ERROR+message,*args,**kwargs)

def subtractHyperrectangles(X_inner,X_outer):
    """
    Assume X_outer and X_inner are two boxes in R^n centered at the origin
    such that X_inner \in X_outer. This function computes the 2*n convex
    polytopes whose union equals X_outer \setminus X_inner.
    
    Parameters
    ----------
    X_outer : Polytope
        Outer box in R^n centered at the origin.
    X_inner : Polytope
        Inner box in R^n centered at the origin.
        
    Returns
    -------
    X_partition : array
        Array of polytopes whose union equals X_outer \setminus X_inner.
    """
    # Offset the hyperrectangles away from the origin
    dist_ax_max = np.max(np.abs(X_outer.V),axis=0)
    ax_max = np.argmax(dist_ax_max)
    dist_ax_max = dist_ax_max[ax_max]
    offset = np.zeros(X_outer.n)
    offset[ax_max] = dist_ax_max
    # Vertices
    vx_original = np.vstack((X_inner.V,X_outer.V))
    vx_offset = np.vstack(([vx+offset for vx in X_inner.V],
                           [vx+offset for vx in X_outer.V]))
    # Get the convex hull of the individual partitions
    X_partition = []
    for ax in range(X_outer.n): # Loop over all axes
        for sign in [-1,1]: # Loop over (+) and (-) direction of the axis
            vertices = vx_offset[sign*vx_original[:,ax]>0]
            polyH = Polytope(V=vertices)
            polyH = Polytope(A=polyH.A,b=polyH.b-polyH.A.dot(offset))
            X_partition.append(polyH)
    return X_partition

class Polytope(object):
    def __init__(self,A=None,b=None,V=None,R=None):
        """
        Parameters
        ----------
        --OR
            A : array
                Matrix whose rows are the facet normals.
            b : array
                Vector whose elements are the facet distances.
            V : {list,bool}, optional
                List of vertices whose convex hull defines the polytope. If
                set to ``False``, no vertex representation is computed!
        --OR
            A : {array,bool}, optional
                Matrix whose rows are the facet normals. If set to ``False``,
                no halfspace representation is computed!
            b : {array,bool}, optional
                Vector whose elements are the facet distances. If set to
                ``False``, no halfspace representation is computed!
            V : list
                List of vertices whose convex hull defines the polytope.
        --OR
            R : list
                List of tuples (lower bound,upper bound) for each coordinate of the
                polytope, which is effectively a hyperrectangle.
        """
        # Define the polytope
        if A is not None:
            # H-rep definition
            self.A = A
            self.b = b
            if V is None:
                self._hrep2vrep()
            elif V is not False:
                self.V = V
        elif V is not None:
            # V-rep definition
            self.V = V
            if A is None:
                self._vrep2hrep()
            elif A is not False or b is not False:
                self.A = A
                self.b = b
        else:
            # Hyperrectangle definition
            n = len(R)
            self.A,self.b = np.zeros((2*n,n)),np.zeros(2*n)
            for i in range(n):
                xi_min = R[i][0]
                xi_max = R[i][1]
                if xi_min > xi_max:
                    raise PolytopeError("x%d_min > x%d_max" % (i,i))
                self.A[2*i,i],self.A[2*i+1,i] = 1,-1
                self.b[2*i],self.b[2*i+1] = xi_max,-xi_min
            self._hrep2vrep()
        if V is not False and len(self.V)==0:
            raise PolytopeError("Empty polytope")
        self.n = self.V[0].size if V is not False else self.A.shape[1] # Polytope \in R^n

    def _hrep2vrep(self):
        """
        Convert polytope halfspace representation to vertex representation.
        """
        bmA = np.column_stack((self.b,-self.A))
        mat = cdd.Matrix(bmA,number_type='fraction')
        mat.rep_type = cdd.RepType.INEQUALITY
        poly = cdd.Polyhedron(mat)
        self.V = [np.array([float(num) for num in v[1:]]) for v in
                  poly.get_generators()]
    
    def _vrep2hrep(self):
        """
        Convert polytope vertex representation to halfspace representation.
        """
        mat = cdd.Matrix([[1]+list(v) for v in self.V],number_type='fraction')
        mat.rep_type = cdd.RepType.GENERATOR
        poly = cdd.Polyhedron(mat)
        poly = poly.get_inequalities()
        bmA = np.array(poly[:])
        self.A = np.array([[float(el) for el in row] for row in -bmA[:,1:]])
        self.b = np.array([float(el) for el in bmA[:,0]])

    def intersect(self,poly):
        """
        Intersect this polytope with another one, poly.
        
        Parameters
        ----------
        poly : Polytope
            Another polytope.
            
        Returns
        -------
        : Polytope
            The result polytope from doing self \cap poly.
        """
        A = np.vstack((self.A,poly.A))
        b = np.concatenate((self.b,poly.b))
        return Polytope(A,b)

    def project(self,coords,only_vertices=False):
        """
        Project polytope onto a subset of its coordinates.
        
        Parameters
        ----------
        coords : list
            List of coordinates to project onto (zero-based).
        only_vertices : bool, optional
            If ``True``, return just the vertices of the projected polytope
            
        Returns
        -------
        --OR
            : list
                Vertices of this polytope's projection onto coords.
        --OR
            : Polytope
                This polytope's projection onto coords, as a polytope.
        """
        Vproj = [V[list(coords)] for V in self.V]
        return Vproj if only_vertices else Polytope(V=Vproj)

    def cond(self):
        """
        Get the condition number of the polytope.

        Returns
        -------
        : float
            Condition number.
        """
        # Division by zero inside la.cond may occur due to **very** badly
        # conditioned polytope, in which case automatically output infinite
        # condition number
        M = np.column_stack([v-self.V[0] for v in self.V[1:]])
        sigma_min = la.svd(M)[1][-1] # Smallest singular value
        cond = la.cond(M) if sigma_min>0 else np.inf
        return cond

    def full(self):
        """
        Check if polytope is full dimensional, i.e. spans its space.
        
        TODO: may be worth changing to the same function as used for Simplex
        (i.e. approximate full based on :func:`self.cond`).
        
        Returns
        -------
        : bool
            ``True`` if full dimensional.
        """
        dirs = [V-self.V[0] for V in self.V[1:]]
        M = np.column_stack(dirs)
        rankM = la.matrix_rank(M)
        rank_check = rankM==self.n
        return rank_check
    
    def almostEmpty(self,tol=1e-8):
        """
        Check if the polytope is quasi-empty.
        """
        x_c = self.averageCenter()
        distances = [la.norm(V-x_c) for V in self.V]
        return np.all(distances<tol)
    
    def averageCenter(self):
        """
        Return the "average" center of the polytope.
        
        Returns
        -------
        x_c : array
            Average polytope center.
        """
        w = 1./len(self.V)
        x_c = sum([w*V for V in self.V])
        return x_c
    
    def radius(self,type='inner'):
        """
        Get the radius of the largest inscribed or smallest covering Eucledian
        ball.

        Parameters
        ----------
        type : {'inner','outer'}, optional
            Setting 'inner' computes the maximum volume inscribed ball while
            'outer' computes the minimum volume covering ball.
        
        Returns
        -------
        R : float
            The radius.
        """
        R = cvx.Variable()
        x = cvx.Variable(self.n)
        if type=='inner':
            cost = cvx.Maximize(R)
            constraints = [self.A[i]*x+R*la.norm(self.A[i])<=self.b[i] for i in
                           range(self.b.size)]
        else:
            cost = cvx.Minimize(R)
            constraints = [cvx.norm(v-x)<=R for v in self.V]
        problem = cvx.Problem(cost,constraints)
        R = problem.solve(**global_vars.SOLVER_OPTIONS)
        if not (problem.status==cvx.OPTIMAL or
                problem.status==cvx.OPTIMAL_INACCURATE):
            # Optimization failed due to ugly polytope
            raise PolytopeError("Radius computation failed (status %s)"
                                       %(problem.status))
        return R
    
    def randomPoint(self, N=1, method='random walk'):
        """
        Return N random points inside the polytope. Implements rejection
        sampling and hit-and-run sampling based on [1,2].
        
        NB: using GUROBI as solver because it is more precise than ECOS. When
        the polytope is very small (order 10^-6 and below), ECOS can give a
        numerical error such that the random point is outside the polytope.
        
        [1] Tim Seguine (https://mathoverflow.net/users/17546/tim-seguine),
            Uniformly Sampling from Convex Polytopes, URL
            (version: 2014-04-03): https://mathoverflow.net/q/162327
        [2] https://www.cc.gatech.edu/~vempala/acg/notes.pdf
        
        Parameters
        ----------
        N : int, optional
            Number of random points to generate (must be ``>=1``).
        method : str, optional
            'random walk' performance hit-and-run sampling while 'rejection'
            performs the less efficient (but perhaps more uniform??) rejection
            sampling.
        """
        if method=='random walk':
            # Hit-and-run sampling
            points = [self.averageCenter()] # Will contain the points
            for i in range(N):
                # Uniformly randomly pick a direction, each component in [-1,1]
                while True:
                    direction = np.random.rand(self.n)*2.-1.
                    direction_norm = la.norm(direction)  
                    if direction_norm < 1e-5:
                        # Very unlucky, the random direction vector is quasi-zero
                        # Try again!
                        continue
                    direction /= direction_norm
                    break
                # Find theta_mina and theta_max such that x[-1]+theta*direction
                # is contained in the polytope
                theta = cvx.Variable()
                cost = cvx.Maximize(theta)
                constraints = [self.A*(points[-1]+theta*direction) <= self.b]
                pbm = cvx.Problem(cost, constraints)
                theta_max = pbm.solve(**global_vars.SOLVER_OPTIONS)
                cost = cvx.Minimize(theta)
                pbm = cvx.Problem(cost, constraints)
                theta_min = pbm.solve(**global_vars.SOLVER_OPTIONS)
                # Sample a random theta in [theta_min, theta_max]
                theta = np.random.uniform(low=theta_min, high=theta_max)
                # Add the new point
                points.append(points[-1]+theta*direction)
            del points[0] # Delete the first "non-random" feasible point
        else:
            # Rejection sampling
            l,u = self.boundingBox()
            gen_point = lambda : np.random.uniform(low=l, high=u)
            points = []
            for i in range(N):
                new_point = gen_point()
                while np.any(self.A.dot(new_point) > self.b):
                    new_point = gen_point()
                points.append(new_point)
        return points[0] if N==1 else points

    def boundingBox(self):
        """
        Find  the smallest volume hyperrectangle that contains the polytope.
        The hyperrectangle is expressed as {x : l <= x <= u}.
        
        Returns
        -------
        l : array
            Hyperrectangle lower bound.
        u : array
            Hyperrectangle upper bound.
        """
        # Find upper bound
        u = cvx.Variable(self.n)
        Y = cvx.Variable((self.n,self.b.size))
        cost = cvx.Minimize(sum(u))
        constraints = [Y*self.b <= u,
                       Y*self.A == np.eye(self.n),
                       Y >= 0]
        problem = cvx.Problem(cost, constraints)
        optimal_value = problem.solve(**global_vars.SOLVER_OPTIONS)
        if optimal_value == np.inf:
            raise PolytopeError("Infeasible problem for bounding box")
        u = np.array(u.value.T).flatten()
        # Find lower bound
        l = cvx.Variable(self.n)
        cost = cvx.Maximize(sum(l))
        constraints = [Y*self.b <= -l,
                       Y*self.A == -np.eye(self.n),
                       Y >= 0]
        problem = cvx.Problem(cost, constraints)
        optimal_value = problem.solve(**global_vars.SOLVER_OPTIONS)
        if optimal_value == np.inf:
            raise PolytopeError("Infeasible problem for bounding box")
        l = np.array(l.value.T).flatten()
        return l,u
    
    def computeScalingMatrix(self):
        """
        Computes a scaling matrix D such that D*xhat=x where x is the
        original variable and xhat is the scaled one, ranging in [-1,1].
        
        Returns
        -------
        D : (n,n) array or NoneType
            The scaling matrix such that D*xhat=x. None if P,p are empty.
        """
        if self.A.size == 0:
            return None
        l,u = self.boundingBox()
        d = np.maximum(np.abs(l),np.abs(u))
        D = np.diag(d)
        return D
    
    def plot(self,ax=None,coords=None,**kwargs):
        """
        2D plot of the polytope. Usual fill plot parameters may be passed in.
        
        Parameters
        ----------
        ax : AxesSubplot, optional
            Existing subplot axes on which to plot.
        coords : list, optional
            Which two dimensions (zero-based) of the polytope to project onto
            for the plot. If not provided, the polytope must be \in R^2.
            
        Returns
        -------
        handle : matplotlib.patches.Polygon
            Handle to the plot.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        # Project polytope
        if coords is not None:
            if len(coords)!=2:
                raise PolytopeError("Precisely two projection axes must be specified")
            vx = np.row_stack(self.project(coords,only_vertices=True))
        elif self.n!=2:
            raise PolytopeError("Polytopes not in R^2 must be projected")
        else:
            vx = np.vstack(self.V)
        co = ss.ConvexHull(vx)
        handle, = ax.fill(vx[co.vertices,0].T.flatten(),
                          vx[co.vertices,1].T.flatten(),**kwargs)
        return handle
