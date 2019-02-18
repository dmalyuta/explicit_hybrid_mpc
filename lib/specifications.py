"""
Specification sets class.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2018 University of Washington. All rights reserved.
"""

import copy
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla

import uncertainty_sets as uc
from polytope import Polytope

class Specifications:
    """
    Control problem specifications.
    """
    def __init__(self, X, U, P=None):
        """
        Parameters
        ----------
        X : Polytope
            Set of allowed states.
        U : tuple of Polytope
            Tuple (U_int,U_ext) such that the admissible inputs are
            \in U_ext \setminus U_int. Both U_int and U_ext are assumed to be
            Polytope type.
        P : UncertaintySet, optional
            Set of uncertainties. If ommitted, an empty set (no uncertainty)
            is used.
        """
        self.X = X
        self.U_int = U[0]
        self.U_ext = U[1]
        self.P = P if P is not None else uc.UncertaintySet()
        
    def getUncertaintyOuterBoundingPolytope(self):
        """
        Find a minimum-volume polytope {x : R*x<=r} containing the uncertainty
        set. Conservatively removes any dependent uncertainty,
        over-approximating it with independent uncertainty.
        
        Returns
        -------
        R : array
            Polytope facet normals.
        r : array
            Polytope facet distances.
        """
        u_max = self.U_ext.V[np.argmax([la.norm(u) for u in self.U_ext.V])]
        x_max = self.X.V[np.argmax([la.norm(x) for x in self.X.V])]
        R = np.eye(0)
        r = np.array([])
        j = 0 # Counter for specs.P.phi
        for i in range(len(self.P.sets)):
            noise_set = self.P.sets[i]
            if isinstance(noise_set,uc.Hyperrectangle):
                R_new, r_new = noise_set.convertToPolytope()[:2]
            elif isinstance(noise_set,uc.NormBall):
                max_norm = self.P.dependency[j].phi_direct(x_max,u_max).value
                R_new, r_new = noise_set.convertToPolytope(max_norm)[:2]
                j += 1
            R = sla.block_diag(R,R_new)
            r = np.concatenate([r,r_new])
        return R, r
