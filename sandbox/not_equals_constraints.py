"""
Test if != integer constraints are possible in CVXPY.
"""

import numpy as np
import cvxpy as cvx

n_z = 3
z = cvx.Variable(n_z, boolean=True)
w = cvx.Variable(n_z, boolean=True)

z0 = [0,0,1]

cost = cvx.Minimize(sum(z))

constraints = []
constraints += [z[i] == z0[i] + (1 if z0[i]==0 else -1)*w[i] for i in range(n_z)]
constraints += [sum(w) >= 1]

problem = cvx.Problem(cost,constraints)

problem.solve(solver=cvx.GUROBI, verbose=False)

print('z = ', z.value)
print('w = ', w.value)