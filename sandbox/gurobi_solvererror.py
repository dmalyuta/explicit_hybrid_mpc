import numpy as np
import numpy.random as nprand
import cvxpy as cvx

z = cvx.Variable(integer=True)

nprand.seed(0)

Q = nprand.randn(2,2) # np.array([[ 1.01092346, -1.246334  ], [-1.246334  , 11.24925237]])
q = np.array([-0.79305302, -0.67245977])
C = np.array([[ 1.74951088, -0.86604018],
       [ 1.83903552,  0.28968686],
       [ 1.85200233,  1.34964664],
       [-0.62858037,  0.15304167]])
c = np.array([ 0.58550308,  0.96431765,  0.40713688, -0.79539809])
w = np.array([ 1.71458648, -1.54797607])

x = cvx.Variable(2)
# obj = cvx.quad_form(x, Q.T @ Q) / 2 + q.T*x + cvx.norm(x-z,2)**2 / 20
obj = cvx.norm(Q*x,2)**2 / 2 + q.T*x + cvx.norm(x-w,2)**2 / 20
constr = [C*x <= z, z>=1, z<=-1]
prob = cvx.Problem(cvx.Minimize(obj), constr)

prob.solve(solver=cvx.GUROBI, verbose=True)