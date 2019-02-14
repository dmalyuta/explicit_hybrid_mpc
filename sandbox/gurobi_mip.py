"""
Tryout of hybrid MPC using Gurobi MIP.
"""

import numpy as np
import numpy.linalg as la
import cvxpy as cvx
import matplotlib.pyplot as plt

m = 1. # [kg] Cart mass
h = 1./20. # [s] Time step
N = 10 # Prediction horizon length

# Discretized dynamics Ax+Bu
n_x,n_u = 2,1
A = np.array([[1.,h],[0.,1.]])
B = np.array([[h**2/2.],[h]])/m

# Control constraints
lb, ub = 0.2, 1.
P = [np.array([[1.],[-1.]]),np.array([[-1.],[1.]]),np.array([[1.],[-1.]])]
p = [np.array([ub*m,-lb*m]),np.array([ub*m,-lb*m]),np.zeros((2))]
M = np.array([ub*m,ub*m])*10

# Control objectives
e_p_max = 0.1 # [m] Max position error
e_v_max = 0.2 # [m/s] Max velocity error
u_max = 3.*m # [N] Max control input

# Cost
D_x = np.diag([e_p_max,e_v_max])
D_u = np.diag([u_max])
Q = 100.*la.inv(D_x).dot(np.eye(n_x)).dot(la.inv(D_x))
R = la.inv(D_u).dot(np.eye(n_u)).dot(la.inv(D_u))

# MPC controller
x = [cvx.Variable(n_x) for k in range(N+1)]
u = [cvx.Variable(n_u) for k in range(N+1)]
delta = [cvx.Variable(3, boolean=True) for k in range(N)]
x0 = cvx.Parameter(n_x)

cost = cvx.Minimize(sum([cvx.quad_form(x[k],Q)+cvx.quad_form(u[k],R) for k in range(N)]))

constraints = []
constraints += [x[k+1] == A*x[k]+B*u[k] for k in range(N)]
constraints += [x[0] == x0]
constraints += [cvx.norm(x[k]) <= 0.14 for k in range(1,N+1)]
for i in range(3):
    constraints += [P[i]*u[k] <= p[i]+M*delta[k][i] for k in range(N)]
constraints += [sum(delta[k]) <= 2 for k in range(N)]
#constraints += [u[k] <= 1.*m for k in range(N)]
#constraints += [u[k] >= -1.*m for k in range(N)]

problem = cvx.Problem(cost,constraints)

# Simulate controller
T = 50 # Time steps to simulate
x_init = np.array([0.1,0.]) # Initial state
x_sim, u_sim = [x_init], []
for t in range(T):
    x0.value = x_sim[t]
    print(x0.value)
    problem.solve(solver=cvx.GUROBI, verbose=False)
    u_opt = u[0].value
    x_next = A.dot(x0.value)+B.dot(u_opt)
    
    x_sim.append(x_next)
    u_sim.append(u_opt)

t_sim = np.array(range(T+1))*h
x_sim = np.row_stack(x_sim)
u_sim = np.row_stack(u_sim)

#%% Plot

fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot(221)
ax.plot(t_sim,x_sim[:,0],label='position')
ax.plot(t_sim,x_sim[:,1],label='velocity')
ax.legend()
ax = fig.add_subplot(222)
ax.plot(t_sim,[la.norm(x_sim[k]) for k in range(T+1)],label='norm2(state)')
ax.legend()
ax = fig.add_subplot(212)
ax.plot(t_sim[:-1],u_sim,label='input')
ax.legend()