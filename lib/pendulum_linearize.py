"""
Linearization of the inverted pendulum with static/kinetic friction.
Call using ``$ sage pendulum_linearize.sage``.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import pickle
import numpy as np
from sage.all import *

var('x theta dxdt dthetadt F')
STATE = vector([x,theta,dxdt,dthetadt])
INPUT = vector([F])

def pendulum_parameters():
    """
    Compile the inverted pendulum MPC parameters.
    Pickles the parameters.

    Returns
    -------
    pars : dict
        Dictionary of parameters.
    """
    pars = dict(mu_s = 0.1, # Static coefficient of friction (steel on steel)
                mu_d = 0.005, # Kinetic coefficient of friction (steel on steel)
                g = 9.81, # [m/s^2] Gravitational acceleration
                l = 3., # [m] Pendulum rod length
                m = 0.3, # [kg] Pendulum tip mass
                M = 1., # [kg] Cart mass
                v_eps = 1e-3, # [m/s] Velocity below which to use static friction
                a_eps = 1e-3, # [m/s^2] Acceleration below which static friction dominates
                T_s = 1/10.) # [s] Discretization sampling time
    with open('sage/pendulum_parameters.pkl','wb') as f:
        pickle.dump(pars,f)
    return pars

def make_linearized_system(d2xdt2,d2thetadt2,state0,input0):
    """
    Given the x and theta dynamics, creates a linearized system about the
    linearization points:

        \ddot x = A*x+B*u+w

    Parameters
    ----------
    d2xdt2 : sage expression
        Sage expression of \ddot x (cart position dynamics).
    d2thetadt2 : Expression
        Sage expression of \ddot\theta (pendulum angle dynamics).
    state0 : list
        Reference state about which to linearize.
    input0 : list
        Reference input about which to linearize.

    Returns
    -------
    A : sage matrix
        Zero-input state dynamics.
    B : sage matrix
        Input to state dynamics map.
    w : sage vector
        Dynamics linearization perturbation term.
    """
    # Linearization reference point
    x0,theta0,dxdt0,dthetadt0 = state0
    F0 = input0

    at_equilibrium = lambda expression: expression.substitute(
        x=x0,theta=theta0,dxdt=dxdt0,dthetadt=dthetadt0,F=F0)

    # Dynamics at the reference point
    dxdt_0 = at_equilibrium(dxdt)
    dthetadt_0 = at_equilibrium(dthetadt)
    d2xdt2_0 = at_equilibrium(d2xdt2)
    d2xdt2_0 = at_equilibrium(d2xdt2)
    d2thetadt2_0 = at_equilibrium(d2thetadt2)
    
    # Jacobians
    # dx/dt
    J_x_dxdt = vector(at_equilibrium(jacobian(dxdt,STATE)))
    J_u_dxdt = vector(at_equilibrium(jacobian(dxdt,INPUT)))
    # dtheta/dt
    J_x_dthetadt = vector(at_equilibrium(jacobian(dthetadt,STATE)))
    J_u_dthetadt = vector(at_equilibrium(jacobian(dthetadt,INPUT)))
    # d^2x/dt^2
    J_x_d2xdt2 = vector(at_equilibrium(jacobian(d2xdt2,STATE)))
    J_u_d2xdt2 = vector(at_equilibrium(jacobian(d2xdt2,INPUT)))
    # d^2theta/dt^2
    J_x_d2thetadt2 = vector(at_equilibrium(jacobian(d2thetadt2,STATE)))
    J_u_d2thetadt2 = vector(at_equilibrium(jacobian(d2thetadt2,INPUT)))
    
    # Linearized dynamics
    A = matrix([J_x_dxdt,J_x_dthetadt,J_x_d2xdt2,J_x_d2thetadt2])
    B = matrix([J_u_dxdt,J_u_dthetadt,J_u_d2xdt2,J_u_d2thetadt2])
    w = (vector([dxdt0,dthetadt0,d2xdt2_0,d2thetadt2_0])-
         A*vector(state0)-B*vector([input0]))
    
    return A,B,w

def main():
    """Create the linearized cases."""
    # Dynamics
    pars = pendulum_parameters()
    mu_s,mu_d,g,l,m,M = (pars['mu_s'],pars['mu_d'],pars['g'],
                         pars['l'],pars['m'],pars['M'])
    d2xdt2 = dict(
        # Case: dx/dt >= v_eps
        opt1=(F-m*g/2*sin(2*theta)+m*l*dthetadt**2*(sin(theta)+mu_d*cos(theta))-
              mu_d*(M+m*cos(theta)**2)*g)/(M+m*(sin(theta)**2+
                                                mu_d*sin(2*theta)/2)),
        # Case: dx/dt <= -v_eps
        opt2=(F-m*g/2*sin(2*theta)+m*l*dthetadt**2*(sin(theta)-mu_d*cos(theta))+
              mu_d*(M+m*cos(theta)**2)*g)/(M+m*(sin(theta)**2-
                                                mu_d*sin(2*theta)/2)),
        # Case: |dx/dt|<=v_eps and d^2x/dt^2>0
        opt3=(F-m*g/2*sin(2*theta)+m*l*dthetadt**2*(sin(theta)+mu_s*cos(theta))-
              mu_s*(M+m*cos(theta)**2)*g)/(M+m*(sin(theta)**2+
                                                mu_s/2*sin(2*theta))),
        # Case: |dx/dt|<=v_eps and d^2x/dt^2<0
        opt4=(F-m*g/2*sin(2*theta)+m*l*dthetadt**2*(sin(theta)-mu_s*cos(theta))+
              mu_s*(M+m*cos(theta)**2)*g)/(M+m*(sin(theta)**2-
                                                mu_s/2*sin(2*theta))),
        # Case: |dx/dt|<=v_eps and d^2x/dt^2=0
        opt5=0*F)
    d2thetadt2 = {opt: g/l*sin(theta)-d2xdt2[opt]/l*cos(theta)
                  for opt in d2xdt2.keys()}

    # Linearization
    state0 = [0,0,0,0]
    input0 = 0
    lin_sys_map = {opt:dict(A=None,B=None,w=None) for opt in d2xdt2.keys()}
    for opt in d2xdt2.keys():
        A,B,w = make_linearized_system(d2xdt2[opt],d2thetadt2[opt],
                                       state0,input0)
        lin_sys_map[opt]['A'] = np.array(A,dtype=np.float)
        lin_sys_map[opt]['B'] = np.array(B,dtype=np.float).flatten()
        lin_sys_map[opt]['w'] = np.array(w,dtype=np.float)

    # Save result
    with open('sage/pendulum_linearization.pkl','wb') as f:
        pickle.dump(lin_sys_map,f)

if __name__=='__main__':
    main()
