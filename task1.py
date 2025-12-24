import numpy as np
import dynamics as dyn
import parameters as par
import animation
import matplotlib.pyplot as plt
import math
from scipy.optimize import fsolve

ns, ni = par.ns, par.ni
dt = par.dt
TT = int(par.tf/par.dt)

#Equilibrium points
#xe1 = np.array([0.0, 0.0, 0.0, 0.0])
#xe2 = np.array([0.0, np.pi, 0.0, 0.0])

def equilibrium_eqs(z):
    """
    z = [theta1, theta2, tau]
    """
    theta1, theta2, tau = z

    x = np.array([theta1, theta2, 0.0, 0.0])
    u = np.array([tau])

    x_next = dyn.dynamics_casadi(x, u)[0]

    # impose zero velocities at next step
    return np.array([
        x_next[2],   # dtheta1_next = 0
        x_next[3],   # dtheta2_next = 0
        tau          # keeps tau bounded / unique
    ])


def find_equilibria():

    # -------- Equilibrium 1 (downward) --------
    z0_1 = np.array([0.0, 0.0, 0.0])
    sol1 = fsolve(equilibrium_eqs, z0_1)

    xe1 = np.array([sol1[0], sol1[1], 0.0, 0.0])
    ue1 = np.array([sol1[2]])

    # -------- Equilibrium 2 (different posture) --------
    z0_2 = np.array([0.0, np.pi, 0.0])
    sol2 = fsolve(equilibrium_eqs, z0_2)

    xe2 = np.array([sol2[0], sol2[1], 0.0, 0.0])
    ue2 = np.array([sol2[2]])

    return xe1, ue1, xe2, ue2

def build_reference(xe1, xe2, TT):
    xxref = np.zeros((ns, TT))
    uuref = np.zeros((ni, TT))

    for kk in range(TT):
        s = kk/(TT-1)
        alpha = 3*s**2 - 2*s**3
        xxref[:,kk] = (1-alpha)*xe1 + alpha*xe2
    
    return xxref, uuref