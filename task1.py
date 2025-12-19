import numpy as np
import dynamics as dyn
import parameters as par
import animation
import matplotlib.pyplot as plt
import math

ns, ni = par.ns, par.ni
dt = par.dt
TT = int(par.tf/par.dt)

#Equilibrium points
#xe1 = np.array([0.0, 0.0, 0.0, 0.0])
#xe2 = np.array([0.0, np.pi, 0.0, 0.0])

def build_reference(xe1, xe2, TT):
    xxref = np.zeros((ns, TT))
    uuref = np.zeros((ni, TT))

    for kk in range(TT):
        s = kk/(TT-1)
        alpha = 3*s**2 - 2*s**3
        xxref[:,kk] = (1-alpha)*xe1 + alpha*xe2
    
    return xxref, uuref