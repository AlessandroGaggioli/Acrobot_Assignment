import numpy as np
import dynamics as dyn
import parameters as par
import matplotlib.pyplot as plt
import math 
from scipy.optimize import least_squares

ns, ni = par.ns, par.ni
dt = par.dt
TT = int(par.tf/par.dt) 
constant_traj = 0.20 # [%]

#Equilibium  points
# xe1 = np.array([0.0, np.radians(25), 0.0, 0.0])
# xe2 = np.array([0.0, np.radians(50), 0.0, 0.0])

def find_equilibria(theta1_guess,theta2): #theta_2 fixed, theta1_guess 

    def equilibrium_eqs(z):
        """
        z = [theta1, tau]
        """
        theta1, tau = z

        xx = np.array([theta1, theta2, 0.0, 0.0])
        uu = np.array([tau])

        xx_next = dyn.dynamics_casadi(xx, uu)[0]

        return (xx_next-xx).flatten() #flatten bc least_squares requires 1D array 

    z_init = np.array([theta1_guess,0.0]) #z = [theta1 guess, u=0]

    res = least_squares(equilibrium_eqs,z_init,verbose=0) #verbose=0 to not print errors

    theta_1_found = res.x[0]
    u_eq = res.x[1:]

    xx_eq = np.array([theta_1_found,theta2,0.0,0.0])

    if res.cost>1e-6: 
        print(f"No equilibria found for theta2: {theta2}. residual is: {res.cost}")

    return xx_eq,u_eq

### --- Testing functions --- ###

# xx_eq1,uu_eq1 = find_equilibria(xe1[0],xe1[1]) 
# print(f"xx_eq1: {xx_eq1*180/np.pi}, uu_eq1: {uu_eq1*180/np.pi}")

# xx_eq2, uu_eq2 = find_equilibria(xe2[0],xe2[1])
# print(f"xx_eq2: {xx_eq2*180/np.pi}, uu_eq2: {uu_eq2*180/np.pi}")

def build_reference(xe1, xe2, TT):
    xxref = np.zeros((ns, TT))
    margin = int(constant_traj*TT)
    
    xxref[:,:margin] = xe1[:,None]

    for kk in range(margin,TT-margin):
        s = (kk-margin)/((TT-2*margin)-1)
        alpha = 3*s**2 - 2*s**3 
        xxref[:,kk] = (1-alpha)*xe1 + alpha*xe2

    xxref[:,TT-margin:] = xe2[:,None]

    return xxref
