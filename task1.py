import numpy as np
import dynamics as dyn
import parameters as par
import matplotlib.pyplot as plt
import math 
from scipy.optimize import least_squares

ns, ni = par.ns, par.ni
dt = par.dt
TT = int(par.tf/par.dt) 

constant_traj = 0.50 # [%] - percentage for the constant part

#Equilibium  points
# xe1 = np.array([0.0, np.radians(25), 0.0, 0.0])
# xe2 = np.array([0.0, np.radians(50), 0.0, 0.0])

def find_equilibria(theta1_guess,theta2): #theta_2 fixed, theta1_guess 

    def equilibrium_eqs(z):
        """
        z = [theta1, tau]. z is the unknown variables, theta2 was fixed as target 
        This function returns the residual between two states [x(k+1) - x(k)]
        """
        theta1, tau = z

        xx = np.array([theta1, theta2, 0.0, 0.0])
        uu = np.array([tau])

        #calculation of the x(k+1) state with the dynamics 
        xx_next = dyn.dynamics_casadi(xx, uu)[0] 

        #flatten bc least_squares requires 1D array 
        return (xx_next-xx).flatten() 

    # initial guess of theta1 and initial guess of u (=0)
    z_init = np.array([theta1_guess,0.0])

    # use the least_squares method to find the minimum residual 
    res = least_squares(equilibrium_eqs,z_init,verbose=0) #verbose=0 to not print errors
    theta_1_found = res.x[0]
    u_eq = res.x[1]
    xx_eq = np.array([theta_1_found,theta2,0.0,0.0]) #theta1, theta2, theta1_dot, theta2_dot

    #check if the residual is near 0, so the equilibria exist
    if res.cost>1e-6: 
        print(f"No equilibria found for theta2: {theta2}. residual is: {res.cost}")

    return xx_eq,u_eq

### --- Testing functions --- ###

# xx_eq1,uu_eq1 = find_equilibria(xe1[0],xe1[1]) 
# print(f"xx_eq1: {xx_eq1*180/np.pi}, uu_eq1: {uu_eq1*180/np.pi}")

# xx_eq2, uu_eq2 = find_equilibria(xe2[0],xe2[1])
# print(f"xx_eq2: {xx_eq2*180/np.pi}, uu_eq2: {uu_eq2*180/np.pi}")

def build_reference(xe1, xe2,ue1,ue2,TT):

    """
    this function create the state trajectory between the two equilbria.
    There are two constant parts, at the beginning and at the end. 
    It uses a 3rd order polynomial function to build the trajectory. 
    """
    xxref = np.zeros((ns, TT))
    uuref =np.zeros((ni,TT))
    margin = int(constant_traj*TT)
    
    xxref[:,:margin] = xe1[:,None]

    if(ni==1):
        ue1 = np.atleast_1d(ue1)
        ue2 = np.atleast_1d(ue2)
    
    uuref[:, :margin] = ue1[:, None]
    uuref[:, TT - margin:] = ue2[:, None]

    for kk in range(margin,TT-margin):
        s = (kk-margin)/((TT-2*margin)-1)
        alpha = 3*s**2 - 2*s**3 
        xxref[:,kk] = (1-alpha)*xe1 + alpha*xe2

    xxref[:,TT-margin:] = xe2[:,None]

    return xxref,uuref
