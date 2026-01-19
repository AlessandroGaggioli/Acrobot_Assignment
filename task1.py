import numpy as np
import dynamics as dyn
import parameters as par
from scipy.optimize import least_squares
import os

def find_equilibria(theta1_guess,theta2): #theta_2 fixed, theta1_guess 

    def equilibrium_eqs(z):
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

def build_smooth_ref(xe1, xe2, ue1, ue2, TT,constant_traj):
    '''Generates a smooth reference trajectory between two equilibria by using a 3rd order poly'''
    ns, ni = par.ns, par.ni
    dt = par.dt
    
    xxref = np.zeros((ns, TT))
    uuref = np.zeros((ni, TT)) 
    
    #percentage time margin at beginning and end
    margin = int(constant_traj * TT) #change factor to have stable equilibrium at beginning and end... (ideally in paramters or main!!!)
    duration = (TT - 2 * margin) * dt
    
    #initial steady-state phase, first equilibrium point constant
    xxref[:, :margin] = xe1[:, None]
    uuref[:, :margin] = np.atleast_1d(ue1)[:, None]
    
    #Smooth transition phase
    for kk in range(margin, TT - margin):
        #Normalized time in [0, 1]
        s = (kk - margin) / (TT - 2 * margin - 1)
        #3rd order poly to ensure initial and final zero velocity
        alpha = 3*s**2 - 2*s**3
        #time derivative of poly
        d_alpha = (6*s - 6*s**2) / duration
        #Smooth interpolation for angles
        xxref[:2, kk] = (1 - alpha) * xe1[:2] + alpha * xe2[:2]
        #Angular velocities, time derivatives of angles
        xxref[2:, kk] = d_alpha * (xe2[:2] - xe1[:2])
        #Smooth interpolation between the two equilibrium inputs
        uuref[:, kk] = (1 - alpha) * ue1 + alpha * ue2

    #final steady-state phase, final equilibrium point constant
    xxref[:, TT - margin:] = xe2[:, None]
    uuref[:, TT - margin:] = np.atleast_1d(ue2)[:, None]

    return xxref, uuref

def equilibria(equilibria_file,current_params):
    if os.path.exists(equilibria_file):
        print(' ----- Loading equilibria from file -----')
        eq_data = np.load(equilibria_file, allow_pickle=True)

        if len(eq_data)>4 and np.array_equal(eq_data[4],current_params):
            xx_eq1, xx_eq2 = eq_data[0], eq_data[1]
            uu_eq1, uu_eq2 = eq_data[2][0], eq_data[3][0]

        else:
            print('----- Parameters changed, recomputing equilibria -----')
            os.path.exists(equilibria_file) and os.remove(equilibria_file)
            
            xe1 = np.array([0.0, par.theta2_start, 0.0, 0.0])
            xe2 = np.array([0.0, par.theta2_end, 0.0, 0.0])

            xx_eq1, uu_eq1 = find_equilibria(xe1[0], xe1[1]) 
            xx_eq2, uu_eq2 = find_equilibria(xe2[0], xe2[1])

            print(f"xx_eq1: {xx_eq1*180/np.pi}, uu_eq1: {uu_eq1}")
            print(f"xx_eq2: {xx_eq2*180/np.pi}, uu_eq2: {uu_eq2}")

            # Salva equilibri CON i parametri usati
            np.save(equilibria_file, np.array([xx_eq1, xx_eq2, [uu_eq1], [uu_eq2], current_params], dtype=object))
    else: 
        print('----- Computing equilibria -----')
        xe1 = np.array([0.0, par.theta2_start, 0.0, 0.0])
        xe2 = np.array([0.0, par.theta2_end, 0.0, 0.0])

        xx_eq1,uu_eq1 = find_equilibria(xe1[0],xe1[1]) 
        xx_eq2, uu_eq2 = find_equilibria(xe2[0],xe2[1])

        print(f"xx_eq1: {xx_eq1*180/np.pi}, uu_eq1: {uu_eq1}")
        print(f"xx_eq2: {xx_eq2*180/np.pi}, uu_eq2: {uu_eq2}")

        #save equilibria
        np.save(equilibria_file, np.array([xx_eq1, xx_eq2, [uu_eq1], [uu_eq2]], dtype=object))

    return xx_eq1,xx_eq2,uu_eq1,uu_eq2