import numpy as np
import cost
import dynamics as dyn
import parameters as par
import matplotlib.pyplot as plt

ns, ni = par.ns, par.ni

Q,R = par.Q,par.R

def lqr(xx_ref, uu_ref):
    """
    Computes time-varying LQR gain matrices Kt along a reference trajectory 
    by solving the Riccati equation backward along the nominal tajectory for the linearized system.
    """
    TT = xx_ref.shape[1]
    
    #Initialize cost-to-go Hessian with terminal cost weight
    P = cost.terminal_grad(xx_ref[:, -1], xx_ref[:, -1])[1]  #terminal cost
    
    Kt = np.zeros((ni, ns, TT-1))

    #Backward Riccati recursion
    for kk in range(TT-2, -1, -1):
        _, fx_T, fu_T = dyn.dynamics_casadi(xx_ref[:, kk], uu_ref[:, kk]) #linearize dynamics at each step along the reference traj
        At = fx_T.T
        Bt = fu_T.T

        #_, _, Q, R = cost.stage_grad(xx_ref[:, kk], xx_ref[:, kk], uu_ref[:, kk], uu_ref[:, kk]) #weighting matrices from cost function

        #Algebraic Riccati eq, K = -(R + B.T * P * B)^-1 * B.T * P * A
        Q_uu = R + Bt.T @ P @ Bt
        Q_ux = Bt.T @ P @ At
        
        K = -np.linalg.inv(Q_uu) @ Q_ux
        Kt[:, :, kk] = K

        P = Q + At.T @ P @ At + Q_ux.T @ K #update P for the previous step

    return Kt

def simulate_fb(x0, xx_ref, uu_ref, Kt):
    """
    Simulates the system in closed-loop: u = u_ref + K(x - x_ref).
    Used to test robustness against initial state perturbations.
    """
    TT = xx_ref.shape[1]
    
    xx_sim = np.zeros((ns, TT))
    uu_sim = np.zeros((ni, TT-1))
    
    xx_sim[:, 0] = x0 #initial state, with perturbation
    
    for kk in range(TT-1):
        #Feedback control law
        error = xx_sim[:, kk] - xx_ref[:, kk]
        uu_sim[:, kk] = uu_ref[:, kk] + Kt[:, :, kk] @ error
        
        #Apply control to the "real" system dynamics
        xx_sim[:, kk+1], _, _ = dyn.dynamics_casadi(xx_sim[:, kk], uu_sim[:, kk])
        
    return xx_sim, uu_sim

