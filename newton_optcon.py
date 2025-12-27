import numpy as np
import dynamics as dyn
import cost
import parameters as par

ns, ni = par.ns, par.ni
max_iters = 10

def backward_passing(xx, uu, xx_ref, uu_ref):
    '''Riccati recursion to find optimal fb gain Kt and ff sigma_t for Newton step'''
    TT = xx.shape[1]

    #Initialize terminal costs, boundary conditions for Riccati
    gx_T, Gxx_T = cost.terminal_grad(xx[:, -1], xx_ref[:, -1])
    P, p = Gxx_T, gx_T
    
    Kt = np.zeros((ni, ns, TT-1))
    sigma_t = np.zeros((ni, TT-1))

    #Backward recursion from terminal to init time
    for kk in range(TT-2, -1, -1):
        #Linearize dynamicd at current trajectoy point
        _, fx_T, fu_T = dyn.dynamics_casadi(xx[:, kk], uu[:, kk]) #fx_T, fu_T are the transpose of state and input Jacobians
        
        At = fx_T.T #(4,4)
        Bt = fu_T.T # 4,1)
        
        #Quadratic stage cost
        qx, ru, Qxx, Ruu = cost.stage_grad(xx[:, kk], xx_ref[:, kk], uu[:, kk], uu_ref[:, kk])

        #Q-function expansion
        Q_x = qx + At.T @ p
        Q_u = ru + Bt.T @ p

        Q_xx = Qxx + At.T @ P @ At
        Q_uu = Ruu + Bt.T @ P @ Bt
        Q_ux = Bt.T @ P @ At

        #Optimal gains by inverting the input Hessian
        invQ_uu = np.linalg.inv(Q_uu)
        
        K = -invQ_uu @ Q_ux #fb gain matrix
        sigma = -invQ_uu @ Q_u #ff step
        
        #Store results
        Kt[:, :, kk] = K
        sigma_t[:, kk] = sigma.flatten() 
        
        #Update for previous time step
        p = Q_x + Q_ux.T @ sigma
        P = Q_xx + Q_ux.T @ K

    return Kt, sigma_t

def armijo_search(xx, uu, xx_ref, uu_ref, Kt, sigma_t, J_old):
    '''Backtracking line search to ensure sufficient cost reduction and maintain stability'''
    
    gamma = 1.0 #initial step size
    beta = 0.5  #reduction factor
    c = 1e-4    #armijo tolerance

    #Calculate expected cost reduction based on gradient of search direction
    descent = 0
    for kk in range(sigma_t.shape[1]):
        descent -= np.linalg.norm(sigma_t[:, kk])**2

    #Iteratively reduce gamma until Armijo condition is met
    for i in range(max_iters):
        xx_new = np.zeros_like(xx)
        uu_new = np.zeros_like(uu)
        xx_new[:, 0] = xx[:, 0] #start from initial state
        
        #Fwd pass: NL simulation with new control law
        for kk in range(xx.shape[1]-1):
            uu_new[:, kk] = uu[:, kk] + Kt[:,:,kk] @ (xx_new[:, kk] - xx[:, kk]) + gamma * sigma_t[:, kk]
            xx_new[:, kk+1] = dyn.dynamics_casadi(xx_new[:, kk], uu_new[:, kk])[0]
            
        #Evealuate total cost of new trajectory    
        J_new = cost.cost_fcn(xx_new, uu_new, xx_ref, uu_ref)
        
        #Verify Armijo condition
        if J_new < J_old + c * gamma * descent:
            return xx_new, uu_new, gamma, J_new
        
        #Decrease step size for next iteration
        gamma *= beta
        
    return xx_new, uu_new, gamma, J_new