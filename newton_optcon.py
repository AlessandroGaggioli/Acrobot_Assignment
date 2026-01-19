import numpy as np
import dynamics as dyn
import cost
import parameters as par
import matplotlib.pyplot as plt

ns, ni = par.ns, par.ni
max_iters = par.Armjio_max_iters

def backward_passing(xx, uu, xx_ref, uu_ref):
    """
    Backward pass: Riccati recursion to compute gains K_t and feedforward sigma_t
    
    Args:
        xx: current state trajectory (ns, TT)
        uu: current input trajectory (ni, TT)
        xx_ref: reference state trajectory (ns, TT)
        uu_ref: reference input trajectory (ni, TT)
    
    Returns:
        Kt: feedback gains (ni, ns, TT-1)
        sigma_t: feedforward terms (ni, TT-1)
        descent_arm: descent direction for Armjio search
        descent_norm: norm of sigma for convergence check
    """
    
    ns, TT = xx.shape
    ni = uu.shape[0]
    
    # Initialize arrays for gains and feedforward
    Kt = np.zeros((ni, ns, TT-1))
    sigma_t = np.zeros((ni, TT-1))

    #Storage 
    At_storage = np.zeros((ns,ns,TT-1))
    Bt_storage = np.zeros((ns,ni,TT-1))
    GradJ_storage = np.zeros((ni,TT-1))
    
    # Initialize P and p for backward recursion
    # Terminal cost derivatives
    lx_T, QT = cost.terminal_grad(xx[:, -1], xx_ref[:, -1])
    
    PT = QT
    pt = lx_T.flatten()
    lamb = lx_T.flatten() # terminal costate equation 
    
    # Backward recursion (from T-1 to 0)
    for tt in range(TT-2, -1, -1):
        
        # Get dynamics at current point
        _, fx, fu = dyn.dynamics_casadi(xx[:, tt], uu[:, tt])
        At = fx.T  # (ns, ns)
        Bt = fu.T  # (ns, ni)

        #Store the matrices 
        At_storage[:,:,tt] = At 
        Bt_storage[:,:,tt] = Bt
        
        # Get cost derivatives at current point
        lx, lu, Qtt, Rtt = cost.stage_grad(xx[:, tt], xx_ref[:, tt], uu[:, tt], uu_ref[:, tt])
        
        qt = lx.flatten()
        rt = lu.flatten()

        #CALCOLO DEL VERO GRADIENTE (usare lambda (Adjoint), non 'pt)
        grad_J_u_Arm = rt + Bt.T @ lamb
        GradJ_storage[:,tt] = grad_J_u_Arm
        
        # Compute Q, R, S matrices (second order terms)
        Qt = Qtt  # (ns, ns) #First method regularization. Consider only the cost Hessian (w.r.t. x)
        Rt = Rtt  # (ni, ni) #First method regularization. Consider only the cost Hessian (w.r.t. u)
        St = np.zeros((ni, ns))  # Cross term (usually zero for our cost)
        
        # Riccati recursion terms
        # K_t = -(R_t + B^T P_{t+1} B)^{-1} (S_t + B^T P_{t+1} A)
        # σ_t = -(R_t + B^T P_{t+1} B)^{-1} (r_t + B^T p_{t+1})
        
        BT_P_B = Bt.T @ PT @ Bt
        BT_P_A = Bt.T @ PT @ At

        Qu_Ric = rt + Bt.T @ pt # (t_c = 0) 
        
        # Compute R matrix
        R_Kt = Rt + BT_P_B
        
        # Add small regularization for numerical stability
        epsilon = 1e-6
        #R_reg += epsilon * np.eye(ni) #Other strategy Regularization
        
        # Compute gain K_t
        try:
            R_inv = np.linalg.inv(R_Kt)
        except np.linalg.LinAlgError:
            print(f"Warning: Singular matrix at t={tt}, using pseudo-inverse")
            R_inv = np.linalg.pinv(R_Kt)
        
        Kt[:, :, tt] = -R_inv @ (St + BT_P_A)
        
        # Compute feedforward σ_t
        sigma_t[:, tt] = -R_inv @ (Qu_Ric)
        
        # Update P and p for next iteration (going backward)
        # p_t = q_t + A^T p_{t+1} - K_t^T (R_t + B^T P_{t+1} B) σ_t
        # P_t = Q_t + A^T P_{t+1} A - K_t^T (R_t + B^T P_{t+1} B) K_t
        
        AT_p = At.T @ pt
        AT_P_A = At.T @ PT @ At
        
        KT_R_sigma = Kt[:, :, tt].T @ R_Kt @ sigma_t[:, tt]
        KT_R_K = Kt[:, :, tt].T @ R_Kt @ Kt[:, :, tt]
        
        pt = qt + AT_p - KT_R_sigma
        PT = Qt + AT_P_A - KT_R_K

        #Update lambda = q + A^T * lambda_next (COSTATE EQUATION)
        lamb = qt + At.T @ lamb

    #Compute Delta_u and Descent_arm 
    delta_x = np.zeros(ns) #delta_x initialized to zero (no variation at initial state)
    descent_arm = 0.0 
    delta_u_seq = np.zeros((ni,TT-1))

    for tt in range(TT-1):
        At = At_storage[:,:,tt]
        Bt = Bt_storage[:,:,tt]
        grad_J_u = GradJ_storage[:,tt]

        #delta_u = K * delta_x + sigma 
        delta_u = Kt[:, :, tt] @ delta_x + sigma_t[:, tt]

        delta_u_seq[:,tt] = delta_u

        #accumulate descent_arm 
        descent_arm += grad_J_u @ delta_u

        #Linear evolution of the state
        #delta_x_next = A*delta_x + B * delta_u 
        delta_x = At @ delta_x + Bt @ delta_u
    
    # Compute norm of sigma for convergence check
    descent_norm = np.linalg.norm(delta_u_seq)

    # # ===== Plot delta_u_seq =====
    # time = np.arange(TT-1)

    # plt.figure()
    # for i in range(ni):
    #     plt.plot(time, delta_u_seq[i, :], label=f'delta_u[{i}]')

    # plt.xlabel('Time step')
    # plt.ylabel('Delta u')
    # plt.title('Control variation delta_u_seq')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    
    return Kt, sigma_t,descent_arm, descent_norm


def armijo_search(xx, uu, xx_ref, uu_ref, Kt, sigma_t, J_old,descent_arm,plot=False):
    '''Backtracking line search to ensure sufficient cost reduction'''
    
    gamma = 1.0 #initial step size
    beta = 0.7  #reduction factor
    c = 0.5    #armijo tolerance

    TT = xx.shape[1]
    gamma_list = [] 
    costs_armijo=[]

    #Iteratively reduce gamma until Armijo condition is met
    for i in range(max_iters):
        xx_new = np.zeros_like(xx)
        uu_new = np.zeros_like(uu)
        xx_new[:, 0] = xx[:, 0] #start from initial state
        
        #Fwd pass: NL simulation with new control law
        for kk in range(TT-1):
            uu_new[:, kk] = uu[:, kk] + Kt[:,:,kk] @ (xx_new[:, kk] - xx[:, kk]) + gamma * sigma_t[:, kk]
            xx_new[:, kk+1] = dyn.dynamics_casadi(xx_new[:, kk], uu_new[:, kk])[0]
            
        #Evaluate total cost of new trajectory    
        J_new = cost.cost_fcn(xx_new, uu_new, xx_ref, uu_ref)

        #print(f"J_new:{J_new},J_old + c * gamma * descent_arm:{J_old + c * gamma * descent_arm}")

        gamma_list.append(gamma)
        costs_armijo.append(J_new)

        #Verify Armijo condition
        if J_new < J_old + c * gamma * descent_arm:
            #return xx_new, uu_new, gamma, J_new
            print('Armjio stepsize found: {:.3}'.format(gamma))
            break 
        
        #Decrease step size for next iteration
        gamma *= beta

    ######################
    # PLOT ARMIJO DESCENT BACKTRACKING SEARCH 
    ######################
    if plot: 
        steps = np.linspace(0,1.0,int(2e1))
        costs = np.zeros(len(steps))
        for ii in range(len(steps)): 
            step = steps[ii]
            xx_temp = np.zeros((ns,TT))
            uu_temp = np.zeros((ni,TT))
            xx_temp[:,0] = xx[:,0] 
            for tt in range(TT-1):
                uu_temp[:,tt]=uu[:,tt] +Kt[:,:,tt]@(xx_temp[:,tt]-xx[:,tt]) + step*sigma_t[:,tt]
                xx_temp[:,tt+1]=dyn.dynamics_casadi(xx_temp[:,tt],uu_temp[:,tt])[0]
            costs[ii] = cost.cost_fcn(xx_temp,uu_temp,xx_ref,uu_ref)
        plt.figure(1)
        plt.clf()
        plt.plot(steps, costs, color='g', label='$J(\\mathbf{u}^k - stepsize*d^k)$')
        plt.plot(steps, J_old + descent_arm*steps, color='r', label='$J(\\mathbf{u}^k) - stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
        plt.plot(steps, J_old + c*descent_arm*steps, color='g', linestyle='dashed', label='$J(\\mathbf{u}^k) - stepsize*c*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
        plt.scatter(gamma_list, costs_armijo, marker='*') # plot the tested stepsize
        plt.grid()
        plt.xlabel('stepsize')
        plt.legend()
        plt.draw()
        plt.show()

    ##########################################
    # save armijo data
    ##########################################
    armijo_data = {
        'gamma_list': gamma_list,
        'costs_armijo': costs_armijo,
        'J_old': J_old,
        'descent_arm': descent_arm,
        'c': c
    }

    return xx_new, uu_new, gamma, J_new, armijo_data

