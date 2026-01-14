import numpy as np
import dynamics as dyn
import cost
import parameters as par
import matplotlib.pyplot as plt

ns, ni = par.ns, par.ni
max_iters = 25

def backward_passing(xx, uu, xx_ref, uu_ref):
    '''Riccati recursion to find optimal fb gain Kt and ff sigma_t for Newton step'''
    TT = xx.shape[1]

    #Initialize terminal costs, boundary conditions for Riccati
    gx_T, Gxx_T = cost.terminal_grad(xx[:, -1], xx_ref[:, -1]) 
    P, p = Gxx_T, gx_T 
    
    Kt = np.zeros((ni, ns, TT-1))
    sigma_t = np.zeros((ni, TT-1))

    descent_arm = 0.0 

    #Backward recursion from terminal to init time
    for kk in range(TT-2, -1, -1):
        #Linearize dynamics at current trajectoy point
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

        #Accumulate descent direction: descent_arm = Q_u^T * sigma
        descent_arm += Q_u.T @sigma
        
        #Update for previous time step
        p = Q_x + Q_ux.T @ sigma
        P = Q_xx + Q_ux.T @ K

        #Norm of descent direction
        descent_norm = np.linalg.norm(sigma_t.flatten())

    return Kt, sigma_t, descent_arm, descent_norm

def armijo_search(xx, uu, xx_ref, uu_ref, Kt, sigma_t, J_old,descent_arm,plot=False):
    '''Backtracking line search to ensure sufficient cost reduction and maintain stability'''
    
    gamma = 1.0 #initial step size
    beta = 0.7  #reduction factor
    c = 0.5    #armijo tolerance

    TT = xx.shape[1]
    descent_arm = float(np.squeeze(descent_arm))
    gamma_list = [] 
    costs_armijo=[]

    #Iteratively reduce gamma until Armijo condition is met
    for i in range(max_iters):
        xx_new = np.zeros_like(xx)
        uu_new = np.zeros_like(uu)
        xx_new[:, 0] = xx[:, 0] #start from initial state
        
        #Fwd pass: NL simulation with new control law
        for kk in range(xx.shape[1]-1):
            uu_new[:, kk] = uu[:, kk] + Kt[:,:,kk] @ (xx_new[:, kk] - xx[:, kk]) + gamma * sigma_t[:, kk]
            xx_new[:, kk+1] = dyn.dynamics_casadi(xx_new[:, kk], uu_new[:, kk])[0]
            
        #Evaluate total cost of new trajectory    
        J_new = cost.cost_fcn(xx_new, uu_new, xx_ref, uu_ref)

        #print(f"descent_arm: {descent_arm},J_new:{J_new},J_old + c * gamma * descent_arm:{J_old + c * gamma * descent_arm}")
        
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

            xx_temp[:,0] = xx[:,0] #??? sarebbe x0

            for tt in range(TT-1):
                uu_temp[:,tt]=uu[:,tt] +Kt[:,:,tt]@(xx_temp[:,tt]-xx[:,tt]) + step*sigma_t[:,tt]
                xx_temp[:,tt+1]=dyn.dynamics_casadi(xx_temp[:,tt],uu_temp[:,tt])[0]

            costs[ii] = cost.cost_fcn(xx_temp,uu_temp,xx_ref,uu_ref)

        plt.figure(1)
        plt.clf()


        plt.plot(steps, costs, color='g', label='$J(\\mathbf{u}^k - stepsize*d^k)$')
        plt.plot(steps, J_old + descent_arm*steps, color='r', label='$J(\\mathbf{u}^k) - stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
        # plt.plot(steps, JJ - descent*steps, color='r', label='$J(\\mathbf{u}^k) - stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
        plt.plot(steps, J_old + c*descent_arm*steps, color='g', linestyle='dashed', label='$J(\\mathbf{u}^k) - stepsize*c*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')

        plt.scatter(gamma_list, costs_armijo, marker='*') # plot the tested stepsize

        plt.grid()
        plt.xlabel('stepsize')
        plt.legend()
        plt.draw()

        plt.show()
    
    armijo_data = {
        'gamma_list': gamma_list,
        'costs_armijo': costs_armijo,
        'J_old': J_old,
        'descent_arm': descent_arm,
        'c': c
    }

    return xx_new, uu_new, gamma, J_new, armijo_data

def plot_armijo_iteration(xx, uu, xx_ref, uu_ref, Kt, sigma_t, armijo_data, newton_iter):
    '''Plot Armijo line search for a specific Newton iteration'''
    
    TT = xx.shape[1]
    J_old = armijo_data['J_old']
    descent_arm = armijo_data['descent_arm']
    c = armijo_data['c']
    gamma_list = armijo_data['gamma_list']
    costs_armijo = armijo_data['costs_armijo']
    
    # Compute full cost curve
    steps = np.linspace(0, 1.0, int(2e1))
    costs = np.zeros(len(steps))

    for ii in range(len(steps)): 
        step = steps[ii]
        xx_temp = np.zeros((ns, TT))
        uu_temp = np.zeros((ni, TT))
        xx_temp[:,0] = xx[:,0]

        for tt in range(TT-1):
            uu_temp[:,tt] = uu[:,tt] + Kt[:,:,tt] @ (xx_temp[:,tt] - xx[:,tt]) + step * sigma_t[:,tt]
            xx_temp[:,tt+1] = dyn.dynamics_casadi(xx_temp[:,tt], uu_temp[:,tt])[0]

        costs[ii] = cost.cost_fcn(xx_temp, uu_temp, xx_ref, uu_ref)

    # Plot
    plt.figure(figsize=(8, 6))
    
    plt.plot(steps, costs, color='g', linewidth=2, label='$J(\\mathbf{u}^k + stepsize*d^k)$')
    plt.plot(steps, J_old + descent_arm*steps, color='r', linewidth=2, label='$J(\\mathbf{u}^k) + stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
    plt.plot(steps, J_old + c*descent_arm*steps, color='g', linestyle='dashed', linewidth=2, label='$J(\\mathbf{u}^k) + stepsize*c*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
    
    plt.scatter(gamma_list, costs_armijo, marker='*', s=150, c='blue', zorder=5, edgecolors='black', linewidths=0.5)
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('stepsize', fontsize=12)
    plt.ylabel('Cost J', fontsize=12)
    plt.title(f'Armijo Line Search - Newton Iteration {newton_iter}', fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()