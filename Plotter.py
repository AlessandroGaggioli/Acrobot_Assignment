import numpy as np
import parameters as par
import matplotlib.pyplot as plt
import cost 
import dynamics as dyn

def plot_armijo_iteration(xx, uu, xx_ref, uu_ref, Kt, sigma_t, armijo_data, newton_iter,task):
    '''Plot Armijo line search for a specific Newton iteration'''
    ns = par.ns
    ni = par.ni
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
    plt.title(f'Armijo Line Search - Newton Iteration {newton_iter} - Task {task}', fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

def newton_plot(TT,xx_history,xx_ref,uu_history,uu_ref,xx_opt,uu_opt,cost_history,descent_norm_history,task):
    
    xx_opt = np.degrees(xx_opt)
    xx_history = np.degrees(xx_history)
    xx_ref = np.degrees(xx_ref)

    plot_method_iterations(TT,xx_history,xx_ref,uu_history,uu_ref,task)

    # Cost evolution 
    print(f"--- Plotting Cost Evolution Task: {task}---\n")
    plt.figure()
    plt.semilogy(cost_history, 'o-', color='b', markersize=4)
    plt.title(f"Cost Evolution - Task {task}")
    plt.xlabel("Iterations")
    plt.ylabel("Total Cost J")
    plt.grid(True, which="both", alpha=0.3)
    plt.show()

    # Norm of descent direction 
    print(f"--- Plotting Descent Norm Direction Task: {task}---\n")
    plt.figure()
    plt.semilogy(descent_norm_history, 'o-', color='r', markersize=4)
    plt.title(f"Norm of Descent Direction - Task {task}")
    plt.xlabel("Iterations")
    plt.ylabel(r"$\|\sigma\|$")
    plt.grid(True, which="both", alpha=0.3)
    plt.show()

def plot_method_iterations(TT, xx_history, xx_ref, uu_history,uu_ref,task):

    t = np.arange(TT) * par.dt

    #iters to plot 
    N = len(xx_history)
    iters_to_plot = [0, N//3, 2*N//3, N-1]

    state_names = [r'$\theta_1$',r'$\theta_2$',r'$\dot{\theta}_1$',r'$\dot{\theta}_2$']
    state_units = ['[deg]', '[deg]', '[deg/s]', '[deg/s]']
    #### states ####
    for i in range(4):

        print(r"\nPlotting evolution of state: {state_names[i]}\n")

        fig, axs = plt.subplots(len(iters_to_plot), 1,figsize=(10, 8),sharex=True)

        for k, it in enumerate(iters_to_plot):

            # Reference (dashed)
            axs[k].plot(t,xx_ref[i, :],'b--',linewidth=1.5,label=f'{state_names[i]} (ref)')

            # Optimal trajectory
            axs[k].plot(t,xx_history[it][i, :],'g',linewidth=2,label=f'{state_names[i]} (opt, k={it})')
            axs[k].set_ylabel(state_units[i])
            axs[k].set_title(f'{state_names[i]} (Iteration {it})')
            axs[k].grid(True, alpha=0.3)
            axs[k].legend(loc='best', fontsize='small')

        axs[-1].set_xlabel('Time [s]')
        fig.suptitle(f'Task {task}: Evolution of {state_names[i]}',fontsize=14)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    #### input ####
    fig, axs = plt.subplots(len(iters_to_plot), 1,figsize=(10, 8),sharex=True)

    for j, it in enumerate(iters_to_plot):
        # Reference input
        axs[j].plot(t[:-1],uu_ref[0, :-1],'k--',linewidth=1.2,label=r'$\tau_{\mathrm{ref}}$')

        # Control at iteration k
        axs[j].plot(t[:-1],uu_history[it][0, :-1],'b',linewidth=2,label=f'Iteration {it}')
        axs[j].set_ylabel('[Nm]')
        axs[j].set_title(rf'$\tau$ (Iteration {it})')
        axs[j].grid(True, alpha=0.3)
        axs[j].legend(fontsize='small')

    axs[-1].set_xlabel('Time [s]')
    fig.suptitle(f'Task {task}: Control input evolution',fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_LQR(TT,xx_opt,xx_lqr,uu_opt,uu_lqr):
    
    xx_opt = np.degrees(xx_opt)
    xx_lqr = np.degrees(xx_lqr)

    #Plot lqr 
    t = np.arange(TT) * par.dt 
    fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    plt.subplots_adjust(hspace=0.3)

    state_labels = [r'$\theta_1$ [deg]', r'$\theta_2$ [deg]', r'$\dot{\theta}_1$ [deg/s]', r'$\dot{\theta}_2$ [deg/s]'] 
    #Plot theta1, theta2, dtheta1, dtheta2 of both optimal and lqr 

    print(f"\n---Plotting Trajectory Tracking via LQR---\n") 

    for i in range(xx_opt.shape[0]): 
        axs[i].plot(xx_opt[i, :], 'r--', linewidth=1.5, label='Optimal trajectory') 
        axs[i].plot(xx_lqr[i, :], 'b', linewidth=1.2, label='LQR trajectory') 
        axs[i].set_ylabel(state_labels[i]) 
        axs[i].grid(True, alpha=0.5) 
        
        if i == 0: 
            axs[i].legend(loc='best') 
            
    #Plot control input 
    axs[4].plot(uu_opt[0, :-1], 'r--', linewidth=1.5, label='Optimal input') 
    axs[4].plot(uu_lqr[0, :-1], 'g', linewidth=1.2, label='LQR input') 
    axs[4].set_ylabel(r'$\tau$ [Nm]') 
    axs[4].set_xlabel('Time steps') 
    axs[4].grid(True, alpha=0.5) 
    axs[4].legend(loc='best') 
    plt.suptitle('Task 3: Trajectory Tracking via LQR', fontsize=14) 
    plt.show()

    # #Plot tracking error 
    # track_error = np.linalg.norm(xx_lqr - xx_opt,axis=0)
    
    # plt.figure(figsize=(10,6))
    # plt.semilogy(t,track_error,linewidth=2,color='green',label='LQR Tracking Error')
    # plt.xlabel('Time [s]',fontsize=2)
    # plt.ylabel(r'Tracking Error $\|x_{LQR} - x_{ref}\|$', fontsize=12)
    # plt.title(f'Task 3: LQR Tracking Error',fontsize=14)
    # plt.grid(True,which='both',alpha=0.3)
    # plt.legend(fontsize=1)
    # plt.tight_layout()
    # plt.show()

def plot_MPC(TT,xx_opt,xx_mpc,uu_opt,uu_mpc):
    #Plot for MPC
    xx_opt = np.degrees(xx_opt)
    xx_mpc = np.degrees(xx_mpc) 

    t = np.arange(TT) * par.dt
    fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    plt.subplots_adjust(hspace=0.3)

    state_labels = [r'$\theta_1$ [deg]', r'$\theta_2$ [deg]',r'$\dot{\theta}_1$ [deg/s]', r'$\dot{\theta}_2$ [deg/s]']

    print(f"\n---Plotting Trajectory Tracking via MPC---\n") 

    for i in range(xx_opt.shape[0]):
        axs[i].plot(t, xx_opt[i, :], 'r--', label='Optimal traj')
        axs[i].plot(t, xx_mpc[i, :-1], 'b', label='MPC tracking')
        axs[i].set_ylabel(f"{state_labels[i]}")
        axs[i].grid(True)
        axs[i].legend(loc='best')

    axs[4].plot(t[:-1], uu_opt[0, :-1], 'g--', label='Optimal input')
    axs[4].plot(t[:-1], uu_mpc[0, :-1], 'b', label='MPC input')
    axs[4].axhline(y=par.umax, color='k', linestyle='--', label='Input constraints')
    axs[4].axhline(y=par.umin, color='k', linestyle='--')
    axs[4].set_ylabel(r'$\tau$ [Nm]')
    axs[4].set_xlabel("Time [s]")
    axs[4].grid(True)
    axs[4].legend(loc='best')

    plt.suptitle(f"Task 4: Trajectory Tracking via MPC (T_pred = {par.T_pred})", fontsize=14)
    plt.tight_layout(rect = [0,0,1,0.97])
    plt.show()

    # #Plot tracking error 
    # track_error = np.linalg.norm(xx_mpc[:,:-1] - xx_opt,axis=0)
    
    # plt.figure(figsize=(10,6))
    # plt.semilogy(t,track_error,linewidth=2,color='green',label='MPC Tracking Error')
    # plt.xlabel('Time [s]',fontsize=2)
    # plt.ylabel(r'Tracking Error $\|x_{MPC} - x_{ref}\|$', fontsize=12)
    # plt.title(f'Task 4: MPC Tracking Error (N_pred) = {par.T_pred})',fontsize=14)
    # plt.grid(True,which='both',alpha=0.3)
    # plt.legend(fontsize=1)
    # plt.tight_layout()
    # plt.show()

def plot_dynamics(TT,xx_test):
    #Plot results
    time_axis = np.arange(TT) * par.dt

    plt.figure(figsize=(10, 8))

    # Subplot 1 (Theta 1)
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, np.degrees(xx_test[0, :]), color='tab:blue', linewidth=1.5, label=r"$\theta_1$")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [deg]")
    plt.title("Test dynamics, zero input")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    #Subplot 2 (Theta 2)
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, np.degrees(xx_test[1, :]), color='tab:orange', linewidth=1.5, label=r"$\theta_2$")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [deg]")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.show()