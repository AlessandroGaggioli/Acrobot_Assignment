import numpy as np
import parameters as par
import dynamics as dyn
import matplotlib.pyplot as plt
import cost
import newton_optcon

def newton_loop(xx_ref,uu_ref,xx_eq1,uu_eq1,max_iters,newton_threshold,descent_norm_threshold,Armijo_Plot,task): 
    #Newton loop
    xx_opt = np.zeros_like(xx_ref)
    xx_opt[:,:] = xx_eq1[:,np.newaxis]
    uu_opt = np.full_like(uu_ref,uu_eq1)
    J_current = cost.cost_fcn(xx_opt, uu_opt, xx_ref, uu_ref)

    xx_history = []
    uu_history = []
    cost_history = [J_current]
    descent_norm_history = []

    # Store data for Armijo plots
    armijo_data_history = []
    Kt_history = []
    sigma_t_history = []
    xx_before_armijo = []
    uu_before_armijo = []

    gamma = 1.0 
    armijo_data = {'gamma_list': []}

    for i in range(max_iters):

        xx_before_armijo.append(xx_opt.copy())
        uu_before_armijo.append(uu_opt.copy())
        
        #Riccati, backward passing
        Kt, sigma_t, descent_arm, descent_norm  = newton_optcon.backward_passing(xx_opt, uu_opt, xx_ref, uu_ref)

        # Save for later plotting
        Kt_history.append(Kt.copy())
        sigma_t_history.append(sigma_t.copy())
        descent_norm_history.append(descent_norm.copy())
        
        #Fwd armijo
        xx_new, uu_new, gamma, J_new, armijo_data = newton_optcon.armijo_search(xx_opt, uu_opt, xx_ref, uu_ref, Kt, sigma_t, J_current, descent_arm)

        cost_diff = abs(J_current - J_new)

        print(
        f"Iter: {i}, "
        f"descent norm: {descent_norm:<10.2f}, "
        f"cost: {J_new}, "
        f"cost diff: {cost_diff}, "
        f"step (gamma): {gamma:<10.2f}, "
        f"Armijo iters: {len(armijo_data['gamma_list'])}"
        )

        # Save history 
        cost_history.append(J_new)
        armijo_data_history.append(armijo_data)
        xx_history.append(xx_new)
        uu_history.append(uu_new)

         # Convergence check
        if cost_diff < newton_threshold or descent_norm < descent_norm_threshold:
            xx_opt, uu_opt = xx_new, uu_new
            J_current = J_new
            break

        xx_opt,uu_opt = xx_new,uu_new
        J_current = J_new
    
    history = {
        'cost_history' : cost_history,
        'armijo_data_history' : armijo_data_history,
        'xx_history' : xx_history,
        'uu_history' : uu_history,
        'descent_norm_history' : descent_norm_history
    }

    if(Armijo_Plot):

        # Plot only first 2 and last 2 Armijo iterations
        total_iters = len(armijo_data_history)
        
        if total_iters <= 4:
            iters_to_plot = list(range(total_iters))
        else:
            iters_to_plot = [0, 1, total_iters-2, total_iters-1]
        
        print(f"\n--- Generating Armijo plots for iterations: {iters_to_plot} ---")
        
        for iter_idx in iters_to_plot:
            print(f"Plotting Armijo for Newton iteration {iter_idx}")
            newton_optcon.plot_armijo_iteration(
                xx_before_armijo[iter_idx], 
                uu_before_armijo[iter_idx],
                xx_ref, 
                uu_ref,
                Kt_history[iter_idx],
                sigma_t_history[iter_idx],
                armijo_data_history[iter_idx],
                iter_idx,
                task
            )
    return xx_opt, uu_opt, gamma, J_new, armijo_data, history

def newton_plot(TT,xx_history,xx_ref,uu_history,xx_opt,uu_opt,cost_history,descent_norm_history,task):
    
    t = np.arange(TT) * par.dt
    state_labels = [r'$\theta_1$ [rad]', r'$\theta_2$ [rad]', 
                    r'$\dot{\theta}_1$ [rad/s]', r'$\dot{\theta}_2$ [rad/s]']

    #choose the iterations to plot
    iters_to_plot = [0, 1, 3, len(xx_history)-1] 
    colors = ['#ff7f0e', '#2ca02c', '#9467bd', 'b'] #Colours: orange, green, purple, blu

    fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    plt.subplots_adjust(hspace=0.3)

    for i in range(xx_ref.shape[0]):
        #Reference plot
        axs[i].plot(t, np.degrees(xx_ref[i, :]), 'r--', linewidth=1.5, label='Reference' if i == 0 else None)
        
        #Plot of selected iterations
        for idx, it in enumerate(iters_to_plot):
            if it < len(xx_history):
                label = f'Iteration {it}' if i == 0 else None
                axs[i].plot(t, np.degrees(xx_history[it][i, :]), color=colors[idx], 
                            linewidth=1.2, label=label)
        
        axs[i].set_ylabel(state_labels[i])
        axs[i].grid(True, alpha=0.3)
        if i == 0:
            axs[i].legend(loc='best', ncol=2, fontsize='small')

    #Control input plot for the iterations
    for idx, it in enumerate(iters_to_plot):
        if it < len(uu_history):
            axs[4].plot(t[:-1], uu_history[it][0, :-1], color=colors[idx], 
                        linewidth=1.2, label=f'Iteration {it}')

    axs[4].set_ylabel(r'$\tau$ [Nm]')
    axs[4].set_xlabel("Time [s]")
    axs[4].grid(True, alpha=0.3)
    axs[4].legend(loc='best', ncol=2, fontsize='small')

    plt.suptitle(f"Task {task}: Optimal trajectory evolution", fontsize=16)
    plt.show()

    fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    plt.subplots_adjust(hspace=0.3)

    for i in range(xx_ref.shape[0]):
        plt.subplot(5, 1, i+1)
        plt.plot(np.degrees(xx_ref[i, :]), 'r--', label=f'Reference')
        plt.plot(np.degrees(xx_opt[i, :]), 'b', label=f'Optimal')
        axs[i].set_ylabel(state_labels[i])
        plt.grid()
        if i == 0:
            axs[i].legend(loc='best')

    plt.subplot(5, 1, 5)
    plt.plot(uu_opt[0, :-1], 'g', label='Optimal input')
    axs[4].set_ylabel(r'$\tau$ [Nm]')
    axs[4].set_xlabel('Time [s]')
    plt.legend(); plt.grid()
    plt.suptitle(f"Task {task}: Smooth ref and Newton optimization", fontsize=14)
    plt.show()

    # Cost evolution 
    plt.figure()
    plt.semilogy(cost_history, 'o-', color='b', markersize=4)
    plt.title(f"Cost Evolution - Task {task}")
    plt.xlabel("Iterations")
    plt.ylabel("Total Cost J")
    plt.grid(True, which="both", alpha=0.3)
    plt.show()

    # Norm of descent direction 
    plt.figure()
    plt.semilogy(descent_norm_history, 'o-', color='r', markersize=4)
    plt.title(f"Norm of Descent Direction - Task {task}")
    plt.xlabel("Iterations")
    plt.ylabel(r"$\|\sigma\|$")
    plt.grid(True, which="both", alpha=0.3)
    plt.show()
    