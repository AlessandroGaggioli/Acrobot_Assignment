import numpy as np
import cost
import newton_optcon
import Plotter

def newton_loop(xx_ref,uu_ref,xx_eq1,uu_eq1,max_iters,newton_threshold,descent_norm_threshold,Armijo_Plot,task): 
    
    """
    Performs Newton's method optimization for trajectory optimization using Riccati-based
    backward/forward passing with Armijo line search.
    """
    
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

    print(f"\nStarting Newton --- Current cost: {J_current}\n")

    for i in range(max_iters):

        xx_before_armijo.append(xx_opt.copy())
        uu_before_armijo.append(uu_opt.copy())

        print(f"Task: {task} Iter: {i}\n")
        
        #Riccati, backward passing
        Kt, sigma_t, descent_arm, descent_norm  = newton_optcon.backward_passing(xx_opt, uu_opt, xx_ref, uu_ref)

        # Save for later plotting
        Kt_history.append(Kt.copy())
        sigma_t_history.append(sigma_t.copy())
        descent_norm_history.append(descent_norm.copy())
        
        #Fwd armijo
        xx_new, uu_new, gamma, J_new, armijo_data = newton_optcon.armijo_search(xx_opt, uu_opt, xx_ref, uu_ref, Kt, sigma_t, J_current, descent_arm)

        cost_diff = abs(J_current - J_new)

        print(f"cost: {J_new}, "f"cost diff: {cost_diff}, "f"step (gamma): {gamma:<10.2f}, "f"Armijo iters: {len(armijo_data['gamma_list'])}\n")

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
        
        print(f"\n--- Generating Armijo plots for iterations: {iters_to_plot} ---\n")
        
        for iter_idx in iters_to_plot:
            print(f"Plotting Armijo for Newton iteration {iter_idx}")
            Plotter.plot_armijo_iteration(
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
    
