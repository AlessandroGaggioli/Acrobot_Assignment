import numpy as np
import parameters as par
import dynamics as dyn
import matplotlib.pyplot as plt
import animation
import task1
import cost
import newton_optcon
import LQR
import MPC

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)
import os

########################################################
#   TASK SELECTION 
tasks_to_run = {
    0: False, #Test dynamics 
    1: False, #Newton - step reference
    2: True, #Newton - smooth reference 
    3: True, #LQR tracking
    4: True, #MPC tracking
    5: False #animation
}

Armijo_Plot = False
########################################################

ns, ni, dt, tf = par.ns, par.ni, par.dt, par.tf
TT = int(tf/dt)

os.makedirs('data',exist_ok=True)
equilibria_file = 'data/equilibria.npy' #equilibria data filename

max_iters = 50
newton_threshold = 1e-7

if tasks_to_run[0]:

    # ---------- TASK 0 ---------
    print(' ----- Task 0 -----')

    #Initialization of state and input arrays
    xx_test = np.zeros((ns, TT))
    uu_test = np.zeros((ni, TT)) #u = 0 to test dynamics +

    #Set initial condition
    xx0 = np.array([np.radians(10), np.radians(10), 0, 0]) 
    xx_test[:, 0] = xx0

    #Simulation Loop
    for kk in range(TT - 1):
        #apply dynamiccs
        xx_test[:, kk+1] = dyn.dynamics_casadi(xx_test[:, kk], uu_test[:, kk])[0]
        
    #Plot results
    time_axis = np.arange(TT) * dt

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

#Equilibrium  points
if tasks_to_run[1] or tasks_to_run[2]:

    current_params = (par.theta2_start,par.theta2_end)

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

            xx_eq1, uu_eq1 = task1.find_equilibria(xe1[0], xe1[1]) 
            xx_eq2, uu_eq2 = task1.find_equilibria(xe2[0], xe2[1])

            print(f"xx_eq1: {xx_eq1*180/np.pi}, uu_eq1: {uu_eq1}")
            print(f"xx_eq2: {xx_eq2*180/np.pi}, uu_eq2: {uu_eq2}")

            # Salva equilibri CON i parametri usati
            np.save(equilibria_file, np.array([xx_eq1, xx_eq2, [uu_eq1], [uu_eq2], current_params], dtype=object))
    else: 
        print('----- Computing equilibria -----')
        xe1 = np.array([0.0, par.theta2_start, 0.0, 0.0])
        xe2 = np.array([0.0, par.theta2_end, 0.0, 0.0])

        xx_eq1,uu_eq1 = task1.find_equilibria(xe1[0],xe1[1]) 
        xx_eq2, uu_eq2 = task1.find_equilibria(xe2[0],xe2[1])

        print(f"xx_eq1: {xx_eq1*180/np.pi}, uu_eq1: {uu_eq1}")
        print(f"xx_eq2: {xx_eq2*180/np.pi}, uu_eq2: {uu_eq2}")

        #save equilibria
        np.save(equilibria_file, np.array([xx_eq1, xx_eq2, [uu_eq1], [uu_eq2]], dtype=object))

#########################################
# START TASK 1
#########################################

if tasks_to_run[1]:

    print(' ----- Task 1 -----')
    constant_traj = 0.50

    xx_ref,uu_ref = task1.build_smooth_ref(xx_eq1, xx_eq2,uu_eq1,uu_eq2, TT, constant_traj)

    # ----- Armijo -----
    #initialization

    xx_opt = np.zeros_like(xx_ref)
    xx_opt[:,:] = xx_eq1[:,np.newaxis]
    uu_opt = np.full_like(uu_ref,uu_eq1)

    xx_history = []
    uu_history = []
    cost_history = []
    descent_norm_history = []

    # Store data for Armijo plots
    armijo_data_history = []
    Kt_history = []
    sigma_t_history = []
    xx_before_armijo = []
    uu_before_armijo = []

    #Newton Loop
    for i in range(max_iters):

        xx_before_armijo.append(xx_opt.copy())
        uu_before_armijo.append(uu_opt.copy())

        J_current = cost.cost_fcn(xx_opt, uu_opt, xx_ref, uu_ref)
        cost_history.append(J_current)
        
        #Riccati, backward pass
        Kt, sigma_t, descent_arm, descent_norm = newton_optcon.backward_passing(xx_opt, uu_opt, xx_ref, uu_ref)
        descent_norm_history.append(descent_norm)

        #Save for later plotting
        Kt_history.append(Kt.copy())
        sigma_t_history.append(sigma_t.copy())
        
        #Fwd armijo
        xx_opt, uu_opt, gamma, J_new, armijo_data = newton_optcon.armijo_search(xx_opt, uu_opt, xx_ref, uu_ref, Kt, sigma_t, J_current, descent_arm)

        armijo_data_history.append(armijo_data)
        xx_history.append(xx_opt)
        uu_history.append(uu_opt)
        
        print(f"Newton iteration: {i}, cost: {J_current:<10.2f}, step (gamma): {gamma:<10.4f}, Armijo iterations: {len(armijo_data['gamma_list'])}")
        
        if i > 0 and abs(cost_history[-2] - J_current) < newton_threshold:
            #print("Convergerge ok")
            break

    if(Armijo_Plot):

        #Plot only first 2 and last 2 armijo iterations
        total_iters = len(armijo_data_history)

        if total_iters<=4:
            iters_to_plot = list(range(total_iters))
        else: 
            iters_to_plot = [0,1,total_iters-2,total_iters-1]

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
                iter_idx
            )
    
    # Save Task 1 results
    np.save('data/task1_xx_opt.npy', xx_opt)
    np.save('data/task1_uu_opt.npy', uu_opt)
    np.save('data/task1_xx_ref.npy', xx_ref)
    np.save('data/task1_uu_ref.npy', uu_ref)

    #Selected Newton iterations plot
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
            axs[4].plot(t, uu_history[it][0, :], color=colors[idx], 
                        linewidth=1.2, label=f'Iteration {it}')

    axs[4].set_ylabel(r'$\tau$ [Nm]')
    axs[4].set_xlabel("Time [s]")
    axs[4].grid(True, alpha=0.3)
    axs[4].legend(loc='best', ncol=2, fontsize='small')

    plt.suptitle("Task 1: Optimal trajectory evolution", fontsize=16)
    plt.show()

    #Plot final results
    t = np.arange(TT) * par.dt
    state_labels = [r'$\theta_1$ [rad]', r'$\theta_2$ [rad]', 
                    r'$\dot{\theta}_1$ [rad/s]', r'$\dot{\theta}_2$ [rad/s]']

    fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    plt.subplots_adjust(hspace=0.3)

    for i in range(xx_ref.shape[0]):
        #plot states
        axs[i].plot(t, (np.degrees(xx_ref[i, :])), 'r--', linewidth=1.5, label=f'Reference')
        axs[i].plot(t, (np.degrees(xx_opt[i, :])), 'b', linewidth=1.2, label=f'Optimal')
        axs[i].set_ylabel(state_labels[i])
        axs[i].grid(True, alpha=0.3)
        if i == 0:
            axs[i].legend(loc='upper right')

    #Plot control input
    axs[4].plot(t, uu_opt[0, :], 'g', linewidth=1.5, label='Optimal input')
    axs[4].set_ylabel(r'$\tau$ [Nm]')
    axs[4].set_xlabel("Time [s]")
    axs[4].grid(True, alpha=0.3)
    axs[4].legend(loc='upper right')

    plt.suptitle('Task 1: Newton optimization', fontsize=14)
    plt.show()

    #Cost evolution
    plt.figure()
    plt.semilogy(cost_history, 'o-', color='b', markersize=4)
    plt.title("Cost Evolution")
    plt.xlabel("Iters")
    plt.ylabel("Total Cost J")
    plt.grid(True, which="both", alpha=0.3)
    plt.show()

    # Norm of descent direction 
    plt.figure()
    plt.semilogy(descent_norm_history, 'o-', color='r', markersize=4)
    plt.title("Norm of Descent Direction - Task 1")
    plt.xlabel("Iterations")
    plt.ylabel(r"$\|\sigma\|$")
    plt.grid(True, which="both", alpha=0.3)
    plt.show()

##############################################################
# END TASK 1 
##############################################################

##############################################################
# START TASK 2
##############################################################
if tasks_to_run[2]:
    print(" ----- Task 2 -----")
    constant_traj = 0.05 
    #Generate smooth reference
    xx_ref, uu_ref = task1.build_smooth_ref(xx_eq1, xx_eq2, uu_eq1, uu_eq2, TT,constant_traj)

    #Newton loop
    xx_opt = np.zeros_like(xx_ref)
    xx_opt[:,:] = xx_eq1[:,np.newaxis]
    uu_opt = np.full_like(uu_ref,uu_eq1)

    xx_history = []
    uu_history = []
    cost_history = []
    descent_norm_history = []

    # Store data for Armijo plots
    armijo_data_history = []
    Kt_history = []
    sigma_t_history = []
    xx_before_armijo = []
    uu_before_armijo = []

    for i in range(max_iters):

        xx_before_armijo.append(xx_opt.copy())
        uu_before_armijo.append(uu_opt.copy())

        J_current = cost.cost_fcn(xx_opt, uu_opt, xx_ref, uu_ref)
        cost_history.append(J_current)
        
        #Riccati, backward passing
        Kt, sigma_t, descent_arm, descent_norm  = newton_optcon.backward_passing(xx_opt, uu_opt, xx_ref, uu_ref)
        descent_norm_history.append(descent_norm)

        # Save for later plotting
        Kt_history.append(Kt.copy())
        sigma_t_history.append(sigma_t.copy())
        
        #Fwd armijo
        xx_opt, uu_opt, gamma, J_new, armijo_data = newton_optcon.armijo_search(xx_opt, uu_opt, xx_ref, uu_ref, Kt, sigma_t, J_current, descent_arm)

        armijo_data_history.append(armijo_data)
        xx_history.append(xx_opt)
        uu_history.append(uu_opt)

        print(f"Newton iteration: {i}, cost: {J_current:<10.2f}, step (gamma): {gamma:<10.4f}, Armijo iterations: {len(armijo_data['gamma_list'])}")
        
        if i > 0 and abs(cost_history[-2] - J_current) < newton_threshold:
            #print("Convergence ok")
            break

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
                iter_idx
            )

    # Save Task 2 results
    np.save('data/task2_xx_opt.npy', xx_opt)
    np.save('data/task2_uu_opt.npy', uu_opt)
    np.save('data/task2_xx_ref.npy', xx_ref)
    np.save('data/task2_uu_ref.npy', uu_ref)

    #plot the theta1, theta2, dtheta1, dtheta2, tau for task 2
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
            axs[4].plot(t, uu_history[it][0, :], color=colors[idx], 
                        linewidth=1.2, label=f'Iteration {it}')

    axs[4].set_ylabel(r'$\tau$ [Nm]')
    axs[4].set_xlabel("Time [s]")
    axs[4].grid(True, alpha=0.3)
    axs[4].legend(loc='best', ncol=2, fontsize='small')

    plt.suptitle("Task 2: Optimal trajectory evolution", fontsize=16)
    plt.show()

    fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    plt.subplots_adjust(hspace=0.3)

    #labels = ['theta1', 'theta2', 'dtheta1', 'dtheta2']
    for i in range(xx_ref.shape[0]):
        plt.subplot(5, 1, i+1)
        plt.plot(np.degrees(xx_ref[i, :]), 'r--', label=f'Reference')
        plt.plot(np.degrees(xx_opt[i, :]), 'b', label=f'Optimal')
        axs[i].set_ylabel(state_labels[i])
        plt.grid()
        if i == 0:
            axs[i].legend(loc='best')

    plt.subplot(5, 1, 5)
    plt.plot(uu_opt[0, :], 'g', label='Optimal input')
    axs[4].set_ylabel(r'$\tau$ [Nm]')
    axs[4].set_xlabel('Time [s]')
    plt.legend(); plt.grid()
    plt.suptitle('Task 2: Smooth ref and Newton optimization', fontsize=14)
    plt.show()

    # Cost evolution 
    plt.figure()
    plt.semilogy(cost_history, 'o-', color='b', markersize=4)
    plt.title("Cost Evolution - Task 2")
    plt.xlabel("Iterations")
    plt.ylabel("Total Cost J")
    plt.grid(True, which="both", alpha=0.3)
    plt.show()

    # Norm of descent direction 
    plt.figure()
    plt.semilogy(descent_norm_history, 'o-', color='r', markersize=4)
    plt.title("Norm of Descent Direction - Task 2")
    plt.xlabel("Iterations")
    plt.ylabel(r"$\|\sigma\|$")
    plt.grid(True, which="both", alpha=0.3)
    plt.show()

    #print("Animation for task 2")
    animation.animate_double_pendolum(
        xx_star = np.degrees(xx_opt), 
        xx_ref  = np.degrees(xx_ref),
        dt = dt,
        title='Task 2: Newton optimization'
    )
##############################################################
# END TASK 2
##############################################################

##############################################################
# START TASK 3
##############################################################
if tasks_to_run[3]:
    print("----- Task 3 -----")

    if not tasks_to_run[2]:
        print('Loading Task2 result...')
        xx_opt = np.load('data/task2_xx_opt.npy')
        uu_opt = np.load('data/task2_uu_opt.npy')
        xx_ref = np.load('data/task2_xx_ref.npy')  
        uu_ref = np.load('data/task2_uu_ref.npy')  

    #Compute LQR gains along the optimized trajectory
    Kt_lqr = LQR.lqr(xx_opt, uu_opt)

    #define an initial perturbation
    xx0_perturbed = xx_opt[:, 0] + par.perturb
    perturbation = np.degrees(xx0_perturbed - xx_opt[:,0])
    print(f'Initial pertubation on theta1: {perturbation[0]:<10.2f}')
    print(f'Initial pertubation on theta2: {perturbation[1]:<10.2f}')

    #Closed-Loop LQR simulation
    xx_lqr, uu_lqr = LQR.simulate_fb(xx0_perturbed, xx_opt, uu_opt, Kt_lqr)

    # Save results
    np.save('data/task3_xx_lqr.npy', xx_lqr)
    np.save('data/task3_uu_lqr.npy', uu_lqr)
    np.save('data/task3_xx0_perturbed.npy', xx0_perturbed)

    #Plot lqr
    t = np.arange(TT) * par.dt
    fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    plt.subplots_adjust(hspace=0.3)

    state_labels = [r'$\theta_1$ [rad]', r'$\theta_2$ [rad]', 
                r'$\dot{\theta}_1$ [rad/s]', r'$\dot{\theta}_2$ [rad/s]']

    #Plot theta1, theta2, dtheta1, dtheta2 of both optimal and lqr
    for i in range(xx_ref.shape[0]):
        axs[i].plot(xx_opt[i, :], 'r--', linewidth=1.5, label='Optimal trajectory')
        axs[i].plot(xx_lqr[i, :], 'b', linewidth=1.2, label='LQR trajectory')
        axs[i].set_ylabel(state_labels[i])
        axs[i].grid(True, alpha=0.5)
        if i == 0:
            axs[i].legend(loc='best')

    #Plot control input
    axs[4].plot(uu_opt[0, :], 'r--', linewidth=1.5, label='Optimal input')
    axs[4].plot(uu_lqr[0, :], 'g', linewidth=1.2, label='LQR input')
    axs[4].axhline(y=par.umax, color='k', linestyle='--', label='Input constraints')
    axs[4].axhline(y=par.umin, color='k', linestyle='--')
    axs[4].set_ylabel(r'$\tau$ [Nm]')
    axs[4].set_xlabel('Time steps')
    axs[4].grid(True, alpha=0.5)
    axs[4].legend(loc='best')

    plt.suptitle('Task 3: LQR', fontsize=14)
    plt.show()

##############################################################
# END TASK 3 
##############################################################

##############################################################
# START TASK 4
##############################################################
if tasks_to_run[4]:

    #Here there will be MPC task part
    print("----- Task 4 -----")

    if not tasks_to_run[2]:
        xx_opt = np.load('data/task2_xx_opt.npy')
        uu_opt = np.load('data/task2_uu_opt.npy')
        xx_ref = np.load('data/task2_xx_ref.npy')  
        uu_ref = np.load('data/task2_uu_ref.npy')  
    if not tasks_to_run[3]:
        xx0_perturbed = np.load('data/task3_xx0_perturbed.npy')
    
    #MPC parameters
    T_sim = TT 
    T_pred = par.T_pred
    
    #simulate mpc 
    xx_mpc, uu_mpc = MPC.simulate_mpc(
        xx_init=xx0_perturbed,           # Stato iniziale perturbato
        xx_ref=xx_opt,          # Traiettoria di riferimento (stati ottimi)
        uu_ref=uu_opt,          # Controlli di riferimento (controlli ottimi)
        T_sim=T_sim,                # Passi di simulazione
        T_pred=T_pred,              # Orizzonte MPC
        verbose=True                # Mostra progresso
    )
    
    if xx_mpc is not None and uu_mpc is not None:
        print("MPC simulation completed successfully!")
        
        # Save results
        np.save('data/task4_xx_mpc.npy', xx_mpc)
        np.save('data/task4_uu_mpc.npy', uu_mpc)
        
        #Tracking error 
        tracking_error = np.linalg.norm(xx_mpc - np.hstack([xx_opt, xx_opt[:, -1:]]), axis=0)
        print(f"Average tracking error: {np.mean(tracking_error):.4f}")
        print(f"Max tracking error: {np.max(tracking_error):.4f}")
    else:
        print("ERROR: MPC simulation failed!")

    #Plot for MPC
    t = np.arange(TT) * par.dt
    fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    plt.subplots_adjust(hspace=0.3)

    state_labels = [r'$\theta_1$ [rad]', r'$\theta_2$ [rad]', 
                    r'$\dot{\theta}_1$ [rad/s]', r'$\dot{\theta}_2$ [rad/s]']

    for i in range(xx_ref.shape[0]):
        axs[i].plot(t, np.degrees(xx_opt[i, :]), 'r--', label='Optimal traj')
        axs[i].plot(t, np.degrees(xx_mpc[i, :-1]), 'b', label='MPC tracking')
        axs[i].set_ylabel(f"{state_labels[i]}")
        axs[i].grid(True)
        axs[i].legend(loc='best')

    axs[4].plot(t, uu_opt[0, :], 'g--', label='Optimal input')
    axs[4].plot(t, uu_mpc[0, :], 'b', label='MPC input')
    axs[4].axhline(y=par.umax, color='k', linestyle='--', label='Input constraints')
    axs[4].axhline(y=par.umin, color='k', linestyle='--')
    axs[4].set_ylabel(r'$\tau$ [Nm]')
    axs[4].set_xlabel("Time [s]")
    axs[4].grid(True)
    axs[4].legend(loc='best')

    plt.suptitle(f"MPC Results (N_pred = {par.T_pred})", fontsize=14)
    plt.tight_layout(rect = [0,0,1,0.97])
    plt.show()
    animation.animate_double_pendolum(np.degrees(xx_mpc[:,:-1]), np.degrees(xx_opt), dt, title='Task 4: MPC Animation')

##############################################################
# END TASK 4 
##############################################################

##############################################################
#START TASK 5 
##############################################################
if tasks_to_run[5]:
    print('----- Task 5 -----')

    if not tasks_to_run[2]:
        xx_opt = np.load('data/task2_xx_opt.npy')
    if not tasks_to_run[3]:
        xx_lqr = np.load('data/task3_xx_lqr.npy')

    #Animation of lqr
    animation.animate_double_pendolum(np.degrees(xx_lqr), np.degrees(xx_opt), dt, title='Task 5: LQR Animation')