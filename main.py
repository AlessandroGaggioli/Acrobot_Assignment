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
import Newton_Loop 
import Test_Dynamics

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)
import os


tasks_to_run = par.tasks_to_run 
Armijo_Plot = par.Armijo_Plot
ns, ni, dt, tf = par.ns, par.ni, par.dt, par.tf
TT = int(tf/dt)
max_iters = par.Newton_max_iters
newton_threshold = par.newton_threshold
descent_norm_threshold = par.descent_norm_threshold

os.makedirs('data',exist_ok=True)
equilibria_file = 'data/equilibria.npy' #equilibria data filename

if tasks_to_run[0]:
    # ---------- TASK 0 ---------
    print(' ----- Task 0 -----')
    Test_Dynamics.test_dynamics(TT,ns,ni,dt)

#Equilibrium  points
if tasks_to_run[1] or tasks_to_run[2]:
    current_params = (par.theta2_start,par.theta2_end)
    #Find equilibria
    xx_eq1,xx_eq2,uu_eq1,uu_eq2 = task1.equilibria(equilibria_file,current_params)

#########################################
# START TASK 1
#########################################

if tasks_to_run[1]:

    print(' ----- Task 1 -----')
    constant_traj = 0.50

    xx_ref,uu_ref = task1.build_smooth_ref(xx_eq1, xx_eq2,uu_eq1,uu_eq2, TT, constant_traj)

    xx_opt, uu_opt, gamma, J_new, armijo_data, history = Newton_Loop.newton_loop(xx_ref,uu_ref,xx_eq1,uu_eq1,max_iters,newton_threshold,descent_norm_threshold,Armijo_Plot)

    cost_history= history['cost_history']
    armijo_data_history = history['armijo_data_history']
    xx_history = history['xx_history']
    uu_history = history['uu_history']
    descent_norm_history = history['descent_norm_history']
    
    # Save Task 1 results
    np.save('data/task1_xx_opt.npy', xx_opt)
    np.save('data/task1_uu_opt.npy', uu_opt)
    np.save('data/task1_xx_ref.npy', xx_ref)
    np.save('data/task1_uu_ref.npy', uu_ref)

    Newton_Loop.newton_plot(TT,xx_history,xx_ref,uu_history,xx_opt,uu_opt,cost_history,descent_norm_history)

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

    #Newton Loop
    xx_opt, uu_opt, gamma, J_new, armijo_data, history = Newton_Loop.newton_loop(xx_ref,uu_ref,xx_eq1,uu_eq1,max_iters,newton_threshold,descent_norm_threshold,Armijo_Plot)

    cost_history= history['cost_history']
    armijo_data_history = history['armijo_data_history']
    xx_history = history['xx_history']
    uu_history = history['uu_history']
    descent_norm_history = history['descent_norm_history']

    # Save Task 2 results
    np.save('data/task2_xx_opt.npy', xx_opt)
    np.save('data/task2_uu_opt.npy', uu_opt)
    np.save('data/task2_xx_ref.npy', xx_ref)
    np.save('data/task2_uu_ref.npy', uu_ref)

    #Plot trajectories, cost, descent 
    Newton_Loop.newton_plot(TT,xx_history,xx_ref,uu_history,xx_opt,uu_opt,cost_history,descent_norm_history)

    #print("Animation for task 2")
    animation.animate_double_pendolum(xx_star = np.degrees(xx_opt), xx_ref  = np.degrees(xx_ref),dt = dt,title='Task 2: Newton optimization')

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

    #Plot LQR 
    LQR.plot_LQR(TT,xx_ref,xx_opt,xx_lqr,uu_opt,uu_lqr)

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

    #Plot MPC 
    MPC.plot_MPC(TT,xx_ref,xx_opt,xx_mpc,uu_opt,uu_mpc)
    
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
    if not tasks_to_run[4]:
        xx_mpc = np.load('data/task4_xx_mpc.npy')

    #Animation of lqr
    animation.animate_double_pendolum(np.degrees(xx_lqr), np.degrees(xx_opt), dt, title='Task 5: LQR Animation')

    #Animate MPC 
    animation.animate_double_pendolum(np.degrees(xx_mpc[:,:-1]), np.degrees(xx_opt), dt, title='Task 4: MPC Animation')
