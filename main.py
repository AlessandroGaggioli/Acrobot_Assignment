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

#task_to_execute = [False, True, True, True, True, True] #Task 0, 1, 2, 3, 4, 5

ns, ni, dt, tf = par.ns, par.ni, par.dt, par.tf
TT = int(tf/dt)

# ---------- TEST DYNAMICS (TASK 0) ---------
print(' ----- Task 0 -----')
#Initialization of state and input arrays
xx_test = np.zeros((ns, TT))
uu_test = np.zeros((ni, TT)) #u = 0 to test dynamics

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

# ---------- TASK 1 ---------
#Equilibrium  points
print(' ----- Task 1 -----')
constant_traj = 0.50

xe1 = np.array([0.0, par.theta2_start, 0.0, 0.0])
xe2 = np.array([0.0, par.theta2_end, 0.0, 0.0])

xx_eq1,uu_eq1 = task1.find_equilibria(xe1[0],xe1[1]) 
print(f"xx_eq1: {xx_eq1*180/np.pi}, uu_eq1: {uu_eq1}")

xx_eq2, uu_eq2 = task1.find_equilibria(xe2[0],xe2[1])
print(f"xx_eq2: {xx_eq2*180/np.pi}, uu_eq2: {uu_eq2}")

xx_ref,uu_ref = task1.build_smooth_ref(xx_eq1, xx_eq2,uu_eq1,uu_eq2, TT, constant_traj)

# ----- Armijo -----
#initialization
#xx_opt = np.copy(xx_ref)
#uu_opt = np.copy(uu_ref)
xx_opt = np.zeros_like(xx_ref)
uu_opt = np.zeros_like(uu_ref)
xx_history = []
uu_history = []
cost_history = []
#sigma_t_history = []

max_iters = 50
armijo_threshold = 1e-3

#Newton Loop
for i in range(max_iters):
    J_current = cost.cost_fcn(xx_opt, uu_opt, xx_ref, uu_ref)
    cost_history.append(J_current)
    
    #Riccati, backward pass
    Kt, sigma_t = newton_optcon.backward_passing(xx_opt, uu_opt, xx_ref, uu_ref)
    
    #Fwd armijo
    xx_opt, uu_opt, gamma, J_new = newton_optcon.armijo_search(xx_opt, uu_opt, xx_ref, uu_ref, Kt, sigma_t, J_current)
    xx_history.append(xx_opt)
    uu_history.append(uu_opt)
    
    print(f"iteration: {i}, cost: {J_current:<10.2f}, step (gamma): {gamma:<10.4f}")
    
    if i > 0 and abs(cost_history[-2] - J_current) < armijo_threshold:
        #print("Convergerge ok")
        break

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
    axs[i].plot(t, xx_ref[i, :], 'r--', linewidth=1.5, label='Reference' if i == 0 else None)
    
    #Plot of selected iterations
    for idx, it in enumerate(iters_to_plot):
        if it < len(xx_history):
            label = f'Iteration {it}' if i == 0 else None
            axs[i].plot(t, xx_history[it][i, :], color=colors[idx], 
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
    axs[i].plot(t, (xx_ref[i, :]), 'r--', linewidth=1.5, label=f'Reference')
    axs[i].plot(t, (xx_opt[i, :]), 'b', linewidth=1.2, label=f'Optimal')
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


# ---- TASK 2 ----
print(" ----- Task 2 -----")
constant_traj = 0.10 
#Generate smooth reference
xx_ref, uu_ref = task1.build_smooth_ref(xx_eq1, xx_eq2, uu_eq1, uu_eq2, TT,constant_traj)

#Newton loop
xx_opt = np.copy(xx_ref)
uu_opt = np.copy(uu_ref)
xx2_hist = []
uu2_hist = []
cost_history = []

for i in range(max_iters):
    J_current = cost.cost_fcn(xx_opt, uu_opt, xx_ref, uu_ref)
    cost_history.append(J_current)
    
    #Riccati, backward passing
    Kt, sigma_t = newton_optcon.backward_passing(xx_opt, uu_opt, xx_ref, uu_ref)
    
    #Fwd armijo
    xx_opt, uu_opt, gamma, J_new = newton_optcon.armijo_search(xx_opt, uu_opt, xx_ref, uu_ref, Kt, sigma_t, J_current)
    xx2_hist.append(xx_opt)
    uu2_hist.append(uu_opt)

    print(f"iteration: {i}, cost: {J_current:<10.2f}, step (gamma): {gamma:<10.4f}")
    
    if i > 0 and abs(cost_history[-2] - J_current) < armijo_threshold:
        #print("Convergence ok")
        break

#plot the theta1, theta2, dtheta1, dtheta2, tau for task 2
state_labels = [r'$\theta_1$ [rad]', r'$\theta_2$ [rad]', 
                r'$\dot{\theta}_1$ [rad/s]', r'$\dot{\theta}_2$ [rad/s]']

#choose the iterations to plot
iters_to_plot = [0, 1, 3, len(xx_history)-1] 
colors = ['#ff7f0e', '#2ca02c', '#9467bd', 'b'] #Colours: orange, green, purple, blu

fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
plt.subplots_adjust(hspace=0.3)

for i in range(xx_ref.shape[0]):
    #Reference plot
    axs[i].plot(t, xx_ref[i, :], 'r--', linewidth=1.5, label='Reference' if i == 0 else None)
    
    #Plot of selected iterations
    for idx, it in enumerate(iters_to_plot):
        if it < len(xx2_hist):
            label = f'Iteration {it}' if i == 0 else None
            axs[i].plot(t, xx2_hist[it][i, :], color=colors[idx], 
                        linewidth=1.2, label=label)
    
    axs[i].set_ylabel(state_labels[i])
    axs[i].grid(True, alpha=0.3)
    if i == 0:
        axs[i].legend(loc='best', ncol=2, fontsize='small')

#Control input plot for the iterations
for idx, it in enumerate(iters_to_plot):
    if it < len(uu2_hist):
        axs[4].plot(t, uu2_hist[it][0, :], color=colors[idx], 
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
    plt.plot(xx_ref[i, :], 'r--', label=f'Reference')
    plt.plot(xx_opt[i, :], 'b', label=f'Optimal')
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


#print("Animation for task 2")
animation.animate_double_pendolum(
    xx_star = np.degrees(xx_opt), 
    xx_ref  = np.degrees(xx_ref),
    dt = dt,
    title='Task 2: Newton optimization'
)


#----- TASK 3 -----
print("----- Task 3 -----")

#Compute LQR gains along the optimized trajectory
Kt_lqr = LQR.lqr(xx_opt, uu_opt)

#define an initial perturbation
xx0_perturbed = xx_opt[:, 0] + np.array([np.radians(5), np.radians(20), 0.0, 0.0])
perturbation = np.degrees(xx0_perturbed - xx_opt[:,0])
print(f'Initial pertubation on theta1: {perturbation[0]:<10.2f}')
print(f'Initial pertubation on theta2: {perturbation[1]:<10.2f}')

#Closed-Loop LQR simulation
xx_lqr, uu_lqr = LQR.simulate_fb(xx0_perturbed, xx_opt, uu_opt, Kt_lqr)

#Plot lqr
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


# ----- TASK 4 -----
#Here there will be MPC task part
print("----- Task 4 -----")
N_pred = 10
xx_mpc, uu_mpc = MPC.simulate_mpc(xx0_perturbed, xx_opt, uu_opt, N_pred) #simulation MPC

#Plot for MPC
t = np.arange(TT) * par.dt
fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
plt.subplots_adjust(hspace=0.3)

for i in range(xx_ref.shape[0]):
    axs[i].plot(t, np.degrees(xx_opt[i, :]), 'r--', label='Optimal traj')
    axs[i].plot(t, np.degrees(xx_mpc[i, :]), 'b', label='MPC tracking')
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

plt.suptitle(f"MPC Results (N_pred = {N_pred})", fontsize=14)
plt.tight_layout(rect = [0,0,1,0.97])
plt.show()
animation.animate_double_pendolum(np.degrees(xx_mpc), np.degrees(xx_opt), dt, title='Task 4: MPC Animation')

# ----- TASK 5 -----
print('----- Task 5 -----')
#Animation of lqr
animation.animate_double_pendolum(np.degrees(xx_lqr), np.degrees(xx_opt), dt, title='Task 5: LQR Animation')