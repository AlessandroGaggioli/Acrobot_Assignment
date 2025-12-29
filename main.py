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

xx_init = np.array([np.radians(90.0),np.radians(40.0),0.0,0.0]) 
uu_init = np.array([0.0]) 

dt = par.dt
ns, ni = par.ns, par.ni 
tf = par.tf

TT = int(tf/dt)
xx = np.zeros((ns,TT))
uu = np.zeros((ni,TT))

xx_temp = np.zeros(ns)
uu_temp = np.zeros(ni)

# ---------- TEST DYNAMICS ----------

# xx_temp = xx_init
# uu_temp = uu_init
# xx[:,0] = xx_init

# kk = 0 

# while kk < TT-1:
    
#     xx_temp= dyn.dynamics_casadi(xx_temp,uu_temp)[0]
#     xx[:,kk+1] = xx_temp
#     kk=kk+1 

# #plot
# x = np.linspace(0, TT, TT)  # X-axis values
# plt.subplot(2, 1, 1)  # (rows, columns, index)
# plt.plot(x, xx[0,:], label="theta1")
# plt.xlabel("TIME")  # Label for x-axis
# plt.ylabel("DEGREE (deg)")  # Label for y-axis
# plt.title("Reference Trajectory")  # Title of the plot
# plt.legend()  # Add legend
# plt.grid(True)  # Add grid for better readability 

# plt.subplot(2, 1, 2)
# plt.plot(x, xx[1,:], label="theta2")
# plt.xlabel("TIME")  # Label for x-axis
# plt.ylabel("DEGREE (deg)")  # Label for y-axis
# plt.legend()  # Add legend
# plt.grid(True)  # Add grid for better readability


# xx_star = np.zeros((ns,ni))#,max_iters)) #?? max iter va aggiutno alle altre vairabili? poi come si usa questa coordinata?
# animation.animate_double_pendolum(xx_star=xx, xx_ref=(xx/math.pi*180), dt=dt)
# plt.show()  # Display the plot 
# # when you have the optimal trajectory put xx_star=xx_star, xx_ref=xx_ref



# ---------- TEST TASK 1 ---------
#Equilibrium  points
print(' ----- Task 1 -----')

xe1 = np.array([np.radians(0), np.radians(90), 0.0, 0.0])
xe2 = np.array([np.radians(0), np.radians(60), 0.0, 0.0])

xx_eq1,uu_eq1 = task1.find_equilibria(xe1[0],xe1[1]) 
print(f"xx_eq1: {xx_eq1*180/np.pi}, uu_eq1: {uu_eq1}")

xx_eq2, uu_eq2 = task1.find_equilibria(xe2[0],xe2[1])
print(f"xx_eq2: {xx_eq2*180/np.pi}, uu_eq2: {uu_eq2}")

xx_ref,uu_ref = task1.build_reference(xx_eq1, xx_eq2,uu_eq1,uu_eq2, TT)

# # Open-loop simulation, input = uu_ref
# xx_sim = np.zeros((ns, TT))
# xx_sim[:, 0] = xx_ref[:, 0]

# uu_sim = np.zeros((ni,TT))
# uu_sim[:,0] = uu_ref[:,0]

# for k in range(TT - 1):
#     xx_sim[:, k+1] = dyn.dynamics_casadi(xx_sim[:, k],uu_sim[:,k])[0]

# # Animation 
# animation.animate_double_pendolum(
#     xx_star = xx_sim * 180 / math.pi,
#     xx_ref  = xx_ref * 180 / math.pi,
#     dt = dt
# )

# Plot theta_ref and theta_sim
#t = np.arange(TT) * dt
#plt.figure()
#plt.subplot(3,1,1)
#plt.plot(t, xx_ref[0,:], 'g--', label='theta1 ref')
#plt.legend()
#plt.grid()

#plt.subplot(3,1,2)
#plt.plot(t, xx_ref[1,:], 'g--', label='theta2 ref')
#plt.legend()
#plt.grid()

#plt.subplot(3,1,3)
#plt.plot(t, uu_ref[0,:], 'g--', label='u ref')
#plt.legend()
#plt.grid()

#plt.xlabel("time [s]")
#plt.show()

# xe1, ue1, xe2, ue2 = task1.find_equilibria()
# xx_ref, uu_ref = task1.build_reference(xe1, xe2, TT)
# uu_ref[:] = ue1

# # Open-loop simulation, input = 0
# xx_sim = np.zeros((ns, TT))
# xx_sim[:, 0] = xx_ref[:, 0]

# for k in range(TT - 1):
#     u = uu_ref[:,k]
#     xx_sim[:, k+1] = dyn.dynamics_casadi(xx_sim[:, k], u)[0]

# # Animation
# animation.animate_double_pendolum(
#     xx_star = xx_sim * 180 / math.pi,
#     xx_ref  = xx_ref * 180 / math.pi,
#     dt = dt
# )

# # Plot theta_ref and theta_sim
# t = np.arange(TT) * dt
# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(t, xx_ref[0,:], 'g--', label='theta1 ref')
# plt.plot(t, xx_sim[0,:], 'b', label='theta1 sim')
# plt.legend()
# plt.grid()

# plt.subplot(2,1,2)
# plt.plot(t, xx_ref[1,:], 'g--', label='theta2 ref')
# plt.plot(t, xx_sim[1,:], 'b', label='theta2 sim')
# plt.legend()
# plt.grid()

# plt.xlabel("time [s]")
# plt.show()

# ----- Armijo -----
#initialization
xx_opt = np.copy(xx_ref)
uu_opt = np.copy(uu_ref)
cost_history = []

max_iters = 20
armijo_threshold = 1e-3


#Newton Loop
for i in range(max_iters):
    J_current = cost.cost_fcn(xx_opt, uu_opt, xx_ref, uu_ref)
    cost_history.append(J_current)
    
    #Riccati, backward pass
    Kt, sigma_t = newton_optcon.backward_passing(xx_opt, uu_opt, xx_ref, uu_ref)
    
    #Fwd armijo
    xx_opt, uu_opt, gamma, J_new = newton_optcon.armijo_search(xx_opt, uu_opt, xx_ref, uu_ref, Kt, sigma_t, J_current)
    
    print(f"iteration: {i}, cost: {J_current:<10.2f}, step (gamma): {gamma:<10.2f}")
    
    if i > 0 and abs(cost_history[-2] - J_current) < armijo_threshold:
        #print("Convergerge ok")
        break

#Plot final results
t = np.arange(TT) * par.dt
state_labels = [r'$\theta_1$ [rad]', r'$\theta_2$ [rad]', 
                r'$\dot{\theta}_1$ [rad/s]', r'$\dot{\theta}_2$ [rad/s]']

fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
plt.subplots_adjust(hspace=0.3)

for i in range(4):
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
#Generate smooth reference
xx_ref, uu_ref = task1.build_smooth_ref(xx_eq1, xx_eq2, uu_eq1, uu_eq2, TT)

#Newton loop
xx_opt = np.copy(xx_ref)
uu_opt = np.copy(uu_ref)
cost_history = []

for i in range(max_iters):
    J_current = cost.cost_fcn(xx_opt, uu_opt, xx_ref, uu_ref)
    cost_history.append(J_current)
    
    #Riccati, backward passing
    Kt, sigma_t = newton_optcon.backward_passing(xx_opt, uu_opt, xx_ref, uu_ref)
    
    #Fwd armijo
    xx_opt, uu_opt, gamma, J_new = newton_optcon.armijo_search(xx_opt, uu_opt, xx_ref, uu_ref, Kt, sigma_t, J_current)
    
    print(f"iteration: {i:<5}, cost: {J_current:<10.2f}, step (gamma): {gamma:<10.4f}")
    
    if i > 0 and abs(cost_history[-2] - J_current) < armijo_threshold:
        #print("Convergence ok")
        break

#plot the theta1, theta2, dtheta1, dtheta2, tau for task 2
state_labels = [r'$\theta_1$ [rad]', r'$\theta_2$ [rad]', 
                r'$\dot{\theta}_1$ [rad/s]', r'$\dot{\theta}_2$ [rad/s]']

fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
plt.subplots_adjust(hspace=0.3)

#labels = ['theta1', 'theta2', 'dtheta1', 'dtheta2']
for i in range(4):
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
for i in range(4):
    axs[i].plot(xx_opt[i, :], 'r--', linewidth=1.5, label='Optimal trajectory')
    axs[i].plot(xx_lqr[i, :], 'b', linewidth=1.2, label='LQR trajectory')
    axs[i].set_ylabel(state_labels[i])
    axs[i].grid(True, alpha=0.5)
    if i == 0:
        axs[i].legend(loc='best')

#Plot control input
axs[4].plot(uu_opt[0, :], 'r--', linewidth=1.5, label='Optimal input')
axs[4].plot(uu_lqr[0, :], 'g', linewidth=1.2, label='LQR input')
axs[4].set_ylabel(r'$\tau$ [Nm]')
axs[4].set_xlabel('Time steps')
axs[4].grid(True, alpha=0.5)
axs[4].legend(loc='best')

plt.suptitle('Task 3: LQR', fontsize=14)
plt.show()


# ----- TASK 4 -----
#Here there will be MPC task part
#print("----- Task 4 -----")



# ----- TASK 5 -----
print(' ----- Task 5 -----')
#Animation of lqr
animation.animate_double_pendolum(np.degrees(xx_lqr), np.degrees(xx_opt), dt, title='Task 5: LQR Animation')