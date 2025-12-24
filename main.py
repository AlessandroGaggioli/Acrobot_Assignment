import numpy as np
import parameters as par
import dynamics as dyn
import matplotlib.pyplot as plt
import math
import animation
import task1

# ---------- TEST DYNAMICS ----------

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

xx_temp = xx_init
uu_temp = uu_init
xx[:,0] = xx_init

kk = 0 

while kk < TT-1:
    
    xx_temp= dyn.dynamics_casadi(xx_temp,uu_temp)[0]
    xx[:,kk+1] = xx_temp
    kk=kk+1 

#plot
x = np.linspace(0, TT, TT)  # X-axis values
plt.subplot(2, 1, 1)  # (rows, columns, index)
plt.plot(x, xx[0,:], label="theta1")
plt.xlabel("TIME")  # Label for x-axis
plt.ylabel("DEGREE (deg)")  # Label for y-axis
plt.title("Reference Trajectory")  # Title of the plot
plt.legend()  # Add legend
plt.grid(True)  # Add grid for better readability 

plt.subplot(2, 1, 2)
plt.plot(x, xx[1,:], label="theta2")
plt.xlabel("TIME")  # Label for x-axis
plt.ylabel("DEGREE (deg)")  # Label for y-axis
plt.legend()  # Add legend
plt.grid(True)  # Add grid for better readability


xx_star = np.zeros((ns,ni))#,max_iters)) #?? max iter va aggiutno alle altre vairabili? poi come si usa questa coordinata?
animation.animate_double_pendolum(xx_star=xx, xx_ref=(xx/math.pi*180), dt=dt)
plt.show()  # Display the plot 
# when you have the optimal trajectory put xx_star=xx_star, xx_ref=xx_ref



# ---------- TEST TASK 1 ---------
#Equilibium  points
xe1 = np.array([0.0, 0.0, 0.0, 0.0])
xe2 = np. array([0.0, np.pi, 0.0, 0.0])

xx_ref, uu_ref = task1.build_reference(xe1, xe2, TT)

# Open-loop simulation, input = 0
xx_sim = np.zeros((ns, TT))
xx_sim[:, 0] = xx_ref[:, 0]

for k in range(TT - 1):
    u = np.zeros(ni)
    xx_sim[:, k+1] = dyn.dynamics_casadi(xx_sim[:, k], u)[0]

# Animation
animation.animate_double_pendolum(
    xx_star = xx_sim * 180 / math.pi,
    xx_ref  = xx_ref * 180 / math.pi,
    dt = dt
)

# Plot theta_ref and theta_sim
t = np.arange(TT) * dt
plt.figure()
plt.subplot(2,1,1)
plt.plot(t, xx_ref[0,:], 'g--', label='theta1 ref')
plt.plot(t, xx_sim[0,:], 'b', label='theta1 sim')
plt.legend()
plt.grid()

plt.subplot(2,1,2)
plt.plot(t, xx_ref[1,:], 'g--', label='theta2 ref')
plt.plot(t, xx_sim[1,:], 'b', label='theta2 sim')
plt.legend()
plt.grid()

plt.xlabel("time [s]")
plt.show()


xe1, ue1, xe2, ue2 = task1.find_equilibria()
xx_ref, uu_ref = task1.build_reference(xe1, xe2, TT)
uu_ref[:] = ue1

# Open-loop simulation, input = 0
xx_sim = np.zeros((ns, TT))
xx_sim[:, 0] = xx_ref[:, 0]

for k in range(TT - 1):
    u = uu_ref[:,k]
    xx_sim[:, k+1] = dyn.dynamics_casadi(xx_sim[:, k], u)[0]

# Animation
animation.animate_double_pendolum(
    xx_star = xx_sim * 180 / math.pi,
    xx_ref  = xx_ref * 180 / math.pi,
    dt = dt
)

# Plot theta_ref and theta_sim
t = np.arange(TT) * dt
plt.figure()
plt.subplot(2,1,1)
plt.plot(t, xx_ref[0,:], 'g--', label='theta1 ref')
plt.plot(t, xx_sim[0,:], 'b', label='theta1 sim')
plt.legend()
plt.grid()

plt.subplot(2,1,2)
plt.plot(t, xx_ref[1,:], 'g--', label='theta2 ref')
plt.plot(t, xx_sim[1,:], 'b', label='theta2 sim')
plt.legend()
plt.grid()

plt.xlabel("time [s]")
plt.show()