import dynamics as dyn
import parameters as par
import numpy as np
import matplotlib.pyplot as plt
import animation 
import math

xx_init = np.array([np.radians(90.0),np.radians(40.0),0.0,0.0]) 
uu_init = np.array([0.0]) 

dt = par.dt
ns = par.ns 
ni = par.ni 
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
    
    xx_temp= dyn.dynamics(xx_temp,uu_temp)[0]
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

