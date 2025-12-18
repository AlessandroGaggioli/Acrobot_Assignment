import dynamics as dyn
import parameters as par
import numpy as np
import matplotlib.pyplot as plt
import animation 

xx_init = np.array(0.01,0.01,0.0,0.0)
uu_init = np.array(0.0)

print(1)

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

kk = 0 
print(TT)
while (TT-kk*dt>0):
    
    xx_temp= dyn.dynamics(xx,uu)[0]
    xx[:,kk] = xx_temp
    kk=kk+1 
    print(kk)
    print(TT-kk*dt)

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

plt.show()  # Display the plot 
xx_star = np.zeros((ns,ni))#,max_iters)) #?? max iter va aggiutno alle altre vairabili? poi come si usa questa coordinata?
animation.animate_double_pendolum(xx_star=xx_temp, xx_ref=xx, dt=dt)
# when you have the optimal trajectory put xx_star=xx_star, xx_ref=xx_ref

