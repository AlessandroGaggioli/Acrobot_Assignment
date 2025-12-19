import numpy as np
import dynamics as dyn
import parameters as par
import animation
import matplotlib.pyplot as plt
import math

ns = par.ns
ni = par.ni
dt = par.dt
TT = int(par.tf/par.dt)

#equilibrium points
xe1 = np.array([0.0, 0.0, 0.0, 0.0])
xe2 = np.array([0.0, np.pi, 0.0, 0.0])

def build_reference(xe1, xe2, TT):
    xxref = np.zeros((ns, TT))
    uuref = np.zeros((ni, TT))

    for kk in range(TT):
        s = kk/(TT-1)
        alpha = 3*s**2 - 2*s**3
        xxref[:,kk] = (1-alpha)*xe1 + alpha*xe2
    
    return xxref, uuref

xx_ref, uu_ref = build_reference(xe1, xe2, TT)

# Open-loop simulation
xx_sim = np.zeros((ns, TT))
xx_sim[:, 0] = xx_ref[:, 0]

for k in range(TT - 1):
    u = np.zeros(ni)  # open-loop input = 0
    xx_sim[:, k+1], _, _ = dyn.dynamics_casadi(xx_sim[:, k], u)

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