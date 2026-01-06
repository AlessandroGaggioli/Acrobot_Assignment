import numpy as np

# dynamic parameters - Set 1 
m1 = 1.0  # mass of link 1 (kg)
m2 = 1.0  # mass of link 2 (kg)
l1 = 1.0  # length of link 1 (m)
l2 = 1.0  # length of link 2 (m)
lc1 = 0.5 # distance between pivot point and center of mass of link 1 (m)
lc2 = 0.5 # distance between pivot point and center of mass of link 2 (m)
I1 = 0.33 # inertia of link 1 (kg*m^2)
I2 = 0.33 # inertia of link 2 (kg*m^2)
g = 9.81  # acceleration due to gravity (m/s^2)
f1 = 1.0 # viscous friction coefficient of joint 1
f2 = 1.0 # viscous friction coefficient of joint 2

# system parameters 
ns = 4 #number of states [theta1, theta2, dtheta1, dtheta2]
ni = 1 # number of inputs [tau] (torque applied at joint 2)

# simulation parameters
dt = 0.05       # time step (s)
tf = 30          # simulation time (s)

#Input constraints (for MPC simulation)
umin = -5
umax = 4