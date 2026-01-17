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
dt = 0.01       # time step (s) #da abbassare 0.01 o 0.001
tf = 5          # simulation time (s)
Newton_max_iters = 50
newton_threshold = 1e-7
Armjio_max_iters = 25

#Input constraints (for MPC simulation)
umin = -2
umax = 2

#Imposed theta2 angles
theta2_start = np.radians(0)
theta2_end = np.radians(45)

#Perturbation on initial condition 
perturb = np.array([np.radians(1), np.radians(1), 0.0, 0.0])

#Cost matrices
Q = np.diag([10,10,1,1]) #weight on theta1, theta2, dtheta1, dtheta2 for stage cost
R = np.array([[0.1]]) #weight on tau
QT = np.diag([500,500,100,100]) #high terminal weight for terminal cost

#MPC parameters
T_pred = 10