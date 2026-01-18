import numpy as np


# TASK SELECTION 
tasks_to_run = {
    0: False, #Test dynamics 
    1: False, #Newton - step reference
    2: False, #Newton - smooth reference 
    3: False, #LQR tracking
    4: True, #MPC tracking
    5: True #animation
}
Armijo_Plot = True

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
descent_norm_threshold = 1e-4

#Input constraints (for MPC simulation)
umin = -3
umax = 3

#Imposed theta2 angles
theta2_start = np.radians(0)
theta2_end = np.radians(45)

#Perturbation on initial condition 
perturb = np.array([np.radians(1), np.radians(1), 0.0, 0.0])

#Cost matrices

# Cost matrices - DEFAULT (task 1-2)
Q = np.diag([10, 10, 1, 1])
R = np.array([[0.1]])
QT = np.diag([500, 500, 100, 100])

# Cost matrices - TASK 3-4 (tracking)
Q_tracking = np.diag([100, 100, 10, 10])
R_tracking = np.array([[1.0]])
QT_tracking = np.diag([1000, 1000, 200, 200])

def set_cost_matrices(task_type='default'):

    global Q, R, QT
    
    if task_type == 'tracking':
        Q = Q_tracking.copy()
        R = R_tracking.copy()
        QT = QT_tracking.copy()
    else:  # 'default'
        Q = np.diag([10, 10, 1, 1])
        R = np.array([[0.1]])
        QT = np.diag([500, 500, 100, 100])

#MPC parameters
T_pred = 30