import numpy as np
import sympy as sy
import parameters as par #import parameters file "parameters.py"

#system dimensions 

ns = par.ns #number of states [theta1, theta2, dtheta1, dtheta2]
ni = par.ni # number of inputs [tau] (torque applied at joint 2)

dt = par.dt # discrete time step

"""
Symbolic generation of the dynamics equations
"""
# Define symbolic variables - sympy simbols 
theta1, theta2, dtheta1, dtheta2 = sy.symbols('theta1 theta2 dtheta1 dtheta2')
u = sy.symbols('u')  # scalar torque

# symbolic vector - organize the variables into vectors
q = sy.Matrix([theta1, theta2])
dq = sy.Matrix([dtheta1, dtheta2])
xx_sym = sy.Matrix([theta1, theta2, dtheta1, dtheta2]) # State x 
uu_sym = sy.Matrix([u]) # Input u

# Model matrices - extract parameters from parameters file
m1,m2 = par.m1, par.m2
l1,l2 = par.l1, par.l2
lc1,lc2 = par.lc1, par.lc2
I1,I2 = par.I1, par.I2
g = par.g
f1,f2 = par.f1, par.f2

# M(q): Inertia matrix 
M11 = I1 + I2 + m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 *lc2 * sy.cos(theta2))
M12 = I2 + m2 * lc2 * (l1 * sy.cos(theta2) + lc2)
M21 = M12
M22 = I2 + m2 * lc2**2
M = sy.Matrix([[M11, M12], [M21, M22]])

# C(q,dq): Coriolis and centrifugal matrix
C11 = -l1 * lc2 * m2 * dtheta2 * sy.sin(theta2)
C12 = -l1 * lc2 * m2 * (dtheta1 + dtheta2) * sy.sin(theta2)
C21 = l1 * lc2 * m2 * dtheta1 * sy.sin(theta2)
C22 = 0
C = sy.Matrix([[C11, C12], [C21, C22]])

# G(q): Gravity vector
G1 = g * m1 * lc1 * sy.sin(theta1) + g * m2 * (l1 * sy.sin(theta1) + lc2 * sy.sin(theta1 + theta2))
G2 = g * m2 * lc2 * sy.sin(theta1 + theta2)
G_vec = sy.Matrix([G1, G2])

# F(dq): Viscous friction vector
F = sy.Matrix([[f1, 0], [0, f2]])

# Input vector
Tau_vec = sy.Matrix([0, u])

# Dynamics computation 
# Eq: M * ddq + C * dq + F * dq + G = Tau
# ddq = M_inv * (Tau - C*dq - F*dq - G)
ddq = M.inv() @ (Tau_vec - C @ dq - F @ dq - G_vec)

# continuous time dynamics
# x_dot = f(x,u)
f_cont = sy.Matrix([dtheta1,dtheta2,ddq[0],ddq[1]]) 

#Runge-Kutta 4th order method for discretization
#step 1 
k1 = f_cont # slope at beginning of interval
#step 2
k2_state = xx_sym + (dt/2) * k1 # state estimation at midpoint (dt/2) using k1
subs_k2 = list(zip(xx_sym, k2_state)) # create substitution list for midpoint state, zip couples each state variable with its estimated value
k2 = f_cont.subs(subs_k2) # dynamics f(...) evaluated at midpoint state: f(x_t + dt/2 * k1, u_t)
#step 3 
k3_state = xx_sym + (dt/2) * k2 # state estimation at midpoint (dt/2) using k2
subs_k3 = list(zip(xx_sym, k3_state)) # create substitution list for midpoint state, zip couples each state variable with its estimated value
k3 = f_cont.subs(subs_k3) # dynamics f(...) evaluated at midpoint state: f(x_t + dt/2 * k2, u_t)
#step 4
k4_state = xx_sym + dt * k3 # state estimation at end of interval (
subs_k4 = list(zip(xx_sym, k4_state)) # create substitution list for end state, zip couples each state variable with its estimated value
k4 = f_cont.subs(subs_k4) # dynamics f(...) evaluated at end

# Combine to get discrete time dynamics
"""
We calculate the state at the next step x_t+1 by taking a weighted average of the four slopes
Now f_sym is a symbolic expression representing the exact evolution of the system for a step dt
"""
f_sym = xx_sym + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

#Gradient of the dynamics
dfx_sym = f_sym.jacobian(xx_sym).T #gradients are the transposed Jacobians
dfu_sym = f_sym.jacobian(uu_sym).T #gradients are the transposed Jacobians

"""
sympy is slow for numerical computations
we convert the symbolic expressions into numerical functions using lambdify
"""
f_dt_lambdified = sy.lambdify((xx_sym, uu_sym), f_sym, 'numpy') 
dfx_lambdified = sy.lambdify((xx_sym, uu_sym), dfx_sym, 'numpy')
dfu_lambdified = sy.lambdify((xx_sym, uu_sym), dfu_sym, 'numpy')

def dynamics(xx,uu): 
    
    xx = np.asarray(xx).reshape(ns,1)
    uu = np.asarray(uu).reshape(ni,1)
    """
    In python a vector can be a flat array or a column vector
    SymPy lambdified functions expect column vectors to do matricial operations correctly
    We reshape the inputs to column vectors before passing them to the lambdified functions
    """
    # next state computation
    xxp = np.array(f_dt_lambdified(xx,uu)).squeeze()
    """
    f_dt_lambdified returns a 2D array, we use squeeze to convert it to a 1D array
    """
    # gradients computation
    fx = np.array(dfx_lambdified(xx,uu))
    fu = np.array(dfu_lambdified(xx,uu))
    """
    calculate the gradients usthising the lambdified functions
    dfx_lambdified: derivative of f w.r.t. x
    dfu_lambdified: derivative of f w.r.t. u
    As we defined them in the simbolic part, this functions return the transposed Jacobians (A_T and B_T)
    """

    # return next state and gradients
    # xxp: next state
    # fx: A_T = df/dx
    # fu: B_T = df/du
    return xxp, fx, fu