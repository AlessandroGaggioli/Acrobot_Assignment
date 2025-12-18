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

# Calcolo Jacobiani Continui (A_c, B_c)
Jac_x_sym = f_cont.jacobian(xx_sym)
Jac_u_sym = f_cont.jacobian(uu_sym)

# Convertiamo in funzioni Numpy la dinamica CONTINUA (non discreta)
f_cont_lambdified = sy.lambdify((xx_sym, uu_sym), f_cont, 'numpy')
Jac_x_lambdified = sy.lambdify((xx_sym, uu_sym), Jac_x_sym, 'numpy')
Jac_u_lambdified = sy.lambdify((xx_sym, uu_sym), Jac_u_sym, 'numpy')

def dynamics(xx,uu): 
    
    xx = np.asarray(xx).flatten()
    uu = np.asarray(uu).flatten()
    
    # --- 1. Integrazione Numerica RK4 (fatta con i numeri, non simboli) ---
    # k1 = f(x, u)
    k1 = np.array(f_cont_lambdified(xx, uu)).flatten()
    # k2 = f(x + dt/2 * k1, u)
    k2 = np.array(f_cont_lambdified(xx + 0.5 * dt * k1, uu)).flatten()
    # k3 = f(x + dt/2 * k2, u)
    k3 = np.array(f_cont_lambdified(xx + 0.5 * dt * k2, uu)).flatten()
    # k4 = f(x + dt * k3, u)
    k4 = np.array(f_cont_lambdified(xx + dt * k3, uu)).flatten()
    # x_next = x + (dt/6) * (k1 + 2k2 + 2k3 + k4)
    xxp = xx + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4).flatten()

    # --- 2. Calcolo Gradienti Discreti ---
    # Calcoliamo A continuo e B continuo
    A_c = np.array(Jac_x_lambdified(xx, uu))
    B_c = np.array(Jac_u_lambdified(xx, uu))

    # Approssimazione discreta (Eulero): A_d = I + A_c * dt
    fx = np.eye(ns) + A_c * dt
    fu = B_c * dt

    # Ritorniamo i gradienti trasposti come nel tuo codice originale
    return xxp, fx.T, fu.T