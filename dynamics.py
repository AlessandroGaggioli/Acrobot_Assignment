import numpy as np
import sympy as sy
import parameters as par #import parameters file "parameters.py"
import casadi as ca

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


#Global variables to store pre-compiled functions
_f_fun = None
_A_fun = None
_B_fun = None

def dynamics_casadi(xx, uu):
    """
    Computes the discrete-time dynamics and Jacobians using CasADi.
    Includes an internal integrator (CVODES) for high-precision mapping.
    Uses a Singleton pattern to compile symbolic functions only once.
    """
    global _f_fun, _A_fun, _B_fun

    #to initialize and compute only once
    if _f_fun is None:
        #print("Compiling CasADi dynamics")
        ns = par.ns
        ni = par.ni
        dt = par.dt

        #Symbolic state and input
        x = ca.SX.sym('x', ns)
        u = ca.SX.sym('u', ni)

        #unpack the state
        theta1, theta2 = x[0], x[1]
        dtheta1, dtheta2 = x[2], x[3]
        dq = ca.vertcat(dtheta1, dtheta2)

        #parameters
        m1, m2 = par.m1, par.m2
        l1, l2 = par.l1, par.l2
        lc1, lc2 = par.lc1, par.lc2
        I1, I2 = par.I1, par.I2
        g, f1, f2 = par.g, par.f1, par.f2

        #M(q)
        M11 = I1 + I2 + m1*lc1**2 + m2*(l1**2 + lc2**2 + 2*l1*lc2*ca.cos(theta2))
        M12 = I2 + m2*lc2*(l1*ca.cos(theta2) + lc2)
        M21 = M12
        M22 = I2 + m2*lc2**2
        M = ca.vertcat(ca.horzcat(M11, M12), ca.horzcat(M21, M22))

        #C(q, dq)
        C11 = -l1*lc2*m2*dtheta2*ca.sin(theta2)
        C12 = -l1*lc2*m2*(dtheta1+dtheta2)*ca.sin(theta2)
        C21 = l1*lc2*m2*dtheta1*ca.sin(theta2)
        C22 = 0
        C = ca.vertcat(ca.horzcat(C11, C12), ca.horzcat(C21, C22))

        #G(q)
        G = ca.vertcat(
            g*m1*lc1*ca.sin(theta1) + g*m2*(l1*ca.sin(theta1) + lc2*ca.sin(theta1 + theta2)),
            g*m2*lc2*ca.sin(theta1 + theta2)
        )

        #F(dq)
        F_vec = ca.vertcat(f1*dtheta1, f2*dtheta2)

        #torque is applied only to the second joint (Acrobot)
        tau_vec = ca.vertcat(0, u[0])
        ddq = ca.solve(M, tau_vec - C @ dq - F_vec - G)

        #x_dot = f(x, u)
        xdot = ca.vertcat(dtheta1, dtheta2, ddq[0], ddq[1])

        #integrate dynamics over dt using CVODES solver
        dae = {'x': x, 'p': u, 'ode': xdot}
        integrator = ca.integrator('int', 'cvodes', dae, {'tf': dt})
        
        #next state (d-t)
        x_next = integrator(x0=x, p=u)['xf']

        #symbolic Jacobians
        A_sym = ca.jacobian(x_next, x)
        B_sym = ca.jacobian(x_next, u)

        _f_fun = ca.Function('f_fun', [x, u], [x_next])
        _A_fun = ca.Function('A_fun', [x, u], [A_sym])
        _B_fun = ca.Function('B_fun', [x, u], [B_sym])
        #print("CasADi compilation complete")

    #Numerical validation, ensure inputs are correctly shaped for CasADi
    xx_num = np.asarray(xx).reshape(-1, 1)
    uu_num = np.asarray(uu).reshape(-1, 1)
    #compute next state and discrete Jacobians, .full() to converte in a numpy array
    xxp = _f_fun(xx_num, uu_num).full().flatten()
    fx = _A_fun(xx_num, uu_num).full()
    fu = _B_fun(xx_num, uu_num).full()

    #return state and transposed Jacobians for the solver
    return xxp, fx.T, fu.T