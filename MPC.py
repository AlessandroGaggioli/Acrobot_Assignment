import numpy as np
import parameters as par
import dynamics as dyn
import cost
import cvxpy as cvx

ns = par.ns
ni = par.ni

def mpc_solver(x_t, x_ref_horizon, u_ref_horizon, T_pred):
    """
    Linear MPC solver using CVXPY.
    """
    #Linearization around current reference point
    #Get A and B matrices from the reference trajectory
    _, fx, fu = dyn.dynamics_casadi(x_ref_horizon[:, 0], u_ref_horizon[:, 0])
    A = np.array(fx.T)
    B = np.array(fu.T)

    #Optimization variables
    x = cvx.Variable((ns, T_pred + 1))
    u = cvx.Variable((ni, T_pred))

    J = 0
    constraints = [x[:, 0] == x_t] #initial conditio

    for kk in range(T_pred):
        #Error states for tracking
        dx = x[:, kk] - x_ref_horizon[:, kk]
        du = u[:, kk] - u_ref_horizon[:, kk]
        
        #Stage cost
        J += cvx.quad_form(dx, cost.Q) + cvx.quad_form(du, cost.R)
        
        #Linear dynamics Constraint
        constraints += [x[:, kk + 1] == A @ x[:, kk] + B @ u[:, kk]]

        #Input cnstraints
        constraints += [u[:, kk] <= par.umax, u[:, kk] >= par.umin]

    #Terminal cost
    dx_T = x[:, T_pred] - x_ref_horizon[:, T_pred]
    J += cvx.quad_form(dx_T, cost.QT)

    #Solve the QP problem
    prob = cvx.Problem(cvx.Minimize(J), constraints)
    prob.solve(solver=cvx.OSQP, warm_start=True)

    if prob.status not in [cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE]:
        return u_ref_horizon[:, 0]

    #apply only the first control input
    return np.array(u[:, 0].value).flatten()

def simulate_mpc(xx_init, xx_ref, uu_ref, T_pred, T_extra=0):
    """
    MPC simulation loop
    """
    TT_nominal = xx_ref.shape[1]
    TT_sim = TT_nominal + T_extra #Total simulation time
    
    xx_mpc = np.zeros((ns, TT_sim))
    uu_mpc = np.zeros((ni, TT_sim))
    xx_mpc[:, 0] = xx_init

    for kk in range(TT_sim - 1):
        #When approaching the end of T_nominal, extend with last equilibrium state
        if kk + T_pred < TT_nominal:
            #Full available horizon
            xx_ref_horizon = xx_ref[:, kk : kk + T_pred + 1]
            uu_ref_horizon = uu_ref[:, kk : kk + T_pred]
        else:
            #Steady-State management
            kk_nominal = max(0, TT_nominal - kk) #steps of optimal trajectory still available
            kk_steady = (T_pred + 1) - kk_nominal #steps to be filled eith terminal state to complete T_pred
            
            #concatenate available trajectory with final equilibrium state
            xx_ref_horizon = np.hstack([xx_ref[:, kk:], np.tile(xx_ref[:, -1:], (1, kk_steady))])
            
            #ensure control input matches terminal equilibrium input
            uu_nominal_steps = max(0, TT_nominal - kk - 1)
            uu_steady_steps = T_pred - uu_nominal_steps
            uu_ref_horizon = np.hstack([uu_ref[:, kk:], np.tile(uu_ref[:, -1:], (1, uu_steady_steps))])

        #Solve and get the first input
        uu_mpc[:, kk] = mpc_solver(xx_mpc[:, kk], xx_ref_horizon, uu_ref_horizon, T_pred)
        
        #Apply dynamics
        xx_mpc[:, kk+1] = dyn.dynamics_casadi(xx_mpc[:, kk], uu_mpc[:, kk])[0].flatten()

    return xx_mpc, uu_mpc