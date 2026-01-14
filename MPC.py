import numpy as np
import parameters as par
import dynamics as dyn
import cost
import cvxpy as cvx
import casadi as ca

ns = par.ns
ni = par.ni

def mpc_solver(x_t,x_ref_horizon,u_ref_horizon,T_pred):

    #Linearization around current reference point
    #Get A and B matrices from the reference trajectory
    # _, fx, fu = dyn.dynamics_casadi(x_ref_horizon[:, 0], u_ref_horizon[:, 0])
    # A = ca.DM(fx.T)
    # B = ca.DM(fu.T)

    Q = ca.DM(par.Q)
    R = ca.DM(par.R)
    QT = ca.DM(par.QT)
    x_t = np.array(x_t).squeeze()

    opti = ca.Opti()
    X = opti.variable(ns,T_pred)
    U = opti.variable(ni,T_pred)

    cost = 0

    for tt in range(T_pred-1):
        xt = X[:,tt]
        ut = U[:,tt]

        x_ref = ca.DM(x_ref_horizon[:, tt])
        u_ref = ca.DM(u_ref_horizon[:, tt])

        _, fx, fu = dyn.dynamics_casadi(x_ref_horizon[:, tt], u_ref_horizon[:, tt])
        A = ca.DM(fx.T)
        B = ca.DM(fu.T)

        x_err = xt - x_ref
        u_err = ut - u_ref

        cost += ca.mtimes([x_err.T, Q, x_err]) + ca.mtimes([u_err.T, R, u_err])

        # x_ref_next = ca.DM(x_ref_horizon[:, tt + 1])
        # opti.subject_to(X[:, tt + 1] == A @ xt + B @ ut + (x_ref_next - A @ x_ref - B @ u_ref))

        x_next_nominal, _, _ = dyn.dynamics_casadi(x_ref, u_ref)
        opti.subject_to(X[:, tt + 1] == A @ (xt - x_ref) + B @ (ut - u_ref) + ca.DM(x_next_nominal))

    #terminal cost
    x_err_final = X[:, T_pred - 1] - ca.DM(x_ref_horizon[:, T_pred - 1])
    cost += ca.mtimes([x_err_final.T, QT, x_err_final])

    #initial condition
    opti.subject_to(X[:,0]==ca.DM(x_t))

    opti.minimize(cost)

    #solver configuration 
    ipopt_opts = {
        "ipopt.print_level": 0, 
        "print_time": 0, 
        "ipopt.max_iter": 1000,
        "ipopt.tol": 1e-6
    }
    opti.solver("ipopt", ipopt_opts)

    try: 
        sol = opti.solve()
    except RuntimeError: 
        opti.set_initial(X,x_ref_horizon)
        opti.set_initial(U,u_ref_horizon)
        try: 
            sol = opti.solve()
        except Exception as e: 
            print("mpc_solver: solver failed",e)
            return None,None,None

    u_t = np.asarray(sol.value(U[:, 0]))
    x_pred = np.asarray(sol.value(X))
    u_pred = np.asarray(sol.value(U))

    return u_t, x_pred, u_pred

def simulate_mpc(xx_init,xx_ref,uu_ref,T_sim,T_pred,verbose=False,save_predictions=False):
    
    xx_mpc = np.zeros((ns,T_sim+1))
    uu_mpc = np.zeros((ni,T_sim))

    predictions = [] if save_predictions else None 

    xx_mpc[:,0] = xx_init

    solve_failures = 0
    solve_times = []

    for tt in range(T_sim):
        if verbose and tt%10==0:
            print(f"Step {tt}/{T_sim}")

            x_t = xx_mpc[:,tt]

            hor_len = min(T_pred,T_sim-tt)

            if tt+hor_len <= T_sim: 
                x_ref_horizon = xx_ref[:,tt:tt+hor_len]
                u_ref_horizon = uu_ref[:,tt:tt+hor_len]
            else: 
                remaining = T_sim -tt
                x_ref_horizon = np.hstack([
                    xx_ref[:,tt:],
                    np.tile(xx_ref[:,-1:],(1,hor_len-remaining))
                ])
                uu_ref = np.hstack([
                    uu_ref[:,tt:],
                    np.tile(uu_ref[:,-1:],(1,hor_len-remaining))
                ])

                u_opt,x_pred,u_pred = mpc_solver(x_t,x_ref_horizon,u_ref_horizon,hor_len)

                if u_opt is None: 
                    solve_failures += 1 
                    if verbose: 
                        print(f"Warning: MPC solver failed at tt={tt}")
                    u_opt = uu_ref[:,tt]
                    x_pred = None 
                    u_pred = None

                if save_predictions: 
                    predictions.append((x_pred,u_pred))

                uu_mpc[:,tt] = u_opt.flatten()

                xx_next,_,_ = dyn.dynamics_casadi(x_t,u_opt)
                xx_mpc[:,tt+1] = xx_next 
    if verbose: 
        print(f"\nSimulation completed\n")
        print(f"Total steps: {T_sim}")
        print(f"Solver failures: {solve_failures}")

    if save_predictions: 
        return xx_mpc,uu_mpc,predictions
    else: 
        return xx_mpc,uu_mpc


# def mpc_solver(x_t, x_ref_horizon, u_ref_horizon, T_pred):
#     """
#     Linear MPC solver using CVXPY.
#     """
#     #Linearization around current reference point
#     #Get A and B matrices from the reference trajectory
#     _, fx, fu = dyn.dynamics_casadi(x_ref_horizon[:, 0], u_ref_horizon[:, 0])
#     A = np.array(fx.T)
#     B = np.array(fu.T)

#     #Optimization variables
#     x = cvx.Variable((ns, T_pred + 1))
#     u = cvx.Variable((ni, T_pred))

#     J = 0
#     constraints = [x[:, 0] == x_t] #initial conditio

#     for kk in range(T_pred):
#         #Error states for tracking
#         dx = x[:, kk] - x_ref_horizon[:, kk]
#         du = u[:, kk] - u_ref_horizon[:, kk]
        
#         #Stage cost
#         J += cvx.quad_form(dx, cost.Q) + cvx.quad_form(du, cost.R)
        
#         #Linear dynamics Constraint
#         constraints += [x[:, kk + 1] == A @ x[:, kk] + B @ u[:, kk]]

#         #Input cnstraints
#         constraints += [u[:, kk] <= par.umax, u[:, kk] >= par.umin]

#     #Terminal cost
#     dx_T = x[:, T_pred] - x_ref_horizon[:, T_pred]
#     J += cvx.quad_form(dx_T, cost.QT)

#     #Solve the QP problem
#     prob = cvx.Problem(cvx.Minimize(J), constraints)
#     prob.solve(solver=cvx.OSQP, warm_start=True)

#     if prob.status not in [cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE]:
#         return u_ref_horizon[:, 0]

#     #apply only the first control input
#     return np.array(u[:, 0].value).flatten()

# def simulate_mpc(xx_init, xx_ref, uu_ref, T_pred, T_extra=0):
#     """
#     MPC simulation loop
#     """
#     TT_nominal = xx_ref.shape[1]
#     TT_sim = TT_nominal + T_extra #Total simulation time
    
#     xx_mpc = np.zeros((ns, TT_sim))
#     uu_mpc = np.zeros((ni, TT_sim))
#     xx_mpc[:, 0] = xx_init.flatten()

#     for kk in range(TT_sim - 1):
#         #When approaching the end of T_nominal, extend with last equilibrium state
#         if kk + T_pred < TT_nominal:
#             #Full available horizon
#             xx_ref_horizon = xx_ref[:, kk : kk + T_pred + 1]
#             uu_ref_horizon = uu_ref[:, kk : kk + T_pred]
#         else:
#             #Steady-State management
#             kk_nominal = max(0, TT_nominal - kk) #steps of optimal trajectory still available
#             kk_steady = (T_pred + 1) - kk_nominal #steps to be filled eith terminal state to complete T_pred
            
#             #concatenate available trajectory with final equilibrium state
#             xx_ref_horizon = np.hstack([xx_ref[:, kk:], np.tile(xx_ref[:, -1:], (1, kk_steady))])
            
#             #ensure control input matches terminal equilibrium input
#             uu_nominal_steps = max(0, TT_nominal - kk - 1)
#             uu_steady_steps = T_pred - uu_nominal_steps
#             uu_ref_horizon = np.hstack([uu_ref[:, kk:], np.tile(uu_ref[:, -1:], (1, uu_steady_steps))])

#         #Solve and get the first input
#         u_opt= mpc_solver(xx_mpc[:, kk], xx_ref_horizon, uu_ref_horizon, T_pred)
        
#         if u_opt is None:
#             print(f"MPC solver failed at time step {kk}")
#             break 
    
#         uu_mpc[:,kk] = u_opt 

#         #Apply dynamics
#         xx_mpc[:, kk+1] = dyn.dynamics_casadi(xx_mpc[:, kk], uu_mpc[:, kk])[0].flatten()

#     return xx_mpc, uu_mpc