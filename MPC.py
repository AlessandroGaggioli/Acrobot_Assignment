import numpy as np
import parameters as par
import dynamics as dyn
import cost
import cvxpy as cvx
import casadi as ca
import matplotlib.pyplot as plt

ns = par.ns
ni = par.ni

def mpc_solver(x_t,x_ref_horizon,u_ref_horizon,T_pred):

    #cost parameters
    Q = ca.DM(par.Q)
    R = ca.DM(par.R)
    QT = ca.DM(par.QT)
    #x_t = np.array(x_t).squeeze()

    # opti object 
    opti = ca.Opti()

    #optimization variables 
    X = opti.variable(ns,T_pred+1)
    U = opti.variable(ni,T_pred)

    cost = 0

    #initial condition 
    opti.subject_to(X[:,0]==x_t)

    for tt in range(T_pred):

        # reference at iteration tt 
        x_ref = x_ref_horizon[:, tt]
        u_ref = u_ref_horizon[:, tt]
        x_ref_next = x_ref_horizon[:,tt+1]

        #Linearization around ref traj 
        #A,B matrices 
        _, fx, fu = dyn.dynamics_casadi(x_ref,u_ref)
        A = ca.DM(fx.T)
        B = ca.DM(fu.T)

        #affine term for linearization: d = x_ref - A*x_ref + B*u_ref)
        #This ensures that if we are exactly on the reference, the error is zero
        affine = x_ref_next - ca.mtimes(A,x_ref)-ca.mtimes(B,u_ref)

        #Stage cost 
        x_err = X[:,tt]- x_ref #error on x 
        u_err = U[:,tt] - u_ref #error on u 

        cost += ca.mtimes([x_err.T, Q, x_err]) + ca.mtimes([u_err.T, R, u_err])

        #dynamics constraints: x_{tt+1} ) A*x+B*u+d
        opti.subject_to(X[:, tt+1] == ca.mtimes(A, X[:, tt]) + ca.mtimes(B, U[:, tt]) + affine)

        #input constraints 
        opti.subject_to(opti.bounded(par.umin, U[:, tt], par.umax))

    #terminal cost
    x_err_final = X[:, T_pred] - x_ref_horizon[:, T_pred]
    cost += ca.mtimes([x_err_final.T, QT, x_err_final])

    #minimize the cost function 
    opti.minimize(cost)

    #solver configuration 
    ipopt_opts = {
        "ipopt.print_level": 0, 
        "print_time": 0, 
        "ipopt.max_iter": 100,
        "ipopt.sb":"yes"
    }
    opti.solver("ipopt", ipopt_opts)

    try: 
        sol = opti.solve()

        #return the first optimal input and predictions for plotting 
        u_opt=sol.value(U[:,0])
        x_pred = sol.value(X)
        return np.array(u_opt).flatten(),x_pred,None
    except RuntimeError: 
        return np.array(u_ref_horizon[:,0]).flatten(),None,None

def simulate_mpc(xx_init,xx_ref,uu_ref,T_sim,T_pred,verbose=False):
    
    xx_mpc = np.zeros((ns,T_sim+1))
    uu_mpc = np.zeros((ni,T_sim))

    xx_mpc[:,0] = xx_init

    print(f"Starting MPC Simulation (Horizon={T_pred}...)")

    for tt in range(T_sim):

        x_t = xx_mpc[:,tt]

        #It requires to supply always T_pred future steps
        #So if it's near the end, pad with the last reference state (equilibrium)

        idx_start = tt
        idx_end = tt + T_pred

        steps = xx_ref.shape[1] - idx_start

        if steps > T_pred: 
            #it has enough future reference 
            x_ref_horizon = xx_ref[:,idx_start:idx_end+1]
            u_ref_horizon = uu_ref[:,idx_start:idx_end]

        else: #pad with the last reference value --- da capire...
            x_chunk = xx_ref[:,idx_start:]
            u_chunk = uu_ref[:,idx_start:]

            steps_miss = (T_pred+1)-x_chunk.shape[1]

            x_pad = np.tile(xx_ref[:,-1:],(1,steps_miss))
            u_pad = np.tile(uu_ref[:,-1:],(1,steps_miss))
            u_miss = T_pred - u_chunk.shape[1]
            u_pad = np.tile(uu_ref[:, -1:], (1, u_miss))

            x_ref_horizon = np.hstack([x_chunk, x_pad])
            u_ref_horizon = np.hstack([u_chunk, u_pad])
        
        #Solve MPC
        u_opt,_,_ = mpc_solver(x_t,x_ref_horizon,u_ref_horizon,T_pred)

        #Save and apply input 
        uu_mpc[:,tt] = u_opt

        xx_next,_,_ = dyn.dynamics_casadi(x_t,u_opt)
        xx_mpc[:,tt+1] = xx_next

        if verbose and tt%50==0:
            print(f"Step{tt}/{T_sim}")
        
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

def plot_MPC(TT,xx_ref,xx_opt,xx_mpc,uu_opt,uu_mpc):
    #Plot for MPC
    t = np.arange(TT) * par.dt
    fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    plt.subplots_adjust(hspace=0.3)

    state_labels = [r'$\theta_1$ [rad]', r'$\theta_2$ [rad]', 
                    r'$\dot{\theta}_1$ [rad/s]', r'$\dot{\theta}_2$ [rad/s]']

    for i in range(xx_ref.shape[0]):
        axs[i].plot(t, np.degrees(xx_opt[i, :]), 'r--', label='Optimal traj')
        axs[i].plot(t, np.degrees(xx_mpc[i, :-1]), 'b', label='MPC tracking')
        axs[i].set_ylabel(f"{state_labels[i]}")
        axs[i].grid(True)
        axs[i].legend(loc='best')

    axs[4].plot(t[:-1], uu_opt[0, :-1], 'g--', label='Optimal input')
    axs[4].plot(t[:-1], uu_mpc[0, :-1], 'b', label='MPC input')
    axs[4].axhline(y=par.umax, color='k', linestyle='--', label='Input constraints')
    axs[4].axhline(y=par.umin, color='k', linestyle='--')
    axs[4].set_ylabel(r'$\tau$ [Nm]')
    axs[4].set_xlabel("Time [s]")
    axs[4].grid(True)
    axs[4].legend(loc='best')

    plt.suptitle(f"MPC Results (N_pred = {par.T_pred})", fontsize=14)
    plt.tight_layout(rect = [0,0,1,0.97])
    plt.show()