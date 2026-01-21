import numpy as np
import parameters as par
import dynamics as dyn
import casadi as ca

ns = par.ns
ni = par.ni

def mpc_solver(x_t,x_ref_horizon,u_ref_horizon,T_pred,QT_mpc=None):

    #cost parameters
    Q = ca.DM(par.Q)
    R = ca.DM(par.R)
    QT = ca.DM(QT_mpc if QT_mpc is not None else par.QT)

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

def simulate_mpc(xx_init,xx_ref,uu_ref,T_sim,T_pred):
    
    xx_mpc = np.zeros((ns,T_sim+1))
    uu_mpc = np.zeros((ni,T_sim))

    xx_mpc[:,0] = xx_init

    print(f"Starting MPC Simulation (Horizon={T_pred}...)")

    if par.use_ARE: 
        import ARE
        QT_mpc = ARE.compute_PT_ARE(xx_ref,uu_ref)
    else: 
        QT_mpc = par.QT 

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

        else: #pad with the last reference value 
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
        u_opt,_,_ = mpc_solver(x_t,x_ref_horizon,u_ref_horizon,T_pred,QT_mpc=QT_mpc)

        #Save and apply input 
        uu_mpc[:,tt] = u_opt

        xx_next,_,_ = dyn.dynamics_casadi(x_t,u_opt)
        xx_mpc[:,tt+1] = xx_next

        if tt%50==0:
            print(f"Step{tt}/{T_sim}")
        
    return xx_mpc,uu_mpc
