import numpy as np
import dynamics as dyn
import parameters as par

if(not par.ARE_with_scipy):
    ##################
    # ARE without scipy 
    ####################
    def compute_PT_ARE(xx_ref, uu_ref):
        """
        Computes the terminal cost-to-go matrix PT using the ARE for infinite time horizon
        Linearizing around the final equilibrium point. 
        """
        _, fx_T, fu_T = dyn.dynamics_casadi(xx_ref[:, -1], uu_ref[:, -1])
        A_inf = fx_T.T
        B_inf = fu_T.T
        Q = par.Q 
        R = par.R 

        print(f"\nSolving Infinite Horizon Riccati at final equilibrium:")
        print(f"  theta1 = {np.degrees(xx_ref[0, -1]):.2f}°, theta2 = {np.degrees(xx_ref[1, -1]):.2f}°")

        try: 
            PT = solve_ARE(A_inf, B_inf, Q, R)

            eigenvalues = np.linalg.eigvals(PT) 
            min_eig = np.min(eigenvalues)
            max_eig = np.max(eigenvalues)

            print(f"  PT eigenvalues: min = {min_eig:.4f}, max = {max_eig:.4f}")

            if min_eig < -1e-6:  
                print(f" Warning: Negative eigenvalue detected, solution may be invalid")
                print(f" Back to QT from parameters")
                return par.QT 
            
            return PT 
            
        except Exception as e: 
            print(f" Error: Riccati solver failed ({e})")
            print(f" Back to QT from parameters")
            return par.QT

    def solve_ARE(A, B, Q, R, max_iters=5000, tol=1e-8):
        """
        Solves the Algebraic Riccati Equation by iterating the Difference Riccati Equation
        until convergence (infinite horizon).
        """
        n = A.shape[0]
        P_next = Q.copy()

        for k in range(max_iters):
            BT_P_B = B.T @ P_next @ B 
            BT_P_A = B.T @ P_next @ A 
            AT_P_A = A.T @ P_next @ A 
            AT_P_B = A.T @ P_next @ B

            # Compute (R + B^T * P * B)^(-1)
            try: 
                R_BPB_inv = np.linalg.inv(R + BT_P_B)
            except np.linalg.LinAlgError: 
                print(f"  Warning: Singular matrix at iteration {k}")
                R_BPB_inv = np.linalg.pinv(R + BT_P_B)

            # Riccati update
            P_current = Q + AT_P_A - AT_P_B @ R_BPB_inv @ BT_P_A

            # Check convergence 
            diff = np.linalg.norm(P_current - P_next, 'fro')

            if diff < tol: 
                print(f"  ARE converged in {k+1} iterations (diff: {diff:.2e})")
                return P_current
            
            P_next = P_current
        
        print(f"  Warning: ARE did not converge in {max_iters} iterations (diff: {diff:.2e})")
        return P_current
    ###############################
    ###############################
else: 
    ###############################
    # ARE with scipy
    from scipy.linalg import solve_discrete_are
    ###############################

    def compute_PT_ARE(xx_ref, uu_ref):
        """
        Computes the terminal cost-to-go matrix PT using the ARE for infinite time horizon
        Linearizing around the final equilibrium point. 
        """
        _, fx_T, fu_T = dyn.dynamics_casadi(xx_ref[:, -1], uu_ref[:, -1])
        A_inf = fx_T.T
        B_inf = fu_T.T

        Q = par.Q 
        R = par.R 

        print(f"\nSolving Infinite Horizon Riccati at final equilibrium:")
        print(f"  theta1 = {np.degrees(xx_ref[0, -1]):.2f}°, theta2 = {np.degrees(xx_ref[1, -1]):.2f}°")

        try: 
            PT = solve_ARE(A_inf, B_inf, Q, R)

            eigenvalues = np.linalg.eigvals(PT) 
            min_eig = np.min(eigenvalues)
            max_eig = np.max(eigenvalues)

            print(f"  PT eigenvalues: min = {min_eig:.4f}, max = {max_eig:.4f}")

            if min_eig < -1e-6:  
                print(f" Warning: Negative eigenvalue detected, solution may be invalid")
                print(f" Back to QT from parameters")
                return par.QT 
            
            residual = PT - (Q + A_inf.T @ PT @ A_inf -
                            A_inf.T @ PT @ B_inf @ np.linalg.inv(R + B_inf.T @ PT @ B_inf) @
                            B_inf.T @ PT @ A_inf)
            residual_norm = np.linalg.norm(residual,'fro')
            if residual_norm>1e-3: print(f"Warning Large residual ({residual_norm:.2e})")
        
            return PT 
            
        except Exception as e: 
            print(f" Error: Riccati solver failed ({e})")
            print(f" Back to QT from parameters")
            return par.QT
        
    def solve_ARE(A, B, Q, R):
        """
        Solves the Algebraic Riccati Equation by iterating the Difference Riccati Equation
        until convergence (infinite horizon).
        """
        return solve_discrete_are(A,B,Q,R)
    ###############################
    ###############################