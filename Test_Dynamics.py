import numpy as np
import dynamics as dyn
import matplotlib.pyplot as plt

def test_dynamics(TT,ns,ni,dt):
    #Initialization of state and input arrays
    xx_test = np.zeros((ns, TT))
    uu_test = np.zeros((ni, TT)) #u = 0 to test dynamics +

    #Set initial condition
    xx0 = np.array([np.radians(10), np.radians(10), 0, 0]) 
    xx_test[:, 0] = xx0

    #Simulation Loop
    for kk in range(TT - 1):
        #apply dynamiccs
        xx_test[:, kk+1] = dyn.dynamics_casadi(xx_test[:, kk], uu_test[:, kk])[0]
        
    #Plot results
    time_axis = np.arange(TT) * dt

    plt.figure(figsize=(10, 8))

    # Subplot 1 (Theta 1)
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, np.degrees(xx_test[0, :]), color='tab:blue', linewidth=1.5, label=r"$\theta_1$")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [deg]")
    plt.title("Test dynamics, zero input")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    #Subplot 2 (Theta 2)
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, np.degrees(xx_test[1, :]), color='tab:orange', linewidth=1.5, label=r"$\theta_2$")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [deg]")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.show()