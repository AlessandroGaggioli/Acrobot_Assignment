import numpy as np
import dynamics as dyn
import Plotter

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
        
    #Plot Dynamics 
    Plotter.plot_dynamics(TT,xx_test)