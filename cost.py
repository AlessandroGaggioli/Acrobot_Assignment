import numpy as np
import parameters as par

Q = par.Q 
R = par.R 
QT = par.QT 

def cost_fcn(xx, uu, xx_ref, uu_ref):
    '''Compute total cost J of a given trajectory'''

    TT = xx.shape[1]
    cost = 0

    #stage cost from t=0 to t=T-1
    for kk in range(TT-1):
        dx = xx[:,kk]-xx_ref[:,kk]
        du = uu[:,kk]-uu_ref[:,kk]
        cost += 0.5*(dx.T@Q@dx + du.T@R@du) #quadratic form

    #terminal cost at final time t=T
    dx_T = xx[:, -1] - xx_ref[:,-1]
    cost += 0.5*(dx_T.T@QT@dx_T)

    return cost

def stage_grad(xx, xx_ref, uu, uu_ref):
    '''Gradient and Hessian of stage cost'''
    dx = (xx-xx_ref).reshape(-1,1)
    du = (uu-uu_ref).reshape(-1,1)
    return Q@dx, R@du, Q, R

def terminal_grad(xx, xx_ref):
    '''Gradient and Hessian of terminal cost'''
    dx = (xx - xx_ref).reshape(-1, 1)
    return QT @ dx, QT