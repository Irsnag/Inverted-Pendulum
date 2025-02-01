# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 22:04:28 2025

@author: 33606
"""

import numpy as np
import cv2
from my_envs.invertedPendulum import InvertedPendulum
from scipy.integrate import solve_ivp

# Pendulum. Cart is fixed and cannot move.
# Y : [ theta, theta_dot ]
# returns expression for Y_dot.
def func(t, y):
    g = 9.8 # Gravitational Acceleration
    L = 1.5 # Length of pendulum
    friction =  -0.5*y[1] 
    return [y[1], -g/L * np.cos( y[0] ) + friction]


# Only the pendulum moves, the cart is stationary
if __name__=="__main__":
    # Solve ODE: theta_dot_dot = -g / L * cos( theta ) + delta * theta_dot
    t_0 = 0
    t_f = 20
    y_0 = [np.pi/2 + 0.1, 0]
    
    sol = solve_ivp(func, [t_0, t_f], y_0, t_eval=np.linspace(t_0, t_f, 300))


    syst = InvertedPendulum()

    for i, t in enumerate(sol.t):
        rendered = syst.step([0, 0, sol.y[0,i], sol.y[1,i]], t)
        cv2.imshow('im', rendered)

        if cv2.waitKey(0) == ord('q'):
            break
        
        