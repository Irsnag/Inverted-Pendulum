# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 00:15:56 2025

@author: 33606
"""

import numpy as np
import cv2

from my_envs.invertedPendulum import InvertedPendulum

from scipy.integrate import solve_ivp

# Pendulum and Cart system.
# Y : [ x, x_dot, theta, theta_dot]
# returns expression for Y_dot.
def func2( t, y ):
    g = 9.8 # Gravitational Acceleration
    L = 1.5 # Length of pendulum
    
    friction_theta = 0
    friction_x =  - 1.0*y[1]

    m = 1.0 #mass of bob (kg)
    M = 5.0  # mass of cart (kg)
    x_ddot = -L * y[3]*y[3] * np.cos( y[2] )  +  g * np.cos(y[2]) *  np.sin(y[2])
    x_ddot = (m + friction_x) / ( m* np.sin(y[2])* np.sin(y[2]) - M -m ) * x_ddot 

    theta_ddot = - g/L * np.cos( y[2] ) + 1./L * np.sin( y[2] ) * x_ddot

    friction_theta =  - 0.5*y[3]
    friction_x =  - 1.0*y[1]
    
    x_ddot = x_ddot + friction_x
    theta_ddot = theta_ddot + friction_theta

    return [y[1], x_ddot, y[3], theta_ddot]


# Both cart and the pendulum can move.
if __name__=="__main__":
    sol = solve_ivp(func2, [0, 20], [ 0, 0, np.pi/2 - 0.2, 0. ],   t_eval=np.linspace( 0, 20, 300)  )

    syst = InvertedPendulum()

    for i, t in enumerate(sol.t):
        rendered = syst.step( [sol.y[0,i], sol.y[1,i], sol.y[2,i], sol.y[3,i] ], t )
        cv2.imshow( 'im', rendered )

        if cv2.waitKey(0) == ord('q'):
            break