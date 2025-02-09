# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 18:10:11 2025

@author: 33606
"""

import numpy as np
import cv2
import sys
import scipy.linalg

sys.path.append('../.')

from my_envs.invertedPendulumRender import InvertedPendulumRenderer
from scipy.integrate import solve_ivp


class MyLinearizedSystem():
    def __init__(self):
        g = 9.8
        L = 1.5
        m = 1.0
        M = 5.0
        d1 = 5.0
        d2 = 2.5

        _q = (m+M) * g / (M*L)
        
        self.A = np.array([\
                    [0, 1, 0, 0], \
                    [0, -d1, g*m/M, 0],\
                    [0, 0, 0, 1],\
                    [0, -d1/(M*L), _q, -d2*_q]])

        self.B = np.expand_dims(np.array([0, 1.0/M, 0, 1/(M*L)] ), 1)
        
        self.Q = np.array([\
                    [1, 0, 0, 0], \
                    [0, 1, 0, 0],\
                    [0, 0, 20, 0],\
                    [0, 0, 0, 100]])
            
        self.R = np.eye(1)

    def compute_K(self):
        P = scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
        self.K = np.linalg.inv(self.R) @ self.B.T @ P

    def get_K(self):
        """Returns K if computed, otherwise computes it first."""
        if self.K is None:
            self.compute_K()
        return self.K
    
# Instantiate the LQR controller
controller = MyLinearizedSystem()
controller.compute_K()

A = controller.A
B = controller.B
Q = controller.Q
R = controller.R
K = controller.get_K()


def cart_pendulum_dynamics(t, state):
    """
    Computes the derivatives for the cart-pendulum system with LQR control.
    
    State vector:
    state[0] = x (cart position)
    state[1] = x_dot (cart velocity)
    state[2] = theta (pendulum angle, theta = 0 is upright)
    state[3] = theta_dot (angular velocity)
    """
    
    x, x_dot, theta, theta_dot = state
    g = 9.8  # Gravitational acceleration
    L = 1.5  # Pendulum length
    m = 1.0  # Mass of pendulum bob (kg)
    M = 5.0  # Mass of cart (kg)
    
    state_vector = np.array([[x], [x_dot], [theta], [theta_dot]])
    
    u = -K @ state_vector  # LQR feedback control law

    next_state = A @ state_vector + B @ u     
    
    return [next_state[0].item(), next_state[1].item(), next_state[2].item(), next_state[3].item()]


if __name__ == "__main__":
    # Simulation parameters
    start_time = 0
    end_time = 20
    initial_state = [0, 0, 0.15, 0]  # Slightly tilted from upright

    # Solve the system of ODEs with LQR control
    solution = solve_ivp(cart_pendulum_dynamics, [start_time, end_time], initial_state, 
                         t_eval=np.linspace(start_time, end_time, 300))

    # Initialize the renderer
    renderer = InvertedPendulumRenderer(env=None)

    # Animate the cart-pendulum motion
    for i, t in enumerate(solution.t):
        frame = renderer.render([solution.y[0, i], solution.y[1, i], solution.y[2, i], 
                                 solution.y[3, i], 0, 0], t)
        cv2.imshow('Cart-Pendulum LQR control', frame)

        if cv2.waitKey(0) == ord('q'):
            break

    cv2.destroyAllWindows()
    
    


