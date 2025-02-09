# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 00:15:56 2025

@author: 33606
"""

import numpy as np
import cv2
from my_envs.invertedPendulumRender import InvertedPendulumRenderer
from scipy.integrate import solve_ivp

def cart_pendulum_dynamics(t, state):
    """
    Computes the derivatives for the cart-pendulum system.
    
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

    # Friction terms
    friction_theta = 0.5 * theta_dot
    friction_x = 1.0 * x_dot

    # Equations of motion
    x_ddot = (friction_x - m * L * (theta_dot ** 2) * np.sin(theta) + m * g * np.cos(theta) * np.sin(theta)) / (m * np.cos(theta)**2 - (M + m))
    theta_ddot = g / L * np.sin(theta) + (1 / L) * np.cos(theta) * x_ddot

    # Apply friction
    theta_ddot += -friction_theta

    return [x_dot, x_ddot, theta_dot, theta_ddot]

if __name__ == "__main__":
    # Simulation parameters
    start_time = 0
    end_time = 20
    initial_state = [0, 0, -0.2, 0]  # Slightly tilted from upright

    # Solve the system of ODEs
    solution = solve_ivp(cart_pendulum_dynamics, [start_time, end_time], initial_state, 
                         t_eval=np.linspace(start_time, end_time, 300))

    # Initialize the renderer
    renderer = InvertedPendulumRenderer(env=None)

    # Animate the cart-pendulum motion
    for i, t in enumerate(solution.t):
        frame = renderer.render([solution.y[0, i], solution.y[1, i], solution.y[2, i], 
                                 solution.y[3, i], 0, 0], t)
        cv2.imshow('Cart-Pendulum System', frame)

        if cv2.waitKey(0) == ord('q'):
            break
