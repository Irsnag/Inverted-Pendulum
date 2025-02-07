# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 22:04:28 2025

@author: 33606
"""

import numpy as np
import cv2
from my_envs.invertedPendulumRender import InvertedPendulumRenderer
from scipy.integrate import solve_ivp

def pendulum_dynamics(t, state):
    """Computes the derivatives for the pendulum system."""
    theta, theta_dot = state
    g = 9.8  # Gravitational acceleration
    L = 1.5  # Pendulum length
    friction = -0.5 * theta_dot  # Damping term
    theta_ddot = (g / L) * np.sin(theta) + friction  # Angular acceleration
    return [theta_dot, theta_ddot]

if __name__ == "__main__":
    # Simulation parameters
    start_time = 0
    end_time = 20
    
    # Initial conditions:
    # theta = 0 means the pendulum is upright.
    # Here, we start slightly tilted from the upright position.
    initial_state = [0.1, 0]  

    # Solve the system of ODEs
    solution = solve_ivp(pendulum_dynamics, [start_time, end_time], initial_state, 
                         t_eval=np.linspace(start_time, end_time, 300))

    # Initialize the renderer
    renderer = InvertedPendulumRenderer(env=None)

    # Animate the pendulum motion
    for i, t in enumerate(solution.t):
        frame = renderer.render([0, 0, solution.y[0, i], solution.y[1, i], 0, 0], t)
        cv2.imshow('Inverted Pendulum', frame)

        if cv2.waitKey(0) == ord('q'):
            break
