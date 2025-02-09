# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 20:39:34 2025

@author: 33606
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class InvertedPendulumEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super(InvertedPendulumEnv, self).__init__()
        
        # Physical constants
        self.gravity = 9.81
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.total_mass = self.mass_cart + self.mass_pole
        self.length = 0.5  # Half the pole length
        self.pole_mass_length = self.mass_pole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # Time step (seconds)
        self.damping_x = 0.99
        self.damping_theta = 0.99

        # Angle at which to fail the episode
        self.theta_threshold_radians = 20 * np.pi / 180  # Angle threshold
        self.x_threshold = 4.4  # Cart position threshold

        # Define action and observation spaces
        #self.action_space = spaces.Discrete(3)  
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=np.array([-self.x_threshold, -np.inf, -self.theta_threshold_radians, -np.inf]), 
            high=np.array([self.x_threshold, np.inf, self.theta_threshold_radians, np.inf]),  
            dtype=np.float32,
        )

        # Rendering
        self.render_mode = render_mode
        self.state = [0, 0., 0, 0]

        # For rendering
        self.screen = None
        self.clock = None

    def reset(self, seed=None, options=None):
        # Reset the environment
        super().reset(seed=seed)
        theta = np.random.uniform(-self.theta_threshold_radians, self.theta_threshold_radians)
        theta_dot = np.random.uniform(-0.1, 0.1)  # Small random initial angular velocity
        x = np.random.uniform(-0.05, 0.05)  # Small random cart position
        x_dot = np.random.uniform(-0.05, 0.05)  # Small random cart velocity
    
        self.state = [x, x_dot, theta, theta_dot]
        return np.array(self.state, dtype=np.float32), {}

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        n = action
        force = self.force_mag * (n-3)

        # Equations of motion
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.pole_mass_length * theta_dot**2 * sintheta) / self.total_mass
        theta_acc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.mass_pole * costheta**2 / self.total_mass)
        )
        x_acc = temp - self.pole_mass_length * theta_acc * costheta / self.total_mass

        # Update state using Euler's method
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * x_acc
        x_dot = x_dot * self.damping_x
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * theta_acc
        theta_dot = theta_dot * self.damping_theta

        self.state = [x, x_dot, theta, theta_dot]

        # Check termination conditions
        terminated = (
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        truncated = False  # For fixed time-limit environments, set this to `True` when reaching the limit

        # Reward penalizes large angles
        reward = 1/(1+abs(theta)) if not terminated else 0.0

        return (
            self.state,
            reward,
            terminated,
            truncated,
            {},
        )

    def render(self):
        if self.render_mode == "human":
            print(
                f"Cart Position: {self.state[0]:.2f}, Pole Angle: {self.state[2]:.2f} rad"
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
            
env = gym.make("InvertedPendulumEnv-v1", render_mode="human")
