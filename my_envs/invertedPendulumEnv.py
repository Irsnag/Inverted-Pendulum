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
        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.total_mass = self.mass_cart + self.mass_pole
        self.length = 0.5  # Half the pole length
        self.pole_mass_length = self.mass_pole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # Time step (seconds)

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * np.pi / 180  # ~12 degrees
        self.x_threshold = 2.4  # Cart position threshold

        # Define action and observation spaces
        self.action_space = spaces.Discrete(2)  # 0 = push left, 1 = push right
        self.observation_space = spaces.Box(
            low=np.array(
                [-self.x_threshold, -np.inf, -self.theta_threshold_radians, -np.inf]
            ),
            high=np.array(
                [self.x_threshold, np.inf, self.theta_threshold_radians, np.inf]
            ),
            dtype=np.float32,
        )

        # Rendering
        self.render_mode = render_mode
        self.state = [0, 0., np.pi/2 + 0.1, 0]

        # For rendering
        self.screen = None
        self.clock = None

    def reset(self, seed=None, options=None):
        # Reset the environment
        super().reset(seed=seed)
        self.state = [0, 0., np.pi/2 + 0.1, 0]
        return np.array(self.state, dtype=np.float32), {}

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = action*self.force_mag if action != 1 else 0

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
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * theta_acc

        self.state = (x, x_dot, theta, theta_dot)

        # Check termination conditions
        terminated = (
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        truncated = False  # For fixed time-limit environments, set this to `True` when reaching the limit

        # Reward is +1 for every time step the pole is balanced
        reward = 1.0 if not terminated else 0.0

        return (
            np.array(self.state, dtype=np.float32),
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
