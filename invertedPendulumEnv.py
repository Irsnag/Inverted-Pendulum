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
        self.state = None

        # For rendering
        self.screen = None
        self.clock = None