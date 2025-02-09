# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 19:37:28 2025

@author: 33606
"""

import cv2
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from my_envs.invertedPendulumEnv import InvertedPendulumEnv
from my_envs.invertedPendulumRender import InvertedPendulumRenderer
from dqn import DQNAgent

# Create environment
env = gym.make("InvertedPendulumEnv-v1", render_mode=None)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Initialize agent
agent = DQNAgent(state_size=state_size, action_size=action_size)
agent.epsilon = 0 #No exploration
agent.q_network = torch.load('DQN_agent.pt')

def control_agent(agent=agent):
    syst = InvertedPendulumRenderer(env)
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    i = 0
    
    while not done:
        action = agent.act(state)  # Get action index
        next_state, reward, terminated, _, _ = env.step(action)
    
        state = next_state
        total_reward += reward
        done = terminated
        
        rendered = syst.render([env.state[0], env.state[1], env.state[2], env.state[3]], i)
        cv2.imshow('Cart-Pendulum RL control', rendered)
        cv2.moveWindow('Cart-Pendulum RL control', 400, 400)
        i += 1

        if cv2.waitKey(0) == ord('q') or i == 200:
            break
        
control_agent(agent)