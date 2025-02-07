# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 19:19:09 2025

@author: 33606
"""

import cv2
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

num_episodes = 800
num_steps = 200


def train_agent(num_episodes=num_episodes, num_steps=num_steps):
    
    rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        for t in range(num_steps):
            action = agent.act(state)
            next_state, reward, terminated, _, _ = env.step(action)

            agent.replay_buffer.add((state, action, reward, next_state, terminated))
            state = next_state
            total_reward += reward

            agent.train()
            if terminated:
                break

        if episode % 10 == 0:
            agent.update_target_network()

        print(f"Episode {episode}: Total Reward = {total_reward}")
        rewards.append(total_reward)
        
    plt.plot(rewards)
    env.close()
    
    
def test_agent(agent):
    syst = InvertedPendulumRenderer(env)
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    i = 0
    
    while not done or i==200:
        action = agent.act(state)  # Get action index
        next_state, reward, terminated, _, _ = env.step(action)
    
        state = next_state
        total_reward += reward
        done = terminated
        
        rendered = syst.render([env.state[0], env.state[1], env.state[2], env.state[3]], i)
        cv2.imshow('im', rendered)
        cv2.moveWindow('im', 400, 400)
        i += 1

        if cv2.waitKey(0) == ord('q'):
            break
    

