# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 20:47:43 2025

@author: 33606
"""

import os 
import sys

sys.path.append('C:/Users/33606/OneDrive/Documents/Polytechnique/Job/Projets/Inverted-Pendulum')

import gymnasium as gym
from my_envs.invertedPendulumEnv import InvertedPendulumEnv
import numpy as np 
from LQR import LQR

Q = np.eye(5)
Q[0, 0] = 10
Q[1, 1] = 15
Q[2, 2] = 30
Q[3, 3] = 6
Q[4, 4] = 1

x_n = 4
u_n = 1

G = np.loadtxt('../G.txt')
H = np.loadtxt('../H.txt').reshape(4, 1)
F_t = np.concatenate([G, H], axis=1)

f_t = np.zeros((x_n, 1))
C_t = Q
c_t = np.zeros((x_n+u_n, 1))

if __name__=="__main__":
    env = InvertedPendulumEnv()
    obs = env.reset()

    lqr = LQR(x_n, u_n, F_t, f_t, C_t, c_t)
    x_0 = obs

    T = 100
    t = 15
    i = 0
    frames = [] # save frames
    try:
        while True:
            # env.render(mode="rgb_array")
            u_seq = lqr(x_0, T)
            for j in range(t):
                if i%4 == 0:
                    frames.append(env.render(mode="rgb_array"))
                obs, _, _, _ = env.step(np.clip(u_seq[j][0], -20, 20))
                print("Step[{}]| Obs:{}| Control:{}".format(i, obs, u_seq[j][0]))
                x_0 = obs
                i += 1
    # Use Ctrl + c to stop
    except KeyboardInterrupt:
        print("stop")
    # display_frames_as_gif(frames) # save frames as gif in './figures/result.gif'