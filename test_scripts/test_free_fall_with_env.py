# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 18:44:26 2025

@author: 33606
"""

import os 
import sys
import cv2
import numpy as np
from my_envs.invertedPendulumRender import InvertedPendulumRenderer

sys.path.append('C:/Users/33606/OneDrive/Documents/Polytechnique/Job/Projets/Inverted-Pendulum')

from my_envs.invertedPendulum import InvertedPendulum
from my_envs.invertedPendulumEnv import InvertedPendulumEnv

env = InvertedPendulumEnv()
env.action_space

syst = InvertedPendulumRenderer(env)

for i in range(700):
    rendered = syst.render([env.state[0], env.state[1], env.state[2], env.state[3]], i)
    cv2.imshow('im', rendered)
    cv2.moveWindow('im', 400, 400)
    env.step(0.01)
    #print(env.state[0])

    if cv2.waitKey(0) == ord('q'):
        break

cv2.destroyAllWindows()
