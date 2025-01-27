# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 18:44:26 2025

@author: 33606
"""

import os 
import sys
import cv2

sys.path.append('../.')

from my_envs.invertedPendulum import InvertedPendulum
from my_envs.invertedPendulumEnv import InvertedPendulumEnv

env = InvertedPendulumEnv()
env.action_space

syst = InvertedPendulum()

for i in range(500):
    rendered = syst.step(env.state, i)
    cv2.imshow( 'im', rendered )
    cv2.moveWindow( 'im', 400, 400 )
    env.step(+0.01)

    if cv2.waitKey(0) == ord('q'):
        break