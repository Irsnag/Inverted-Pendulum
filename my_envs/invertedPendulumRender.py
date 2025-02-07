# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 02:50:50 2025

@author: 33606
"""

import numpy as np
import cv2

class InvertedPendulumRenderer:
    def __init__(self, env):
        self.env = env
        self.length = 100  # Scale length of the pendulum for visualization
        self.image_size = (512, 512)

    def render(self, state, t=None):
        """
        state vector:
            state[0] : cart position (x)
            state[1] : cart velocity (x_dot)
            state[2] : pole angle (theta) from vertical (radians)
            state[3] : pole angular velocity (theta_dot)
        """
        CART_POS = state[0]
        BOB_ANG = state[2]  # Already in radians
        BOB_ANG_DEG = np.degrees(BOB_ANG)  # Convert to degrees

        IM = np.zeros((512, 512, 3), dtype='uint8')

        # Ground line
        cv2.line(IM, (0, 450), (IM.shape[1], 450), (19, 69, 139), 4)

        # Scale position for visualization
        XSTART, XEND = -2.4, 2.4  # Cart position limits
        cart_x = int((CART_POS - XSTART) / (XEND - XSTART) * IM.shape[1])

        # Draw Cart
        cart_w, cart_h = 80, 40
        cv2.rectangle(IM, (cart_x - cart_w // 2, 400), (cart_x + cart_w // 2, 400 - cart_h), (255, 255, 255), -1)

        # Draw Wheels
        wheel_radius = 10
        cv2.circle(IM, (cart_x - 30, 420), wheel_radius, (255, 255, 255), -1)
        cv2.circle(IM, (cart_x + 30, 420), wheel_radius, (255, 255, 255), -1)

        # Pendulum Hinge
        hinge_x, hinge_y = cart_x, 380
        cv2.circle(IM, (hinge_x, hinge_y), 6, (0, 0, 255), -1)

        # Compute Pendulum Bob Position
        pendulum_x = int(hinge_x + self.length * np.sin(BOB_ANG))
        pendulum_y = int(hinge_y - self.length * np.cos(BOB_ANG))

        # Draw Pendulum
        cv2.line(IM, (hinge_x, hinge_y), (pendulum_x, pendulum_y), (255, 255, 255), 3)
        cv2.circle(IM, (pendulum_x, pendulum_y), 10, (255, 255, 255), -1)

        # Display Information
        cv2.putText(IM, f"Theta: {BOB_ANG_DEG:.2f} deg", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 250), 2)
        cv2.putText(IM, f"Cart Pos: {CART_POS:.2f} m", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 250), 2)
        if t is not None:
            cv2.putText(IM, f"Time: {t:.2f} sec", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 250), 2)

        return IM