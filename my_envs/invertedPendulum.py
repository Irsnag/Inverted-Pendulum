# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 22:40:57 2025

@author: 33606
"""

import cv2
import numpy as np
import time

class InvertedPendulum:
    def __init__(self, length=100, width=500, height=300):
        self.length = length  # Length of the pendulum (scaled for visualization)
        self.width = width    # Width of the window
        self.height = height  # Height of the window
        self.cart_width = 50
        self.cart_height = 20
        self.cart_x = width // 2
        self.cart_y = height // 2 + 50
        self.theta = np.pi / 4  # Initial angle (45 degrees)
        self.running = True
    
    def update(self, theta):
        """Update the angle of the pendulum and redraw."""
        self.theta = theta
    
    def draw(self):
        """Draw the cart and pendulum using OpenCV."""
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        
        # Cart position (centered at cart_x, cart_y)
        cart_top_left = (self.cart_x - self.cart_width // 2, self.cart_y - self.cart_height // 2)
        cart_bottom_right = (self.cart_x + self.cart_width // 2, self.cart_y + self.cart_height // 2)
        cv2.rectangle(frame, cart_top_left, cart_bottom_right, (0, 0, 255), -1)
        
        # Pendulum end position
        pendulum_x = int(self.cart_x + self.length * np.sin(self.theta))
        pendulum_y = int(self.cart_y - self.length * np.cos(self.theta))
        
        # Draw pendulum
        cv2.line(frame, (self.cart_x, self.cart_y), (pendulum_x, pendulum_y), (0, 255, 0), 5)
        cv2.circle(frame, (pendulum_x, pendulum_y), 8, (255, 0, 0), -1)
        
        # Show frame
        cv2.imshow("Inverted Pendulum", frame)
        
    def run(self):
        """Run the animation loop."""
        while self.running:
            self.draw()
            key = cv2.waitKey(50)  # 50ms delay (adjust for speed)
            
            # Simulate a small oscillation (just for visualization purposes)
            self.theta += np.random.uniform(-0.05, 0.05)
            
            if key == 27:  # ESC key to exit
                self.running = False
        
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    pendulum = InvertedPendulum()
    pendulum.run()
