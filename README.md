# Inverted pendulum on moving cart

## Introduction 

The Inverted Pendulum is a classic problem in dynamics and control systems, often used to demonstrate control theory concepts. The goal of this project is to balance an inverted pendulum on a moving cart using LQR control and compare it with Deep Reinforcement Learning. The simulation was made using python.

## Equations

The equations governing the system are :

$$ (M + m)\ddot{x} + b_x\dot{x} - ml\ddot{\theta}\cos\theta + ml\dot{\theta}^2\sin\theta = F $$
$$l\ddot{\theta} + b_\theta\dot{\theta} - g\sin\theta = \ddot{x}\cos\theta $$

Which can be linearized as :

```math
\begin{bmatrix}\dot{x}\\\ddot{x}\\\dot{\theta}\\\ddot{\theta}\end{bmatrix} = \begin{bmatrix}0 & 1 & 0 & 0\\ 0 & -b_x & \frac{g*m}{M} & 0 \\ 0 & \frac{-b_x}{(M*l)} & \frac{(m+M)*g}{(M*l)} & \frac{-b_\theta*(m+M)*g}{(M*l)} \\ 0 & 0 & 0 & 1 \end{bmatrix} \begin{bmatrix}x\\\dot{x}\\\theta\\\dot{\theta}\end{bmatrix} + \begin{bmatrix}0\\\\frac{1}{M}\\0\\\frac{1}{M*l}\end{bmatrix} u
```


