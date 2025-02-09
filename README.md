# Inverted pendulum on moving cart

## Introduction 

The Inverted Pendulum is a classic problem in dynamics and control systems, often used to demonstrate control theory concepts. The goal of this project is to balance an inverted pendulum on a moving cart using LQR control and compare it with Deep Reinforcement Learning. The simulation was made using python.

## Equations

The equations governing the system are :

$$ (M + m)\ddot{x} + b_x\dot{x} - ml\ddot{\theta}\cos\theta + ml\dot{\theta}^2\sin\theta = F $$
$$l\ddot{\theta} + b_\theta\dot{\theta} - g\sin\theta = \ddot{x}\cos\theta $$


```math
\begin{bmatrix}X\\Y\end{bmatrix}
```


