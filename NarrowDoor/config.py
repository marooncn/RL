#!/usr/bin/env python
# -*- coding: UTF-8 -*-

""" parameters setting """

import numpy as np

n_episodes = 6000
restore = False
algorithm = "policyGradient" #"qlearning"  # or "sarsa",
alpha = 0.1  # learning rate
gamma = 0.9  # discounted factor
epsilon = 0.3 # epsilon-greedy
valid_actions = ['forward', 'backward', 'left_forward', 'right_forward']
speed = 0.5  # rad/s (pioneer 3dx: 0.5 rad/s: ~ 0.05m/s)

wait_response = False # True: Synchronous response(too much delay)
ultra_distribution = ['left_ultra', 'right_ultra']
n_ultra = 2
valid_actions_dict = {valid_actions[0]: np.array([speed, speed]),
                      valid_actions[1]: np.array([-speed, -speed]),
                      valid_actions[2]: np.array([0, speed]),
                      valid_actions[3]: np.array([speed, 0])}
tolerance = 0.01
time_step = 0.05
grid_width = 0.002
