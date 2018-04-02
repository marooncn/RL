#!/usr/bin/env python
# -*- coding: UTF-8 -*-

""" parameters setting """

import numpy as np

n_episodes = 6000
restore = True
algorithm = "qlearning" # or "sarsa"
alpha = 0.1
gamma = 0.9
epsilon = 0.3
valid_actions = ['forward', 'backward', 'turn_left', 'turn_right', 'left_forward', 'right_forward']
speed = 0.5  # rad/s (pioneer 3dx: 0.5 rad/s: ~ 0.05m/s)

wait_response = False # True: Synchronous response(too much delay)
ultra_distribution = ['left_ultra', 'right_ultra']
n_ultra = 2
valid_actions_dict = {valid_actions[0]: np.array([speed, speed]),
                      valid_actions[1]: np.array([-speed, -speed]),
                      valid_actions[2]: np.array([-speed, speed]),
                      valid_actions[3]: np.array([speed, -speed]),
                      valid_actions[4]: np.array([0, speed]),
                      valid_actions[5]: np.array([speed, 0])}
tolerance = 0.01
time_step = 1
grid_width = 0.002
