#!/usr/bin/env python
# -*- coding: UTF-8 -*-

""" parameters setting """

import numpy as np

restore = False

imitation_learning = True

valid_actions = ['forward', 'backward', 'left_forward', 'right_forward', 'stop']
speed = 5.6  # rad/s (pioneer 3dx: 5.6 rad/s: ~ 0.56m/s)  # similar to human's normal speed

wait_response = False # True: Synchronous response(too much delay)
valid_actions_dict = {valid_actions[0]: np.array([1.8*speed, 1.8*speed]),
                      valid_actions[1]: np.array([-speed*1.8, -speed*1.8]),
                      valid_actions[2]: np.array([0, speed*0.5]),
                      valid_actions[3]: np.array([speed*0.5, 0]),
                      valid_actions[4]: np.array([0, 0])}

# network
batch_size = 32  # How many experiences to use for each training step.
update_freq = 4  # How often to perform a training step.
gamma = .99  # Discount factor on the target Q-values
startE = 1  # Starting chance of random action
endE = 0.1  # Final chance of random action
path = "./trainedModel"   # The path to save our model to.
annealing_steps = 10000.  # How many steps of training to reduce startE to endE.
num_episodes = 10000      # How many episodes of game environment to train network with.
pre_train_steps = 10000   # How many steps of random actions before training begins.
max_epLength = 50         # The max allowed length of our episode.
tau = 0.001               # Rate to update target network toward primary network
replay_memory = 50000

time_step = 0.001
best_distance = 3.5/2
