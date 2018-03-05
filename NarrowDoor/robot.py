#!/usr/bin/env python
# -*- coding: UTF-8 -*-

""" main script """

import numpy as np
import config
import vrepInterface
import os
import pickle
import threading
from collections import namedtuple
algorithm = __import__(config.algorithm)

states = namedtuple('states', 'x y')


class LearningAgent(object):
    def __init__(self, restore):
        self.n_episodes = config.n_episodes
        self.state = None
        self.action = None
        self.reward = 0.0
        self.next_state = None
        self.next_action = None

        self.done = False
        self.t = 0
        self.flag = 0
        self.flag_prev = 0
        self.deadline = 300
        self.num_out_of_time = 0
        self.hit_wall_time = 0
        self.ok_time = 0
        self.far_time = 0
        self.ai = algorithm.RL()

        parent_path = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(parent_path, 'trainedData')
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        self.data = os.path.join(data_path, 'data.pkl')

        if restore:
            if os.path.exists(self.data):
                print('restoring trained data from {}'.format(data_path))
                with open(self.data, 'rb') as f:
                    self.Q_values = pickle.load(f)
                    print('trainedData length:', len(self.Q_values))
                    print('restoring done ...')

    def run(self):
        vrepInterface.connect()
        train_thread = threading.Thread(name="train", target=self.train())
        train_thread.daemon = True
        train_thread.start()
        while True:
            continue

    def train(self):
        for epi in range(self.n_episodes):
            print("Simulator.run(): Train {}".format(epi))
            if (epi > 8000) and (epi < 15000):
                self.ai.epsilon = 0.3
            elif (epi > 15000) and (epi < 25000):
                self.ai.epsilon = 0.2
            elif epi > 25000:
                self.ai.epsilon = 5000 / float(epi)
            vrepInterface.start()
            vrepInterface.reset()
            self.reset()
            self.done = False
            self.t = 0

            while True:
                quit_flag = False
                try:
                    self.step()
                except KeyboardInterrupt:
                    quit_flag = True
                finally:
                    if quit_flag or self.done:
                        break

            if epi > 0 and epi % 500 == 0:
                print("Train {} done, saving Q table...".format(epi))
                with open(self.data,'wb') as f:
                    pickle.dump(self.ai.q, f, protocol=pickle.HIGHEST_PROTOCOL)
                print("Success: {} / {}".format(self.ok_time, self.t))
                print("Collision: {} / {}".format(self.hit_wall_time, self.t))
                print("too far: {} / {}".format(self.far_time, self.t))
                print("Out of time: {} / {}".format(self.num_out_of_time, self.t))

    def reset(self):
        self.state = None
        self.action = None
        self.reward = None

    def step(self):
        self.update()
        if self.done:
            return
        if self.t >= self.deadline:
            print("Agent ran out of time limit!")
            self.done = True
            self.num_out_of_time += 1
        self.t += 1

    def update(self):
        if self.state is None:
            s, self.flag = vrepInterface.get_ultra_distance()
            self.state = states(x=s[0], y=s[1])
            self.action = self.ai.chooseAction(self.state)

        self.get_next_state()
        self.get_reward()
        self.next_action = self.ai.chooseAction(self.next_state)
        if config.algorithm == "qlearning":
            self.ai.learn(self.state, self.action, self.reward, self.next_state)
        elif config.algorithm == "sarsa":
            self.ai.learn(self.state, self.action, self.reward, self.next_state, self.next_action)
        print("LearningAgent.update(): step = {}, state = {}, action = {}, reward = {}".format(self.t, self.state,
                                                                                            self.action, self.reward))
        self.state = self.next_state
        self.action = self.next_action

    def get_next_state(self):
        v_left = config.valid_actions_dict[self.action][0]
        v_right = config.valid_actions_dict[self.action][1]
        vrepInterface.move_wheels(v_left, v_right)
        self.flag_prev = self.flag
        n_s, self.flag = vrepInterface.get_ultra_distance()
        self.next_state = states(x=n_s[0], y=n_s[1])

    def get_reward(self):
        collision = vrepInterface.if_collision()
        if collision == 1:
            self.hit_wall_time += collision
            self.done = True
            self.reward = -300
            return
        reward_ref = vrepInterface.get_reward_distance()
        if reward_ref < config.tolerance:
            print("Success!")
            self.done = True
            self.ok_time += 1
            self.reward = 200
        elif reward_ref > 2 or self.state == states(x=-1.0, y=-1.0):
            print("too far!")
            self.done = True
            self.far_time += 1
            self.reward = -50
        elif self.flag_prev == 0 and self.flag == 1:
            self.reward = -50 + 100 / (10 * reward_ref + 1)
        elif self.flag_prev == 1 and self.flag == 0:
            self.reward = 50 + 100 / (10 * reward_ref + 1)
        else:
            self.reward = -20 * self.flag + 100 / (10 * reward_ref + 1)


if __name__ == '__main__':
    agent = LearningAgent(restore=False)
    agent.run()
