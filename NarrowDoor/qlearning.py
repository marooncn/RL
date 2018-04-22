#!/usr/bin/env python
# -*- coding: UTF-8 -*-

""" Q-learning algorithm """

import random
import config


class RL:
    def __init__(self, actions=config.valid_actions, epsilon=config.epsilon, alpha=config.alpha, gamma=config.gamma):
        self.q = {}             # store Q values

        self.actions = actions  # valid actions set
        self.epsilon = epsilon  # exploration constant of epsilon-greedy algorithm
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # learning rate

    def getQ(self, state, action):    # get Q value
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state):    # get action using epsilon-greedy algorithm
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q = [self.getQ(state,a) for a in self.actions]
            maxQ = max(q)
            count = q.count(maxQ)
            # in case there're several state-action max values, we select a random one among them
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)

            action = self.actions[i]
        return action, None

    # after choosing action, the agent needs to interact with the environment to get reward and the next state
    # then it can update Q value
    def learn(self, last_state, last_action, reward, new_state):  # update Q value according to reward and new state
        '''
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + gamma * maxQ(s') - Q(s,a))
        '''
        oldq = self.getQ(last_state, last_action)
        q_ = max([self.getQ(new_state, a) for a in self.actions]) # get the Q value of s' according to greedy algorithm.
        self.q[(last_state, last_action)] = oldq + self.alpha * (reward + self.gamma * q_ - oldq)
