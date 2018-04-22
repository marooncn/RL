#!/usr/bin/env python
# -*- coding: UTF-8 -*-

""" Policy Gradient algorithm """

import random
import config
import numpy as np

from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam


class RL:
    def __init__(self, actions=config.valid_actions, alpha=config.alpha, gamma=config.gamma):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.model = self.buildNetwork()
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.states = []

    def buildNetwork(self):
        input = Input(shape=(2,))
        x = Dense(40, activation='relu')(input)
        policy = Dense(len(self.actions),activation='softmax')(x)
        model = Model(inputs=input, outputs = policy)
        opt = Adam(lr=self.alpha)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model

    def remember(self, state, action, prob, reward):
        y = np.zeros([len(self.actions)])
        action = [i for i in range(len(self.actions)) if self.actions[i]==action][0]
        y[action] = 1  # one-hot encode
        self.gradients.append(y.astype('float32') - prob)
        self.rewards.append(reward)
        self.states.append(np.reshape(state, [1,2]))

    def discount_rewards(self, r):
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(r.size)):
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def chooseAction(self, state):
        prob = self.model.predict_on_batch(np.reshape(state, [1,2])).flatten()
        self.probs.append(prob)
        action = int(np.random.choice(len(self.actions), 1, p=prob))
        action = self.actions[action]
        return action, prob

    def train(self):
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        rewards = rewards / np.std(rewards - np.mean(rewards))  # normalize rewards ~ N(0,1)
        gradients *= rewards
        X = np.squeeze(np.vstack([self.states]))
        Y = self.probs + self.alpha*np.squeeze(np.vstack([gradients])) # make sure loss=model.output-Y=-log(prob)*r
        self.model.train_on_batch(X, Y)
        self.states, self.probs, self.gradients, self.rewards = [], [], [], []

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
