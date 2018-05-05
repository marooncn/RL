#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from collections import deque
import config
import random
import env
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # build TensorFlow from source, it can be faster on your machine.

'''
# for imitation learning using joystick
import sys, termios
fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)
new = termios.tcgetattr(fd)
# turn off echo and press-enter
new[3] = new[3] & ~termios.ECHO & ~termios.ICANON
termios.tcsetattr(fd, termios.TCSADRAIN, new)
aList = [4]
def readKeyboard():
    char = sys.stdin.read(1)
    if char in ['w', 's', 'a', 'd']:
        print('get:', char)
        if char == 'w':
            act = 0
        elif char == 's':
            act = 1
            env.endPause()
        elif char == 'a':
            act = 2
        elif char == 'd':
            act = 3
        aList.append(act)
        return 1
    return 0
'''

class Qnetwork():
    def __init__(self):
        # input layer
        # The network receives a frame from V-REP, flattened into an array
        # It then resize it and process it through three convolutional layers.
        self.scalarInput = tf.placeholder(shape=[None,16384], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1,64,64,4])
        # hidden layer
        self.conv1 = slim.conv2d(inputs=self.imageIn, num_outputs=32, kernel_size=[8,8], stride=[4,4], padding='VALID',
                                 biases_initializer=None)
        self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=64, kernel_size=[4,4], stride=[2,2], padding='VALID',
                                 biases_initializer=None)
        self.conv3 = slim.conv2d(inputs=self.conv2, num_outputs=64, kernel_size=[3,3], stride=[1,1], padding='VALID',
                                 biases_initializer=None)
        self.conv4 = slim.conv2d(inputs=self.conv3, num_outputs=512, kernel_size=[4,4], stride=[1,1], padding='VALID',
                                 biases_initializer=None)
        self.streamAC, self.streamVC = tf.split(self.conv4, 2, 3)  # tf.split(input, num_split, dimension_split)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([256, len(config.valid_actions)]))  # xavier initialization
        self.VW = tf.Variable(xavier_init([256, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Advantage = tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1,
                                                                    keep_dims=True))
                                                                 # Advantage为action与其它action的比较值，设计为0均值
        self.Value = tf.matmul(self.streamV, self.VW)

        # Then combine them together to get our final Q-values.
        self.Qout = self.Value + self.Advantage
        self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, len(config.valid_actions), dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)


# store the previous observations in replay memory
my_buffer = deque()  # bi-directional efficient list


# update target network
def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder


def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)


batch_size = config.batch_size
update_freq = config.update_freq
y = config.gamma
startE = config.startE
endE = config.endE
path = config.path
annealing_steps = config.annealing_steps
num_episodes = config.num_episodes
pre_train_steps = config.pre_train_steps
tau = config.tau
replay_memory = config.replay_memory

tf.reset_default_graph()
mainQN = Qnetwork()    # define main network
targetQN = Qnetwork()  # define target network

init = tf.global_variables_initializer()
saver = tf.train.Saver()
trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables,tau)

# Set the rate of random action decrease.
e = startE
stepDrop = (startE - endE)/annealing_steps


# create lists to contain total rewards and steps per episode
rList = []
total_steps = 0

# Make a path for our model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)


global_step= tf.Variable(0, name='global_step', trainable=False)  # 计数器变量


if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(init)
        if config.restore:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(path)
            if ckpt and ckpt.model_checkpoint_path:
                print(ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(num_episodes):
            # Reset environment and get first new observation
            s = env.start(i)
            d = False
            rAll = 0
            # The Q-Network
            while not d:
                if config.imitation_learning:
		    a = env.auto_move();                   
                else:
                    # Choose an action by greedily (with e chance of random action) from the Q-network
                    if np.random.rand(1) < e or total_steps < pre_train_steps:
                       a = np.random.randint(0, len(config.valid_actions))
                    else:
                       a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [s]})[0]
                s1, r, d = env.step(config.valid_actions[a])
                total_steps += 1
                my_buffer.append([s, a, r, s1, d])  # Save the experience to our episode buffer.
                if len(my_buffer) > replay_memory:
                    my_buffer.popleft()
                if total_steps > pre_train_steps:
                    if e > endE:
                        e -= stepDrop

                    if total_steps % update_freq == 0:
                        trainBatch = random.sample(list(my_buffer), batch_size)  # Get a random batch of experiences.
                        # print(trainBatch)
                        state = [trainBatch[k][0] for k in range(batch_size)]
                        action = [trainBatch[k][1] for k in range(batch_size)]
                        reward = [trainBatch[k][2] for k in range(batch_size)]
                        next_state = [trainBatch[k][3] for k in range(batch_size)]
                        # print(len(np.vstack(next_state)))
                    # Below we perform the Double-DQN update to the target Q-values
                        A = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput:np.vstack(next_state)})
                                                      # 回放下一个状态s1传入mainQN执行predict，得到主模型选择的action
                        Q = sess.run(targetQN.Qout, feed_dict={targetQN.scalarInput: np.vstack(next_state)})
                                                      # 回放下一个状态s1传入targetQN执行predict，得到目标模型的输出Q值
                        doubleQ = Q[range(batch_size), A] # 评估mainQN选择的action
                        targetQ = reward + y * doubleQ  # 回放reward加上doubleQ乘以衰减系数y得到学习目标
                    # Update the network with our target values.
                        _ = sess.run(mainQN.updateModel,feed_dict={mainQN.scalarInput: np.vstack(state),
                                                               mainQN.targetQ: targetQ, mainQN.actions: action})

                        updateTarget(targetOps, sess)  # Update the target network toward the primary network.
                rAll += r
                s = s1

            rList.append(rAll)
            global_step.assign(i).eval()  # update calculator
            if i > 0 and i % 25 == 0:
                print("episode", i, ', average reward of last 25 episodes', np.mean(rList[-25]))
                # Periodically save the model.
            if i > 0 and i % 1000 == 0:
                saver.save(sess, path + '/model-' + str(i) + '.ckpt', global_step=global_step)
                print("Saved Model")
        saver.save(sess, path + '/model-' + str(i) + '.ckpt', global_step=global_step)
        print("Percent of succesful episodes: " + str(sum(rList) / num_episodes) + "%")
