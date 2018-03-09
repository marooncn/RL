#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
Policy network
Policy Based(输出选取action的概率)与Value Based(输出某个action的期望价值）相比，有更好的收敛性，对于高维或连续值的action非常有效。
黄文坚《Tensorflow实战》一书第八章的示例程序，设计了一个4X50X1的全连接神经网络来训练。定义交叉熵成本函数作为损失函数。训练过程中，
训练次数达到batch_size的整数倍时，gradBuffer中累计了足够的梯度，使用updateGrads将梯度更新到模型中。当平均的奖励值超过200时性能已经良好，
训练结束。此外，在选择action的时候体现了强化学习中的一些思想，神经网络输出probability后，通过选择随机数与probability比较，大于随机数则取1,
小于则0，这种方法相比取定值（如probability>0.5取1）的方法更加exploratory
'''

import numpy as np
import tensorflow as tf
import gym
env = gym.make('CartPole-v0')
# 定义参数
H = 50
batch_size = 25
learning_rate = 1e-1
D = 4
gamma = 0.99
# 定义神经网络
observations = tf.placeholder(tf.float32, [None, D], name='input_x')
W1 = tf.get_variable('W1', shape=[D, H], initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations, W1))
W2 = tf.get_variable('W2', shape=[H, 1], initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1, W2)
probability = tf.nn.sigmoid(score)
# 定义优化器和网络梯度参数
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
W1Grad = tf.placeholder(tf.float32, name='batch_grad1')
W2Grad = tf.placeholder(tf.float32, name='batch_grad2')
batchGrad = [W1Grad, W2Grad]
‘’‘
定义折扣奖励，输入数据r为每一个action实际获得的reward，在该问题中判断越靠前的action的期望价值越高，因为它们保持了Pole的长期稳定，
而越靠后的Pole越有可能是导致试验结束的原因，期望价值越低。因此使用折扣累计奖赏，定义running_add为除直接获得的reward外的潜在价值，
每一个action的潜在价值为后一个action的潜在价值乘以衰减系数gamma再加上其直接获得的reward，即running_add*gamma+r[t]。这样可以
从最后一个action向前累计：
’‘’
def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r
# 定义损失函数
input_y = tf.placeholder(tf.float32, [None, 1], name='input_y')
advantages = tf.placeholder(tf.float32, name='reward_signal')
loglik = tf.log(input_y*(input_y - probability) + (1 - input_y)*(input_y + probability))
loss = -tf.reduce_mean(loglik* advantages)
tvars = tf.trainable_variables()
newGrads = tf.gradients(loss, tvars)
updateGrads = adam.apply_gradients(zip(batchGrad, tvars))

xs, ys, drs = [], [], []
reward_sum = 0
episode_number = 1
total_episodes = 400
with tf.Session() as sess:
    rendering = False  # 初始阶段模型还不成熟，关闭render减小内存占用
    init = tf.global_variables_initializer()
    sess.run(init)
    observation = env.reset()
    gradBuffer = sess.run(tvars)
    for ix, grad in enumerate(gradBuffer):  # 初始化梯度参数
        gradBuffer[ix] = grad * 0

    while episode_number <= total_episodes: 
        if reward_sum/batch_size > 100 or rendering == True:  # 平均奖励值大于100,开启render
            env.render()
            rendering = True

        x = np.reshape(observation, [1, D])

        tfprob = sess.run(probability, feed_dict={observations: x})  # 运行神经网络，计算probability
        action = 1 if np.random.uniform() < tfprob else 0  

        xs.append(x)
        y = 1 - action
        ys.append(y)

        observation, reward, done, info = env.step(action)
        reward_sum += reward
        drs.append(reward)

        if done:
            episode_number += 1
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            xs, ys, drs = [],[],[]
# discount_rewards计算每一个action的潜在价值，然后正则化（减去均值再除以标准差），得到零均值标准差为1的分布均值标准差为1的分布
            discounted_epr = discount_rewards(epr)   
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)
            tGrad = sess.run(newGrads, feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})# 计算梯度
            for ix, grad in enumerate(tGrad):
                gradBuffer[ix] += grad   # 累计梯度

            if episode_number % batch_size == 0:  # 经过batch_size的训练
                sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0],W2Grad:gradBuffer[1]})  # 更新神经网络参数
                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0     # 清空，为一个batch作准备
                print('Average reward for episode %d: %f.' % (episode_number, reward_sum/batch_size))
                if reward_sum/batch_size > 200:   # 平均奖励高于200则认为达到要求，退出
                    print("Task solved in ", episode_number, 'episodes!')
                    break
                reward_sum = 0
            observation = env.reset()


