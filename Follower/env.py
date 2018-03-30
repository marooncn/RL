#!/usr/bin/env python
# -*- coding: UTF-8 -*-

""" main script """
import config
import vrepInterface
import numpy as np


def start(i):
    if i == 0:
        vrepInterface.connect()
    vrepInterface.start()
    state = vrepInterface.fetch_kinect()
    state = np.tile(state, 4)
    state = np.reshape(state, [16384])
    return state


def step(action):
    v_left = config.valid_actions_dict[action][0]
    v_right = config.valid_actions_dict[action][1]
    next_state = vrepInterface.move_wheels(v_left, v_right)
    next_state = np.reshape(next_state, [16384])
    reward, done = get_reward()
    return next_state, reward, done


def get_reward():
    done = False
    collision = vrepInterface.if_collision()
    if collision == 1:
        done = True
        reward = -500
        return reward, done
    distance, out_flag = vrepInterface.if_in_range()
    if out_flag == 1:
        done = True
        reward = -100
        return reward, done
    reward = 10/(2*np.abs(distance-config.best_distance)+1)
    return reward, done
