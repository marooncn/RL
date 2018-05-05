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
    distance, out_flag, _ = vrepInterface.if_in_range()
    if out_flag == 1:
        done = True
        reward = -100
        return reward, done
    reward = 10/(2*np.abs(distance-config.best_distance)+1)
    return reward, done


def auto_move():  
    dis, _, ang = vrepInterface.if_in_range() # dis(m), ang(rad)
    print(dis, 180*ang/np.pi)
    if dis < 2:
       if np.abs(ang) < 15*np.pi/180:
           a = 1 # backward
       elif ang > 15*np.pi/180:
           a = 2 # left_forward
       else:
           a = 3 # right_forward
       
    elif dis > 2.5:
       if np.abs(ang) < 18*np.pi/180:
           a = 0 # forward
       elif ang > 18*np.pi/180:
           a = 2 # left_forward
       else:
           a = 3 # right_forward
    elif ang > 10*np.pi/180:
       a = 2 # left_forward
    elif ang < -10*np.pi/180:
       a = 3 # right_forward
    else:
       a = 4
    print("action", a)
    return a


def startPause():
    vrepInterface.pause()

def endPause():
    vrepInterface.start()


