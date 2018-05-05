#!/usr/bin/env python
# -*- coding: UTF-8 -*-

""" vrep interface script that connects to vrep, reads and sets data to objects through vrep remote API  """

import time
import numpy as np
import vrep
import config
import math

# V-REP data transmission modes:
WAIT = vrep.simx_opmode_oneshot_wait
ONESHOT = vrep.simx_opmode_oneshot
STREAMING = vrep.simx_opmode_streaming
BUFFER = vrep.simx_opmode_buffer
BLOCKING = vrep.simx_opmode_blocking


if config.wait_response:
    MODE_INI = WAIT
    MODE = WAIT
else:
    MODE_INI = STREAMING
    MODE = BUFFER

robotID = -1
peopleID = -1
kinect_depthID = -1
left_motorID = -1
right_motorID = -1
collisionID = -1
clientID = -1

pos_absolute = np.full(3, -1, dtype = np.float64)  # Robot pose in the world coordination: x(m), y(m), theta(rad)
pos_relative = np.full(3, -1, dtype = np.float64)  # People pose relative to robot: x(m), y(m), theta(rad)


def show_msg(message):
    """ send a message for printing in V-REP """
    vrep.simxAddStatusbarMessage(clientID, message, WAIT)
    return


def connect():
    """ Connect to the simulator"""
    ip = '127.0.0.1'
    port = 19997
    vrep.simxFinish(-1)  # just in case, close all opened connections
    global clientID
    clientID = vrep.simxStart(ip, port, True, True, 3000, 5)
    # Connect to V-REP
    if clientID == -1:
        import sys
        sys.exit('\nV-REP remote API server connection failed (' + ip + ':' +
                 str(port) + '). Is V-REP running?')
    print('Connected to Remote API Server')  # show in the terminal
    show_msg('Python: Hello')    # show in the VREP
    time.sleep(0.5)
    return


def disconnect():
    """ Disconnect from the simulator"""
    # Make sure that the last command sent has arrived
    vrep.simxGetPingTime(clientID)
    show_msg('ROBOT: Bye')
    # Now close the connection to V-REP:
    vrep.simxFinish(clientID)
    time.sleep(0.5)
    return


def start():
    """ Start the simulation (force stop and setup)"""
    stop()
    setup_devices()
    vrep.simxStartSimulation(clientID, ONESHOT)
    time.sleep(0.5)
    # Solve a rare bug in the simulator by repeating:
    setup_devices()
    vrep.simxStartSimulation(clientID, ONESHOT)
    time.sleep(0.5)
    return


def stop():
    """ Stop the simulation """
    vrep.simxStopSimulation(clientID, ONESHOT)
    time.sleep(0.5)

def pause():
    """ pause the simulation """
    vrep.simxPauseSimulation(clientID, ONESHOT)
    time.sleep(0.5)

def setup_devices():
    """ Assign the devices from the simulator to specific IDs """
    global robotID, left_motorID, right_motorID, peopleID, kinect_depthID, collisionID
    # res: result (1(OK), -1(error), 0(not called))
    # robot
    res, robotID = vrep.simxGetObjectHandle(clientID, 'robot#', WAIT)
    # people
    res, peopleID = vrep.simxGetObjectHandle(clientID, 'Bill_collisionShapeForPathPlanning#0', WAIT)
    # motors
    res, left_motorID = vrep.simxGetObjectHandle(clientID, 'leftMotor#', WAIT)
    res, right_motorID = vrep.simxGetObjectHandle(clientID, 'rightMotor#', WAIT)
    # kinetic
    res, kinect_depthID = vrep.simxGetObjectHandle(clientID, 'kinect_depth#', WAIT)
    # if res == vrep.simx_return_ok:  # [debug]
    #    print("vrep.simxGetDistanceHandle executed fine")
    # collision object
    res, collisionID = vrep.simxGetCollisionHandle(clientID, "Collision#", BLOCKING)

    # start up devices
    # wheels
    vrep.simxSetJointTargetVelocity(clientID, left_motorID, 0, STREAMING)
    vrep.simxSetJointTargetVelocity(clientID, right_motorID, 0, STREAMING)
    # pose
    vrep.simxGetObjectPosition(clientID, robotID, -1, MODE_INI)
    vrep.simxGetObjectOrientation(clientID, robotID, -1, MODE_INI)
    vrep.simxGetObjectPosition(clientID, peopleID, robotID, MODE_INI)
    vrep.simxGetObjectOrientation(clientID, peopleID, robotID, MODE_INI)
    # collision
    vrep.simxReadCollision(clientID, collisionID, STREAMING)
    # kinect
    vrep.simxGetVisionSensorDepthBuffer(clientID, kinect_depthID, STREAMING)
    return


def fetch_kinect():
    res, resolution, depth = vrep.simxGetVisionSensorDepthBuffer(clientID, kinect_depthID, BUFFER)
    depthData = np.array(depth)
    return depthData


def get_robot_pose2d():
    """ return the pose of the robot relative to world coordination:  [ x(m), y(m), Theta(rad) ] """
    global pos_absolute
    res, pos = vrep.simxGetObjectPosition(clientID, robotID, -1, MODE)
    res, ori = vrep.simxGetObjectOrientation(clientID, robotID, -1, MODE)
    pos_absolute = np.array([pos[0], pos[1], ori[2]])
    return pos_absolute


def get_relative_pose2d():
    """ return the pose of the people relative to robot:  [ x(m), y(m), Theta(rad) ] """
    global pos_relative
    res, pos = vrep.simxGetObjectPosition(clientID, peopleID, robotID, MODE)
    # print(pos)
    res, ori = vrep.simxGetObjectOrientation(clientID, peopleID, robotID, MODE)
    pos_relative = np.array([pos[0], pos[1], ori[2]])
    return pos_relative


def if_collision():
    """ judge if collision happens"""
    res, collision = vrep.simxReadCollision(clientID, collisionID, BUFFER)
    if collision == 1:
        print("Collision!")
    return collision


def if_in_range():
    """ judge if the people in the the range of kinect on the robot"""
    pos1 = get_relative_pose2d()
    out_flag = 0
    distance = np.sqrt(pos1[0] * pos1[0] + pos1[1] * pos1[1])
    if distance > 3.5 or distance < 0.01:
        print("Distance is out of range!")
        out_flag = 1
    # pos2 = get_robot_pose2d()
    angle = np.arctan2(pos1[1], pos1[0])
    # print(angle*180/math.pi)
    if angle < -28.5*np.pi/180 or angle > 28.5*np.pi/180:
        print("Angle is out of range!")
        out_flag = 1
    return distance, out_flag, angle


def move_wheels(v_left, v_right):
    """ move the wheels. Input: Angular velocities in rad/s """
    vrep.simxSetJointTargetVelocity(clientID, left_motorID, v_left, STREAMING)
    vrep.simxSetJointTargetVelocity(clientID, right_motorID, v_right,
                                    STREAMING)
    depthData = []
    for i in range(4):
        time.sleep(config.time_step)
        depthData1 = fetch_kinect()
        depthData.append(depthData1)
    return depthData


def stop_motion():
    """ stop the base wheels """
    vrep.simxSetJointTargetVelocity(clientID, left_motorID, 0, STREAMING)
    vrep.simxSetJointTargetVelocity(clientID, right_motorID, 0, STREAMING)
    return
