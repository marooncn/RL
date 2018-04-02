## Introduction
    This package uses Reinforcement Learning to train a differential drive robot with only two ultrasonic 
    sensors to go through a narrow door which is just a little wider than the robot.
    
<img alt="Introduction" src="image/Environment.png" width="800">

`config.py:` configuration file<br>
`robot.py:` the main file<br>
`vrepInterface.py:` interface file between python code with V-REP<br>
`sarsa.py:` SARSA algorithm<br>
`qlearning.py:` Q-learning algorithm<br>
`robot.ttt:` the scene to de loaded in V-REP<br>
`vrep.py, vrepConst.py, remoteApi.so:` the remote API bridge offered by V-REP

## Reinforcement Learning
* [x] [SARSA](./sarsa.py)
* [x] [Q-learning](./qlearning.py)
* [ ] Policy Gradient Methods

`Environment：` V-REP<br>
`State：` the data from two ultrasonic sensors<br>
`Action：` six valid actions(can be changed in config.py)<br>
`Reward:` get_reward function of class LearningAgent(robot.py)
 
## Run the demo by yourself
Tested on Ubuntu 16.04 (64 bits).<br>
1> Open V-REP 
~~~
# in your V-REP installation folder
./vrep.sh
~~~
load the scene: `File -> Open Scene -> robot.ttt` 

Recommended simulation settings for V-REP scenes:
* Simulation step time: 50 ms  (default) 
* Real-Time Simulation: Enabled

2> Execute the learning algorithm
~~~
python3 robot.py
~~~
## Reference
[RL-ROBOT](https://github.com/angelmtenor/RL-ROBOT)<br>
[basic_reinforcement_learning](https://github.com/vmayoral/basic_reinforcement_learning)<br>
[Automatic-Parking](https://github.com/taochenshh/Automatic-Parking)<br>
