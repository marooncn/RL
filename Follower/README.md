## Introduction
    This package uses Deep Reinforcement Learning(DQN) to train a differential drive robot with a kinect 
    to follow a worker.
      
<img alt="Introduction" src="image/DQN_Environment.png" width="800">

`config.py:` configuration file<br>
`DQN.py:` the main file<br>
`vrepInterface.py:` interface file between python code with V-REP<br>
`env.py:` environment file<br>
`robot_DQN.ttt:` the scene to de loaded in V-REP<br>
`vrep.py, vrepConst.py, remoteApi.so:` the remote API bridge offered by V-REP

## Reinforcement Learning
* [x] [DQN](./DQN.py)
* [x] [Imitation Learning](./imitationLearning.py)
* [ ] Continous Space

`Environment：` V-REP<br>
`State：` the depth image from kinect <br>
`Action：` five valid actions(can be changed in config.py)<br>
`Reward:` get_reward function (env.py)
 
## Run the demo by yourself
Tested on Ubuntu 16.04 (64 bits) and V-REP PRO EDU 3.4.0.<br>
1> Open V-REP 
~~~
# in your V-REP installation folder
./vrep.sh
~~~
load the scene: `File -> Open Scene -> robot_DQN.ttt` 

Recommended simulation settings for V-REP scenes:
* Simulation step time: 50 ms  (default) 
* Real-Time Simulation: Enabled

2> Execute the learning algorithm
~~~
python DQN.py
~~~
## Reference
[Double-Dueling-DQN](https://github.com/awjuliani/DeepRL-Agents/blob/master/Double-Dueling-DQN.ipynb)<br>
[DeepLearningFlappyBird](https://github.com/yenchenlin/DeepLearningFlappyBird)
