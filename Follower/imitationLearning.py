import config
import DQN
import env
import os
import tensorflow as tf
from collections import deque
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # build TensorFlow from source, it can be faster on your machine.

total_steps = 0
il_update_freq = config.il_update_freq
my_buffer = deque()  # bi-directional efficient list
replay_memory = config.replay_memory
# Make a path for our model to be saved in.
path = config.path
if not os.path.exists(path):
    os.makedirs(path)
i = 0
init = tf.global_variables_initializer()
saver = tf.train.Saver()

if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(init)
        if(total_steps < config.il_steps):
            s = env.start(total_steps)
            d = False
            # The Q-Network
            while not d:
                a = env.auto_move(); 
                s1, r, d = env.step(config.valid_actions[a])
                total_steps += 1
                my_buffer.append([s, a, r, s1, d])  # Save the experience to our episode buffer.
                if len(my_buffer) > replay_memory:
                    my_buffer.popleft()
  
		if total_steps % il_update_freq == 0:
	            DQN.update_network(sess, my_buffer)

                s = s1
		# Periodically save the model.
		if total_steps > 0 and total_steps % 1000 == 0:
                    i = i+1
		    saver.save(sess, path + '/model-' + str(i) + '.ckpt')
		    print("Saved Model")
        saver.save(sess, path + '/model-' + str(i) + '.ckpt')

