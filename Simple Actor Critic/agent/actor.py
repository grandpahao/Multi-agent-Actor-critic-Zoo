import tensorflow as tf

from .tfagent import TFAgent


class Actor(TFAgent):
    def __init__(self, n_ac, lr):
        super(Actor, self).__init__(n_ac, lr)
        self.update_target()

    def _net(self, input, trainable):
        conv1 = tf.contrib.layers.conv2d(
            input, 32, 8, 4, activation_fn=tf.nn.relu, trainable=trainable)
        conv2 = tf.contrib.layers.conv2d(
            conv1, 64, 4, 2, activation_fn=tf.nn.relu, trainable=trainable)
        conv3 = tf.contrib.layers.conv2d(
            conv2, 64, 3, 1, activation_fn=tf.nn.relu, trainable=trainable)
        )

    def _build_net(self):
        self.input=tf.placeholder(
            shape = [None, 84, 84, 4], dtype = tf.float32, name = 'inputs')
