import numpy as np
import tensorflow as tf

from .tfagent import TFAgent


class Critic(TFAgent):
    def __init__(self, lr, discount, test=False):
        super(Critic, self).__init__(lr=lr, test=test)
        self.discount = discount
        self.update_target()

    def _build_net(self):
        if self.test:
            input_shape = [None, 4]
        else:
            input_shape = [None, 84, 84, 4]
        self.input = tf.placeholder(
            shape=input_shape, dtype=tf.float32, name='inputs')
        self.target = tf.placeholder(
            shape=[None], dtype=tf.float32, name='target')

        with tf.variable_scope('critic_accurate'):
            fc1 = self._net(self.input, trainable=True)
            score = tf.contrib.layers.fully_connected(
                fc1, 1, trainable=True)
            self.score = tf.reshape(score, [-1])

        with tf.variable_scope('critic_target'):
            fc1_target = self._net(self.input, trainable=False)
            score_target = tf.contrib.layers.fully_connected(
                fc1_target, 1, trainable=False)
            self.score_target = tf.reshape(score_target, [-1])

        self.update_target_opr = self._update_target_opr()
        # given the sum of scores for the next state and reward for the transition
        trainable_variables = tf.trainable_variables('critic_accurate')
        self.advantage = self.target - self.score
        self.loss = tf.reduce_mean(
            tf.squared_difference(self.score, self.target))

        self.train_opr = self.optimizer.minimize(
            self.loss,
            global_step=tf.train.get_global_step(),
            var_list=trainable_variables
        )

    def get_target(self, state_batch):
        return self.sess.run(self.score_target, feed_dict={self.input: state_batch})

    def _update_target_opr(self):
        params = tf.trainable_variables('ciritc_accurate')
        params = sorted(params, key=lambda v: v.name)
        target_params = tf.global_variables('critic_target')
        target_params = sorted(target_params, key=lambda v: v.name)

        update_opr = []
        for (param, target_param) in zip(params, target_params):
            update_opr.append(target_param.assign(param))

        return update_opr

    def update_target(self):
        self.sess.run(self.update_target_opr)

    def update(self, state_batch, target_batch):
        _, total_t, critic_loss, advantage = self.sess.run(
            [
                self.train_opr,
                tf.train.get_global_step(), self.loss, self.advantage
            ],
            feed_dict={
                self.input: state_batch,
                self.target: target_batch
            }
        )
        return total_t, {'critic_loss': critic_loss}, advantage
