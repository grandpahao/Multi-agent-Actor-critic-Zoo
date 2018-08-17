import numpy as np
import tensorflow as tf

from .tfagent import TFAgent


class Actor(TFAgent):
    def __init__(self, n_ac, lr, test=False):
        super(Actor, self).__init__(n_ac, lr, test)
        self.update_target()

    def _build_net(self):
        if self.test:
            self.input = tf.placeholder(
                shape=[None, 4], dtype=tf.float32, name='inputs')
        else:
            self.input = tf.placeholder(
                shape=[None, 84, 84, 4], dtype=tf.float32, name='inputs')

        self.action_select = tf.placeholder(
            shape=[None], dtype=tf.float32, name='selected_action')
        self.advantage = tf.placeholder(
            shape=[None], dtype=tf.float32, name='advantage')

        with tf.variable_scope('actor_accurate'):
            fc1 = self._net(self.input, trainable=True)
            action = tf.contrib.layers.fully_connected(
                fc1, self.n_ac, trainable=True)
            self.action_prob = tf.nn.softmax(action)

        with tf.variable_scope('actor_target'):
            fc1_target = self._net(self.input, trainable=False)
            action_target = tf.contrib.layers.fully_connected(
                fc1_target, self.n_ac, trainable=False)
            self.target_action_prob = tf.nn.softmax(action_target)

        self.update_target_opr = self._update_target_opr()

        trainable_variables = tf.trainable_variables('actor_accurate')
        self.loss = self._eval_loss()
        self.train_opr = self.optimizer.minimize(
            self.loss,
            global_step=tf.train.get_global_step(),
            var_list=trainable_variables
        )

    def update(self, input_batch, action_batch, advantage_batch):
        _, total_t, actor_loss, actor_max_prob = self.sess.run(
            [
                self.train_opr,
                tf.train.get_global_step(), self.loss, tf.max(self.action_target_prob)
            ],
            feed_dict={
                self.input: input_batch,
                self.action_select: action_batch,
                self.advantage: advantage_batch
            }
        )
        return total_t, {'actor_loss': actor_loss, 'actor_max_prob': actor_max_prob}

    def get_action(self, input_state, epsilon):
        action_prob = self.sess.run(self.target_action_prob, feed_dict={
                                    self.input: input_state})
        action_prob = epsilon / self.n_ac + (1.0 - epsilon) * action - prob
        return np.choice(np.arange(self.n_ac), size=1, p=action_prob)

    def _eval_loss(self):
        if self.n_ac > 1:
            batch_size = tf.shape(self.input)[0]
            gather_indices = tf.range(batch_size) * \
                self.n_ac + self.action_select
            action_prob = tf.gather(tf.reshape(
                self.action_prob, [-1]), gather_indices)
            # policy gradient should ascent
            ad_log_prob = tf.neg(tf.log(action_prob) * self.advantage)
            return tf.reduce_mean(ad_log_prob)
        else:
            raise NotImplementedError

    def _update_target_opr(self):
        params = tf.trainable_variables('actor_accurate')
        params = sorted(params, key=lambda v: v.name)
        target_params = tf.global_variables('actor_target')
        target_params = sorted(params, key=lambda v: v.name)

        update_opr = []
        for param, target_param in zip(params, target_params):
            update_opr.append(target_param.assign(param))

        return update_opr

    def update_target(self):
        self.sess.run(self.update_target_opr)
