import tensorflow as tf

from .tfagent import TFAgent


class Critic(TFAgent):
    def __init__(self, lr, discount):
        super(Critic, self).__init__(lr=lr)
        self.discount = discount
        self.update_target()

    def _build_net(self):
        self.input = tf.placeholder(
            shape=[None, 84, 84, 4], dtype=tf.float32, name='input')
        self.target = tf.placeholder(
            shape=[None], dtype=tf.float32, name='target')

        with tf.variable_scope('critic_accurate'):
            fc1 = self._net(self.input, trainable=True)
            self.score = tf.contrib.layers.fully_connected(
                fc1, 1, trainable=True)

        with tf.variable_scope('critic_target'):
            fc1_target = self._net(self.input, trainable=False)
            self.score_target = tf.contrib.layers.fully_connected(
                fc1_target, 1, trainable=False)

        self.update_target_opr = self._update_target_opr()
        # given the sum of scores for the next state and reward for the transition
        trainable_variables = tf.trainable_variables('critic_accurate')
        self.advantage = self.target - self.score
        self.loss = tf.squared_difference(self.target, self.score)
        self.train_opr = self.optimizer.minimize(
            self.loss,
            global_step=tf.train.get_global_step(),
            var_list=trainable_variables
        )

    def _update_target_opr(self):
        params = tf.trainable_variable('ciritc_accurate')
        params = sorted(params, key=lambda v: v.name)
        target_params = tf.global_variable('critic_target')
        target_params = sorted(target_params, key=lambda v: v.name)

        update_opr = []
        for (param, target_param) in zip(params, target_params):
            update_opr.append(target_param.assign(param))

        return update_opr

    def update_target(self):
        self.sess.run(self.update_target_opr)

    def update(self, input_batch, reward_batch, reward_batch, next_batch, done_batch):
        next_score = self.sess.run(self.score_target, feed_dict={
                                   self.input: next_batch})
        target = next_score * self.discount + (1 - done_batch) * reward_batch
        _, total_t, critic_loss, advantage = self.sess.run(
            [
                self.train_opr,
                tf.train.get_global_step(), self.loss, self.advantage
            ],
            feed_dict={
                self.input: input_batch,
                self.target: target
            }
        )
        return total_t, {'critic_loss': loss}, advantage
