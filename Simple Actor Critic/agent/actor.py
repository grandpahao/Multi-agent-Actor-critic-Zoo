import tensorflow as tf

from .tfagent import TFAgent


class Actor(TFAgent):
    def __init__(self, n_ac, lr):
        super(Actor, self).__init__(n_ac, lr)
        self.update_target()

    def _build_net(self):
        self.input = tf.placeholder(
            shape=[None, 84, 84, 4], dtype=tf.float32, name='inputs')
        self.action_select = tf.placeholder(
            shape=[None], dtype=tf.float32, name='selected_action')
        self.advantage = tf.placeholder(
            shape=[None], dtype=tf.float32, name='advantage')

        with tf.variable_scope('actor_accurate'):
            fc1 = self._net(self.input, trainable=True)
            self.action = tf.layers.dense(fc1, self.n_ac, trainable=True)

        with tf.variable_scope('actor_target'):
            fc1_target = self._net(self.input, trainable=False)
            self.action_target = tf.layers.dense(
                fc1_target, self.n_ac, trainable=False)

        self.update_target_opr = self._update_target_opr()

        trainable_variables = tf.trainable_variables('actor_accurate')
        self.loss = self._eval_loss()
        self.train_opr = self.optimizer.minimize(
            self.loss,
            global_step=tf.train.get_global_step(),
            var_list=trainable_variables
        )

    def update(self, ):

    def _eval_loss(self):
        if self.n_ac > 1:
            batch_size = tf.shape(self.input)[0]
            gather_indices = tf.range(batch_size) * \
                self.n_ac + self.action_select
            action_prob = tf.gather(tf.reshape(
                self.action, [-1]), gather_indices)
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
