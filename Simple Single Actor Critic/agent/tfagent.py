import os

import tensorflow as tf


class TFAgent(object):
    def __init__(self, n_ac=0, lr=1e-4, test=False):
        self.optimizer = tf.train.AdamOptimizer(lr, epsilon=1.5e-4)
        self._log_prepare()
        self.n_ac = n_ac
        self.test = test
        self._net_prepare()

    def _log_prepare(self):
        tf.reset_default_graph()
        tf.Variable(0, name='global_step', trainable=False)
        self.saver = tf.train.Saver(max_to_keep=5)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def _net_prepare(self):
        self._build_net()
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        raise NotImplementedError

    def update_target(self, *args, **kwargs):
        raise NotImplementedError

    def _net(self, input, trainable):
        if self.test:
            fc1 = tf.contrib.layers.fully_connected(
                input, 10, activation_fn=tf.nn.selu, trainable=trainable)
            fc2 = tf.contrib.layers.fully_connected(
                fc1, 10, activation_fn=tf.nn.selu, trainable=trainable)
            return fc2

        conv1 = tf.contrib.layers.conv2d(
            input, 32, 8, 4, activation_fn=tf.nn.relu, trainable=trainable)
        conv2 = tf.contrib.layers.conv2d(
            conv1, 64, 4, 2, activation_fn=tf.nn.relu, trainable=trainable)
        conv3 = tf.contrib.layers.conv2d(
            conv2, 64, 3, 1, activation_fn=tf.nn.relu, trainable=trainable)

        flat1 = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(
            flat1, 256, trainable=trainable)
        return fc1

    def save_model(self, outdir, cur_step):
        # cur_step=self.get_global_step()
        self.saver.save(
            self.sess,
            os.path.join(outdir, 'model'),
            cur_step,
            write_meta_graph=False
        )

    def load_model(self, outdir):
        latest_log = tf.train.latest_checkpoint(outdir)
        if latest_log:
            print("Loading model from {}".format(latest_log))
            self.saver.restore(self.sess, latest_log)
        else:
            print("No history record!")

    def get_global_step(self):
        return self.sess.run(tf.train.get_global_step())
