import os

import tensorflow as tf


class TFAgent(object):
    def __init__(self, lr):
        self.optimizer = tf.train.AdamOptimizer(lr, epsilon=1.5e-4)
        self._logd_prepare()
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

    def _conv_net(self):
        raise NotImplementedError

    def _lstm_net(self):
        raise NotImplementedError

    def save_model(self, outdir):
        cur_step = get_global_step()
        self.saver.save(
            self.sess,
            os.path.join(outdir, 'model'),
            cur_step,
            write_meta_graph=False
        )

    def load_model(self, outdir):
        latest_log = tf.train.last_checkpoint(outdir)
        if latest_log:
            print("Loading model from {}".format(latest_log))
            self.saver.restore(self.sess, latest_log)
        else:
            print("No history record!")

    def get_global_step(self):
        return self.sess.run(tf.train.get_global_step())
