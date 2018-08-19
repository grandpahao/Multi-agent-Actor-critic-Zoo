
class Config(object):
    def __init__(self):
        self.game_name = 'CartPole-v1'
        self.base_path = './test/'
        self.batch_size = 32
        self.epsilon = 0.01
        self.save_interval = 1000
        self.update_interval = 1000
        self.learning_starts = 200
        self.memory_size = 500000
        self.max_epoch = 100000
        self.max_iter = 10000
        self.n_ac = 2
        self.lr = 1e-4
        self.discount = 0.99

config = Config()
