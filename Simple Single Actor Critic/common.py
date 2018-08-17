
class Config(object):
    def __init__(self):
        self.game_name = 'CartPole-v1'
        self.store_path = './test/'
        self.batch_size = 32
        self.epsilon = 0.01
        self.save_interval = 1000
        self.update_interval = 1000
        self.learning_starts = 200
        self.memory_size = 500000
        self.total_iter = 10000000
