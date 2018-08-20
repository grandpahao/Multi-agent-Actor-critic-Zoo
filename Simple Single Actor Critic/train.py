import gym
from tensorboardX import SummaryWriter

from agent import get_agent
from common import config
from train_process import single_ac_train
from wrapper import atari_env

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    # env = atari_env(config.game_name)

    actor = get_agent('actor', n_ac=config.n_ac, lr=1e-2, test=True)
    critic = get_agent('critic', lr=1e-2,
                       discount=config.discount, test=True)

    single_ac_train(env, actor, critic, config.base_path,
                    config.batch_size, config.epsilon, config.save_interval, config.update_interval,
                    config.learning_starts, config.memory_size, config.max_epoch, config.max_iter)
