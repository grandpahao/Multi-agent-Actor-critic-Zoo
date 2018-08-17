import gym
from tensorboardX import SummaryWriter

from agent import get_agent
from common import config
from train_process import single_ac_train
from wrapper import atari_env

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    # env = atari_env(config.game_name)

    actor = get_agent('actor', n_ac=config.n_ac, lr=config.lr, test=True)
    critic = get_critic('critic', lr=config.lr,
                        discount=config.discount, test=True)

    single_ac_train(env, actor, critic, config.base_path,
                    config.batch_size, config.epsilon, config.save_interval, config.update_interval,
                    config.learning_starts, config.memory_size, config.total_iter)
