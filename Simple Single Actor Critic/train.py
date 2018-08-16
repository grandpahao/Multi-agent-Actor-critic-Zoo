from tensorboardX import SummaryWriter

from agent import get_agent
from common import config
from train_process import single_ac_train
from wrapper import atari_env

if __name__ == '__main__':
    env = atari_env(config.game_name)

    actor = get_agent('actor', n_ac=config.n_ac, lr=config.lr)
    critic = get_critic('critic', lr=config.lr, discount=config.discount)

    single_ac_train(env, actor, critic, config.base_path,
                    config.batch_size, config.epsilon)
