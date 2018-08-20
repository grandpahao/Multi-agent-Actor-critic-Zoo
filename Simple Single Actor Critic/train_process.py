import os
from collections import deque

import numpy as np
from tensorboardX import SummaryWriter

from utils import Episode_Record, Memory, ResultsBuffer, checkpath


def single_ac_train(env,
                    actor,
                    critic,
                    store_path='./',
                    batch_size=32,
                    epsilon=0.01,
                    save_interval=1000,
                    update_interval=1000,
                    learning_starts=200,
                    memory_size=50000,
                    max_epoch=100000,
                    max_iter=10000):
    event_path = os.path.join(store_path, 'actor_events')
    actor_model_path = os.path.join(store_path, 'actor_models')
    critic_model_path = os.path.join(store_path, 'critic_models')
    checkpath(event_path)
    checkpath(actor_model_path)
    checkpath(critic_model_path)

    actor.load_model(actor_model_path)
    critic.load_model(critic_model_path)

    summary_writer = SummaryWriter(event_path)
    memory_buffer = Memory(memory_size)
    results_buffer = ResultsBuffer()

    states = env.reset()

    for i in range(max_epoch):
        states = env.reset()
        episode_buffer = Episode_Record()
        episode_buffer.append('state', states)
        while True:
            actions = actor.get_action(states, epsilon)
            next_states, rewards, dones, info = env.step(actions)

            episode_buffer.append('reward', rewards)
            episode_buffer.append('action', actions)

            if dones:
                state_batch, reward_batch, action_batch = episode_buffer.dump()

                score_batch = critic.get_target(state_batch)
                target_batch = np.zeros_like(reward_batch)
                target_batch[-1] = reward_batch[-1]
                for idx in range(len(reward_batch) - 2, -1, -1):
                    target_batch[idx] = reward_batch[idx] + \
                        0.95 * target_batch[idx + 1]
                global_step, critic_summary, advantage_batch = critic.update(
                    state_batch, target_batch)

                # advantage_batch = np.zeros_like(reward_batch)
                # R = 0.0
                # for idx in range(len(reward_batch) - 1, -1, -1):
                #     R = R * 0.95 + reward_batch[idx]
                #     advantage_batch[idx] = R
                # advantage_batch -= np.mean(advantage_batch)
                # advantage_batch /= np.std(advantage_batch)
                actor_summary = actor.update(
                    state_batch, action_batch, advantage_batch)
                # results_buffer.add_summary(summary_writer, global_step)
                actor.update_target()
                critic.update_target()
                # actor.save_model(actor_model_path, global_step)
                # critic.save_model(critic_model_path, global_step)
                print("Epoch {} earns a reward of {}.".format(
                    i, np.sum(reward_batch)))
                break
            else:
                episode_buffer.append('state', next_states)
                states = next_states
