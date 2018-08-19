import os

from tensorboardX import SummaryWriter

from utils import Memory, ResultsBuffer, checkpath


def single_ac_train(env,
                    actor,
                    critic,
                    store_path='./',
                    batch_size=32,
                    epsilon=0.01,
                    save_interval=1000,
                    update_interval=1000,
                    learning_starts=200,
                    memory_size=500000,
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
    memory = Memory(memory_size)
    results_buffer = ResultsBuffer()

    states = env.reset()
    for i in range(learning_starts):
        actions = actor.get_action(states, epsilon)
        next_states, rewards, dones, info = env.step(actions)
        memory_buffer.extend(zip(states, actions, rewards, next_states, dones))
        states = next_states

    states = env.reset()

    for i in range(max_epoch):
        states = env.reset()
        total_reward = 0.0
        for j in range(num_iterations):
            actions = actor.get_action(states, epsilon)
            next_states, rewards, dones, info = env.step(actions)
            total_reward += rewards
            memory_buffer.extend(
                zip(states, actions, rewards, next_states, dones))
            cur_batch, action_batch, reward_batch, next_batch, done_batch = memory_buffer.critic_sample(
                batch_size)
            global_step, critic_summaries, ad_batch = critic.update(
                cur_batch, action_batch, reward_batch, next_batch, done_batch)

            _, actor_summaries = actor.update(
                cur_batch, action_batch, ad_batch)

            summaries = dict(critic_summaries, **actor_summaries)
            results_buffer.update_summaries(summaries)

            if global_step % update_interval:
                actor.update_target()
                critic.update_target()

            if global_step % save_interval:
                actor.save_model(actor_model_path)
                critic.save_model(critic_model_path)
                results_buffer.add_summary(summary_writer, global_step)

            if dones:
                print("Epoch {} earns a reward of {}".format(i, total_reward))
            else:
                states = next_states
