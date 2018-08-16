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
                    total_iter=10000000):
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

    for i in range(num_iterations):
        actions = actor.get_action(states, epsilon)
        next_states, rewards, dones, info = env.step(actions)
        memory_buffer.extend(zip(states, actions, rewards, next_states, dones))

        critic.update(*memory_buffer.sample(batch_size))
