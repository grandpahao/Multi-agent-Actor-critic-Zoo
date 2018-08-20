import os
import random
from collections import defaultdict, deque

import numpy as np


def checkpath(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Memory(object):
    def __init__(self, capacity):
        self.mem = deque(maxlen=capacity)

    def append(self, transition):
        self.mem.append(transition)

    def extend(self, transitions):
        for t in transitions:
            self.append(t)

    def sample(self, batch_size):
        samples = random.sample(self.mem, batch_size)
        return map(np.array, zip(*samples))


class Episode_Record(object):
    def __init__(self):
        self.states = deque()
        self.actions = deque()
        self.rewards = deque()

    def append(self, type, transition):
        if type == 'state':
            self.states.append(transition)
        elif type == 'action':
            self.actions.append(transition)
        elif type == 'reward':
            self.rewards.append(transition)

    def dump(self):
        return map(np.array, [self.states, self.rewards, self.actions])


class ResultsBuffer(object):
    def __init__(self, rewards_history=[]):
        self.buffer = defaultdict(list)
        assert isinstance(rewards_history, list)
        self.rewards_history = rewards_history

    def update_infos(self, info, total_t):
        for key in info:
            msg = info[key]
            if b'real_reward' in msg:
                self.buffer['reward'].append(msg[b'real_reward'])
                self.buffer['length'].append(msg[b'real_length'])
                self.rewards_history.append(
                    [total_t, key, msg[b'real_reward']])

    def update_summaries(self, summaries):
        for key in summaries:
            self.buffer[key].append(summaries[key])

    def add_summary(self, summary_writer, total_t):
        s = {}
        for key in self.buffer:
            if self.buffer[key]:
                s[key] = np.mean(self.buffer[key])
                self.buffer[key].clear()

        for key in s:
            summary_writer.add_scalar(key, s[key], total_t)
