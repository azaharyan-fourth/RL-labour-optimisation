from collections import deque, namedtuple
import random
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        transition = Transition(*args)
        self.memory.append(transition)

        return transition

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def sample_batch_from_idxs(self, indices):
        array = [self.memory[i] for i in indices] # O(n) lookup -> very inefficient
        return array

    def __len__(self):
        return len(self.memory)
