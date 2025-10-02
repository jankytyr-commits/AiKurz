import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity:int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size:int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(lambda x: np.array(x), zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
