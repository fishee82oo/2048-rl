from collections import deque
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state.copy(), action, reward, next_state.copy(), done))

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        batch = random.sample(self.buffer, batch_size)
        return tuple(np.asarray(values) for values in zip(*batch))

    def __len__(self) -> int:
        return len(self.buffer)
