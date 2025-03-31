import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def push(self, experience):
        """Push (state, action, reward, next_state) to the buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer.pop(0)  # Remove the oldest experience
            self.buffer.append(experience)

    def sample(self, batch_size):
        """Sample a batch of experiences."""
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)
