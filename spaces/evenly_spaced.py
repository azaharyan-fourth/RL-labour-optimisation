from gym.spaces.space import Space
import numpy as np

class EvenlySpaced(Space):
    def __init__(self, start, stop, num, **kwargs):
        self.values = np.linspace(start, stop, num, **kwargs)
        self.n = int(num)
        super().__init__(self.values.shape, self.values.dtype)

    def sample(self):
        return np.random.choice(self.values)

    def contains(self, x):
        return x in self.values

    def __getitem__(self, key):
        assert key < len(self.values)

        return self.values[key]