import os
import numpy as np

# Global variable used to store the random seed
globalRandomSeed = None

class TrueRandomGenerator:
    """
    This class is meant for handling a random generator that is both a true and reproducible. If initialized without
    the seed argument the initial seed is chosen with a true random function and saved as an attribute. After the
    initialization alla the following calls are reproducible.
    """
    def __init__(self, seed=None):
        if seed is None:
            seed = int.from_bytes(os.urandom(4))  # 4-bytes int generated with a true random function
        self.initialSeed = seed
        self.randomState = np.random.RandomState(seed)

    def uniform(self, low=None, high=None, size=None):
        return self.randomState.uniform(low=low, high=high, size=size)

    def int(self, low, high=None, size=None, dtype=int):
        return self.randomState.randint(low=low, high=high, size=size, dtype=dtype)
