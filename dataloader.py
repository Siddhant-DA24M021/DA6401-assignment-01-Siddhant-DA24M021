import numpy as np


class Dataloader:
    def __init__(self, X, y, batch_size = 32, shuffle = False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = X.shape[0]
        self.indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.seed(42)
            np.random.shuffle(self.indices)

        
    def __iter__(self):
        self.current_index = 0
        if self.shuffle:
            np.random.seed(42)
            np.random.shuffle(self.indices)
        return self
    
    def __next__(self):
        if self.current_index >= self.n_samples:
            raise StopIteration
        
        sample_indices = self.indices[self.current_index: min(self.current_index + self.batch_size, self.n_samples-1)]
        batch_X = self.X[sample_indices]
        batch_y = self.y[sample_indices]

        self.current_index += self.batch_size
        return batch_X, batch_y