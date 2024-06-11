import numpy as np
from torch.utils.data import Subset
from .coresetmethod import CoresetMethod


class Full(CoresetMethod):
    def __init__(self, dst_train):
        self.dst_train = dst_train
        self.n_train = len(dst_train)

    def select(self, return_indices=False):
        indices = np.arange(self.n_train)
        self.weights = np.ones((len(indices.tolist())))
        if return_indices:
            return Subset(self.dst_train, indices), indices
        else:
            return Subset(self.dst_train, indices)
