import numpy as np
from torch.utils.data import Subset
from .coresetmethod import CoresetMethod


class Uniform(CoresetMethod):
    def __init__(self, dst_train, fraction=0.5, seed=None, per_class=False, replace=False, logger=None):
        super().__init__(dst_train=dst_train, fraction=fraction, seed=seed, logger=logger)
        self.per_class = per_class
        self.replace = replace
        self.n_train = len(dst_train)

    def select_per_class(self):
        """The same sampling proportions were used in each class separately."""
        np.random.seed(self.random_seed)
        self.index = np.array([], dtype=np.int64)
        all_index = np.arange(self.n_train)
        for c in range(self.num_classes):
            c_index = (self.dst_train.targets == c)
            self.index = np.append(self.index,
                                   np.random.choice(all_index[c_index], round(self.fraction * c_index.sum().item()),
                                                    replace=self.replace))
        return self.index

    def select_no_per_class(self):
        np.random.seed(self.random_seed)
        self.index = np.random.choice(np.arange(self.n_train), round(self.n_train * self.fraction),
                                      replace=self.replace)

        return self.index

    def select(self, return_indices=False):
        indices = self.select_per_class() if self.per_class else self.select_no_per_class()
        self.weights = np.ones((len(indices.tolist())))
        if return_indices:
            return Subset(self.dst_train, indices), indices
        else:
            return Subset(self.dst_train, indices)
