import torch
import numpy as np
from src.utils.log import DumbLogger
from src.datasets import CIFAR10, CIFAR100, SVHN, FashionMNIST


class CoresetMethod(object):
    def __init__(self, dst_train, fraction, seed=None, logger=None):
        if fraction <= 0.0 or fraction > 1.0:
            raise ValueError("Illegal Coreset Size.")
        self.dst_train = dst_train
        self.dst_train.targets = torch.LongTensor(np.array(dst_train.targets).tolist())
        # self.num_classes = len(dst_train.classes)
        if isinstance(dst_train, CIFAR100):
            self.num_classes = 100
        elif isinstance(dst_train, CIFAR10):
            self.num_classes = 10
        elif isinstance(dst_train, SVHN):
            self.num_classes = 10
        elif isinstance(dst_train, FashionMNIST):
            self.num_classes = 10
        else:
            raise ValueError('Dataset type not recognized!')
        self.fraction = fraction
        self.random_seed = seed
        self.index = []

        self.n_train = len(dst_train)
        self.coreset_size = round(self.n_train * fraction)
        self.weights = np.ones((self.coreset_size,))
        
        self.logger = logger if logger is not None else DumbLogger()

    def select(self):
        ...

    def get_weights(self):
        return self.weights

