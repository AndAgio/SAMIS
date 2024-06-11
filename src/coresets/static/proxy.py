from ..base.coresetmethod import CoresetMethod
import torch, time
import math
import numpy as np
from torch.utils.data import Subset
from src.datasets import CIFAR10, CIFAR100, SVHN, FashionMNIST
from src.metrics import Proxy


# Acknowledgement to
# https://github.com/mtoneva/example_forgetting

class ProxyCoreset(CoresetMethod):
    def __init__(self, dst_train, mode, coreset_mode='few_shot', fraction=0.5, seed=None, per_class=True, logger=None):
        super().__init__(dst_train=dst_train, fraction=fraction, seed=seed, logger=logger)
        self.mode = mode
        self.coreset_mode = coreset_mode
        self.per_class = per_class
        if isinstance(dst_train, CIFAR100):
            self.dataset_name = 'cifar100'
        elif isinstance(dst_train, CIFAR10):
            self.dataset_name = 'cifar10'
        elif isinstance(dst_train, SVHN):
            self.dataset_name = 'svhn'
        elif isinstance(dst_train, FashionMNIST):
            self.dataset_name = 'fmnist'
        else:
            raise ValueError('Dataset type not recognized!')
        self.proxy_obj = Proxy(mode=mode, dataset=self.dataset_name)
        self.sorted_proxy_dict = self.proxy_obj.get_dict_form(sort=True)
        self.sorted_indices = list(self.sorted_proxy_dict.keys())
        self.sorted_scores = list(self.sorted_proxy_dict.values())

    def scale_sigmoid(self, index):
        sample_sigmoid_val = 1 / (1 + math.pow(math.e, 8 * (self.sorted_proxy_dict[index] - 0.5)))
        return sample_sigmoid_val
    
    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def select(self, model, loss_func, optimizer, scheduler, batch_size, epochs, return_indices=False):
        train_indx = np.arange(self.n_train)
        if self.per_class:
            sampled_indices = np.array([], dtype=np.int64)
            for c in range(self.num_classes):
                c_indx = train_indx[self.dst_train.targets == c]
                class_index = np.arange(self.n_train)[self.dst_train.targets == c]
                class_budget = math.ceil(self.fraction * len(class_index))
                # class_budget = int(self.coreset_size/self.num_classes)
                class_scores_dict = {idx: score for idx, score in self.sorted_proxy_dict.items() if idx in c_indx}
                if self.coreset_mode == 'few_shot':
                    class_samples = train_indx[list(class_scores_dict.keys())[:class_budget]]
                elif self.coreset_mode == 'large':
                    class_samples = train_indx[list(class_scores_dict.keys())[::-1][:class_budget]]
                else:
                    raise ValueError('Coreset mode "{}" not available!'.format(self.coreset_mode))
                sampled_indices = np.append(sampled_indices, class_samples)
        else:
            if self.coreset_mode == 'few_shot':
                sampled_indices = train_indx[list(self.sorted_proxy_dict.keys())[:self.coreset_size]]
            elif self.coreset_mode == 'large':
                sampled_indices = train_indx[list(self.sorted_proxy_dict.keys())[::-1][:self.coreset_size]]
            else:
                raise ValueError('Coreset mode "{}" not available!'.format(self.coreset_mode))
            
        self.weights = np.ones((len(sampled_indices.tolist())))
        sampled_scores = [self.sorted_proxy_dict[ind] for ind in sampled_indices.tolist()]
        print('Number of samples: {} -> Min {} score: {} -> Max {} score: {}'.format(len(sampled_scores),
                                                                                     self.mode,
                                                                                       np.min(sampled_scores),
                                                                                       self.mode,
                                                                                       np.max(sampled_scores)))
        if return_indices:
            return Subset(self.dst_train, sampled_indices), sampled_indices
        else:
            return Subset(self.dst_train, sampled_indices)
