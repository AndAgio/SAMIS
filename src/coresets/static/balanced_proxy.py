from ..base.coresetmethod import CoresetMethod
import torch, time
import math
import numpy as np
from torch.utils.data import Subset
from src.datasets import CIFAR10, CIFAR100, SVHN, FashionMNIST
from src.metrics import Proxy


# Acknowledgement to
# https://github.com/mtoneva/example_forgetting

class BalancedProxyCoreset(CoresetMethod):
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
        self.sampled_indices = None

    def scale_sigmoid(self, index):
        sample_sigmoid_val = 1 / (1 + math.pow(math.e, 8 * (self.sorted_proxy_dict[index] - 0.5)))
        return sample_sigmoid_val
    
    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def select(self, model, loss_func, optimizer, scheduler, batch_size, epochs):
        train_indx = np.arange(self.n_train)
        if self.per_class:
            sampled_indices = np.array([], dtype=np.int64)
            for c in range(self.num_classes):
                c_indx = train_indx[self.dst_train.targets == c]
                if self.coreset_mode == 'few_shot':
                    pass
                elif self.coreset_mode == 'large':
                    c_indx = c_indx[::-1]
                else:
                    raise ValueError('Coreset mode "{}" not available!'.format(self.coreset_mode))
                probs = [self.scale_sigmoid(index) for index in c_indx]
                probs = self.softmax(np.asarray(probs))
                class_index = np.arange(self.n_train)[self.dst_train.targets == c]
                class_budget = math.ceil(self.fraction * len(class_index))
                class_samples = np.random.choice(c_indx, class_budget, p=probs)
                sampled_indices = np.append(sampled_indices, class_samples)
        else:
            if self.coreset_mode == 'few_shot':
                c_indx = train_indx
            elif self.coreset_mode == 'large':
                c_indx = train_indx[::-1]
            else:
                raise ValueError('Coreset mode "{}" not available!'.format(self.coreset_mode))
            probs = [self.scale_sigmoid(index) for index in c_indx]
            probs = self.softmax(np.asarray(probs))
            sampled_indices = np.random.choice(c_indx, self.coreset_size, p=probs)
        self.weights = np.ones((len(sampled_indices.tolist())))
        sampled_scores = [self.sorted_proxy_dict[ind] for ind in sampled_indices.tolist()]
        print('Number of samples: {} -> Min {} score: {} -> Max {} score: {}'.format(len(sampled_scores),
                                                                                     self.mode,
                                                                                       np.min(sampled_scores),
                                                                                       self.mode,
                                                                                       np.max(sampled_scores)))
        self.sampled_indices = sampled_indices
        return Subset(self.dst_train, sampled_indices)

    def get_weights(self, epoch_fract=None):
        if epoch_fract is None:
            self.weights = np.asarray([math.exp(-5*self.sorted_proxy_dict[ind]) for ind in self.sampled_indices.tolist()])
            # self.weights = np.asarray([math.cos(math.pi/2*(self.sorted_proxy_dict[ind])) for ind in self.sampled_indices.tolist()])
        else:
            assert 0 <= epoch_fract <= 1
            self.weights = np.asarray([(self.sorted_proxy_dict[ind]+(1-epoch_fract))*((1-self.sorted_proxy_dict[ind])+epoch_fract) * ((self.sorted_proxy_dict[ind] - 0.5) ** 2 + (epoch_fract - 0.5) ** 2 + 0.5) for ind in self.sampled_indices.tolist()])
        return self.weights