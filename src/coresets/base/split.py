import os
import math
import numpy as np
from torch.utils.data import Subset
from .coresetmethod import CoresetMethod

SPLITS = 10

class SplitReader(CoresetMethod):
    def __init__(self, dst_train, settings, valid_split):
        self.dst_train = dst_train
        self.n_train = len(dst_train)
        self.settings = settings
        self.valid_split = valid_split

    def select(self, return_indices=False):
        try:
            val_indices = np.loadtxt('{}/splits/{}/seed_{}/val_{}.txt'.format(self.settings.datasets_folder, self.settings.dataset, self.settings.seed, self.valid_split), dtype=int)
        except FileNotFoundError:

            if self.settings.dataset == 'cifar10':
                num_classes = 10
            elif self.settings.dataset == 'cifar100':
                num_classes = 100
            elif self.settings.dataset == 'svhn':
                num_classes = 10
            elif self.settings.dataset == 'fmnist':
                num_classes = 10
            elif self.settings.dataset == 'imagenet':
                num_classes = 1000
            elif self.settings.dataset == 'tiny_imagenet':
                num_classes = 200
            else:
                raise ValueError('Dataset "{}" is not available!'.format(self.settings.dataset))

            # Get indices for each class
            if self.settings.dataset == 'svhn':
                class_indices = [np.where(np.array(self.dst_train.labels) == i)[0] for i in range(num_classes)]
                indices_per_class_for_val = [math.floor(len(np.where(np.array(self.dst_train.labels) == i)[0]) / SPLITS) for i in range(num_classes)]
            else:
                class_indices = [np.where(np.array(self.dst_train.targets) == i)[0] for i in range(num_classes)]
                indices_per_class_for_val = [math.floor(len(np.where(np.array(self.dst_train.targets) == i)[0]) / SPLITS) for i in range(num_classes)]
            print('Generating validation splits...')
            r = np.random.RandomState(self.settings.seed)
            # Generate val1 to val10
            for val_index in range(1, SPLITS+1):
                # Sample 50 indices for each class for validation and keep their original indices
                if val_index == SPLITS:
                    indices_per_class_for_val = [len(class_indices[ind]) for ind in range(len(class_indices))]
                val_indices = np.concatenate([r.choice(class_indices[i], indices_per_class_for_val[i], replace=False) for i in range(num_classes)])

                # Save the validation indices
                print('length of split {}: {}'.format(val_index,len(val_indices)))
                os.makedirs('{}/splits/{}/seed_{}'.format(self.settings.datasets_folder, self.settings.dataset, self.settings.seed), exist_ok=True)
                np.savetxt('{}/splits/{}/seed_{}/val_{}.txt'.format(self.settings.datasets_folder, self.settings.dataset, self.settings.seed, val_index), val_indices, fmt='%d')

                # Exclude the used indices from available class indices
                for i in range(num_classes):
                    class_indices[i] = np.setdiff1d(class_indices[i], val_indices)
        
        train_indices = np.setdiff1d(np.arange(len(self.dst_train)), val_indices)
        self.weights = np.ones((len(train_indices.tolist())))
        if return_indices:
            return Subset(self.dst_train, train_indices), train_indices
        else:
            return Subset(self.dst_train, train_indices)
