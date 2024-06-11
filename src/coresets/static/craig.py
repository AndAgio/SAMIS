from .earlytrain import EarlyTrain
import torch
from ..utils import FacilityLocation, submodular_optimizer
import numpy as np
import math
from ..utils.euclidean import euclidean_dist_pair_np
from torch.utils.data import Subset
from tqdm import tqdm


class Craig(EarlyTrain):
    def __init__(self, dst_train, fraction, seed=None, per_class=True, submodular_greedy="LazyGreedy", logger=None, ckpts_folder=None):
        super().__init__(dst_train=dst_train, fraction=fraction, seed=seed, logger=logger, ckpts_folder=ckpts_folder)

        if submodular_greedy not in submodular_optimizer.optimizer_choices:
            raise ModuleNotFoundError("Greedy optimizer not found.")
        self._greedy = submodular_greedy
        self.per_class = per_class
        self.weights = None

    def calc_gradient(self, model, loss_func, optimizer, index=None, batch_size=256):
        # Set model to evaluation mode
        model.eval()
        device = next(model.parameters()).device
        # Define batch loader
        batch_loader = torch.utils.data.DataLoader(
            self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
            batch_size=batch_size, num_workers=0)
        sample_num = len(self.dst_val.targets) if index is None else len(index)
        self.embedding_dim = model.get_last_layer().in_features
        # Keep gradients list
        gradients = []
        # Compute gradient over each batch
        for i, (input, targets) in enumerate(batch_loader):
            optimizer.zero_grad()
            outputs = model(input.to(device))
            loss = loss_func(outputs.requires_grad_(True), targets.to(device)).sum()
            batch_num = targets.shape[0]
            with torch.no_grad():
                bias_parameters_grads = torch.autograd.grad(loss, outputs)[0]
                weight_parameters_grads = model.embedding_recorder.embedding.view(batch_num, 1,
                                                                                       self.embedding_dim).repeat(1,
                                                                                                                  self.num_classes,
                                                                                                                  1) * bias_parameters_grads.view(
                    batch_num, self.num_classes, 1).repeat(1, 1, self.embedding_dim)
                gradients.append(
                    torch.cat([bias_parameters_grads, weight_parameters_grads.flatten(1)], dim=1).cpu().numpy())
        # Convert gradients to array
        gradients = np.concatenate(gradients, axis=0)
        # Reset model to train
        model.train()
        return euclidean_dist_pair_np(gradients)

    def calc_weights(self, matrix, result):
        min_sample = np.argmax(matrix[result], axis=0)
        weights = np.ones(np.sum(result) if result.dtype == bool else len(result))
        for i in min_sample:
            weights[i] = weights[i] + 1
        return weights

    def select(self, model, loss_func, optimizer, scheduler, batch_size, epochs, return_indices=False):
        self.run(model, loss_func, optimizer, scheduler, batch_size, epochs)

        self.model.no_grad = True
        with self.model.embedding_recorder:
            print('Selecting using craig...')
            if self.per_class:
                # Do selection by class
                selection_result = np.array([], dtype=np.int32)
                weights = np.array([])
                for c in tqdm(range(self.num_classes), ascii=True, desc='CRAIG computing gradients and submodular optimizer for each class', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
                    class_index = np.arange(self.n_train)[self.dst_train.targets == c]
                    matrix = -1. * self.calc_gradient(self.model, loss_func, optimizer, class_index, batch_size)
                    matrix -= np.min(matrix) - 1e-3
                    submod_function = FacilityLocation(index=class_index, similarity_matrix=matrix)
                    submod_optimizer = submodular_optimizer.__dict__[self._greedy](index=class_index,
                                                                                   budget=math.ceil(self.fraction * len(
                                                                                       class_index)))
                    class_result = submod_optimizer.select(gain_function=submod_function.calc_gain,
                                                           update_state=submod_function.update_state)
                    selection_result = np.append(selection_result, class_result)
                    weights = np.append(weights, self.calc_weights(matrix, np.isin(class_index, class_result)))
            else:
                print('CRAIG computing gradients and submodular optimizer over whole dataset...')
                matrix = np.zeros([self.n_train, self.n_train])
                all_index = np.arange(self.n_train)
                for c in tqdm(range(self.num_classes), ascii=True, desc='CRAIG processing dataset classes'):
                    class_index = np.arange(self.n_train)[self.dst_train.targets == c]
                    matrix[np.ix_(class_index, class_index)] = -1. * self.calc_gradient(self.model, loss_func, optimizer, class_index, batch_size)
                    matrix[np.ix_(class_index, class_index)] -= np.min(matrix[np.ix_(class_index, class_index)]) - 1e-3
                submod_function = FacilityLocation(index=all_index, similarity_matrix=matrix)
                submod_optimizer = submodular_optimizer.__dict__[self._greedy](index=all_index,
                                                                               budget=self.coreset_size)
                selection_result = submod_optimizer.select(gain_function=submod_function.calc_gain_batch,
                                                           update_state=submod_function.update_state,
                                                           batch=batch_size)
                weights = self.calc_weights(matrix, selection_result)
        self.weights = weights
        self.model.no_grad = False
        print('selection_result: {}'.format(selection_result))
        if return_indices:
            return Subset(self.dst_train, selection_result), selection_result
        else:
            return Subset(self.dst_train, selection_result)
