from .earlytrain import EarlyTrain
import numpy as np
import torch
from tqdm import tqdm
import math
from ..utils import cossim_np, submodular_function, submodular_optimizer
from torch.utils.data import Subset
import time


class GraphCut(EarlyTrain):
    def __init__(self,  dst_train, fraction=0.5, seed=None, per_class=True, greedy="NaiveGreedy", metric="cossim", logger=None, ckpts_folder=None):
        super(GraphCut, self).__init__(dst_train=dst_train, fraction=fraction, seed=seed, logger=logger, ckpts_folder=ckpts_folder)

        if greedy not in submodular_optimizer.optimizer_choices:
            raise ModuleNotFoundError("Greedy optimizer not found.")
        self._greedy = greedy
        self._metric = metric
        self._function = "GraphCut"

        self.per_class = per_class

    def calc_gradient(self, model, loss_func, optimizer, index=None, batch_size=256):
        '''
        Calculate gradients matrix on current network for specified training dataset.
        '''
        model.eval()
        device = next(model.parameters()).device
        self.logger.print_it('device: {}'.format(device))

        batch_loader = torch.utils.data.DataLoader(
                self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
                batch_size=batch_size,
                num_workers=0)
        sample_num = self.n_train if index is None else len(index)

        self.embedding_dim = model.get_last_layer().in_features

        # Initialize a matrix to save gradients.
        # (on cpu)
        gradients = []

        for i, (input, targets) in enumerate(batch_loader):
            optimizer.zero_grad()
            outputs = model(input.to(device))
            loss = loss_func(outputs.requires_grad_(True), targets.to(device)).sum()
            batch_num = targets.shape[0]
            with torch.no_grad():
                bias_parameters_grads = torch.autograd.grad(loss, outputs)[0]
                weight_parameters_grads = model.embedding_recorder.embedding.view(batch_num, 1,
                                        self.embedding_dim).repeat(1, self.num_classes, 1) *\
                                        bias_parameters_grads.view(batch_num, self.num_classes,
                                        1).repeat(1, 1, self.embedding_dim)
                gradients.append(torch.cat([bias_parameters_grads, weight_parameters_grads.flatten(1)],
                                            dim=1).cpu().numpy())

        gradients = np.concatenate(gradients, axis=0)
        return gradients

    def select(self, model, loss_func, optimizer, scheduler, batch_size, epochs, return_indices=False):
        self.run(model, loss_func, optimizer, scheduler, batch_size, epochs)

        # Turn on the embedding recorder and the no_grad flag
        with self.model.embedding_recorder:
            self.model.no_grad = True
            self.train_indx = np.arange(self.n_train)

            if self.per_class:
                selection_result = np.array([], dtype=np.int64)
                self.logger.print_it('GraphCut computing gradients and submodular function for each class...')
                for c in range(self.num_classes):
                # for c in tqdm(range(self.num_classes), ascii=True, desc='GraphCut computing gradients and submodular function for each class', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
                    self.logger.print_it('class {} out of {}...'.format(c+1, self.num_classes))
                    c_indx = self.train_indx[self.dst_train.targets == c]
                    # Calculate gradients into a matrix
                    self.logger.print_it('Computing gradient')
                    grad_start = time.time()
                    gradients = self.calc_gradient(self.model, loss_func, optimizer, c_indx, batch_size)
                    self.logger.print_it('Gradient was computed in {:.3f}'.format(time.time() - grad_start))
                    # Instantiate a submodular function
                    self.logger.print_it('Starting to run submodular function')
                    submod_start = time.time()
                    submod_function = submodular_function.__dict__[self._function](index=c_indx,
                                        similarity_kernel=lambda a, b:cossim_np(gradients[a], gradients[b]))
                    submod_optimizer = submodular_optimizer.__dict__[self._greedy](index=c_indx, budget=math.ceil(self.fraction * len(c_indx)), already_selected=[])
                    c_selection_result = submod_optimizer.select(gain_function=submod_function.calc_gain,
                                                                 update_state=submod_function.update_state)
                    self.logger.print_it('Submod function was computed in {:.3f}'.format(time.time() - submod_start))
                    selection_result = np.append(selection_result, c_selection_result)
            else:
                self.logger.print_it('GraphCut computing gradients and submodular function over whole dataset...')
                # Calculate gradients into a matrix
                gradients = self.calc_gradient(self.model, loss_func, optimizer, batch_size)
                # Instantiate a submodular function
                submod_function = submodular_function.__dict__[self._function](index=self.train_indx,
                                            similarity_kernel=lambda a, b: cossim_np(gradients[a], gradients[b]))
                submod_optimizer = submodular_optimizer.__dict__[self._greedy](index=self.train_indx,
                                                                                  budget=self.coreset_size)
                selection_result = submod_optimizer.select(gain_function=submod_function.calc_gain,
                                                           update_state=submod_function.update_state)
        self.model.no_grad = False
        self.weights = np.ones((len(selection_result.tolist())))
        if return_indices:
            return Subset(self.dst_train, selection_result), selection_result
        else:
            return Subset(self.dst_train, selection_result)



