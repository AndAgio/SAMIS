from .earlytrain import EarlyTrain
from ..utils import submodular_optimizer
import torch
from torch.utils.data import Subset
import numpy as np
import math
from ..base.uniform import Uniform
from tqdm import tqdm


class Glister(EarlyTrain):
    def __init__(self, dst_train, dst_val, fraction=0.5, seed=None, per_class=True, greedy="LazyGreedy", eta=0.1, logger=None, ckpts_folder=None):
        super().__init__(dst_train=dst_train, fraction=fraction, seed=seed, logger=logger, ckpts_folder=ckpts_folder)
        if dst_val is None or dst_train == dst_val:
            raise ValueError('Glister requires to have validation dataset!')
        self.dst_val = dst_val
        self.dst_val.targets = torch.LongTensor(np.array(self.dst_val.targets).tolist())
        self.val_num_classes = self.num_classes
        self.n_val = len(self.dst_val)
        self.per_class = per_class
        self.eta = eta
        if greedy not in submodular_optimizer.optimizer_choices:
            raise ModuleNotFoundError("Greedy optimizer not found.")
        self._greedy = greedy

    def calc_gradient(self, model, loss_func, optimizer, index=None, batch_size=256, val=False, record_val_detail=False):
        '''
        Calculate gradients matrix on current network for training or validation dataset.
        '''

        model.eval()

        if val:
            batch_loader = torch.utils.data.DataLoader(
                self.dst_val if index is None else torch.utils.data.Subset(self.dst_val, index),
                batch_size=batch_size, num_workers=0)
        else:
            batch_loader = torch.utils.data.DataLoader(
                self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
                batch_size=batch_size, num_workers=0)

        self.embedding_dim = model.get_last_layer().in_features
        gradients = []
        if val and record_val_detail:
            self.init_out = []
            self.init_emb = []
            self.init_y = []

        device = next(model.parameters()).device

        for i, (input, targets) in enumerate(batch_loader):
            optimizer.zero_grad()
            outputs = model(input.to(device))
            loss = loss_func(outputs.requires_grad_(True), targets.to(device)).sum()
            batch_num = targets.shape[0]
            with torch.no_grad():
                bias_parameters_grads = torch.autograd.grad(loss, outputs)[0]
                weight_parameters_grads = model.embedding_recorder.embedding.view(batch_num, 1,
                                                self.embedding_dim).repeat(1, self.num_classes, 1) *\
                                                bias_parameters_grads.view(
                                                batch_num, self.num_classes, 1).repeat(1, 1, self.embedding_dim)
                gradients.append(torch.cat(
                    [bias_parameters_grads, weight_parameters_grads.flatten(1)], dim=1).cpu())

                if val and record_val_detail:
                    self.init_out.append(outputs.cpu())
                    self.init_emb.append(model.embedding_recorder.embedding.cpu())
                    self.init_y.append(targets)

        gradients = torch.cat(gradients, dim=0)
        if val:
            self.val_grads = torch.mean(gradients, dim=0)
        else:
            self.train_grads = gradients
        if val and record_val_detail:
            with torch.no_grad():
                self.init_out = torch.cat(self.init_out, dim=0)
                self.init_emb = torch.cat(self.init_emb, dim=0)
                self.init_y = torch.cat(self.init_y)

        model.train()

    def update_val_gradients(self, new_selection, selected_for_train, selection_batch=256):

        sum_selected_train_gradients = torch.mean(self.train_grads[selected_for_train], dim=0)

        new_outputs = self.init_out - self.eta * sum_selected_train_gradients[:self.num_classes].view(1,
                      -1).repeat(self.init_out.shape[0], 1) - self.eta * torch.matmul(self.init_emb,
                      sum_selected_train_gradients[self.num_classes:].view(self.num_classes, -1).T)

        sample_num = new_outputs.shape[0]
        gradients = torch.zeros([sample_num, self.num_classes * (self.embedding_dim + 1)], requires_grad=False)
        i = 0
        while i * selection_batch < sample_num:
            batch_indx = np.arange(sample_num)[i * selection_batch:min((i + 1) * selection_batch,
                                                                                 sample_num)]
            new_out_puts_batch = new_outputs[batch_indx].clone().detach().requires_grad_(True)
            loss = self.criterion(new_out_puts_batch, self.init_y[batch_indx])
            batch_num = len(batch_indx)
            bias_parameters_grads = torch.autograd.grad(loss.sum(), new_out_puts_batch, retain_graph=True)[0]

            weight_parameters_grads = self.init_emb[batch_indx].view(batch_num, 1, self.embedding_dim).repeat(1,
                                      self.num_classes, 1) * bias_parameters_grads.view(batch_num,
                                      self.num_classes, 1).repeat(1, 1, self.embedding_dim)
            gradients[batch_indx] = torch.cat([bias_parameters_grads, weight_parameters_grads.flatten(1)], dim=1).cpu()
            i += 1

        self.val_grads = torch.mean(gradients, dim=0)

    def select(self, model, loss_func, optimizer, scheduler, batch_size, epochs, return_indices=False):
        self.run(model, loss_func, optimizer, scheduler, batch_size, epochs)
        
        self.model.embedding_recorder.record_embedding = True
        self.model.no_grad = True

        self.train_indx = np.arange(self.n_train)
        self.val_indx = np.arange(self.n_val)
        if self.per_class:
            selection_result = np.array([], dtype=np.int64)
            #weights = np.array([], dtype=np.float32)
            for c in tqdm(range(self.num_classes), ascii=True, desc='GLISTER computing gradients and submodular optimizer for each class', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
                c_indx = self.train_indx[self.dst_train.targets == c]
                c_val_inx = self.val_indx[self.dst_val.targets == c]
                self.calc_gradient(self.model, loss_func, optimizer, index=c_val_inx, batch_size=batch_size, val=True, record_val_detail=True)
                self.calc_gradient(self.model, loss_func, optimizer, index=c_indx, batch_size=batch_size)
                submod_optimizer = submodular_optimizer.__dict__[self._greedy](index=c_indx,
                                                            budget=math.ceil(self.fraction * len(c_indx)))
                c_selection_result = submod_optimizer.select(gain_function=lambda idx_gain, selected,
                                                             **kwargs: torch.matmul(self.train_grads[idx_gain],
                                                             self.val_grads.view(-1, 1)).detach().cpu().numpy().
                                                             flatten(), upadate_state=self.update_val_gradients)
                selection_result = np.append(selection_result, c_selection_result)

        else:
            print('GLISTER computing gradients and submodular optimizer over whole dataset...')
            self.calc_gradient(self.model, loss_func, optimizer, batch_size=batch_size, val=True, record_val_detail=True)
            self.calc_gradient(self.model, loss_func, optimizer, batch_size=batch_size)

            submod_optimizer = submodular_optimizer.__dict__[self._greedy](index=np.arange(self.n_train), budget=self.coreset_size)
            selection_result = submod_optimizer.select(gain_function=lambda idx_gain, selected,
                                                       **kwargs: torch.matmul(self.train_grads[idx_gain],
                                                       self.val_grads.view(-1, 1)).detach().cpu().numpy().flatten(),
                                                       upadate_state=self.update_val_gradients)

        self.model.embedding_recorder.record_embedding = False
        self.model.no_grad = False
        self.weights = np.ones((len(selection_result.tolist())))
        if return_indices:
            return Subset(self.dst_train, selection_result), selection_result
        else:
            return Subset(self.dst_train, selection_result)
