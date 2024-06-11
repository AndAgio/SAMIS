from .earlytrain import EarlyTrain
import torch, time
import numpy as np
from torch.utils.data import Subset


# Acknowledgement to
# https://github.com/mtoneva/example_forgetting

class ForgettingCoreset(EarlyTrain):
    def __init__(self, dst_train, fraction=0.5, seed=None, dst_test=None, per_class=True, logger=None):
        super().__init__(dst_train=dst_train, fraction=fraction, seed=seed, dst_test=dst_test, logger=logger)

        self.per_class = per_class

    def after_loss(self, outputs, loss, targets, batch_inds, epoch):
        with torch.no_grad():
            _, predicted = torch.max(outputs.data, 1)

            cur_acc = (predicted == targets).clone().detach().requires_grad_(False).type(torch.float32)
            self.forgetting_events[torch.tensor(batch_inds)[(self.last_acc[batch_inds]-cur_acc)>0.01]]+=1.
            self.last_acc[batch_inds] = cur_acc

    def before_run(self, device):
        self.best_acc = 0
        self.logger.print_it('Starting pretraining to find coreset. This will take a while...')
        self.train_start_time = time.time()
        self.test_accs = []

        self.forgetting_events = torch.zeros(self.n_train, requires_grad=False).to(device)
        self.last_acc = torch.zeros(self.n_train, requires_grad=False).to(device)

    def select(self, model, loss_func, optimizer, scheduler, batch_size, epochs, return_indices=False):
        self.run(model, loss_func, optimizer, scheduler, batch_size, epochs)

        if not self.per_class:
            top_examples = self.train_indx[np.argsort(self.forgetting_events.cpu().numpy())][::-1][:self.coreset_size]
        else:
            top_examples = np.array([], dtype=np.int64)
            for c in range(self.num_classes):
                c_indx = self.train_indx[self.dst_train.targets == c]
                budget = round(self.fraction * len(c_indx))
                top_examples = np.append(top_examples,
                                    c_indx[np.argsort(self.forgetting_events[c_indx].cpu().numpy())[::-1][:budget]])

        if return_indices:
            return Subset(self.dst_train, top_examples), top_examples
        else:
            return Subset(self.dst_train, top_examples)
