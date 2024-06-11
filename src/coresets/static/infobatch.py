from .earlytrain import EarlyTrain
import torch, time
from tqdm import tqdm
import math
import numpy as np
from torch.utils.data import Subset


class InfoBatch(EarlyTrain):
    def __init__(self, dst_train, fraction=0.5, seed=None, dst_test=None, per_class=True, logger=None, ckpts_folder=None):
        super().__init__(dst_train=dst_train, fraction=fraction, seed=seed, dst_test=dst_test, logger=logger, ckpts_folder=ckpts_folder)

        self.per_class = per_class

    
    def compute_metrics(self, model, loss_func, optimizer, scheduler, batch_size=256):
        model.embedding_recorder.record_embedding = True  # recording embedding vector

        model.eval()
        device = next(model.parameters()).device

        batch_loader = torch.utils.data.DataLoader(
            self.dst_train, batch_size=batch_size, num_workers=0)
        sample_num = self.n_train

        for i, (input, targets) in enumerate(batch_loader):
            optimizer.zero_grad()
            outputs = model(input.to(device))
            loss = loss_func(outputs.requires_grad_(True), targets.to(device)).sum()

            self.scores[i * batch_size:min((i + 1) * batch_size, sample_num)] = loss.detach().cpu().numpy()

        model.train()

        model.embedding_recorder.record_embedding = False

    def select(self, model, loss_func, optimizer, scheduler, batch_size, epochs, return_indices=False):
        self.train_indx = np.arange(self.n_train)
        init_model, init_loss_func, init_optimizer, init_scheduler = model, loss_func, optimizer, scheduler

        # Initialize a matrix to save norms of each sample on idependent runs
        device = next(model.parameters()).device
        self.scores = torch.zeros([self.n_train], requires_grad=False).to(device).detach().cpu().numpy()

        self.run(init_model, init_loss_func, init_optimizer, init_scheduler, batch_size, epochs)
        self.compute_metrics(self.model, loss_func, optimizer, scheduler, batch_size)
        # self.scores = self.scores.cpu().detach().numpy()

        if not self.per_class:
            print('InfoBatch computing gradients and top examples over whole dataset...')
            loss_mean = np.mean(self.scores)
            print('loss mean: {}'.format(loss_mean))
            prob_val = 1./np.sum((self.scores < loss_mean))
            probabilities = prob_val * (self.scores < loss_mean)
            print('probabilities: {}'.format(probabilities))
            top_examples = np.random.choice(self.train_indx, self.coreset_size, p=probabilities)
        else:
            top_examples = np.array([], dtype=np.int64)
            for c in tqdm(range(self.num_classes), ascii=True, desc='InfoBatch computing gradients and top examples for each class', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
                c_indx = self.train_indx[self.dst_train.targets == c]
                budget = math.ceil(self.fraction * len(c_indx))
                # top_examples = np.append(top_examples, c_indx[np.argsort(self.scores[c_indx])[::-1][:budget]])

                loss_mean = np.mean(self.scores[c_indx])
                print('loss mean: {}'.format(loss_mean))
                prob_val = 1./np.sum((self.scores[c_indx] < loss_mean))
                probabilities = prob_val * (self.scores[c_indx] < loss_mean)
                print('probabilities: {}'.format(probabilities))
                top_examples = np.append(top_examples, np.random.choice(c_indx, budget, p=probabilities))
        self.weights = np.ones((len(top_examples.tolist())))
        if return_indices:
            return Subset(self.dst_train, top_examples), top_examples
        else:
            return Subset(self.dst_train, top_examples)
