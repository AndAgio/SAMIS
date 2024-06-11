from ..base.coresetmethod import CoresetMethod
import torch, time
from torch import nn
import numpy as np
from copy import deepcopy
import os


class EarlyTrain(CoresetMethod):
    '''
    Core code for training related to coreset selection methods when pre-training is required.
    '''

    def __init__(self, dst_train, fraction=0.5, seed=None, dst_test=None, logger=None, ckpts_folder=None):
        super().__init__(dst_train=dst_train, fraction=fraction, seed=seed, logger=logger)
        self.dst_test = dst_test
        self.ckpts_folder = ckpts_folder
        if dst_test is not None:
            self.dst_test.targets = torch.LongTensor(np.array(dst_test.targets).tolist())
            self.n_test = len(dst_test)
        else:
            self.n_test = 0

    def train_epoch(self, batch_size, epoch, epochs):
        """ Train model for one epoch """

        train_loss = 0.
        correct = 0.
        total = 0.
        total_time = 0.
        self.model.train()
        device = next(self.model.parameters()).device

        # Get permutation to shuffle trainset
        trainset_permutation_inds = np.random.permutation(np.arange(len(self.dst_train)))

        for batch_idx, batch_start_ind in enumerate(range(0, len(self.dst_train), batch_size)):

            # Get trainset indices for batch
            batch_inds = trainset_permutation_inds[batch_start_ind: batch_start_ind + batch_size]

            # Get batch inputs and targets, transform them appropriately
            transformed_trainset = []
            targets = []
            for ind in batch_inds:
                transformed_trainset.append(self.dst_train.__getitem__(ind)[0])
                targets.append(self.dst_train.__getitem__(ind)[1])
            inputs = torch.stack(transformed_trainset)
            targets = torch.stack(targets)

            inputs, targets = inputs.to(device), targets.to(device)

            # Forward propagation, compute loss, get predictions
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_func(outputs, targets)

            samples_indices = [np.array(range(len(self.dst_train.targets)))[index] for index in batch_inds]
            self.after_loss(outputs, loss, targets, samples_indices, epoch)

            # Update loss, backward propagate, update optimizer
            loss = loss.mean()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            loss.backward()
            self.optimizer.step()

            # Print message on console
            metrics = {'loss': train_loss / total,
                        'acc': correct.item() / total}
            self.logger.print_train_test_message(index_batch=batch_idx+1, 
                                                 total_batches=len(self.dst_train.targets) // batch_size + 1,
                                                 metrics=metrics, 
                                                 mode='train',
                                                 epoch_index=epoch+1,
                                                 total_epochs=epochs,
                                                 pre_msg='PRETRAIN')
        self.logger.set_logger_newline()
        self.finish_train()

    def run(self, model, loss_func, optimizer, scheduler, batch_size=256, epochs=100):
        available = self.check_model_available()
        if available:
            self.load_last_model()
        else:
            self._run(model, loss_func, optimizer, scheduler, batch_size, epochs)

    def _run(self, model, loss_func, optimizer, scheduler, batch_size=256, epochs=100, save=True):
            self.model = model
            self.loss_func = loss_func
            self.optimizer = optimizer
            self.scheduler = scheduler

            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
            self.train_indx = np.arange(self.n_train)

            self.before_run(device=next(self.model.parameters()).device)
            for epoch in range(epochs):
                self.before_epoch()

                self.train_epoch(batch_size, epoch, epochs)
                if self.dst_test is not None:
                    self.test_acc = self.test_epoch(epoch, epochs)
                
                self.after_epoch()
                if save:
                    self.save_model('epoch_{}.pt'.format(epoch))

            self.finish_run()
            if save:
                self.save_model('last.pt')

    def test_epoch(self, epoch, epochs):
        self.model.no_grad = True
        self.model.eval()
        device = next(self.model.parameters()).device

        test_loss = 0.
        correct = 0.
        total = 0.
        test_batch_size = 32

        for batch_idx, batch_start_ind in enumerate(range(0, len(self.dst_test.targets), test_batch_size)):

            # Get batch inputs and targets
            transformed_testset = []
            for ind in range(batch_start_ind, min(len(self.dst_test.targets), batch_start_ind + test_batch_size)):
                transformed_testset.append(self.dst_test.__getitem__(ind)[0])
            inputs = torch.stack(transformed_testset)
            targets = torch.LongTensor(np.array(self.dst_test.targets)[batch_start_ind:batch_start_ind + test_batch_size].tolist())

            # Map to available device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward propagation, compute loss, get predictions
            outputs = self.model(inputs)
            loss = self.loss_func(outputs, targets)
            loss = loss.mean()
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            self.total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            # Print message on console
            metrics = {'loss': test_loss / total,
                        'acc': correct.item() / total}
            self.logger.print_train_test_message(index_batch=batch_idx+1, 
                                                 total_batches=len(self.dst_test.targets) // test_batch_size + 1,
                                                 metrics=metrics, 
                                                 mode='test',
                                                 epoch_index=epoch+1,
                                                 total_epochs=epochs,
                                                 pre_msg='PRETRAIN')
        self.logger.set_logger_newline()
        self.model.no_grad = False

    def convert_to_hms(self, seconds):
        # Format time for printing purposes
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return int(h), int(m), int(s)

    def after_loss(self, outputs, loss, targets, batch_inds, epoch):
        # with torch.no_grad():
        #     _, predicted = torch.max(outputs.data, 1)
        #     cur_acc = (predicted == targets).clone().detach().requires_grad_(False).type(torch.float32)
        #     self.forgetting_events[torch.tensor(batch_inds)[(self.last_acc[batch_inds]-cur_acc)>0.01]]+=1.
        #     self.last_acc[batch_inds] = cur_acc
        pass

    def before_epoch(self):
        self.epoch_start_time = time.time()

    def after_epoch(self):
        epoch_time = time.time() - self.epoch_start_time
        total_time = time.time() - self.train_start_time
        h, m, s = self.convert_to_hms(epoch_time)
        self.logger.print_it('Elapsed time for epoch: {}:{:02d}:{:02d}'.format(h,m,s))
        h, m, s = self.convert_to_hms(total_time)
        self.logger.print_it('Cumulative elapsed time: {}:{:02d}:{:02d}'.format(h,m,s))
        # Save checkpoint when best model
        try:
            self.test_accs.append(self.test_acc)
            if self.test_acc > self.best_acc:
                self.logger.print_it('New Best model: \t Top1-acc = {:.2f}'.format(self.test_acc*100))
                self.best_acc = self.test_acc
        except AttributeError:
            pass

    def before_run(self, device):
        self.best_acc = 0
        self.logger.print_it('Starting pretraining to find coreset. This will take a while...')
        self.train_start_time = time.time()
        self.test_accs = []

        # self.forgetting_events = torch.zeros(self.n_train, requires_grad=False).to(self.args.device)
        # self.last_acc = torch.zeros(self.n_train, requires_grad=False).to(self.args.device)

    def finish_run(self):
        try:
            self.logger.print_it('Last model: \t Top1-acc = {:.2f}'.format(self.test_acc*100))
        except AttributeError:
            pass
        h, m, s = self.convert_to_hms(time.time() - self.train_start_time)
        self.logger.print_it('Pretrain completed in: {}:{:02d}:{:02d}'.format(h, m, s))

    def finish_train(self):
        pass        

    def select(self, model, loss_func, optimizer, scheduler, batch_size=256, epochs=100):
        ...

    def save_model(self, model_name):
        os.makedirs(self.ckpts_folder, exist_ok=True)
        torch.save(self.model, os.path.join(self.ckpts_folder, model_name))
    
    def load_best_model(self):
        self.model = torch.load(os.path.join(self.ckpts_folder, 'best.pt'))
        
    def load_last_model(self):
        self.model = torch.load(os.path.join(self.ckpts_folder, 'last.pt'))
    
    def check_model_available(self):
        if os.path.exists(os.path.join(self.ckpts_folder, 'last.pt')):
            return True
        else:
            return False