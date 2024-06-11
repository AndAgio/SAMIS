import torch
import numpy as np

from .runner import Runner


class Tester(Runner):
    def __init__(self, settings):
        super().__init__(settings=settings)

    def test(self, dataset, mode='best'):
        if mode == 'best':
            self.load_best_model()
        elif mode == 'last':
            self.load_last_model()
        else:
            raise ValueError('Mode "{}" for running Tester.test() is not available!'.format(mode))

        correct = 0.
        total = 0.
        test_batch_size = 32

        self.model.eval()
        self.logger.print_it('Testing of "{}" for "{}"...'.format(self.settings.model, self.settings.dataset))

        for batch_idx, batch_start_ind in enumerate(range(0, len(dataset.targets), test_batch_size)):

            # Get batch inputs and targets
            transformed_testset = []
            for ind in range(batch_start_ind, min(len(dataset.targets), batch_start_ind + test_batch_size)):
                transformed_testset.append(dataset.__getitem__(ind)[0])
            inputs = torch.stack(transformed_testset)
            targets = torch.LongTensor(np.array(dataset.targets)[batch_start_ind:batch_start_ind + test_batch_size].tolist())

            # Map to available device
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward propagation, compute loss, get predictions
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss = loss.mean()
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            # Print message on console
            metrics = {'loss': loss / total,
                        'acc': correct.item() / total}
            self.print_message(index_batch=batch_idx+1, 
                            total_batches=len(self.train_dataset.targets) // test_batch_size + 1,
                            metrics=metrics, mode='test')

        # Add test accuracy to dict
        loss = test_loss / total
        acc = correct.item() / total
        
        self.logger.print_it('Test of {} over {} gave: Loss = {}\tTop1-Acc = {}%'.format(self.settings.model,
                                                                                        self.settings.dataset,
                                                                                        loss,
                                                                                        acc * 100))

        return acc
    

    def print_message(self, index_batch, total_batches, metrics, mode='test'):
        bar_length = 20
        progress = float(index_batch) / float(total_batches)
        if progress >= 1.:
            progress = 1
        block = int(round(bar_length * progress))
        message += '[{}]'.format('=' * block + ' ' * (bar_length - block))
        message += '| {}: '.format(mode.upper())
        if metrics is not None:
            train_metrics_message = ''
            index = 0
            for metric_name, metric_value in metrics.items():
                train_metrics_message += '{}={:.5f}{} '.format(metric_name, metric_value,
                                                            ',' if index < len(metrics.keys()) - 1 else '')
                index += 1
            message += train_metrics_message
        message += '|'
        self.logger.print_it_same_line(message)