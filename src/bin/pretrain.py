import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import time
import pickle
import glob
# Import custom stuff
from src.datasets import Cutout, CIFAR10, CIFAR100, SVHN, FashionMNIST, ImageNet, TinyImageNet
from src.models import ResNet18, ResNet50, Inception, enable_running_stats, disable_running_stats
from src.models import VGG19, MobileNetV3Small, MobileNetV3Large, InceptionV3, WRN2810
from src.optimizers import SAM, SGD, Adam
from src.optimizers.schedulers import GradualWarmupScheduler, CosineAnnealingWarmupRestarts, CustomLRSchedule
from .runner import Runner


class Pretrainer(Runner):
    def __init__(self, settings):
        super().__init__(settings=settings, experiment_name='pretrain_{}_over_{}_with_{}_sched_{}'.format(settings.model, settings.dataset, settings.optimizer, settings.lr_sched))
        self.gather_dataset()
        self.setup_model()
        self.setup_training()

    def save_model(self, model_name):
        models_folder = os.path.join(self.settings.models_folder, 'pretrain', self.settings.dataset, self.settings.model, 'seed_{}'.format(self.settings.seed))
        os.makedirs(models_folder, exist_ok=True)
        torch.save(self.model, os.path.join(models_folder, model_name))

    def gather_dataset(self):
        self.logger.print_it('Gathering dataset "{}". This may take a while...'.format(self.settings.dataset))
        # Image Preprocessing
        if self.settings.dataset in ['cifar10', 'cifar100']:
            normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                            std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            # Setup train transforms
            train_transform = transforms.Compose([])
            if self.settings.data_augmentation:
                train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
                train_transform.transforms.append(transforms.RandomHorizontalFlip())
            train_transform.transforms.append(transforms.ToTensor())
            train_transform.transforms.append(normalize)
            if self.settings.cutout_holes > 0:
                train_transform.transforms.append(
                    Cutout(n_holes=self.settings.cutout_holes, length=self.settings.cutout_length))
            # Setup test transforms
            test_transform = transforms.Compose([transforms.ToTensor(), normalize])
        elif self.settings.dataset == 'svhn':
            mean = [0.4377, 0.4438, 0.4728]
            std = [0.1980, 0.2010, 0.1970]
            train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        elif self.settings.dataset == 'fmnist':
            mean = [0.2861]
            std = [0.3530]
            train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        elif self.settings.dataset == 'imagenet':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            train_transform = transforms.Compose([])
            train_transform.transforms.append(transforms.RandomResizedCrop(224))
            if self.settings.data_augmentation:
                train_transform.transforms.append(transforms.RandomHorizontalFlip())
            train_transform.transforms.append(transforms.ToTensor())
            train_transform.transforms.append(normalize)
            test_transform = transforms.Compose([transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    normalize])
        elif self.settings.dataset == 'tiny_imagenet':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                             std=[0.229, 0.224, 0.225])
            # Setup train transforms for Tiny ImageNet
            train_transform = transforms.Compose([])
            if self.settings.data_augmentation:
                train_transform.transforms.append(transforms.RandomCrop(64, padding=4))
                train_transform.transforms.append(transforms.RandomHorizontalFlip())
            train_transform.transforms.append(transforms.ToTensor())
            train_transform.transforms.append(normalize)
            if self.settings.cutout_holes > 0:
                train_transform.transforms.append(
                    Cutout(n_holes=self.settings.cutout_holes, length=self.settings.cutout_length))
            # Setup test transforms for Tiny ImageNet
            test_transform = transforms.Compose([transforms.ToTensor(), normalize])
        else:
            raise ValueError('Dataset "{}" is not available!'.format(self.settings.dataset))

        # Load the appropriate train and test datasets
        if self.settings.dataset == 'cifar10':
            self.num_classes = 10
            self.im_size = (32,32)
            self.im_channels = 3
            root = os.path.join(self.settings.datasets_folder, 'cifar10')
            self.train_dataset = CIFAR10(root=root, train=True, transform=train_transform, download=True)
            self.test_dataset = CIFAR10(root=root, train=False, transform=test_transform, download=True)
        elif self.settings.dataset == 'cifar100':
            self.num_classes = 100
            self.im_size = (32,32)
            self.im_channels = 3
            root = os.path.join(self.settings.datasets_folder, 'cifar100')
            self.train_dataset = CIFAR100(root=root, train=True, transform=train_transform, download=True)
            self.test_dataset = CIFAR100(root=root, train=False, transform=test_transform, download=True)
        elif self.settings.dataset == 'svhn':
            self.im_channels = 3
            self.im_size = (32, 32)
            self.num_classes = 10
            root = os.path.join(self.settings.datasets_folder, 'svhn')
            self.train_dataset = SVHN(root=root, split='train', download=True, transform=train_transform)
            self.train_dataset.targets = self.train_dataset.labels
            self.test_dataset = SVHN(root=root, split='test', download=True, transform=test_transform)
            self.test_dataset.targets = self.test_dataset.labels
        elif self.settings.dataset == 'fmnist':
            self.im_channels = 1
            self.im_size = (28, 28)
            self.num_classes = 10
            root = os.path.join(self.settings.datasets_folder, 'fmnist')
            self.train_dataset = FashionMNIST(root=root, train=True, download=True, transform=train_transform)
            self.test_dataset = FashionMNIST(root=root, train=False, download=True, transform=test_transform)
        elif self.settings.dataset == 'imagenet':
            self.im_channels = 3
            self.im_size = (224, 224)
            self.num_classes = 1000
            root = os.path.join(self.settings.datasets_folder, 'imagenet')
            self.train_dataset = ImageNet(root=root, split='train', download=True, transform=train_transform)
            self.test_dataset = ImageNet(root=root, split='val', download=True, transform=test_transform)
        elif self.settings.dataset == 'tiny_imagenet':
            self.im_channels = 3
            self.im_size = (64, 64)
            self.num_classes = 200
            root = os.path.join(self.settings.datasets_folder, 'tiny_imagenet')
            self.train_dataset = TinyImageNet(root=root, train=True, transform=train_transform)
            self.test_dataset = TinyImageNet(root=root, train=False, transform=test_transform)
        else:
            raise ValueError('Dataset "{}" is not available!'.format(self.settings.dataset))
        
        # # Get indices of examples that should be used for training
        # self.train_indx = np.array(range(len(self.train_dataset.targets)))
        # # Reassign train data and labels
        # if self.settings.dataset == 'tiny_imagenet':
        #     self.train_dataset.data = self.train_dataset.data[self.train_indx]
        # elif self.settings.dataset == 'fmnist':
        #     self.train_dataset.data = self.train_dataset.data[self.train_indx, :, :]
        # else:
        #     self.train_dataset.data = self.train_dataset.data[self.train_indx, :, :, :]
        # # self.train_dataset.data = self.train_dataset.data[self.train_indx, :, :, :]
        # self.train_dataset.targets = np.array(self.train_dataset.targets)[self.train_indx].tolist()

        # # Get indices of examples that should be used for testing
        # self.test_indx = np.array(range(len(self.test_dataset.targets)))
        # # Reassign test data and labels
        # if self.settings.dataset == 'tiny_imagenet':
        #     self.test_dataset.data = self.test_dataset.data[self.test_indx]
        # elif self.settings.dataset == 'fmnist':
        #     self.test_dataset.data = self.test_dataset.data[self.test_indx, :, :]
        # else:
        #     self.test_dataset.data = self.test_dataset.data[self.test_indx, :, :, :]
        # # self.test_dataset.data = self.test_dataset.data[self.test_indx, :, :, :]
        # self.test_dataset.targets = np.array(self.test_dataset.targets)[self.test_indx].tolist()

        self.logger.print_it('Gathered dataset "{}":\tTraining samples = {} & Testing samples = {}'.format(self.settings.dataset,
                                                                                            len(self.train_dataset),
                                                                                            len(self.test_dataset)))

    def setup_model(self):
        self.logger.print_it('Setting up model "{}"...'.format(self.settings.model))
        # Setup model
        if self.settings.model == 'resnet18':
            model = ResNet18(channel=self.im_channels, num_classes=self.num_classes)
        elif self.settings.model == 'resnet50':
            model = ResNet50(channel=self.im_channels, num_classes=self.num_classes)
        elif self.settings.model == 'wideresnet':
            model = WRN2810(channel=self.im_channels, num_classes=self.num_classes, im_size=self.im_size)
            # model = WideResNet(depth=28, num_classes=self.num_classes, widen_factor=10, dropRate=0.3)
        elif self.settings.model == 'vgg19':
            model = VGG19(channel=self.im_channels, num_classes=self.num_classes, im_size=self.im_size)
        elif self.settings.model == 'mobile_small':
            model = MobileNetV3Small(channel=self.im_channels, num_classes=self.num_classes, im_size=self.im_size)
        elif self.settings.model == 'mobile_large':
            model = MobileNetV3Large(channel=self.im_channels, num_classes=self.num_classes, im_size=self.im_size)
        elif self.settings.model == 'inception':
            model = InceptionV3(channel=self.im_channels, num_classes=self.num_classes, im_size=self.im_size)
            # model = Inception(num_classes=self.num_classes)
        else:
            print('Specified model "{}" not recognized!'.format(self.settings.model))
        # Move model to device
        self.model = model.to(self.device)
        self.logger.print_it('Model setup done!')

    def setup_training(self):
        self.logger.print_it('Setting up training with "{}" optimizer...'.format(self.settings.optimizer))
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)

        # Setup optimizer
        if self.settings.optimizer == 'adam':
            self.optimizer = Adam(self.model.parameters(),
                                            lr=self.settings.learning_rate)
        elif self.settings.optimizer == 'sgd':
            self.optimizer = SGD(self.model.parameters(), 
                                            lr=self.settings.learning_rate,
                                            momentum=0.9,
                                            nesterov=False)    
            self.scheduler = CustomLRSchedule(self.optimizer)
        elif self.settings.optimizer == 'sam':
            self.optimizer = SAM(params=self.model.parameters(),
                                base_optimizer=SGD,
                                lr=self.settings.learning_rate,
                                momentum=0.9,
                                nesterov=False,
                                rho=0.05)
            self.scheduler = CustomLRSchedule(self.optimizer)  
        else:
            raise ValueError('Specified optimizer "{}" not supported. Options are: adam and sgd and sam'.format(self.settings.optimizer))

        self.logger.print_it('Training setup done!')

    def train(self):
        if self.settings.resume:
            self.load_last_resume_ckpt()
        else:
            # Initialize dictionary to save statistics for every example presentation
            self.best_acc = 0
            self.elapsed_time = 0
            self.logger.print_it('Starting training of "{}" for "{}" from scratch. This will take a while...'.format(self.settings.model,
                                                                                                                    self.settings.dataset))
            train_start_time = time.time()
            self.test_accs = []
            self.epoch = 1

        while(self.epoch <= self.settings.epochs):
            start_time = time.time()

            self.train_epoch()
            test_acc = self.test_epoch()
            self.test_accs.append(test_acc)

            epoch_time = time.time() - start_time
            self.elapsed_time += epoch_time
            h, m, s = self.convert_to_hms(self.elapsed_time)
            self.logger.print_it('Elapsed time for epoch {}: {}:{:02d}:{:02d}'.format(self.epoch,h,m,s))

            # Update optimizer step
            if self.settings.optimizer != 'adam':
                self.scheduler.step()

            self.save_model('epoch_{}.pt'.format(self.epoch))

            # Save checkpoint when best model
            if test_acc > self.best_acc:
                self.logger.print_it('New Best model at epoch {}: \t Top1-acc = {:.2f}'.format(self.epoch, test_acc*100))
                self.save_model('best.pt')
                self.best_acc = test_acc
                # self.best_model = self.model
            
            # Save model when last epoch
            if self.epoch == (self.settings.epochs):
                self.logger.print_it('Saving last model: \t Top1-acc = {:.2f}'.format(test_acc*100))
                self.save_model('last.pt')
                # self.last_model = self.model

            self.epoch += 1

            self.store_resume_ckpt()

        h, m, s = self.convert_to_hms(self.elapsed_time)
        self.logger.print_it('Training of "{}" for "{}" completed in: {}:{:02d}:{:02d}'.format(self.settings.model, self.settings.dataset, h, m, s))
        return self.test_accs

    def train_epoch(self):
        train_loss = 0.
        correct = 0.
        total = 0.

        self.model.train()

        # Get permutation to shuffle trainset
        trainset_permutation_inds = np.random.permutation(np.arange(len(self.train_dataset)))

        batch_size = self.settings.batch_size
        for batch_idx, batch_start_ind in enumerate(range(0, len(self.train_dataset), batch_size)):

            # Get trainset indices for batch
            batch_inds = trainset_permutation_inds[batch_start_ind: batch_start_ind + batch_size]

            # Get batch inputs and targets, transform them appropriately
            transformed_trainset = []
            for ind in batch_inds:
                transformed_trainset.append(self.train_dataset.__getitem__(ind)[0])
                if self.settings.dataset in ['cifar10', 'cifar100', 'imagenet', 'tiny_imagenet']:
                    targets.append(self.train_dataset.__getitem__(ind)[1])
                elif self.settings.dataset in ['svhn']:
                    targets.append(torch.tensor(self.train_dataset.__getitem__(ind)[1]))
            inputs = torch.stack(transformed_trainset)
            targets = torch.stack(targets)
            # targets = torch.LongTensor(np.array(self.train_dataset.targets)[batch_inds].tolist())

            # Map to available device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Compute loss and predictions
            if self.settings.optimizer=='sam':
                # first forward-backward step
                enable_running_stats(self.model)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.mean().backward()
                self.optimizer.first_step(zero_grad=True)
                # second forward-backward step
                disable_running_stats(self.model)
                self.criterion(self.model(inputs), targets).mean().backward()
                self.optimizer.second_step(zero_grad=True)  
            else:
                # Forward propagation, compute loss, get predictions
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            _, predicted = torch.max(outputs.data, 1)

            # Update loss, backward propagate, update optimizer
            loss = loss.mean()
            train_loss += loss.item()
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            if self.settings.optimizer != 'sam':
                loss.backward()
                self.optimizer.step()

            # Print message on console
            metrics = {'loss': train_loss / total,
                        'acc': correct.item() / total}
            self.print_message(epoch_index=self.epoch, total_epochs=self.settings.epochs,
                            index_batch=batch_idx+1, total_batches=len(self.train_dataset.targets) // batch_size + 1,
                            metrics=metrics, mode='train')

        self.logger.set_logger_newline()

    def test_epoch(self):
        test_loss = 0.
        correct = 0.
        total = 0.
        test_batch_size = 32

        self.model.eval()

        for batch_idx, batch_start_ind in enumerate(range(0, len(self.test_dataset), test_batch_size)):

            # Get batch inputs and targets
            transformed_testset = []
            for ind in range(batch_start_ind, min(len(self.test_dataset), batch_start_ind + test_batch_size)):
                transformed_testset.append(self.test_dataset.__getitem__(ind)[0])
                if self.settings.dataset in ['cifar10', 'cifar100', 'imagenet', 'tiny_imagenet']:
                    targets.append(self.test_dataset.__getitem__(ind)[1])
                elif self.settings.dataset in ['svhn']:
                    targets.append(torch.tensor(self.test_dataset.__getitem__(ind)[1]))
            inputs = torch.stack(transformed_testset)
            targets = torch.stack(targets)
            # targets = torch.LongTensor(np.array(self.test_dataset.targets)[batch_start_ind:batch_start_ind + test_batch_size].tolist())

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
            metrics = {'loss': test_loss / total,
                        'acc': correct.item() / total}
            self.print_message(epoch_index=self.epoch, total_epochs=self.settings.epochs,
                            index_batch=batch_idx+1, total_batches=len(self.test_dataset.targets) // test_batch_size + 1,
                            metrics=metrics, mode='test')
        self.logger.set_logger_newline()

        # Add test accuracy to dict
        acc = correct.item() / total
        return acc

    def print_message(self, epoch_index, total_epochs, index_batch, total_batches, metrics, mode='train'):
        message = '| EPOCH: {}/{} |'.format(epoch_index, total_epochs)
        bar_length = 10
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

    def store_resume_ckpt(self):
        state = {
                'epoch': self.epoch,
                'arch': self.settings.model,
                'self.best_acc': self.self.best_acc,
                'self.elapsed_time': self.self.elapsed_time,
                'self.test_accs': self.self.test_accs,
                'state_dict': self.model.state_dict(),
                'optimizer' : self.optimizer.state_dict(),
                'scheduler' : self.scheduler.state_dict(),
            }
        checkpoint_folder = os.path.join(self.settings.resume_ckpts_folder, self.experiment_name, 'seed_{}'.format(self.settings.seed))
        os.makedirs(checkpoint_folder, exist_ok=True)
        torch.save(state, os.path.join(checkpoint_folder, 'epoch={}.pth.tar'.format(self.epoch)))

    def load_last_resume_ckpt(self):
        checkpoint_folder = os.path.join(self.settings.resume_ckpts_folder, self.experiment_name, 'seed_{}'.format(self.settings.seed))
        checkpoints = glob.glob(os.path.join(checkpoint_folder, '*.pth.tar'))
        found_epochs = [int(check.split('epoch=')[-1].split('.pth.tar')[0]) for check in checkpoints]
        if found_epochs == []:
            raise FileNotFoundError('No checkpoints found!')
        last_epoch = max(found_epochs)
        checkpoint_to_load = os.path.join(self.settings.resume_ckpts_folder, self.experiment_name, 'seed_{}'.format(self.settings.seed), 'epoch={}.pth.tar'.format(last_epoch))
        # Read variables
        checkpoint = torch.load(checkpoint_to_load, map_location=self.device)
        self.epoch = checkpoint['epoch']
        self.self.best_acc = checkpoint['self.best_acc']
        self.self.elapsed_time = checkpoint['self.elapsed_time']
        self.self.test_accs = checkpoint['self.test_accs']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])