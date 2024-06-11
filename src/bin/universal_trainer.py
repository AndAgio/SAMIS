import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import math
import time
import pickle
import glob
import copy
import random
# Import custom stuff
from src.datasets import Cutout, CIFAR10, CIFAR100, SVHN, FashionMNIST, ImageNet, TinyImageNet, DatasetWrapper
from src.models import ResNet18, ResNet50, enable_running_stats, disable_running_stats
from src.models import VGG19, MobileNetV3Small, MobileNetV3Large, InceptionV3, WRN2810, ViT
from src.optimizers import SAM, SGD, Adam
from src.optimizers.schedulers import GradualWarmupScheduler, CosineAnnealingWarmupRestarts, CustomLRSchedule
from src.metrics import HessianTraceCurvature, EpsilonFlatness, Forgettability, EpochsToLearn
from src.coresets import StatCraig, StatGlister, GraNd, EL2N, InfoBatch, GraphCut, ProxyCoreset, Uniform, Full, SplitReader
from src.utils import get_logger


class UniversalTrainer():
    def __init__(self, settings, experiment_name=None):
        self.settings = settings
        if experiment_name is None:
            self.experiment_name = '{}_over_{}'.format(self.settings.model, self.settings.dataset)
        else:
            self.experiment_name = experiment_name
        self.logger = get_logger(arguments=self.settings, experiment_name=self.experiment_name)

        if self.settings.distributed_training:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.global_rank = int(os.environ["RANK"])
        else:
            self.local_rank = self.settings.device if torch.cuda.is_available() else 'cpu'
            self.global_rank = self.settings.device if torch.cuda.is_available() else 'cpu'

        self.set_seed()
        self.setup_device()

    def setup_device(self):
        # Set appropriate devices
        if self.settings.distributed_training:
            dev_str = 'cuda:{}'.format(self.local_rank)
            self.device = torch.device(dev_str)
        else:
            cuda_available = torch.cuda.is_available()
            if cuda_available and self.settings.device != 'cpu':
                dev_str = 'cuda:{}'.format(self.settings.device)
                self.device = torch.device(dev_str)
                self.logger.print_it('Runner is setup using CUDA enabled device: {}'.format(torch.cuda.get_device_name(dev_str)))
            else:
                self.device = torch.device('cpu')
                self.logger.print_it('Runner is setup using CPU! Training and inference will be very slow!')
            cudnn.benchmark = True  # Should make training go faster for large models

    def set_seed(self):
        # Set random seed for initialization
        torch.manual_seed(self.settings.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.settings.seed)
        np.random.seed(self.settings.seed)
    
    def save_model(self, model_name):
        models_folder = os.path.join(self.settings.models_folder, self.experiment_name, 'seed_{}'.format(self.settings.seed))
        os.makedirs(models_folder, exist_ok=True)
        torch.save(self.model, os.path.join(models_folder, model_name))
    
    def load_best_model(self):
        models_folder = os.path.join(self.settings.models_folder, self.experiment_name, 'seed_{}'.format(self.settings.seed))
        self.model = torch.load(os.path.join(models_folder, 'best.pt'))
        self.model = self.model.to(self.device)
    
    def load_last_model(self):
        models_folder = os.path.join(self.settings.models_folder, self.experiment_name, 'seed_{}'.format(self.settings.seed))
        self.model = torch.load(os.path.join(models_folder, 'last.pt'))
        self.model = self.model.to(self.device)

    def setup_model(self):
        self.logger.print_it('Setting up model "{}"...'.format(self.settings.model))
        # Setup model
        if self.settings.model == 'resnet18':
            model = ResNet18(channel=self.im_channels, num_classes=self.num_classes)
        elif self.settings.model == 'resnet50':
            model = ResNet50(channel=self.im_channels, num_classes=self.num_classes)
        elif self.settings.model == 'wideresnet':
            model = WRN2810(channel=self.im_channels, num_classes=self.num_classes, im_size=self.im_size)
        elif self.settings.model == 'vgg19':
            model = VGG19(channel=self.im_channels, num_classes=self.num_classes, im_size=self.im_size)
        elif self.settings.model == 'mobile_small':
            model = MobileNetV3Small(channel=self.im_channels, num_classes=self.num_classes, im_size=self.im_size)
        elif self.settings.model == 'mobile_large':
            model = MobileNetV3Large(channel=self.im_channels, num_classes=self.num_classes, im_size=self.im_size)
        elif self.settings.model == 'inception':
            model = InceptionV3(channel=self.im_channels, num_classes=self.num_classes, im_size=self.im_size)
        elif self.settings.model == 'vit':
            model = ViT(pretrained=True, in_channels=self.im_channels, num_classes=self.num_classes, image_size=self.im_size)
        else:
            print('Specified model "{}" not recognized!'.format(self.settings.model))
        # Move model to device
        if self.settings.distributed_training:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        else:
            self.model = model.to(self.device)
        self.logger.print_it('Model setup done!')

    def setup_loss(self):
        self.logger.print_it('Setting up crossentropy loss...')
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)

    def setup_optimizer(self):
        self.logger.print_it('Setting up "{}" optimizer...'.format(self.settings.optimizer))
        if self.settings.optimizer == 'adam':
            self.optimizer = Adam(self.model.parameters(),
                                            lr=self.settings.learning_rate)
        elif self.settings.optimizer == 'sgd':
            self.optimizer = SGD(self.model.parameters(), 
                                            lr=self.settings.learning_rate,
                                            momentum=0.9,
                                            nesterov=False)
        elif self.settings.optimizer == 'sam':
            self.optimizer = SAM(params=self.model.parameters(),
                                base_optimizer=SGD,
                                lr=self.settings.learning_rate,
                                momentum=0.9,
                                nesterov=False,
                                rho=0.05)
        else:
            raise ValueError('Specified optimizer "{}" not supported. Options are: adam and sgd and sam'.format(self.settings.optimizer))

    def setup_lr_scheduler(self):
        self.logger.print_it('Setting up "{}" learning rate scheduler...'.format(self.settings.lr_sched))
        if self.settings.lr_sched == 'const':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=1)
        elif self.settings.lr_sched == 'warmup_step':
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=math.ceil(self.settings.epochs/3), gamma=0.1)
            self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=math.ceil(self.settings.epochs/40), after_scheduler=scheduler)
            self.scheduler.step()
        elif self.settings.lr_sched == 'warmup_exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
            self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=math.ceil(self.settings.epochs/40), after_scheduler=scheduler)
            self.scheduler.step()
        elif self.settings.lr_sched == 'warmup_cosine':
            cycle_steps = math.ceil(self.settings.epochs/5)
            warmup_steps = math.ceil(cycle_steps/10)
            max_lr=self.settings.lr
            min_lr=max_lr/100
            self.scheduler = CosineAnnealingWarmupRestarts(self.optimizer, first_cycle_steps=cycle_steps, cycle_mult=1.0, max_lr=max_lr, min_lr=min_lr, warmup_steps=warmup_steps, gamma=0.5)
        elif self.settings.lr_sched == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=math.ceil(self.settings.epochs/3), gamma=0.1)
        elif self.settings.lr_sched == 'exp':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
        elif self.settings.lr_sched == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.settings.epochs)
        elif self.settings.lr_sched == 'custom':
            self.scheduler = CustomLRSchedule(self.optimizer)
        else:
            raise ValueError('Learning rate scheduler "{}" not available!'.format(self.settings.lr_sched))

    def setup_training(self):
        self.logger.print_it('Setting up training...')
        self.setup_loss()
        self.setup_optimizer()
        self.setup_lr_scheduler()
        self.logger.print_it('Training setup done!')

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

        # Get indices of examples that should be used for training
        self.train_indx = np.array(range(len(self.train_dataset)))

        # Get indices of examples that should be used for testing
        self.test_indx = np.array(range(len(self.test_dataset)))

        self.logger.print_it('Gathered dataset "{}":\tTraining samples = {} '
                            '& Testing samples = {}'.format(self.settings.dataset,
                                                            len(self.train_dataset),
                                                            len(self.test_dataset)))

    def define_loaders(self):
        if self.settings.distributed_training:
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.settings.batch_size,
                                            pin_memory=True, shuffle=False,
                                            sampler=DistributedSampler(self.train_dataset))
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.settings.batch_size,
                                            pin_memory=True, shuffle=False,
                                            sampler=DistributedSampler(self.test_dataset))
        else:
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.settings.batch_size, shuffle=True)
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.settings.batch_size, shuffle=False)

    def wrap_dataset(self, indices=None, weights=None):
        if indices is None:
            indices = np.arange(self.n_train)
        if weights is None:
            weights = np.ones((len(indices.tolist())))
        self.train_dataset = DatasetWrapper(Subset(self.train_dataset, indices), sample_indices=indices, sample_weights=weights)

    def setup_coreset_selector(self):
        if self.settings.coreset_func in ['glister']:
            # Get indices of examples that should be used for training
            full_train_indx = np.array(range(len(self.train_dataset)))
            samp = set(random.sample(range(len(full_train_indx)), k=int(len(full_train_indx)*0.9)))
            self.train_indx = np.array([v for i,v in enumerate(full_train_indx) if i in samp])
            self.val_indx = np.array([v for i,v in enumerate(full_train_indx) if i not in samp])
            self.val_dataset = Subset(self.train_dataset, self.val_indx)
            self.train_dataset = Subset(self.train_dataset, self.train_indx)

        ckpts_folder = os.path.join(self.settings.models_folder,
                                    'pretrain',
                                    self.settings.dataset,
                                    self.settings.model,
                                    'seed_{}'.format(self.settings.seed))
        
        if self.settings.coreset_func == 'full':
            self.coreset_selector = Full(dst_train=self.train_dataset)
        elif self.settings.coreset_func == 'split':
            self.coreset_selector = SplitReader(dst_train=self.train_dataset,
                                                settings=self.settings, 
                                                valid_split=self.settings.samis_validation_split_index)
        elif self.settings.coreset_func == 'craig':
            self.coreset_selector = StatCraig(dst_train=self.train_dataset,
                                            fraction=self.settings.coreset_fraction,
                                            seed=self.settings.seed,
                                            per_class=self.settings.per_class,
                                            logger=self.logger,
                                            ckpts_folder=ckpts_folder)
        elif self.settings.coreset_func == 'glister':
            self.coreset_selector = StatGlister(dst_train=self.train_dataset,
                                            dst_val=self.val_dataset, 
                                            fraction=self.settings.coreset_fraction,
                                            seed=self.settings.seed,
                                            per_class=self.settings.per_class,
                                            logger=self.logger,
                                            ckpts_folder=ckpts_folder)
        elif self.settings.coreset_func == 'graph_cut':
            self.coreset_selector = GraphCut(dst_train=self.train_dataset,
                                            fraction=self.settings.coreset_fraction,
                                            seed=self.settings.seed,
                                            per_class=self.settings.per_class,
                                            logger=self.logger,
                                            ckpts_folder=ckpts_folder)
        elif self.settings.coreset_func == 'grand':
            self.coreset_selector = GraNd(dst_train=self.train_dataset,
                                        fraction=self.settings.coreset_fraction,
                                        seed=self.settings.seed,
                                        per_class=self.settings.per_class,
                                        logger=self.logger,
                                        ckpts_folder=ckpts_folder)
        elif self.settings.coreset_func == 'el2n':
            self.coreset_selector = EL2N(dst_train=self.train_dataset,
                                        fraction=self.settings.coreset_fraction,
                                        seed=self.settings.seed,
                                        per_class=self.settings.per_class,
                                        logger=self.logger,
                                        ckpts_folder=ckpts_folder)
        elif self.settings.coreset_func == 'infobatch':
            self.coreset_selector = InfoBatch(dst_train=self.train_dataset,
                                        fraction=self.settings.coreset_fraction,
                                        seed=self.settings.seed,
                                        per_class=self.settings.per_class,
                                        logger=self.logger,
                                        ckpts_folder=ckpts_folder)
        elif self.settings.coreset_func == 'random':
            self.coreset_selector = Uniform(dst_train=self.train_dataset,
                                            fraction=self.settings.coreset_fraction,
                                            seed=self.settings.seed,
                                            per_class=self.settings.per_class,
                                            logger=self.logger)
        elif self.settings.coreset_func == 'min_mem':
            self.coreset_selector = ProxyCoreset(dst_train=self.train_dataset,
                                                mode='mem',
                                                coreset_mode='large',
                                                fraction=self.settings.coreset_fraction,
                                                seed=self.settings.seed,
                                                per_class=self.settings.per_class,
                                                logger=self.logger)
        elif self.settings.coreset_func == 'min_forg':
            self.coreset_selector = ProxyCoreset(dst_train=self.train_dataset,
                                                mode='forg',
                                                coreset_mode='large',
                                                fraction=self.settings.coreset_fraction,
                                                seed=self.settings.seed,
                                                per_class=self.settings.per_class,
                                                logger=self.logger)
        elif self.settings.coreset_func == 'min_flat':
            self.coreset_selector = ProxyCoreset(dst_train=self.train_dataset,
                                                mode='flat',
                                                coreset_mode='large',
                                                fraction=self.settings.coreset_fraction,
                                                seed=self.settings.seed,
                                                per_class=self.settings.per_class,
                                                logger=self.logger)
        elif self.settings.coreset_func == 'min_eps':
            self.coreset_selector = ProxyCoreset(dst_train=self.train_dataset,
                                                mode='eps',
                                                coreset_mode='large',
                                                fraction=self.settings.coreset_fraction,
                                                seed=self.settings.seed,
                                                per_class=self.settings.per_class,
                                                logger=self.logger)
        elif self.settings.coreset_func == 'min_etl':
            self.coreset_selector = ProxyCoreset(dst_train=self.train_dataset,
                                                mode='etl',
                                                coreset_mode='large',
                                                fraction=self.settings.coreset_fraction,
                                                seed=self.settings.seed,
                                                per_class=self.settings.per_class,
                                                logger=self.logger)
        elif self.settings.coreset_func == 'min_sam-sgd-loss':
            self.coreset_selector = ProxyCoreset(dst_train=self.train_dataset,
                                                mode='samis_loss',
                                                coreset_mode='large',
                                                fraction=self.settings.coreset_fraction,
                                                seed=self.settings.seed,
                                                per_class=self.settings.per_class,
                                                logger=self.logger)
        elif self.settings.coreset_func == 'min_sam-sgd-prob':
            self.coreset_selector = ProxyCoreset(dst_train=self.train_dataset,
                                                mode='samis_prob',
                                                coreset_mode='large',
                                                fraction=self.settings.coreset_fraction,
                                                seed=self.settings.seed,
                                                per_class=self.settings.per_class,
                                                logger=self.logger)
        else:
            raise ValueError('Coreset contructor "{}" is not available!'.format(self.settings.coreset_func))

    def prune_dataset(self):
        self.logger.print_it('Pruning dataset depending on experiment context...')
        self.setup_coreset_selector()
        pretrain_epochs = self.settings.coreset_pretrain_epochs
        if self.settings.coreset_func in ['random', 'full', 'split']:
            _, indices = self.coreset_selector.select(return_indices=True)
        else:
            model = copy.deepcopy(self.model)
            optimizer = SGD(model.parameters(), 
                            lr=self.settings.coreset_pretrain_lr,
                            momentum=0.9,
                            nesterov=True,
                            weight_decay=5e-4)
            if self.settings.coreset_pretrain_lr_sched == 'const':
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
            elif self.settings.coreset_pretrain_lr_sched == 'step':
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=math.ceil(pretrain_epochs/3), gamma=0.1)
            else:
                raise ValueError('Learning rate scheduler "{}" not available!'.format(self.settings.coreset_pretrain_lr_sched))
            criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)
            _, indices = self.coreset_selector.select(model=model,
                                                    loss_func=criterion,
                                                    optimizer=optimizer,
                                                    scheduler=scheduler,
                                                    batch_size=self.settings.coreset_pretrain_batch_size,
                                                    epochs=pretrain_epochs,
                                                    return_indices=True)
        self.logger.print_it('Selected indices: {}'.format(indices))
        weights = self.coreset_selector.get_weights()
        self.wrap_dataset(indices=indices, weights=weights)
        self.define_loaders()
        self.logger.print_it('Pruned dataset "{}":\tTraining samples = {} '
                            '& Testing samples = {}'.format(self.settings.dataset,
                                                            len(self.train_dataset),
                                                            len(self.test_dataset)))

    def initialize_train(self):
        self.logger.print_it('Initializing training...')
        self.gather_dataset()
        self.setup_model()
        self.setup_training()
        self.prune_dataset()
        self.epoch = 1
        self.best_acc = 0.
        self.elapsed_time = 0.
        self.test_accs = []
        if self.settings.measure_proxies:
            self.setup_proxy_metrics()
        self.logger.print_it('Training initialization completed!')

    def train(self):
        while(self.epoch <= self.settings.epochs):
            start_time = time.time()

            self.train_epoch()
            test_acc = self.test_epoch()
            self.test_accs.append(test_acc)

            epoch_time = time.time() - start_time
            self.elapsed_time += epoch_time
            h, m, s = UniversalTrainer.convert_to_hms(self.elapsed_time)
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
            
            # Save model when last epoch
            if self.epoch == (self.settings.epochs):
                self.logger.print_it('Saving last model: \t Top1-acc = {:.2f}'.format(test_acc*100))
                self.save_model('last.pt')

            if self.local_rank == 0 or not self.settings.distributed_training:
                    self.store_resume_ckpt()

            self.epoch += 1

        h, m, s = UniversalTrainer.convert_to_hms(self.elapsed_time)
        self.logger.print_it('Training of "{}" for "{}" completed in: {}:{:02d}:{:02d}'.format(self.settings.model, self.settings.dataset, h, m, s))

        return self.test_accs

    def train_epoch(self):
        self.train_loss = 0.
        self.train_correct = 0.
        self.train_total = 0.

        self.model.train()

        if self.settings.distributed_training:
            self.train_loader.sampler.set_epoch(self.epoch)
        
        for batch_idx, (inputs, targets, indices, weights) in enumerate(self.train_loader):
            self.train_step(inputs, targets, indices, weights, batch_idx=batch_idx, total_batches=len(self.train_loader))

        self.logger.set_logger_newline()

    def train_step(self, inputs, targets, indices, weights, batch_idx=0, total_batches=0):
        # Map to available device
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        weights = weights.to(self.device)

        # Compute loss and predictions
        if self.settings.optimizer=='sam':
            # first forward-backward step
            enable_running_stats(self.model)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss = torch.sum(loss * weights) / torch.sum(weights)
            loss.backward()
            self.optimizer.first_step(zero_grad=True)
            # second forward-backward step
            disable_running_stats(self.model)
            second_loss = self.criterion(self.model(inputs), targets)
            second_loss = torch.sum(second_loss * weights) / torch.sum(weights)
            second_loss.mean().backward()
            self.optimizer.second_step(zero_grad=True)                 
            
        else:
            # Forward propagation, compute loss, get predictions
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss = torch.sum(loss * weights) / torch.sum(weights)
        _, predicted = torch.max(outputs.data, 1)

        if self.settings.measure_proxies:
            # Update forgettability and epochs to learn metrics
            samples_indices = indices
            self.forgettability_met.update_history(outputs=outputs,
                                                targets=targets,
                                                samples_indices=samples_indices,
                                                mode='train')
            self.epochs_to_learn_met.update_history(outputs=outputs,
                                                targets=targets,
                                                samples_indices=samples_indices,
                                                mode='train')

        # Update loss, backward propagate, update optimizer
        loss = loss.mean()
        self.train_loss += loss.item()
        self.train_total += targets.size(0)
        self.train_correct += predicted.eq(targets.data).cpu().sum().item()
        if self.settings.optimizer != 'sam' and self.settings.optimizer != 'sharp' and self.settings.optimizer != 'lsam':
            loss.backward()
            self.optimizer.step()

        # Print message on console
        metrics = {'loss': self.train_loss / self.train_total,
                    'acc': self.train_correct / self.train_total}
        self.print_message(epoch_index=self.epoch, total_epochs=self.settings.epochs,
                        index_batch=batch_idx+1, total_batches=total_batches,
                        metrics=metrics, mode='train')

    def test_epoch(self):
        self.test_loss = 0.
        self.test_correct = 0.
        self.test_total = 0.

        self.model.eval()

        if self.settings.distributed_training:
            self.test_loader.sampler.set_epoch(self.epoch)
        
        for batch_idx, (inputs, targets) in enumerate(self.test_loader):
            self.test_step(inputs, targets, batch_idx=batch_idx, total_batches=len(self.test_loader))
            
        self.logger.set_logger_newline()

        # Add test accuracy to dict
        acc = self.test_correct.item() / self.test_total
        return acc
    
    def test_step(self, inputs, targets, batch_idx=0, total_batches=0):
        # Map to available device
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        # Forward propagation, compute loss, get predictions
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss = loss.mean()
        self.test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        self.test_total += targets.size(0)
        self.test_correct += predicted.eq(targets.data).cpu().sum()

        # Print message on console
        metrics = {'loss': self.test_loss / self.test_total,
                    'acc': self.test_correct.item() / self.test_total}
        self.print_message(epoch_index=self.epoch, total_epochs=self.settings.epochs,
                        index_batch=batch_idx+1, total_batches=total_batches,
                        metrics=metrics, mode='test')

    def setup_proxy_metrics(self):
        # Setting up metrics objects
        self.forgettability_met = Forgettability()
        self.epochs_to_learn_met = EpochsToLearn()
        self.hessian_flatness_met = HessianTraceCurvature(model=self.model, loss_function=self.criterion, device=self.settings.device)
        self.epsilon_flatness_met = EpsilonFlatness(model=self.model, loss_function=self.criterion, device=self.settings.device)

    def compute_and_store_final_metrics(self):
        metrics_dir = os.path.join(self.settings.metrics_folder, '{}_over_{}'.format(self.settings.model, self.settings.dataset), 'seed_{}'.format(self.settings.seed))
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir, exist_ok=True)
        # Forgettability metrics
        forgettability_train = self.forgettability_met.get_scores(mode='train')
        with open(os.path.join(metrics_dir, 'forgettability_train.pkl'), 'wb') as file:
            pickle.dump(forgettability_train, file)
        del forgettability_train
        self.forgettability_met.store_history(os.path.join(metrics_dir, 'forgettability_history.pkl'))
        del self.forgettability_met
        # Epochs to learn metrics
        epochs_to_learn_train = self.epochs_to_learn_met.get_scores(mode='train')
        with open(os.path.join(metrics_dir, 'epochs_to_learn_train.pkl'), 'wb') as file:
            pickle.dump(epochs_to_learn_train, file)
        del epochs_to_learn_train
        self.epochs_to_learn_met.store_history(os.path.join(metrics_dir, 'epochs_to_learn_history.pkl'))
        del self.epochs_to_learn_met
        # Hessian flatness metrics
        hessian_flat_train = self.hessian_flatness_met.compute_dataset(self.train_dataset, sample_indices=self.train_indx)
        with open(os.path.join(metrics_dir, 'hessian_flat_train.pkl'), 'wb') as file:
            pickle.dump(hessian_flat_train, file)
        del hessian_flat_train
        # Epsilon flatness metrics
        epsilon_flat_train = self.epsilon_flatness_met.compute_dataset(self.train_dataset, sample_indices=self.train_indx)
        with open(os.path.join(metrics_dir, 'epsilon_flat_train.pkl'), 'wb') as file:
            pickle.dump(epsilon_flat_train, file)
        del epsilon_flat_train

    def store_resume_ckpt(self):
        state = {
                'epoch': self.epoch,
                'best_acc': self.best_acc,
                'elapsed_time': self.elapsed_time,
                'test_accs': self.test_accs,
                'state_dict': self.model.state_dict(),
                'optimizer' : self.optimizer.state_dict(),
                'scheduler' : self.scheduler.state_dict()
            }
        if self.settings.measure_proxies:
            state['forg_met'] = self.forgettability_met
            state['etl_met'] = self.epochs_to_learn_met,
            state['hess_met'] = self.hessian_flatness_met,
            state['eps_met'] = self.epsilon_flatness_met,
        checkpoint_folder = os.path.join(self.settings.resume_ckpts_folder, self.experiment_name, 'seed_{}'.format(self.settings.seed))
        os.makedirs(checkpoint_folder, exist_ok=True)
        torch.save(state, os.path.join(checkpoint_folder, 'epoch={}.pth.tar'.format(self.epoch)))

    def load_last_resume_ckpt(self):
        checkpoint_folder = os.path.join(self.settings.resume_ckpts_folder, self.experiment_name, 'seed_{}'.format(self.settings.seed))
        checkpoints = glob.glob(os.path.join(checkpoint_folder, '*.pth.tar'))
        found_epochs = [int(check.split('epoch=')[-1].split('.pth.tar')[0]) for check in checkpoints]
        if found_epochs == []:
            self.logger.print_it('No resume checkpoint found! Are you sure you wanted to resume?')
            self.initialize_train()
        else:
            self.initialize_train()
            last_epoch = max(found_epochs)
            checkpoint_to_load = os.path.join(self.settings.resume_ckpts_folder, self.experiment_name, 'seed_{}'.format(self.settings.seed), 'epoch={}.pth.tar'.format(last_epoch))
            # Read variables
            checkpoint = torch.load(checkpoint_to_load, map_location=self.device)
            self.epoch = checkpoint['epoch'] + 1
            self.best_acc = checkpoint['best_acc']
            self.elapsed_time = checkpoint['elapsed_time']
            self.test_accs = checkpoint['test_accs']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            if self.settings.measure_proxies:
                self.forgettability_met = checkpoint['forg_met']
                self.epochs_to_learn_met = checkpoint['etl_met']
                self.hessian_flatness_met = checkpoint['hess_met']
                self.epsilon_flatness_met = checkpoint['eps_met']

    def print_message(self, epoch_index, total_epochs, index_batch, total_batches, metrics, mode='train'):
        if self.global_rank == 'cpu':
            message = '| CPU | EPOCH: {}/{} |'.format(epoch_index, total_epochs)
        else:
            message = '| GPU-{} | EPOCH: {}/{} |'.format(self.global_rank, epoch_index, total_epochs)
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

    

    @staticmethod
    # Format time for printing purposes
    def convert_to_hms(seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return int(h), int(m), int(s)