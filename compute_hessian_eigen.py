import os
import torch
import time
from torch.utils.data import DataLoader
from src.datasets import CIFAR10, CIFAR100, SVHN, FashionMNIST
import torchvision.transforms as transforms
import numpy as np
from src.utils import gather_settings
import pickle
from src.pyhessian import Hessian, get_esd_plot

settings = gather_settings()

if settings.wait_start:
    import time
    time.sleep(25*60)

if settings.deactivate_bn:
    experiment_name='train_{}_{}_over_{}_nobn'.format(settings.high_low_spec, settings.model, settings.dataset)
else:
    experiment_name='train_{}_{}_over_{}_bn'.format(settings.high_low_spec, settings.model, settings.dataset)

model_path = os.path.join(settings.models_folder, experiment_name, 'seed_{}'.format(settings.seed), 'last.pt')
model = torch.load(model_path)
for module in model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        if hasattr(module, 'weight'):
            module.weight.requires_grad_(True)
        if hasattr(module, 'bias'):
            module.bias.requires_grad_(True)


# Load the appropriate train and test datasets
if settings.dataset == 'cifar10':
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    train_transform = transforms.Compose([])
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)
    dataset = CIFAR10(root=settings.datasets_folder, train=True if settings.eigen_split=='train' else False, transform=train_transform, download=True)
elif settings.dataset == 'cifar100':
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    train_transform = transforms.Compose([])
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)
    dataset = CIFAR100(root=settings.datasets_folder, train=True if settings.eigen_split=='train' else False, transform=train_transform, download=True)
elif settings.dataset == 'svhn':
    normalize = transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])
    train_transform = transforms.Compose([])
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)
    dataset = SVHN(root=settings.datasets_folder, split=settings.eigen_split, download=True, transform=train_transform)
elif settings.dataset == 'fmnist':
    normalize = transforms.Normalize(mean=[0.2861], std=[0.3530])
    train_transform = transforms.Compose([])
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)
    dataset = FashionMNIST(root=settings.datasets_folder, train=True if settings.eigen_split=='train' else False, download=True, transform=train_transform)


dataloader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
# loss = torch.nn.functional.cross_entropy
loss = torch.nn.CrossEntropyLoss()

with torch.cuda.device(int(settings.device)):
    hessian_comp = Hessian(model, loss, dataloader=dataloader, cuda=True)
    eigenvals, _ = hessian_comp.eigenvalues(top_n=5)
    trace = hessian_comp.trace()
    st = time.time()
    density_eigen, density_weight = hessian_comp.density()
    print('Time to compute density estimation is {:.4f}'.format(time.time() - st))

print('Top Eigenvalue = {}'.format(eigenvals))
print('Trace = {}'.format(np.mean(trace)))

# Define the file name where you want to save the eigenvalues
directory = 'hessian_eigenvalues/{}/{}'.format(experiment_name.replace('train', settings.eigen_split), settings.seed)
os.makedirs(directory, exist_ok=True)
filename = "pyhessian_eigen_{}.pkl".format(settings.eigen_mode)
with open(os.path.join(directory, filename), 'wb') as file:
    pickle.dump({'eigenvals': eigenvals,
                'trace': trace,
                'density_eigen': density_eigen,
                'density_weight': density_weight}, file)

# get_esd_plot(density_eigen, density_weight)