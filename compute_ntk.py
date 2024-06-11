import shutil
import os
import numpy as np
import copy
import time
import pickle
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torch.func import functional_call, vmap, jacrev
# import qml
device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
# Import custom stuff
from src.datasets import CIFAR10, CIFAR100, FashionMNIST, SVHN
from src.models import disable_running_stats
from src.metrics import Proxy
from src.utils import gather_settings


settings = gather_settings()
DATASET = 'cifar100'
MODEL = 'resnet18'
SEEDS = [2026, 4113, 5977, 7481, 9833]
M_SAMPLES = 50
CHOSEN_METRIC = 'mem'


# Load dataset
if DATASET == 'cifar10':
    train_dataset = CIFAR10(root=settings.datasets_folder, train=True, transform=None, download=True)
    epochs = [1,5,10,20,50]
    lower_indices = np.arange(0, len(train_dataset), int(len(train_dataset)/20))
elif DATASET == 'cifar100':
    train_dataset = CIFAR100(root=settings.datasets_folder, train=True, transform=None, download=True)
    epochs = [10,40,70,100,150]
    lower_indices = [0, 1000, 2500, 5000, 7500, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 47500, 49000, 49500]
elif DATASET == 'svhn':
    train_dataset = SVHN(root=settings.datasets_folder, train=True, transform=None, download=True)
    epochs = [1,5,10,20]
    lower_indices = np.arange(0, len(train_dataset), int(len(train_dataset)/20))


def compute_ks(settings, lower_index, epoch):

    # Load dataset
    print('Loading dataset...')
    if settings.dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])
        num_classes = 10
        im_channels = 3
        im_size = (32, 32)
        train_dataset = CIFAR10(root=settings.datasets_folder, train=True, transform=test_transform, download=True)
    elif settings.dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])
        num_classes = 100
        im_channels = 3
        im_size = (32, 32)
        train_dataset = CIFAR100(root=settings.datasets_folder, train=True, transform=test_transform, download=True)
    elif settings.dataset == 'svhn':
        mean = [0.4377, 0.4438, 0.4728]
        std = [0.1980, 0.2010, 0.1970]
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        num_classes = 10
        im_channels = 3
        im_size = (32, 32)
        train_dataset = SVHN(root=settings.datasets_folder, train=True, transform=test_transform, download=True)


    print('Loading model checkpoint...')
    models_folder = os.path.join(settings.models_folder, 'standard_{}_over_{}'.format(settings.model, settings.dataset), 'seed_{}'.format(settings.seed))
    model = torch.load(os.path.join(models_folder, 'epoch_{}.pt'.format(epoch)), map_location='cpu')
    model = model.to(device)
    print('Model loaded to device: {}'.format(device))

    # Load metric
    print('Loading metric...')
    sorted_proxy_dict = Proxy(mode=CHOSEN_METRIC, dataset=settings.dataset).get_dict_form(sort=True)
    sorted_indices = list(sorted_proxy_dict.keys())

    # Copy model
    print('Disabling running stats of model...')
    model_copy = copy.deepcopy(model)
    disable_running_stats(model_copy)
    params = {k: v.detach() for k, v in model_copy.named_parameters()}
    tot_params = sum([len(j.flatten().squeeze().tolist()) for j in params.values()])
    print('Required matrix of shape: ({}x{})'.format(num_classes*M_SAMPLES, tot_params))

    def fnet_single(params, x):
        return functional_call(model_copy, params, (x.unsqueeze(0),)).squeeze(0)

    def compute_jacobian(fnet_single, params, x1):
        # Compute J(x1)
        jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
        jac1 = torch.hstack([j.flatten(2).squeeze() for j in jac1.values()]).cpu()
        return jac1


    st_time = time.time()
    print('Computing K for samples [{}:{}]'.format(lower_index, lower_index+M_SAMPLES))
    # Gather samples to compute velocity
    indices_to_use = sorted_indices[lower_index: lower_index+M_SAMPLES]
    subset = Subset(train_dataset, indices_to_use)
    subset_loader = DataLoader(subset, batch_size=1, shuffle=False)
    # Iterate over each sample and compute jacobian
    psi = torch.empty((M_SAMPLES*num_classes, tot_params))
    for batch_ind, (inputs, targets) in enumerate(subset_loader):
        print('Sample: {}/{}'.format(batch_ind+1, M_SAMPLES), end='\r')
        inputs = inputs.to(device)
        jac = compute_jacobian(fnet_single, params, inputs)
        psi[batch_ind*num_classes:(batch_ind+1)*num_classes,:] = jac
        del jac
    k = psi @ psi.t()
    del psi
    print('Time taken to compute single value of K: {}'.format(time.time() - st_time))


    print('Saving velocity for {} over {} at epoch {} with li = {} and m = {}'.format(CHOSEN_METRIC, settings.dataset, epoch, lower_index, M_SAMPLES))
    storing_folder = os.path.join(settings.ntk_folder, settings.dataset, settings.model, '{}'.format(CHOSEN_METRIC), 'epoch_{}'.format(epoch), 'seed_{}'.format(settings.seed), 'l_index_{}'.format(lower_index), 'm_{}'.format(M_SAMPLES))
    os.makedirs(storing_folder, exist_ok=True)
    file_name_ks = os.path.join(storing_folder, 'k.pkl')
    with open(file_name_ks, 'wb') as f:
        pickle.dump(k, f)



# COMPUTE KS REQUIRED FOR NTK VELOCITY

for seed in SEEDS:
    for epoch in epochs:
        for li in lower_indices:
            compute_ks(settings=settings, lower_index=li, epoch=epoch)



# COMPUTE NTK VELOCITY FROM KS

def frob_prod(a, b):
    return np.trace(np.matmul(a.numpy(), b.numpy().transpose()))
    # return torch.inner(a, b)

def frob_norm(a):
    return np.sqrt(np.trace(np.matmul(a.numpy(), a.numpy().transpose())))
    # return np.linalg.norm(a.numpy(), ord='fro')
    # return torch.norm(a, p='fro').to_numpy()

for seed in SEEDS:
    for epoch in epochs:
        print('epoch: {}'.format(epoch))
        vs = {li: None for li in lower_indices}
        for lower_index in lower_indices:
            storing_folder = os.path.join(settings.ntk_folder, settings.dataset, settings.model, '{}'.format(CHOSEN_METRIC), 'epoch_{}'.format(epoch), 'seed_{}'.format(seed), 'l_index_{}'.format(lower_index), 'm_{}'.format(M_SAMPLES))
            file_name_ks = os.path.join(storing_folder, 'k.pkl')
            with open(file_name_ks, 'rb') as f:
                k_li_t = pickle.load(f)
            storing_folder = os.path.join(settings.ntk_folder, settings.dataset, settings.model, '{}'.format(CHOSEN_METRIC), 'epoch_{}'.format(epoch+1), 'seed_{}'.format(seed), 'l_index_{}'.format(lower_index), 'm_{}'.format(M_SAMPLES))
            file_name_ks = os.path.join(storing_folder, 'k.pkl')
            with open(file_name_ks, 'rb') as f:
                k_li_tp1 = pickle.load(f)
            v = 1 - (frob_prod(k_li_t, k_li_tp1))/(frob_norm(k_li_t)*frob_norm(k_li_tp1))
            print('v = {}'.format(v))
            vs[lower_index] = v.item()
        
        storing_folder = os.path.join(settings.ntk_folder, settings.dataset, settings.model, '{}'.format(CHOSEN_METRIC), 'epoch_{}'.format(epoch), 'seed_{}'.format(seed))
        file_name_ks = os.path.join(storing_folder, 'vs.pkl')
        with open(file_name_ks, 'wb') as f:
            pickle.dump(vs, f)

