import os
import numpy as np
import torch
from torchvision import transforms
import scipy.stats
import torch.nn.functional as F
# Import custom stuff
from src.datasets import CIFAR10, CIFAR100, FashionMNIST, SVHN
from torch.utils.data import DataLoader, Subset
from src.metrics import Memorization


from src.utils import gather_settings
settings = gather_settings()
mem = Memorization()
device = 'cuda'

def min_max_normalization(data, new_min=0, new_max=1):
    """
    Perform min-max normalization on a list of numbers.
    
    :param data: List of numbers to be normalized
    :param new_min: Minimum value of the new range
    :param new_max: Maximum value of the new range
    :return: List of normalized numbers
    """
    min_val = min(data)
    max_val = max(data)
    
    normalized_data = [new_min + (x - min_val) * (new_max - new_min) / (max_val - min_val) for x in data]
    return normalized_data


def spearmanr(normalized_diffs, all_val_indices):
    rho, p_value = scipy.stats.spearmanr(normalized_diffs, mem.get(all_val_indices))

    print("Spearman's rho:", rho)
    print("p-value:", p_value)
    return rho

def process_diff(diffs, all_val_indices):
    diffs = [element for sublist in diffs for element in sublist]
    _, values = zip(*diffs)
    diff_sorted = sorted(diffs, key=lambda x: x[1])
    indices, _ = zip(*diff_sorted)
    mid_ind = len(diff_sorted) // 2
    low_diff = indices[:mid_ind]
    high_diff = indices[mid_ind:]
    
    combined = list(zip(all_val_indices, mem.get(all_val_indices)))
    mem_sorted = sorted(combined, key=lambda x: x[1])
    indices, _ = zip(*mem_sorted)
    mid_ind = len(mem_sorted) // 2
    low_mem = indices[:mid_ind]
    high_mem = indices[mid_ind:]

    low_diff = set(low_diff)
    high_diff = set(high_diff)
    low_mem = set(low_mem)
    high_mem = set(high_mem)

    print('Low Intersection: {}'.format(len(low_diff.intersection(low_mem))))
    print('High Intersection: {}'.format(len(high_diff.intersection(high_mem))))

    rho = spearmanr(values, all_val_indices)
    
    return rho, values, all_val_indices

def save_file(mode, values, indices):
    sorted_values = np.zeros(50000)

    # Place each value in the sorted_values array at its corresponding index
    for value, index in zip(values, indices):
        sorted_values[index] = value
    
    os.makedirs('{}/{}/{}/{}'.format(settings.metrics_folder, settings.dataset, settings.model, settings.seed), exist_ok=True)
    # Save the sorted_values array to an NPZ file
    if mode == 'loss':
        np.savez('{}/{}/{}/{}/nn_loss_diff_score.npz'.format(settings.metrics_folder, settings.dataset, settings.model, settings.seed), sorted_values)
    elif mode == 'prob':
        np.savez('{}/{}/{}/{}/nn_prob_diff_score.npz'.format(settings.metrics_folder, settings.dataset, settings.model, settings.seed), sorted_values)

loss_spearmans = []
prob_spearmans = []

for fin_epoch in range(settings.epochs,settings.epochs+1):
    print('epoch: {}'.format(fin_epoch))
    loss_diffs_list = []
    prob_diffs_list = [] 
    all_val_indices = []
    for split_ind in range(1,11):    
        sgd_models_folder = os.path.join(settings.models_folder, '{}_over_{}'.format(settings.model, settings.dataset), 'seed_{}'.format(settings.seed), 'sgd', 'epochs_{}'.format(settings.epochs),'coreset_0', 'reweighting_1')
        sgd_model = torch.load(os.path.join(sgd_models_folder, '{}wd_last.pt'.format(split_ind)))
        sgd_model = sgd_model.to(device)

        sam_models_folder = os.path.join(settings.models_folder, '{}_over_{}'.format(settings.model, settings.dataset), 'seed_{}'.format(settings.seed), 'sam', 'rho_0.05', 'epochs_{}'.format(settings.epochs), 'coreset_0', 'reweighting_1')
        sam_model= torch.load(os.path.join(sam_models_folder, '{}wd_last.pt'.format(split_ind)))
        sam_model = sam_model.to(device)

        if settings.dataset == 'cifar10':
            normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                            std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            test_transform = transforms.Compose([transforms.ToTensor(), normalize])
            num_classes = 10
            train_dataset = CIFAR10(root=settings.datasets_folder, train=True, transform=test_transform, download=True)
        elif settings.dataset == 'cifar100':
            normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                            std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            test_transform = transforms.Compose([transforms.ToTensor(), normalize])
            num_classes = 100
            train_dataset = CIFAR100(root=settings.datasets_folder, train=True, transform=test_transform, download=True)
        elif settings.dataset == 'fmnist':
            mean = [0.2861]
            std = [0.3530]
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
            num_classes = 10
            train_dataset = FashionMNIST(root=settings.datasets_folder, train=True, transform=test_transform, download=True)
        elif settings.dataset == 'svhn':
            mean = [0.4377, 0.4438, 0.4728]
            std = [0.1980, 0.2010, 0.1970]
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
            num_classes = 10
            train_dataset = SVHN(root=settings.datasets_folder, train=True, transform=test_transform, download=True)

        val_indices = np.loadtxt('{}/{}/{}/{}/val_ind/val{}.txt'.format(settings.datasets_folder, settings.dataset, settings.model, settings.seed, split_ind), dtype=int)
        val_indices = val_indices.tolist()
        all_val_indices.append(val_indices)
        # Create a DataLoader for the validation set
        val_dataset = Subset(train_dataset, val_indices)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        criterion = torch.nn.CrossEntropyLoss().to(device)

        # Initialize variables to store total loss and number of correct predictions
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        total_loss2 = 0.0
        total_correct2 = 0
        total_samples2 = 0
        prob_diffs= []
        loss_diffs= []
        for batch_ind, (inputs, targets) in enumerate(val_dataloader):
            # Move inputs and targets to the same device as the model if needed
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            if settings.dataset in ['fmnist', 'svhn']:
                outputs = sgd_model(inputs)
                outputs_sam = sam_model(inputs)
            else:
                outputs,_ = sgd_model(inputs)
                outputs_sam,_ = sam_model(inputs)

            prob_sgd = F.softmax(outputs,dim=1)
            prob_sam = F.softmax(outputs_sam,dim=1)

            prob_diffs.append(torch.abs(prob_sgd - prob_sam).sum().item())

            # Calculate the loss
            loss = criterion(outputs, targets)
            loss_sam = criterion(outputs_sam, targets)

            loss = loss.item()
            loss_sam = loss_sam.item()
            # Update total loss
            total_loss += loss
            total_loss2 += loss_sam
            loss_diffs.append(abs(loss - loss_sam))
            # Get the predicted labels
            _, predicted = torch.max(outputs, 1)
            _, predicted2 = torch.max(outputs_sam, 1)

            # Update total number of correct predictions and total samples
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)    

            # Update total number of correct predictions and total samples
            total_correct2 += (predicted2 == targets).sum().item()
            total_samples2 += targets.size(0)       

        # Calculate average loss and accuracy
        average_loss = total_loss / len(val_dataloader)
        accuracy = total_correct / total_samples

        # Calculate average loss and accuracy
        average_loss2 = total_loss2 / len(val_dataloader)
        accuracy2 = total_correct2 / total_samples2

        loss_diffs_list.append(list(zip(val_indices, loss_diffs)))
        prob_diffs_list.append(list(zip(val_indices, prob_diffs)))
        print(f'SGD Validation Loss: {average_loss:.4f}, Validation Accuracy: {accuracy * 100:.2f}%')
        print(f'SAM Validation Loss: {average_loss2:.4f}, Validation Accuracy: {accuracy2 * 100:.2f}%')


    all_val_indices = [element for sublist in all_val_indices for element in sublist]

    print('Statistics using absolute loss difference')
    normalized_loss_diffs, values, indices = process_diff(loss_diffs_list, all_val_indices)
    save_file('loss', values, indices)
    loss_spearmans.append(normalized_loss_diffs)
    print('\n\nStatistics using mean absolute prediction vector difference')
    normalized_prob_diffs, values, indices = process_diff(prob_diffs_list, all_val_indices)
    save_file('prob', values, indices)
    prob_spearmans.append(normalized_prob_diffs)

