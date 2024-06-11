import glob
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from src.metrics import Proxy
from src.datasets import CIFAR100

n_images = 5
metrics = ["mem", "forg", "etl", "eps", "flat", "samis_loss", "samis_prob"]
metrics_fancy_names = ["$m(x_{i})$", "$F(x_{i})$", "$E2L(x_{i})$", "$\\epsilon(x_{i})$", "$\\gamma(x_{i})$", "$S_{L}(x_{i})$", "$S_{P}(x_{i})$"]
classes = [i for i in range(100)]
labels = [
        "apple",
        "aquarium_fish",
        "baby",
        "bear",
        "beaver",
        "bed",
        "bee",
        "beetle",
        "bicycle",
        "bottle",
        "bowl",
        "boy",
        "bridge",
        "bus",
        "butterfly",
        "camel",
        "can",
        "castle",
        "caterpillar",
        "cattle",
        "chair",
        "chimpanzee",
        "clock",
        "cloud",
        "cockroach",
        "couch",
        "crab",
        "crocodile",
        "cup",
        "dinosaur",
        "dolphin",
        "elephant",
        "flatfish",
        "forest",
        "fox",
        "girl",
        "hamster",
        "house",
        "kangaroo",
        "keyboard",
        "lamp",
        "lawn_mower",
        "leopard",
        "lion",
        "lizard",
        "lobster",
        "man",
        "maple_tree",
        "motorcycle",
        "mountain",
        "mouse",
        "mushroom",
        "oak_tree",
        "orange",
        "orchid",
        "otter",
        "palm_tree",
        "pear",
        "pickup_truck",
        "pine_tree",
        "plain",
        "plate",
        "poppy",
        "porcupine",
        "possum",
        "rabbit",
        "raccoon",
        "ray",
        "road",
        "rocket",
        "rose",
        "sea",
        "seal",
        "shark",
        "shrew",
        "skunk",
        "skyscraper",
        "snail",
        "snake",
        "spider",
        "squirrel",
        "streetcar",
        "sunflower",
        "sweet_pepper",
        "table",
        "tank",
        "telephone",
        "television",
        "tiger",
        "tractor",
        "train",
        "trout",
        "tulip",
        "turtle",
        "wardrobe",
        "whale",
        "willow_tree",
        "wolf",
        "woman",
        "worm"
    ]
labels = {i: label for i, label in enumerate(labels)}

out_folder = os.path.join('plots', 'proxies_comparison', 'img_visualization', 'cifar100')

train_dataset = CIFAR100(root='data/', train=True, transform=None, download=True)

# Get indices of examples that should be used for training
train_indx = np.array(range(len(train_dataset.targets)))
# Reassign train data and labels
train_dataset.data = train_dataset.data[train_indx, :, :, :]
train_dataset.targets = np.array(train_dataset.targets)[train_indx].tolist()

for class_ind in classes:
    fig, axs = plt.subplots(len(metrics), n_images)
    class_indx = [ind for ind, targ in enumerate(train_dataset.targets) if targ==class_ind]
    for met_ind, metric in enumerate(metrics):
        proxy_obj = Proxy(mode=metric)
        sorted_proxy_dict = proxy_obj.get_dict_form(sort=True)
        sorted_indices = list(sorted_proxy_dict.keys())
        class_sorted_indices = [ind for ind in sorted_indices if ind in class_indx]
        picked_indices = class_sorted_indices[:n_images]
        for plchldr, image_indx in enumerate(picked_indices):
            axs[met_ind, plchldr].imshow(train_dataset.__getitem__(image_indx)[0])
            axs[met_ind, plchldr].set_xticks([])
            axs[met_ind, plchldr].set_yticks([])
    for ax, row in zip(axs[:,0], metrics_fancy_names):
        pad = 5
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                fontsize=25, ha='right', va='center')
    plt.subplots_adjust(hspace=0, wspace=0)
    fig.tight_layout()
    plot_name = '{}.pdf'.format(labels[class_ind])
    os.makedirs(os.path.join(out_folder, 'lowest_{}'.format(n_images)), exist_ok=True)
    plt.savefig(os.path.join(out_folder, 'lowest_{}'.format(n_images), plot_name))
    # plt.show()
    plt.close()


for class_ind in classes:
    fig, axs = plt.subplots(len(metrics), n_images)
    class_indx = [ind for ind, targ in enumerate(train_dataset.targets) if targ==class_ind]
    for met_ind, metric in enumerate(metrics):
        proxy_obj = Proxy(mode=metric)
        sorted_proxy_dict = proxy_obj.get_dict_form(sort=True)
        sorted_indices = list(sorted_proxy_dict.keys())
        class_sorted_indices = [ind for ind in sorted_indices if ind in class_indx][::-1]
        picked_indices = class_sorted_indices[:n_images]
        for plchldr, image_indx in enumerate(picked_indices):
            axs[met_ind, plchldr].imshow(train_dataset.__getitem__(image_indx)[0])
            axs[met_ind, plchldr].set_xticks([])
            axs[met_ind, plchldr].set_yticks([])
    for ax, row in zip(axs[:,0], metrics_fancy_names):
        pad = 5
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                fontsize=25, ha='right', va='center')
    plt.subplots_adjust(hspace=0, wspace=0)
    fig.tight_layout()
    plot_name = '{}.pdf'.format(labels[class_ind])
    os.makedirs(os.path.join(out_folder, 'highest_{}'.format(n_images)), exist_ok=True)
    plt.savefig(os.path.join(out_folder, 'highest_{}'.format(n_images), plot_name))
    plt.close()
    # plt.show()