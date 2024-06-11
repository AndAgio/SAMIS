# SAMIS

This repository contains the source code for the experiments presented in the paper **Approximating Memorization Using Loss Surface Geometry for Dataset Pruning and Summarization** to appear at the 2024 SIGKDD Conference on Knowledge Discovery and Data Mining (KDD).

## Usage

### Compute SAMIS

To compute SAMIS scores it is required to compute the difference in distribution between a model trained using the SAM optimizer and the same model trained using standard SGD. To do so, it is possible to run twice the same `train_split.py` python script twice, using `--optimizer='sgd'` and `--optimizer='sam'`. An example of how to run the code is as follows:

```
python train_split.py --dataset='cifar100' --data_augmentation --model='resnet50' --seed=1234 --optimizer='sgd' --samis_validation_split_index=1 --epochs=160 --batch_size=128 --learning_rate=0.1 --lr_sched='step' --resume
```

The `dataset` option is used to define the dataset considered. Available options are: cifar10, cifar100, svhn, tiny_imagenet, fmnist (not fully tested), imagenet (not fully tested). The `model` option is used to define the architecture considered. Available options are: resnet18 and resnet50, vgg19, wideresnet, inception, mobile_small and mobile_large, vit. The `optimizer` option is used to define the optimization algorithm. Available options are: sgd, adam, sam.  The `data_augmentation` option is used to enable data augmentation, while `seed`, `epochs`, `batch_size`, `learning_rate`, `lr_sched` are used to set the training hyperparameters. Finally, the resume option is used to store the last training iteration, allowing to resume from it and avoid time wastes.

The code will automatically split the dataset into the subsets used to compute SAMIS scores over validation data. Therefore, the `samis_validation_split_index` is used to define which data split should be left out for validation. To enable computing SAMIS scores over the whole dataset this procedure should be repeated $n$ times, where $n$ is the number of splits considered (10 in the paper), varying the `samis_validation_split_index` accordingly.

Premature training can be used to compute SAMIS scores more efficiently (see Section 4.2 in the paper). To do so, we can execute the `train_full.py` python script twice, using `--optimizer='sgd'` and `--optimizer='sam'` as follows:

```
python train_full.py --dataset='cifar100' --data_augmentation --model='resnet50' --seed=1234 --optimizer='sgd' --epochs=20 --batch_size=128 --learning_rate=0.1 --lr_sched='const' --resume
```

Note that this time the number of training epochs is set to 20 rather than 160 and that the `samis_validation_split_index` parameter is not used.

The `train_full.py` and `train_split.py` scripts enable training of any model architecture over any of the considered datasets. These scripts store the final model checkpoints in the folder specified using the `--models_folder` parameter. The checkpoints are then used to compute the SAMIS scores, using the `compute_sam_sgd.py` script as follows:

```
python compute_sam_sgd.py --dataset='cifar100' --model='resnet50' --seed=1234 --epochs=160
```

Here, the `epochs` parameter is used to select at which epoch the SAMIS scores should be computed. The `compute_sam_sgd.py` script computes the SAMIS scores over the whole dataset for a specific seed and stores the obtained scores in a npz file. The obtained npz files can then be used to measure the average SAMIS scores over multiple seeds, thus obtaining more accurate approximations of the memorization scores.


### Data pruning and few-shot data summarization training

```
python train_prune.py --dataset='cifar100' --data_augmentation --model='resnet50' --seed=1234 --optimizer='sgd' --coreset_mode='large' --coreset_func='min_mem' --coreset_fraction=0.8 --coreset_epochs=80 --coreset_pretrain_epochs=40 --batch_size=256 --per_class=1 --learning_rate=0.1 --lr_sched='step'
```

The `coreset_mode` parameter is used to distinguish between data pruning (large) and few-shot data summarization (few_shot). Meanwhile, the `coreset_func` parameter is used to select the approach used to prune data and the available options are: craig, glister, gradmatch, forget, grand, el2n, infobatch, graph_cut, random, min_mem, min_forg, min_etl, min_eps, min_flat, min_sam-sgd-loss, min_sam-sgd-prob. Finally, `coreset_fraction` defines the ratio of data to be used ($\mathcal{P}$ in the paper) and `coreset_epochs` defines the number of epochs to be used to train the model on the downstream (e.g., data pruning) task.

For the baselines that do not rely on the memorization proxies -- such as CRAIG, Glister, InfoBatch, etc. --, the script automatically starts a pretraining phase in which the corresponding scores used to select the data are computed. The `coreset_pretrain_epochs` parameter is used to select the number of epochs used in this pretraining phase. Meanwhile, for the proxy-based data pruning and data summarization approaches, the script relies on the pre-computed scores. Therefore, before running the data pruning approaches using the SAMIS-L or SAMIS-P scores it is necessary to compute the scores following the previous steps.

### Eigenspectrum analysis

Once a model is trained, the `compute_hessian_eigen.py` code can be used to compute (and store in a pickle file) the Hessian eigenvalues and their distribution. This code is based on the [PyHessian](https://github.com/amirgholami/PyHessian) library and can be used to compare the flatness of the solutions obtained using SGD and SAM (Figure 4 in the paper).

### NTK velocity

Once a model is trained and once the memorization or its proxies scores are available, the NTK velocity can be computed using the `compute_ntk.py` script. Depending on the selected model and dataset, this script may cause issues related to memory usage. Indeed, computing NTK velocity requires computing an $mC \times N$ matrix $\Psi$, where $m$ is the number of samples in the dataset, $C$ is the number of classes in the dataset and $N$ is the number of parameters in the model selected. Therefore, computing $\Psi$ on large datasets and big models may become unfeasible. Using the `compute_ntk.py` script it is possible to reproduce Figure 2 in the paper, showcasing the correlation between NTK velocity and sample relevance.

## Readily available SAMIS scores

The SAMIS-L and SAMIS-P scores for the CIFAR10, CIFAR100, and SVHN datasets are made available as pickle files under the `src > metrics > sam_sgd` folder. These files contain a python dictionary mapping the sample index (following the original ordering of the corresponding dataset) to the SAMIS scores obtained by averaging the computation over 10 seeds (see Section 4 of the paper).
Similarly, the scores computed using the other proxies considered in the paper are available under the `src > metrics > flatness` ($\gamma(x_i)$ and $\epsilon(x_i)$) and `src > metrics > forgettability` ($F(x_i)$ and $E2L(x_i)$) folders.
The scores can be loaded in the python code using the `Proxy` class made available in the `src > metrics` package. Inside `Proxy` objects it is possible to use the `mode` parameter to specify the considered proxy and compare them plotting their distributions (Figures 6 or 12 in the paper). Similarly, `plot_images.py` contains an example of the code used to plot the cleanest or most cumbersome samples of the CIFAR100 dataset (Figures 7, 10 and 11 in the paper).

## Citation

If you use this code for your research, please consider citing our KDD paper.

```
@inproceedings{AgiolloKdd2024
    title={Approximating Memorization Using Loss Surface Geometry for Dataset Pruning and Summarization},
    author={Agiollo, Andrea and Kim, Young In and Khanna, Rajiv},
    booktitle={SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
    year={2024}
}
```


For any question, please email Andrea at [andrea.agiollo@gmail.com](mailto:andrea.agiollo@gmail.com?subject=[GitHub]%20Source%20Code%20SAMIS%20KDD2024)