from .utils import str_to_bool
import argparse


def gather_settings():
    parser = argparse.ArgumentParser(description="CNN")
    parser.add_argument("--dataset", default="cifar100",
                        choices=["cifar10", "cifar100", "svhn", "fmnist", "imagenet", "tiny_imagenet"])
    parser.add_argument("--model", default="resnet50",
                        choices=["resnet18", "resnet50", "wideresnet", "inception", "vgg19", "mobile_small", "mobile_large", "vit"])
    parser.add_argument("--batch_size", type=int, default=512,
                        help="input batch size for training (default: 512)",)
    parser.add_argument("--epochs", type=int, default=160,
                        help="number of epochs to train (default: 160)",)
    parser.add_argument("--learning_rate", type=float, default=0.4,
                        help="learning rate")
    parser.add_argument('--lr_sched', type=str, required=False, default='cosine',
                        help='lr scheduler', choices=['const', 'step', 'exp', 'cosine', 'warmup_step', 'warmup_exp', 'warmup_cosine', 'custom'])
    parser.add_argument("--data_augmentation", action="store_true", default=True,
                        help="augment data by flipping and cropping",)
    parser.add_argument("--cutout_holes", type=int, default=0,
                        help="number of holes to cut out from image")
    parser.add_argument("--cutout_length", type=int, default=16,
                        help="length of the holes")
    parser.add_argument("--device", default='0',
                        help="Set to 0 or 1 to enable CUDA training, cpu otherwise")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--optimizer", default="sgd",
                        help="optimizer to use, default is sgd. Can also use adam",)
    parser.add_argument('--datasets_folder', type=str, default='data')
    parser.add_argument('--log_folder', type=str, default='logs')
    parser.add_argument('--models_folder', type=str, default='ckpts')
    parser.add_argument('--metrics_folder', type=str, default='metrics')
    parser.add_argument('--out_folder', type=str, default='outs')
    parser.add_argument('--plots_folder', type=str, default='plots')

    parser.add_argument("--cumulative", type=int, default=1,
                        help="cumulative option in mem splits training")
    parser.add_argument("--weight_func", type=str, default='inv_lin',
                        choices=["lin", "inv_lin", "exp", "inv_exp", "cos_decay", "inv_cos_decay", "cent_cos", "inv_cent_cos", "custom", "dynamic"],
                        help="weighting function to be used during sample weighting")
    
    parser.add_argument("--mode", default="mem_scale", 
                        choices=["mem_scale", "avg_dist", "mem_dist_only"])
    parser.add_argument("--pivots", type=int, default=10,
                        help="number of pivots")
    parser.add_argument("--mid_point", default="pivots", 
                        choices=["pivots", "centroid"])
    parser.add_argument("--pivot_sample", type=int, default=10,
                        help="number of pivot samples")
    parser.add_argument("--pretrain_epoch", type=int, default=10,
                        help="number of epochs to pretrain with sgd")
    parser.add_argument("--late_epoch", type=int, default=160,
                        help="epoch to start train with sgd late")
    parser.add_argument('--pretrain_splits', type=int, default=1,
                        help="number of coresets splits to use with CE before starting cluster loss")
    parser.add_argument("--coeff_reg", type=float, default=1.0,
                        help="regularization for dist coeff")
    parser.add_argument("--mem_scale_coeff", type=float, default=0.01,
                        help="coeff for mem scaling")
    parser.add_argument('--coreset_splits', type=int, default=10,
                        help='number of splits to use in training')
    parser.add_argument("--alpha", type=float, default=1, help="alpha")
    parser.add_argument("--proxy", default="mem", 
                        choices=["mem", "etl", "forg", "flat", "eps", "sam-sgd"])
    
    # Coresets parameters
    parser.add_argument("--coreset_mode", default="large",
                        choices=['large', 'few_shot'])
    parser.add_argument("--coreset_func", default="full", 
                        choices=["full", "split", "craig", "glister", "gradmatch", "forget", "grand", "el2n", "infobatch", "graph_cut", "random",
                                "min_mem", "min_forg", "min_etl", "min_eps", "min_flat", "min_sam-sgd-loss", "min_sam-sgd-prob",
                                "balance_mem", "balance_forg", "balance_etl", "balance_eps", "balance_flat", "balance_sam-sgd-loss", "balance_sam-sgd-prob"])
    parser.add_argument('--coreset_fraction', default=0.1, type=float, help='fraction of data to be selected (default: 0.1)')
    parser.add_argument('--per_class', default=True, type=str_to_bool,
                        help="whether balance selection is performed per class")
    parser.add_argument('--coreset_epoch_selection', type=int, default=1,
                        help='Every n epochs compute coresets')
    parser.add_argument("--coreset_epochs", type=int, default=160,
                        help="number of epochs to train (default: 160)",)
    parser.add_argument("--coreset_warmup_ratio", type=float, default=0.0,
                        help="number of epochs to train (default: 160)",)
    parser.add_argument("--coreset_pretrain_epochs", type=int, default=160,
                        help="number of epochs to train (default: 160)",)
    parser.add_argument("--coreset_pretrain_lr", type=float, default=0.4,
                        help="learning rate")
    parser.add_argument('--coreset_pretrain_lr_sched', type=str, required=False, default='const',
                        help='lr scheduler', choices=['const', 'step'])
    parser.add_argument("--coreset_pretrain_batch_size", type=int, default=256,
                        help="batch size to be used while pretraining model for coreset baselines",)
    
    parser.add_argument(
        "--starting_index", type=int, default=1, help="starting index for splits (default: 1)")
    
    parser.add_argument('--ntk_folder', type=str, default='ntks')
    parser.add_argument('--lower_index', type=int, default=0)
    parser.add_argument('--m_samples', type=int, default=100)
    parser.add_argument('--ntk_velocity_epoch', type=str, default=10)
    parser.add_argument('--ntk_chosen_metric', type=str, default='mem')
    
    parser.add_argument('--high_low_spec', type=str, default='mem:low:0.3')
    parser.add_argument('--deactivate_bn', action='store_true')
    parser.add_argument('--wait_start', action='store_true')
    parser.add_argument('--eigen_mode', type=str, default='lanczos',
                        help='mode to use for computing eigenvalues', choices=['lanczos', 'power_iter'])
    parser.add_argument('--eigen_split', type=str, default='train',
                        help='dataset split to use for computing eigenvalues', choices=['train', 'test'])
    
    parser.add_argument('--resume_ckpts_folder', type=str, default='resume_ckpts')
    parser.add_argument("--resume", action="store_true", default=False,
                        help="resume training from last checkpoint found",)
    parser.add_argument('--samis_validation_split_index', type=int, default=1)
    parser.add_argument('--distributed_training', action="store_true")

    parser.add_argument('--measure_proxies', action="store_true")

    # Parse arguments and setup name of output file with forgetting stats
    args = parser.parse_args()
    return args


ordered_settings = [
    "dataset",
    "model",
    "seed",
]
