import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from src.utils import ordered_settings, get_logger


class Runner():
    def __init__(self, settings, experiment_name=None):
        self.settings = settings
        if experiment_name is None:
            self.experiment_name = '{}_over_{}'.format(self.settings.model, self.settings.dataset)
        else:
            self.experiment_name = experiment_name
        self.logger = get_logger(arguments=self.settings, experiment_name=self.experiment_name)
        self.set_seed()
        self.setup_device()
        settings_dict = vars(self.settings)
        self.stats_file = '__'.join('{}_{}'.format(arg, settings_dict[arg]) for arg in ordered_settings)

    def setup_device(self):
        # Set appropriate devices
        if self.settings.distributed_training:
            self.local_rank = int(os.environ["LOCAL_RANK"])
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

    @staticmethod
    # Format time for printing purposes
    def convert_to_hms(seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return int(h), int(m), int(s)