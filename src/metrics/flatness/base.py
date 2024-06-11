import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import collections
import pickle


class FlatMetric():
    def __init__(self, model, loss_function, device='cpu', mode='hessian'):
        # Set appropriate devices
        cuda_available = torch.cuda.is_available()
        if cuda_available and device != 'cpu':
            dev_str = 'cuda:{}'.format(device)
            self.device = torch.device(dev_str)
            print('Using CUDA enabled device: {}'.format(torch.cuda.get_device_name(dev_str)))
        else:
            self.device = torch.device('cpu')
            print('Using CPU! Training and inference will be very slow!')
        cudnn.benchmark = True  # Should make training go faster for large model

        self.model = model.to(self.device)
        self.loss_function = loss_function.to(self.device)
        assert mode in ['hessian', 'epsilon'], f"Invalid mode for FlatMetric class!"
        self.mode = mode

    def compute(self, inputs, targets):
        raise NotImplementedError

    def compute_dataset(self, dataset, sample_indices=None):
        # Set batch_size to 1 to compute curvature over every single data
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        metrics = {}
        for index, (inputs, targets) in enumerate(dataloader):
            # TODO: Refactor print message
            print('Computing {} flatness for sample {} out of {}...'.format(self.mode, index, len(dataset.targets)), end='\r')
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            if sample_indices is None:
                metrics[index] = self.compute(inputs=inputs, targets=targets)
            else:
                metrics[sample_indices[index]] = self.compute(inputs=inputs, targets=targets)
            del inputs
            del targets
        print()
        return metrics
    
    @staticmethod
    def zero_gradients(x):
        if isinstance(x, torch.Tensor):
            if x.grad is not None:
                x.grad.detach_()
                x.grad.zero_()
        elif isinstance(x, collections.abc.Iterable):
            for elem in x:
                FlatMetric.zero_gradients(elem)
    
    def compute_and_store(self, dataset, filepath):
        metrics = self.compute_dataset(dataset)
        with open(filepath, 'wb') as file:
            pickle.dump(metrics, file)