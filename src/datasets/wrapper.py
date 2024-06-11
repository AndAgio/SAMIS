from torch.utils.data import Dataset


class DatasetWrapper(Dataset):
    def __init__(self, dataset: Dataset, sample_indices, sample_weights):
        self.wrapped_dataset = dataset
        self.sample_indices = sample_indices
        self.sample_weights = sample_weights

    def __len__(self):
        return len(self.wrapped_dataset)

    def __getitem__(self, idx):
        image, label = self.wrapped_dataset.__getitem__(idx)
        indices = self.sample_indices[idx]
        weights = self.sample_weights[idx]
        return image, label, indices, weights
    

