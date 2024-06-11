import torch
import pickle


class ForgetBase():
    def __init__(self):
        self.history_dict = {'train': {},
                            'test': {}}

    def update_history(self, outputs, targets, samples_indices, mode='train'):
        # Compute predictions from outputs
        _, predicted = torch.max(outputs.data, 1)
        # Update statistics
        acc = predicted == targets
        batch_size = targets.shape[0]
        for j in range(batch_size):
            # Get index in original dataset (not sorted by forgetting)
            index_in_original_dataset = samples_indices[j]
            # Add the statistics of the current training example to dictionary
            try:
                self.history_dict[mode][index_in_original_dataset].append(acc[j].sum().item())
            except KeyError:
                self.history_dict[mode][index_in_original_dataset] = [acc[j].sum().item()]

    def store_history(self, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump(self.history_dict, file)

    def load_history(self, filepath):
        with open(filepath, 'rb') as file:
            self.history_dict = pickle.load(file)

    def get_scores(self, mode='train'):
        return self.history_dict[mode]