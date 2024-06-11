import pathlib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


DEFAULT_MEM_FILE_PATH = os.path.join(pathlib.Path(__file__).parent, 'memorization', 'dataset_infl_matrix.npz')
DEFAULT_SAM_SGD_FILE_PATH = os.path.join(pathlib.Path(__file__).parent, 'sam_sgd', 'dataset_nn_loss_diff_score.npz')
DEFAULT_METRICS_DIR_PATH = os.path.join(pathlib.Path(__file__).parent.parent.parent, 'metrics', 'resnet50_over_dataset')


def normalize_dict_by_values(my_dict):
    values = [val for _, val in my_dict.items()]
    minimum = min(values)
    maximum = max(values)
    normalized_dict = {key:(value-minimum)/float(maximum-minimum) for key, value in my_dict.items()}
    return normalized_dict

def divide_dict_by_value(my_dict, value):
    divided_dict = {key: val/value if not np.isnan(val) else 1 for key, val in my_dict.items()}
    return divided_dict

def sort_dict_by_values(my_dict, reverse=True):
    return dict(sorted(my_dict.items(), key=lambda x:x[1], reverse=reverse))

def sort_dict_by_key(my_dict, reverse=False):
    return dict(sorted(my_dict.items(), key=lambda x:x[0], reverse=reverse))

def normalize_list(given_list):
    minimum = min(given_list)
    maximum = max(given_list)
    normalized_list = [(value-minimum)/float(maximum-minimum) for value in given_list]
    return normalized_list


class Proxy:
    def __init__(self, mode: str = 'mem', dataset: str = 'cifar100', normalize: bool = True):
        assert mode in ["mem", "etl", "forg", "flat", "eps", "samis_loss", "samis_prob"]
        assert dataset in ["cifar10", "cifar100", "svhn"]
        self.dataset = dataset
        self.mode = mode
        if self.mode == 'mem':
            self.load_mem_scores()
        else:
            self.load_scores_from_file()
        if normalize:
            self.scores = np.asarray(self.scores)
            self.scores = (self.scores - self.scores.min()) / (self.scores.max() - self.scores.min())

    def load_scores_from_file(self):
        metrics_file_name = '{}_{}.pkl'.format(self.mode, self.dataset)
        subfolder_name = self.get_subfolder_name_from_mode()
        summary_file = os.path.join(pathlib.Path(__file__).parent, subfolder_name, metrics_file_name)
        with open(summary_file, 'rb') as f:
            normalized_scores_dict = pickle.load(f)
        scores_dict = sort_dict_by_key(normalized_scores_dict)
        self.scores = list(scores_dict.values())

    def load_sam_sgd_scores(self, mode='loss'):
        if mode == 'loss':
            self.all_data = np.load(DEFAULT_SAM_SGD_FILE_PATH.replace('dataset', self.dataset))
        elif mode == 'prob':
            self.all_data = np.load(DEFAULT_SAM_SGD_FILE_PATH.replace('loss', 'prob').replace('dataset', self.dataset))
        else:
            raise ValueError('Mode "{}" is not available!'.format(mode))
        self.scores = self.all_data['arr_0']

    def load_mem_scores(self):
        self.all_data = np.load(DEFAULT_MEM_FILE_PATH.replace('dataset', self.dataset))
        self.scores = self.all_data['tr_mem']
        self.scores = self.scores / np.sum(self.scores)

    def load_scores_from_runs_outputs(self):
        metrics_folder = DEFAULT_METRICS_DIR_PATH.replace('dataset', self.dataset)
        name = self.get_metrics_filename_from_mode()
        subfolders = [x[0] for x in os.walk(metrics_folder)][1:]
        seeds = [subf.split('_')[-1] for subf in subfolders]
        score_dictionaries = []
        for subfolder in subfolders:
            if subfolder.split('_')[-1] in seeds:
                with open(os.path.join(subfolder, name), 'rb') as f:
                    scores = pickle.load(f)
                if self.mode=='forget':
                    scores = dict(sorted(scores.items(), key=lambda x:x[0]))
                score_dictionaries.append(scores)
        n_experiments = len(score_dictionaries)
        avg_scores = {key: 0 for key in list(score_dictionaries[0].keys())}
        for scores_dict in score_dictionaries:
            for idx, score in scores_dict.items():
                avg_scores[idx] += score
        for idx, score in avg_scores.items():
            avg_scores[idx] /= n_experiments
        if self.mode=='flat':
            normalized_scores_dict = normalize_dict_by_values(avg_scores)
        else:
            normalized_scores_dict = divide_dict_by_value(avg_scores, value=160)
        filename = os.path.join('src', 'metrics', self.get_subfolder_name_from_mode(), 'summary_{}_{}'.format(self.dataset, self.get_metrics_filename_from_mode()))
        if os.path.exists(filename):
            os.remove(filename)
        with open(filename, 'xb') as f:
            pickle.dump(normalized_scores_dict, f)
        scores_dict = sort_dict_by_key(normalized_scores_dict)
        self.scores = list(scores_dict.values())

    def load_scores_from_summary_file(self):
        metrics_file_name = 'summary_{}_{}'.format(self.dataset, self.get_metrics_filename_from_mode())
        subfolder_name = self.get_subfolder_name_from_mode()
        summary_file = os.path.join(pathlib.Path(__file__).parent, subfolder_name, metrics_file_name)
        with open(summary_file, 'rb') as f:
            normalized_scores_dict = pickle.load(f)
        scores_dict = sort_dict_by_key(normalized_scores_dict)
        self.scores = list(scores_dict.values())

    def metrics_runs_files_available(self):
        subfolders = [x[0] for x in os.walk(DEFAULT_METRICS_DIR_PATH.replace('dataset', self.dataset))][1:]
        metrics_file_name = self.get_metrics_filename_from_mode()
        seeds = [subf.split('_')[-1] for subf in subfolders]
        found_files = 0
        for subfolder in subfolders:
            if subfolder.split('_')[-1] in seeds:
                if os.path.exists(os.path.join(subfolder, metrics_file_name)):
                    found_files += 1
        return found_files > 0
    
    def summary_file_available(self):
        metrics_file_name = 'summary_{}_{}'.format(self.dataset, self.get_metrics_filename_from_mode())
        subfolder_name = self.get_subfolder_name_from_mode()
        return os.path.exists(os.path.join(pathlib.Path(__file__).parent, subfolder_name, metrics_file_name))

    def get_metrics_filename_from_mode(self):
        if self.mode == 'flat':
            return 'hessian_flat_train.pkl'
        elif self.mode == 'eps':
            return 'epsilon_flat_train.pkl'
        elif self.mode == 'forg':
            return 'forgettability_train.pkl'
        elif self.mode == 'etl':
            return 'epochs_to_learn_train.pkl'
        else:
            raise ValueError('Mode "{}" not available!'.format(self.mode))
    
    def get_subfolder_name_from_mode(self):
        if self.mode in ['flat', 'eps']:
            return 'flatness'
        elif self.mode in ['forg', 'etl']:
            return 'forgettability'
        elif self.mode in ['samis_loss', 'samis_prob']:
            return 'sam_sgd'
        else:
            raise ValueError('Mode "{}" not available!'.format(self.mode))
    
    def get(self, idx):
        """Returns the memorization score at the given index."""
        return self.scores[idx]   

    def full_list(self):
        """Returns a list of all memorization scores in the file."""
        return list(self.scores)
    
    def get_dict_form(self, sort: bool = True):
        scores_dict = {i: self.scores[i] for i in range(len(self.scores))}
        if sort:
            return dict(sorted(scores_dict.items(), key=lambda x:x[1], reverse=False))
        else:
            return scores_dict

    def get_sorted_indices(self, idx=None):
        scores_dict = {i: self.scores[i] for i in range(len(self.scores))}
        sorted_dict = dict(sorted(scores_dict.items(), key=lambda x:x[1], reverse=False))
        sorted_indices = list(sorted_dict.keys())
        # sorted_list = sorted(list(enumerate(self.scores)))
        # sorted_indices = [x[0] for x in sorted_list]
        if idx is None:
            return sorted_indices 
        else:
            return sorted_indices[idx] 
    
    def fixed_pastel_colors(self, n):
        """Returns a fixed set of pastel colors, cycling through if n is greater than the list length."""
        base_colors = [
            "#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF", 
            "#e0bbe4", "#957DAD", "#D291BC", "#FEC8D8", "#FFDFD3"
        ]
        return [base_colors[i % len(base_colors)] for i in range(n)]
    
    def plot_scores_hist(self, idx=None):
        """Plots the histogram of self.scores and saves to a file."""
        if idx is None:
            plt.figure(figsize=(10, 6))
            n_bins = 50
            colors = self.fixed_pastel_colors(n_bins)

            counts, bins, patches = plt.hist(self.scores, bins=n_bins, density=False, alpha=0.6)

            # Setting the color for each bar
            for i in range(len(patches)):
                patches[i].set_facecolor(colors[i])
            plt.title('Histogram of scores')
            plt.xlabel('scores')
            plt.ylabel('Number of data points')
            plt.grid(axis='y', alpha=0.75)
            filename = f"all_scores_hist.png"
            plt.savefig(filename)
            print(f"Plot saved as {filename}")                
            
            
        else:
            plt.figure(figsize=(10, 6))
            n_bins = 50
            colors = self.fixed_pastel_colors(n_bins)

            counts, bins, patches = plt.hist(self.get(idx), bins=n_bins, density=False, alpha=0.6)

            # Setting the color for each bar
            for i in range(len(patches)):
                patches[i].set_facecolor(colors[i])
            plt.title('Histogram of scores')
            plt.xlabel('scores')
            plt.ylabel('Number of data points')
            plt.grid(axis='y', alpha=0.75)
            filename = f"scores_hist.png"
            plt.savefig(filename)
            print(f"Plot saved as {filename}")    
            
            
    def plot_scores_kde(self, idx=None):
        """Plots the probability distribution of self.scores using KDE and saves to a file."""
        if idx is None:
            plt.figure(figsize=(10, 6))

            sns.kdeplot(self.scores, fill=True, color="skyblue", alpha=0.6)

            plt.title('KDE of scores')
            plt.xlabel('scores')
            plt.ylabel('Density')
            plt.grid(axis='y', alpha=0.75)

            filename = f"all_scores_distribution.png"
            plt.savefig(filename)
            print(f"Plot saved as {filename}")            
            
        else: 
            
            plt.figure(figsize=(10, 6))

            sns.kdeplot(self.scores[idx], fill=True, color="skyblue", alpha=0.6)

            plt.title('KDE of scores')
            plt.xlabel('scores')
            plt.ylabel('Density')
            plt.grid(axis='y', alpha=0.75)

            filename = f"scores_distribution.png"
            plt.savefig(filename)
            print(f"Plot saved as {filename}")   
            
            
    def plot_prob_dist(self, idx=None, plot='scatter'):
        
        if idx is None:
            
            """Plots scores against indices from 1 to 50,000."""
            plt.figure(figsize=(12, 6))

            indices = np.arange(1, len(self.normalized_scores) + 1)
            if plot == 'scatter':
                plt.scatter(indices, self.normalized_scores, color='skyblue', alpha=0.6, s=1)  # 's' sets the size of the points
            elif plot == 'line':
                plt.plot(indices, self.normalized_scores, color='skyblue', alpha=0.6)
            else: 
                print("Not valid option")
                return

            plt.title('Probability Distribution')
            plt.xlabel('Index')
            plt.ylabel('Normalized Mem Score')
            plt.grid(True, which='both', linestyle='--', linewidth=0.1)

            filename = f"all_prob_dist.png"
            plt.savefig(filename)
            print(f"Plot saved as {filename}")      
            
            
        else: 
            plt.figure(figsize=(12, 6))

            indices = np.arange(1, len(idx) + 1)
            probs = self.get(idx) / np.sum(self.get(idx))
            if plot == 'scatter':             
                plt.scatter(idx, probs, color='skyblue', alpha=0.6, s=1)
            elif plot == 'line':
                plt.plot(idx, probs, color='skyblue', alpha=0.6)
            else: 
                print("Not valid option")
                return    
            

            plt.title('Probability Distribution')
            plt.xlabel('Index')
            plt.ylabel('Normalized Mem Score')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)

            filename = f"prob_dist.png"
            plt.savefig(filename)
            print(f"Plot saved as {filename}")   
            
            
            
        