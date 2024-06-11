import pathlib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


DEFAULT_FILE_PATH = os.path.join(pathlib.Path(__file__).parent, 'cifar100_infl_matrix.npz')

class Memorization:
    
    def __init__(self, file_path=DEFAULT_FILE_PATH):
        self.all_data = np.load(file_path)
        self.mem_scores = self.all_data['tr_mem']
        self.normalized_mem_scores = self.mem_scores / np.sum(self.mem_scores)

    def get(self, idx):
        """Returns the memorization score at the given index."""
        return self.mem_scores[idx]
    
    def get_prob(self, idx):
        """Returns the memorization score at the given index."""
        return self.normalized_mem_scores[idx]    

    def full_list(self):
        """Returns a list of all memorization scores in the file."""
        return list(self.mem_scores)

    def full_prob(self):
        """Returns a list of all memorization scores in the file."""
        return list(self.normalized_mem_scores)
    
    def get_dict_form(self, sort: bool = True):
        mem_scores_dict = {i: self.mem_scores[i] for i in range(len(self.mem_scores))}
        if sort:
            return dict(sorted(mem_scores_dict.items(), key=lambda x:x[1], reverse=False))
        else:
            return mem_scores_dict

    def get_sorted_indices(self, idx=None):
        mem_scores_dict = {i: self.mem_scores[i] for i in range(len(self.mem_scores))}
        sorted_dict = dict(sorted(mem_scores_dict.items(), key=lambda x:x[1], reverse=False))
        sorted_indices = list(sorted_dict.keys())
        # sorted_list = sorted(list(enumerate(self.mem_scores)))
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
        """Plots the histogram of self.mem_scores and saves to a file."""
        if idx is None:
            plt.figure(figsize=(10, 6))
            n_bins = 50
            colors = self.fixed_pastel_colors(n_bins)

            counts, bins, patches = plt.hist(self.mem_scores, bins=n_bins, density=False, alpha=0.6)

            # Setting the color for each bar
            for i in range(len(patches)):
                patches[i].set_facecolor(colors[i])
            plt.title('Histogram of mem_scores')
            plt.xlabel('Mem_scores')
            plt.ylabel('Number of data points')
            plt.grid(axis='y', alpha=0.75)
            filename = f"all_mem_scores_hist.png"
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
            plt.title('Histogram of mem_scores')
            plt.xlabel('Mem_scores')
            plt.ylabel('Number of data points')
            plt.grid(axis='y', alpha=0.75)
            filename = f"mem_scores_hist.png"
            plt.savefig(filename)
            print(f"Plot saved as {filename}")    
            
            
    def plot_scores_kde(self, idx=None):
        """Plots the probability distribution of self.mem_scores using KDE and saves to a file."""
        if idx is None:
            plt.figure(figsize=(10, 6))

            sns.kdeplot(self.mem_scores, fill=True, color="skyblue", alpha=0.6)

            plt.title('KDE of mem_scores')
            plt.xlabel('Mem_scores')
            plt.ylabel('Density')
            plt.grid(axis='y', alpha=0.75)

            filename = f"all_mem_scores_distribution.png"
            plt.savefig(filename)
            print(f"Plot saved as {filename}")            
            
        else: 
            
            plt.figure(figsize=(10, 6))

            sns.kdeplot(self.mem_scores[idx], fill=True, color="skyblue", alpha=0.6)

            plt.title('KDE of mem_scores')
            plt.xlabel('Mem_scores')
            plt.ylabel('Density')
            plt.grid(axis='y', alpha=0.75)

            filename = f"mem_scores_distribution.png"
            plt.savefig(filename)
            print(f"Plot saved as {filename}")   
            
            
    def plot_prob_dist(self, idx=None, plot='scatter'):
        
        if idx is None:
            
            """Plots mem_scores against indices from 1 to 50,000."""
            plt.figure(figsize=(12, 6))

            indices = np.arange(1, len(self.normalized_mem_scores) + 1)
            if plot == 'scatter':
                plt.scatter(indices, self.normalized_mem_scores, color='skyblue', alpha=0.6, s=1)  # 's' sets the size of the points
            elif plot == 'line':
                plt.plot(indices, self.normalized_mem_scores, color='skyblue', alpha=0.6)
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
            
            
            
        