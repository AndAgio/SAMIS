import numpy as np
from .base import ForgetBase


class EpochsToLearn(ForgetBase):
    def __init__(self):
        super().__init__()

    def get_scores(self, mode='train'):
        scores = {}
        for example_id, example_stats in self.history_dict[mode].items():
            # Forgetting event is a transition in accuracy from 1 to 0
            presentation_acc = np.array(example_stats)
            # Find the presentation at which the example was first learned, 
            # e.g. first presentation when acc is 1 and remains 1 forever
            reversed_accs = np.array(presentation_acc[::-1])
            # print('reversed_accs: {}'.format(reversed_accs))
            # print('np.where(reversed_accs==0): {}'.format(np.where(reversed_accs==0)))
            reversed_indices = np.where(reversed_accs==0)[0].tolist()
            if len(reversed_indices) == 0:
                first_epoch_learned = 1
            else:
                reversed_epoch_learnt = reversed_indices[0]
                if reversed_epoch_learnt == 0:
                    first_epoch_learned = np.nan
                else:
                    first_epoch_learned = len(example_stats) - reversed_epoch_learnt + 1
            scores[example_id] = first_epoch_learned
        return scores

