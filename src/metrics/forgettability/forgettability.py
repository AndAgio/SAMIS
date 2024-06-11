import numpy as np
from .base import ForgetBase


class Forgettability(ForgetBase):
    def __init__(self):
        super().__init__()

    def get_scores(self, mode='train'):
        scores = {}
        for example_id, example_stats in self.history_dict[mode].items():
            # Forgetting event is a transition in accuracy from 1 to 0
            presentation_acc = np.array(example_stats)
            transitions = presentation_acc[1:] - presentation_acc[:-1]
            # Find all presentations when forgetting occurs
            if len(np.where(transitions == -1)[0]) > 0:
                forgettings = np.where(transitions == -1)[0] + 2
            else:
                forgettings = []
            scores[example_id] = len(forgettings)
        return scores

