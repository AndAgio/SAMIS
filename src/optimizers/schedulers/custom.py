from torch.optim.lr_scheduler import _LRScheduler


class CustomLRSchedule(_LRScheduler):
    def __init__(self, optimizer, total_epochs=160, base_lr=0.4, last_epoch=-1):
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        super(CustomLRSchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        increase_iters = int(0.15 * self.total_epochs)
        if self.last_epoch < increase_iters:
            return [self.base_lr * (self.last_epoch / increase_iters) for _ in self.optimizer.param_groups]
        else:
            return [self.base_lr * (1 - (self.last_epoch - increase_iters) / (self.total_epochs - increase_iters)) for _ in self.optimizer.param_groups]
