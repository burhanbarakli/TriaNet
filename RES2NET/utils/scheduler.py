import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineScheduler(_LRScheduler):
    """
    Warmup + Cosine Annealing Learning Rate Scheduler
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.warmup_epochs > 0 and self.last_epoch < self.warmup_epochs:
            # Warmup phase: linear increase using (last_epoch+1)/warmup_epochs, clamped to [0,1]
            warmup_factor = (self.last_epoch + 1) / max(1, self.warmup_epochs)
            warmup_factor = max(0.0, min(1.0, warmup_factor))
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            cosine_epochs = max(1, self.total_epochs - max(1, self.warmup_epochs))
            cosine_progress = (self.last_epoch - self.warmup_epochs) / cosine_epochs
            cosine_progress = max(0.0, min(1.0, cosine_progress))

            return [
                self.min_lr + (base_lr - self.min_lr) *
                (1 + math.cos(math.pi * cosine_progress)) / 2
                for base_lr in self.base_lrs
            ]
