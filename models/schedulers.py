#!/usr/bin/env python3
from torch.optim.lr_scheduler import LRScheduler
import math


class TriangularCyclicLR(LRScheduler):
    """LR from 'Fast Geometric Esembling' (see Garipov et al. 2018)"""

    def __init__(self, optimizer, cycle_size=20, lr_hi=0.1, lr_lo=0.01):
        assert(0 < lr_lo <= lr_hi)
        assert(cycle_size > 1)

        self.cycle_size = cycle_size
        self.lr_hi = lr_hi
        self.lr_lo = lr_lo
        self.batch_n = 0

        super().__init__(optimizer)

    def get_lr(self):
        C_progress = (self.batch_n % self.cycle_size) / self.cycle_size
        lr = self.lr_lo + 2 * (self.lr_hi - self.lr_lo) * abs(1 / 2 - C_progress)
        return [lr for _ in self.optimizer.param_groups]

    def step(self):
        # Should step every batch
        self.batch_n += 1

class CyclicCosineAnnealingLR(LRScheduler):
    """LR for 'Snapshot Ensembles' (see Huang et al. 2017)"""

    def __init__(self, optimizer, cycle_size=20, lr_hi=0.1):
        assert(0 < lr_hi)
        assert(cycle_size > 1)

        self.cycle_size = cycle_size
        self.lr_hi = lr_hi
        self.epoch_n = 0

        super().__init__(optimizer)

    def get_lr(self):
        C_progress = (self.epoch_n % self.cycle_size) / self.cycle_size
        lr = self.lr_hi / 2 * (1 + math.cos(math.pi * C_progress))
        return [lr for _ in self.optimizer.param_groups]

    def step(self):
        # Should step every epoch
        self.epoch_n += 1
