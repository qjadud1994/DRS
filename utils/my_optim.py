import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np


class PolyOptimizer(optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)
        self.param_groups = params
        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult
        super().step(closure)

        self.global_step += 1


def lr_poly(base_lr, iter,max_iter,power=0.9):
    return base_lr*((1-float(iter)/max_iter)**(power))

def reduce_lr_poly(args, optimizer, global_iter, max_iter):
    base_lr = args.lr
    for g in optimizer.param_groups:
        g['lr'] = lr_poly(base_lr=base_lr, iter=global_iter, max_iter=max_iter, power=0.9)


def reduce_lr(args, optimizer, epoch, factor=0.1):
    values = args.decay_points.strip().split(',')
    try:
        change_points = map(lambda x: int(x.strip()), values)
    except ValueError:
        change_points = None

    if change_points is not None and epoch in change_points:
        for g in optimizer.param_groups:
            g['lr'] = g['lr']*factor
            print("Reduce Learning Rate : ", epoch, g['lr'])
        return True
