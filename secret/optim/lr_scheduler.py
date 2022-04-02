from __future__ import absolute_import
import torch
from bisect import bisect_right

__factory = ['single_step', 'multi_step', 'cosine', 'warmupmultisteplr']

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / float(self.warmup_iters)
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

def build_lr_scheduler(cfg, optimizer, last_epoch = -1):
    """A function wrapper for building a learning rate scheduler.
    Args:
        optimizer (Optimizer): an Optimizer.
        lr_scheduler (str, optional): learning rate scheduler method. Default is single_step.
        stepsize (int or list, optional): step size to decay learning rate. When ``lr_scheduler``
            is "single_step", ``stepsize`` should be an integer. When ``lr_scheduler`` is
            "multi_step", ``stepsize`` is a list. Default is 1.
        gamma (float, optional): decay rate. Default is 0.1.
        max_epoch (int, optional): maximum epoch (for cosine annealing). Default is 1.
    Examples::
        >>> # Decay learning rate by every 20 epochs.
        >>> scheduler = torchreid.optim.build_lr_scheduler(
        >>>     optimizer, lr_scheduler='single_step', stepsize=20
        >>> )
        >>> # Decay learning rate at 30, 50 and 55 epochs.
        >>> scheduler = torchreid.optim.build_lr_scheduler(
        >>>     optimizer, lr_scheduler='multi_step', stepsize=[30, 50, 55]
        >>> )
    """
    stepsize = cfg.OPTIM.STEPS

    if cfg.OPTIM.SCHED not in __factory:
        raise ValueError(
            'Unsupported scheduler: {}. Must be one of {}'.format(
                cfg.OPTIM.SCHED, __factory
            )
        )

    if cfg.OPTIM.SCHED == 'single_step':
        if isinstance(stepsize, list):
            stepsize = stepsize[-1]

        if not isinstance(stepsize, int):
            raise TypeError(
                'For single_step lr_scheduler, stepsize must '
                'be an integer, but got {}'.format(type(stepsize))
            )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=stepsize, gamma=cfg.OPTIM.GAMMA, last_epoch = last_epoch
        )

    elif cfg.OPTIM.SCHED == 'multi_step':
        if not isinstance(stepsize, list):
            raise TypeError(
                'For multi_step lr_scheduler, stepsize must '
                'be a list, but got {}'.format(type(stepsize))
            )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=stepsize, gamma=cfg.OPTIM.GAMMA, last_epoch = last_epoch
        )

    elif cfg.OPTIM.SCHED == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(cfg.OPTIM.COSINE_MAX_EPOCH), last_epoch = last_epoch
        )

    elif cfg.OPTIM.SCHED == 'warmupmultisteplr':
        if not isinstance(stepsize, list):
            raise TypeError(
                'For WarmupMultiStepLR lr_scheduler, stepsize must '
                'be a list, but got {}'.format(type(stepsize))
            )

        scheduler = WarmupMultiStepLR(
            optimizer, milestones = stepsize, gamma = cfg.OPTIM.GAMMA,
            warmup_factor = cfg.OPTIM.WARMUP_FACTOR,
            warmup_iters = cfg.OPTIM.WARMUP_ITERS,
            warmup_method = cfg.OPTIM.WARMUP_METHOD,
            last_epoch = last_epoch
        )

    return scheduler
