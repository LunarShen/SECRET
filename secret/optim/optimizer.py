from __future__ import print_function, absolute_import
import warnings
import torch
import torch.nn as nn


__factory = ['adam', 'amsgrad', 'sgd', 'rmsprop']


def build_optimizer(cfg, model, LR = None):
    """A function wrapper for building an optimizer.
    Args:
        model (nn.Module): model.
        optim (str, optional): optimizer. Default is "adam".
        lr (float, optional): learning rate. Default is 0.0003.
        weight_decay (float, optional): weight decay (L2 penalty). Default is 5e-04.
        momentum (float, optional): momentum factor in sgd. Default is 0.9,
        sgd_dampening (float, optional): dampening for momentum. Default is 0.
        sgd_nesterov (bool, optional): enables Nesterov momentum. Default is False.
        rmsprop_alpha (float, optional): smoothing constant for rmsprop. Default is 0.99.
        adam_beta1 (float, optional): beta-1 value in adam. Default is 0.9.
        adam_beta2 (float, optional): beta-2 value in adam. Default is 0.99,
        staged_lr (bool, optional): uses different learning rates for base and new layers. Base
            layers are pretrained layers while new layers are randomly initialized, e.g. the
            identity classification layer. Enabling ``staged_lr`` can allow the base layers to
            be trained with a smaller learning rate determined by ``base_lr_mult``, while the new
            layers will take the ``lr``. Default is False.
        new_layers (str or list): attribute names in ``model``. Default is empty.
        base_lr_mult (float, optional): learning rate multiplier for base layers. Default is 0.1.
    Examples::
        >>> # A normal optimizer can be built by
        >>> optimizer = torchreid.optim.build_optimizer(model, optim='sgd', lr=0.01)
        >>> # If you want to use a smaller learning rate for pretrained layers
        >>> # and the attribute name for the randomly initialized layer is 'classifier',
        >>> # you can do
        >>> optimizer = torchreid.optim.build_optimizer(
        >>>     model, optim='sgd', lr=0.01, staged_lr=True,
        >>>     new_layers='classifier', base_lr_mult=0.1
        >>> )
        >>> # Now the `classifier` has learning rate 0.01 but the base layers
        >>> # have learning rate 0.01 * 0.1.
        >>> # new_layers can also take multiple attribute names. Say the new layers
        >>> # are 'fc' and 'classifier', you can do
        >>> optimizer = torchreid.optim.build_optimizer(
        >>>     model, optim='sgd', lr=0.01, staged_lr=True,
        >>>     new_layers=['fc', 'classifier'], base_lr_mult=0.1
        >>> )
    """

    if cfg.OPTIM.OPT not in __factory:
        raise ValueError(
            'Unsupported optim: {}. Must be one of {}'.format(
                cfg.OPTIM.OPT, __factory
            )
        )

    if not isinstance(model, nn.Module):
        raise TypeError(
            'model given to build_optimizer must be an instance of nn.Module'
        )

    if LR is None:
        LR = cfg.OPTIM.LR
    # param_groups = []
    # for _, value in model.named_parameters():
    #     if not value.requires_grad:
    #         continue
    #     param_groups += [{"params": [value]}]
    param_groups = [{'params': model.parameters(), 'initial_lr': LR}] # model.parameters()

    # if len(cfg.GPU_Device) == 1:
    #     n = cfg.DATALOADER.BATCH_SIZE // cfg.OPTIM.FORWARD_BATCH_SIZE
    #     LR = LR/n

    if cfg.OPTIM.OPT == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=LR,
            weight_decay=cfg.OPTIM.WEIGHT_DECAY,
            betas=(cfg.OPTIM.ADAM_BETA1, cfg.OPTIM.ADAM_BETA2),
        )

    elif cfg.OPTIM.OPT == 'amsgrad':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=LR,
            weight_decay=cfg.OPTIM.WEIGHT_DECAY,
            betas=(cfg.OPTIM.ADAM_BETA1, cfg.OPTIM.ADAM_BETA2),
            amsgrad=True,
        )

    elif cfg.OPTIM.OPT == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=LR,
            momentum=cfg.OPTIM.MOMENTUM,
            weight_decay=cfg.OPTIM.WEIGHT_DECAY,
            dampening=cfg.OPTIM.SGD_DAMPENING,
            nesterov=cfg.OPTIM.SGD_NESTEROV,
        )

    elif cfg.OPTIM.OPT == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            param_groups,
            lr=LR,
            momentum=cfg.OPTIM.MOMENTUM,
            weight_decay=cfg.OPTIM.WEIGHT_DECAY,
            alpha=cfg.OPTIM.RMSPROP_ALPHA,
        )

    return optimizer
