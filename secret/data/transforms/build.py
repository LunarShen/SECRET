from __future__ import absolute_import
from .transforms import *
import torchvision

def build_transforms(cfg, is_train=True):
    res = []

    if is_train:
        size_train = cfg.INPUT.SIZE_TRAIN

        # horizontal filp
        do_flip = cfg.INPUT.DO_FLIP
        flip_prob = cfg.INPUT.FLIP_PROB

        # padding
        do_pad = cfg.INPUT.DO_PAD
        padding = cfg.INPUT.PADDING

        # random erasing
        do_rea = cfg.INPUT.REA.ENABLED
        rea_prob = cfg.INPUT.REA.PROB
        rea_mean = cfg.INPUT.REA.MEAN

        # res.append(Resize(size_train, interpolation=3))
        res.append(Resize(size_train, torchvision.transforms.InterpolationMode.BICUBIC))
        if do_flip:
            res.append(RandomHorizontalFlip(p=flip_prob))
        if do_pad:
            res.extend([Pad(padding),
                        RandomCrop(size_train)])
    else:
        size_test = cfg.INPUT.SIZE_TEST
        # res.append(Resize(size_test, interpolation=3))
        res.append(Resize(size_test, torchvision.transforms.InterpolationMode.BICUBIC))

    res.append(ToTensor())
    res.append(Normalize(mean=cfg.INPUT.PIXEL_MEAN,
                             std=cfg.INPUT.PIXEL_STD))

    if is_train and do_rea:
        res.append(RandomErasing(probability=rea_prob, mean=rea_mean))

    return Compose(res)
