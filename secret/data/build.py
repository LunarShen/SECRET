from __future__ import absolute_import
import os.path as osp
from .datasets import *
from .samplers import *
from .preprocessor import *
from .transforms import build_transforms
from torch.utils.data import DataLoader

class IterLoader:
    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None

    def __len__(self):
        if (self.length is not None):
            return self.length
        return len(self.loader)

    def new_epoch(self):
        self.iter = iter(self.loader)

    def next(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)

def build_data(name, data_dir, mode='ReID'):
    root = osp.join(data_dir, name)
    dataset = create(name, root)
    return dataset

def build_loader(cfg, dataset, inputset=None, num_instances = 4, is_train = True, mode = None):
    if mode is None: mode = cfg.MODE

    transform = build_transforms(cfg, is_train)

    if is_train:

        if mode == 'mutualrefine':
            dataset = sorted(dataset.train) if inputset is None else inputset

            rmgs_flag = num_instances > 0
            if rmgs_flag:
                sampler = PartRandomMultipleGallerySampler(dataset, num_instances)
            else:
                sampler = None

            loader = DataLoader(Preprocessor(dataset, transform=transform),
                        batch_size=cfg.DATALOADER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, sampler=sampler,
                        shuffle=not rmgs_flag, pin_memory=True, drop_last=True)
            if cfg.DATALOADER.ITER_MODE:
                loader = IterLoader(loader, length = cfg.DATALOADER.ITERS)

        elif mode == 'pretrain':
            dataset = sorted(dataset.train) if inputset is None else inputset

            rmgs_flag = num_instances > 0
            if rmgs_flag:
                sampler = RandomMultipleGallerySampler(dataset, num_instances)
            else:
                sampler = None

            loader = DataLoader(Preprocessor(dataset, transform=transform),
                        batch_size=cfg.DATALOADER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, sampler=sampler,
                        shuffle=not rmgs_flag, pin_memory=True, drop_last=True)
            if cfg.DATALOADER.ITER_MODE:
                loader = IterLoader(loader, length = cfg.DATALOADER.ITERS)
        else:
            raise KeyError('NotImplementedError')
    else:
        dataset = list(set(dataset.query) | set(dataset.gallery)) if inputset is None else inputset
        loader = DataLoader(
            Preprocessor(dataset, transform=transform),
            batch_size=cfg.DATALOADER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS,
            shuffle=False, pin_memory=True)

    return loader
