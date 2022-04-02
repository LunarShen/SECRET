from __future__ import absolute_import

from .mutualrefine import mutualrefine
from .pretrain import pretrain

__factory = {
    'mutualrefine': mutualrefine,
    'pretrain': pretrain,
}

def names():
    return sorted(__factory.keys())

def create_engine(cfg):
    if cfg.MODE not in __factory:
        raise KeyError("Unknown Engine:", cfg.MODE)
    engine = __factory[cfg.MODE](cfg)
    return engine
