from __future__ import print_function, absolute_import
import json
import os.path as osp
import shutil
import logging

import torch
from torch.nn import Parameter

def save_checkpoint(state, is_top1_best, is_mAP_best, fpath='checkpoint.pth.tar', remain=False):
    if remain is False: state['state_dict'] = {k: v for k, v in state['state_dict'].items() if 'classifier' not in k}
    if 'student_dict' in state and remain is False: state['student_dict'] = {k: v for k, v in state['state_dict'].items() if 'classifier' not in k}
    torch.save(state, fpath, _use_new_zipfile_serialization=False)
    if is_top1_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_top1_best.pth.tar'))
    if is_mAP_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_mAP_best.pth.tar'))

def save_checkpoint_idattr(state, is_top1_best, is_mAP_best, fpath='checkpoint.pth.tar', remain=False):
    if remain is False: state['state_dict'] = {k: v for k, v in state['state_dict'].items() if '.classifier' not in k}
    if 'student_dict' in state and remain is False: state['student_dict'] = {k: v for k, v in state['state_dict'].items() if 'classifier' not in k}
    torch.save(state, fpath, _use_new_zipfile_serialization=False)
    if is_top1_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_top1_best.pth.tar'))
    if is_mAP_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_mAP_best.pth.tar'))


def save_checkpoint_Attr(state, is_mA_best, is_F1_best, fpath='checkpoint.pth.tar', remain=False):
    if remain is False: state['state_dict'] = {k: v for k, v in state['state_dict'].items() if 'classifier' not in k}
    torch.save(state, fpath, _use_new_zipfile_serialization=False)
    if is_mA_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_mA_best.pth.tar'))
    if is_F1_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_F1_best.pth.tar'))

def load_checkpoint(fpath):
    logger = logging.getLogger('UnReID')
    if osp.isfile(fpath):
        # checkpoint = torch.load(fpath)
        checkpoint = torch.load(fpath, map_location=torch.device('cpu'))
        logger.info("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))

def copy_state_dict(state_dict, model, strip=None):
    logger = logging.getLogger('UnReID')
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            # logger.info('mismatch: {} {} {}'.format(name, param.size(), tgt_state[name].size()))
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    # if len(missing) > 0:
    #     logger.info("missing keys in state_dict: {}".format(missing))

    return model
