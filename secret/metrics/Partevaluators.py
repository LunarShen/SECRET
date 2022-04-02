from __future__ import print_function, absolute_import
import time
import collections
from collections import OrderedDict
import numpy as np
import torch
import random
import copy
import logging

from .rank_c import evaluate_rank
from .ranking import cmc, mean_ap
from .rerank import re_ranking
from ..utils import to_torch
from ..utils.meters import AverageMeter

def extract_cnn_feature(model, inputs, GenLabel = True):
    inputs = to_torch(inputs).cuda()
    if GenLabel:
        outputs = model(inputs, finetune=True)[1]
    else:
        outputs = model(inputs, finetune=False)
    outputs = [x.data.cpu() for x in outputs]
    return outputs

def extract_features(model, data_loader, print_freq=100, GenLabel = True):
    logger = logging.getLogger('UnReID')

    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            outputs = extract_cnn_feature(model, imgs, GenLabel)
            for index, fname in enumerate(fnames):
                features[fname] = [x[index] for x in outputs]

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % print_freq == 0) or ((i + 1) % len(data_loader) == 0) :
                logger.info('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features

def pairwise_distance(x, y):
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(x, y.t(), beta=1, alpha=-2)
    return dist_m

def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False, use_cython = False):
    logger = logging.getLogger('UnReID')

    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    if use_cython is True:
        return evaluate_rank(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    logger.info('Mean AP: {:4.1%}'.format(mAP))

    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True),}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    logger.info('CMC Scores:')
    for k in cmc_topk:
        logger.info('  top-{:<4}{:12.1%}'.format(k, cmc_scores['market1501'][k-1]))
    return cmc_scores['market1501'][0], mAP


class Evaluator(object):
    def __init__(self, cfg, model):
        super(Evaluator, self).__init__()
        self.cfg = cfg
        self.model = model
        self.logger = logging.getLogger('UnReID')

    def evaluate(self, data_loader, query, gallery, use_cython = False):
        features = extract_features(self.model, data_loader, print_freq = self.cfg.TEST.PRINT_PERIOD, GenLabel=False)
        eval_nums = len(features[query[0][0]])
        results = []
        for i in range(eval_nums):
            x = torch.cat([features[f][i].unsqueeze(0) for f, _, _ in query], 0)
            y = torch.cat([features[f][i].unsqueeze(0) for f, _, _ in gallery], 0)
            distmat = pairwise_distance(x,y)
            results.append(evaluate_all(distmat, query=query, gallery=gallery, use_cython = use_cython))
        return results
