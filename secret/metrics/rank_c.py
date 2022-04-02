from __future__ import division, print_function, absolute_import
import numpy as np
import warnings
from collections import defaultdict
import logging

try:
    from .rank_cylib.rank_cy import evaluate_cy
    IS_CYTHON_AVAI = True
except ImportError:
    IS_CYTHON_AVAI = False
    warnings.warn(
        'Cython evaluation (very fast so highly recommended) is '
        'unavailable, now use python evaluation.'
    )

def evaluate_rank(
    distmat,
    q_pids,
    g_pids,
    q_camids,
    g_camids,
    max_rank=10,
    use_cython=True
):
    """Evaluates CMC rank.

    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        q_pids (numpy.ndarray): 1-D array containing person identities
            of each query instance.
        g_pids (numpy.ndarray): 1-D array containing person identities
            of each gallery instance.
        q_camids (numpy.ndarray): 1-D array containing camera views under
            which each query instance is captured.
        g_camids (numpy.ndarray): 1-D array containing camera views under
            which each gallery instance is captured.
        max_rank (int, optional): maximum CMC rank to be computed. Default is 50.
        use_metric_cuhk03 (bool, optional): use single-gallery-shot setting for cuhk03.
            Default is False. This should be enabled when using cuhk03 classic split.
        use_cython (bool, optional): use cython code for evaluation. Default is True.
            This is highly recommended as the cython code can speed up the cmc computation
            by more than 10x. This requires Cython to be installed.
    """
    use_metric_cuhk03=False

    cmc_scores, mAP = evaluate_cy(
        distmat, q_pids, g_pids, q_camids, g_camids, max_rank,
        use_metric_cuhk03
    )
    cmc_topk = (1, 5, 10)
    logger = logging.getLogger('UnReID')
    logger.info('Mean AP: {:4.1%}'.format(mAP))
    logger.info('CMC Scores:')
    for k in cmc_topk:
        logger.info('  top-{:<4}{:12.1%}'.format(k, cmc_scores[k-1]))
    return cmc_scores[0], mAP
