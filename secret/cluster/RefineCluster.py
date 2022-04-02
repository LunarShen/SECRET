from __future__ import print_function, absolute_import
import numpy
import numpy as np
import collections
import torch

def RefineClusterProcess(Reference_Cluster_result, Target_Cluster_result, divide_ratio):
    L = len(Reference_Cluster_result)
    assert L == len(Target_Cluster_result)

    Target_Cluster_nums = len(set(Target_Cluster_result)) - (1 if -1 in Target_Cluster_result else 0)

    Final_Cluster = np.zeros(L, dtype=np.int64) - 1
    assert len(np.where(Final_Cluster == -1)[0]) == L

    ban_cluster = 0
    for Target_Cluster in range(Target_Cluster_nums):
        Target_Cluster_index = np.where(Target_Cluster_result == Target_Cluster)[0]

        zero_index = np.where(Reference_Cluster_result == -1)[0]
        Target_Cluster_index = np.setdiff1d(Target_Cluster_index, zero_index)

        if np.size(Target_Cluster_index) == 0:
            ban_cluster+=1
            continue
        num_ID = len(Target_Cluster_index)
        num_Part = np.bincount(Reference_Cluster_result[Target_Cluster_index])
        ban_flag = True

        for i in range(int(1/divide_ratio)):
            _max = np.argmax(num_Part)

            if num_Part[_max] > 0 and num_Part[_max] > num_ID * divide_ratio:
                Reference_Cluster_index = np.where(Reference_Cluster_result == _max)[0]
                fit_condition = np.intersect1d(Target_Cluster_index, Reference_Cluster_index)
                Final_Cluster[fit_condition] = Target_Cluster - ban_cluster
                num_Part[_max] = 0
                ban_flag = False
            else:
                break
        if ban_flag:
            ban_cluster += 1

    return Final_Cluster
