from __future__ import absolute_import
import logging
import time
import datetime
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import numpy as np
from sklearn.cluster import DBSCAN

from ..models import create_model
from ..data.build import build_data, build_loader
from ..metrics.Partevaluators import Evaluator, extract_features
from ..metrics.ranking import accuracy
from ..optim.optimizer import build_optimizer
from ..optim.lr_scheduler import build_lr_scheduler
from ..loss import CrossEntropyLabelSmooth, SoftTripletLoss, SoftEntropy
from ..utils.meters import AverageMeter
from ..utils.serialization import save_checkpoint, load_checkpoint, copy_state_dict
from ..cluster.faiss_utils import compute_jaccard_distance
from ..cluster.RefineCluster import RefineClusterProcess
from ..utils.osutils import PathManager

class mutualrefine(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = logging.getLogger('UnReID')
        self.best_mAP = 0
        self.best_top1 = 0
        self.cluster_list = []

    def _build_dataset(self):
        self.Target_dataset = build_data(self.cfg.DATASETS.TARGET, self.cfg.DATASETS.DIR)
        self.Target_cluster_loader = build_loader(self.cfg, None, inputset=sorted(self.Target_dataset.train), is_train=False)
        self.Target_test_loader = build_loader(self.cfg, self.Target_dataset, is_train=False)
        self.num_classes = len(self.Target_dataset.train)

    def _build_model(self):
        self.model = create_model(self.cfg, self.num_classes)
        self.model_ema = create_model(self.cfg, self.num_classes)


        if self.cfg.CHECKPOING.PRETRAIN_PATH:
            initial_weights = load_checkpoint(self.cfg.CHECKPOING.PRETRAIN_PATH)
            copy_state_dict(initial_weights['state_dict'], self.model)
            copy_state_dict(initial_weights['state_dict'], self.model_ema)

        if self.cfg.CHECKPOING.EVAL:
            initial_weights = load_checkpoint(self.cfg.CHECKPOING.EVAL)
            copy_state_dict(initial_weights['state_dict'], self.model_ema)

        start_epoch = 0

        self.model = nn.DataParallel(self.model)
        self.model_ema = nn.DataParallel(self.model_ema)

        for param in self.model_ema.parameters():
            param.detach_()

        self.evaluator = Evaluator(self.cfg, self.model_ema)

        return start_epoch

    def _build_optim(self, epoch):
        if self.cfg.OPTIM.SCHED == 'single_step':
            scale = 1.0
            if self.cfg.CHECKPOING.PRETRAIN_PATH:
                scale = 1.0
            else:
                if epoch < 40:
                    scale = 1.0
                elif epoch < 60:
                    scale = 0.3
                elif epoch < 80:
                    scale = 0.1
            LR = self.cfg.OPTIM.LR * scale
        else:
            raise NotImplementedError("NO {} for UDA".format(self.cfg.OPTIM.SCHED))
        self.optimizer = build_optimizer(self.cfg, self.model, LR = LR)
        self.logger.info('lr: {:.8f}'.format(self.optimizer.param_groups[0]['lr']))

    def run(self):
        self._build_dataset()
        start_epoch = self._build_model()
        if self.cfg.CHECKPOING.EVAL:
            self.eval()
            return

        self.start_epoch = start_epoch
        for epoch in range(start_epoch, self.cfg.OPTIM.EPOCHS):
            epoch_time = time.time()

            self.generate_pseudo_dataset(epoch)
            self._build_optim(epoch)
            self.init_train()
            self.train(epoch)
            self.eval_save(epoch)

            eta_seconds = (time.time()-epoch_time) * (self.cfg.OPTIM.EPOCHS - (epoch + 1))
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            self.logger.info('eta: {}'.format(eta_str))

    def generate_pseudo_dataset(self, epoch):

        self.logger.info('Extract feat and Calculate dist...')
        dict_f = extract_features(self.model_ema, self.Target_cluster_loader, print_freq = self.cfg.TEST.PRINT_PERIOD, GenLabel = True)
        part_num = len(dict_f[self.Target_dataset.train[0][0]])
        model_FC = ['classifier', 'classifier_partup', 'classifier_partdown']
        cf = [torch.cat([dict_f[f][i].unsqueeze(0) for f, _, _ in sorted(self.Target_dataset.train)], 0) for i in range(part_num)]
        self.num_clusters_list = []
        self.labels_list = []

        for i in range(part_num):
            rerank_dist = compute_jaccard_distance(cf[i])

            if (epoch==0 or epoch == self.start_epoch):
                # # DBSCAN cluster
                if self.cfg.CHECKPOING.PRETRAIN_PATH:
                    tri_mat = np.triu(rerank_dist, 1)
                    tri_mat = tri_mat[np.nonzero(tri_mat)]
                    tri_mat = np.sort(tri_mat,axis=None)
                    rho = 1.6e-3
                    top_num = np.round(rho*tri_mat.size).astype(int)
                    eps = tri_mat[:top_num].mean()
                else:
                    eps = self.cfg.CLUSTER.EPS

                self.logger.info('eps for cluster: {:.3f}'.format(eps))
                self.cluster_list.append(DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1))

            self.logger.info('Clustering and labeling...')
            labels = self.cluster_list[i].fit_predict(rerank_dist)
            num_ids = len(set(labels)) - (1 if -1 in labels else 0)
            self.num_clusters_list.append(num_ids)
            self.labels_list.append(labels)
            self.logger.info('Clustered into {} classes'.format(num_ids))

        self.new_num_clusters_list = []
        self.new_labels_list = []
        self.refine_index = [[1,2],[0],[0]]
        self.logger.info(self.refine_index)
        for i in range(part_num):
            target_labels = self.labels_list[i]

            new_labels_list = []
            tmp_refine_index = self.refine_index[i]
            for j in tmp_refine_index:
                new_labels_list.append(RefineClusterProcess(self.labels_list[j], target_labels, divide_ratio = self.cfg.CLUSTER.REFINE_K))

            labels = []
            dictReindex = dict()
            label_index = 0
            useful_nums = 0
            for j in range(len(target_labels)):
                tmp_label = [int(target_labels[j])]
                tmp_label += [int(x[j]) for x in new_labels_list]
                if -1 in tmp_label:
                    labels.append(-1)
                else:
                    labels.append(int(target_labels[j]))
                    useful_nums += 1
                    if int(target_labels[j]) not in dictReindex.keys():
                        dictReindex[int(target_labels[j])] = label_index
                        label_index += 1

            for index in range(len(labels)):
                if labels[index]==-1: continue
                labels[index] = dictReindex[labels[index]]

            del_samples = (len(labels) - useful_nums)-len(np.where(target_labels == -1)[0])
            self.logger.info('useful samples {}, mutual refine del {} samples, del {} noisy cluster'.format(useful_nums, del_samples, self.num_clusters_list[i] - label_index))
            self.new_labels_list.append(labels)
            self.new_num_clusters_list.append(label_index)

            cluster_centers_dict = collections.defaultdict(list)
            for index, ((fname, _, cid), label) in enumerate(zip(sorted(self.Target_dataset.train), labels)):
                if label==-1: continue
                cluster_centers_dict[label].append(cf[i][index])

            cluster_centers = [torch.stack(cluster_centers_dict[idx]).mean(0) for idx in sorted(cluster_centers_dict.keys())]
            cluster_centers = torch.stack(cluster_centers)
            model_param = getattr(self.model.module, model_FC[i])
            model_param.weight.data[:label_index].copy_(F.normalize(cluster_centers, dim=1).float().cuda())
            model_ema_param = getattr(self.model_ema.module, model_FC[i])
            model_ema_param.weight.data[:label_index].copy_(F.normalize(cluster_centers, dim=1).float().cuda())

        self.num_clusters_list = self.new_num_clusters_list
        self.labels_list = self.new_labels_list

        new_dataset = []
        for i, (fname, _, cid) in enumerate(sorted(self.Target_dataset.train)):
            label = []
            for L in range(part_num):
                label.append(int(self.labels_list[L][i]))
            if -1 in label:
                continue
            new_dataset.append((fname, label, cid))

        self.logger.info('new dataset length {}'.format(len(new_dataset)))
        self.Target_train_loader = build_loader(self.cfg, None, inputset=new_dataset, num_instances = self.cfg.DATALOADER.NUM_INSTANCES, is_train=True)
        assert self.cfg.DATALOADER.ITERS == len(self.Target_train_loader)

    def eval(self):
        self.logger.info('bn_x')
        use_cython = True
        self.logger.info('Use Cython Eval...')
        return self.evaluator.evaluate(self.Target_test_loader,
                self.Target_dataset.query, self.Target_dataset.gallery, use_cython = use_cython)

    def eval_save(self, epoch):
        if (epoch+1) != self.cfg.OPTIM.EPOCHS and self.cfg.CHECKPOING.SAVE_STEP[0] > 0 and (epoch+1) not in self.cfg.CHECKPOING.SAVE_STEP:
            return
        elif (epoch+1) != self.cfg.OPTIM.EPOCHS and self.cfg.CHECKPOING.SAVE_STEP[0] < 0  and (epoch+1) % -self.cfg.CHECKPOING.SAVE_STEP[0] != 0:
            return

        top1, mAP = self.eval()[0]

        is_top1_best = top1 > self.best_top1
        self.best_top1 = max(top1, self.best_top1)
        is_mAP_best = mAP > self.best_mAP
        self.best_mAP = max(mAP, self.best_mAP)

        _state_dict = self.model_ema.module.state_dict()

        save_checkpoint({
            'state_dict': _state_dict,
            'epoch': epoch + 1,
            'best_top1': self.best_top1,
            'best_mAP': self.best_mAP
        }, is_top1_best, is_mAP_best, fpath=osp.join(self.cfg.OUTPUT_DIR, 'checkpoint_new.pth.tar'), remain=self.cfg.CHECKPOING.REMAIN_CLASSIFIER)

        self.logger.info('Finished epoch {:3d}\n Target mAP: {:5.1%}  best: {:5.1%}{}\nTarget top1: {:5.1%}  best: {:5.1%}{}'.
              format(epoch + 1, mAP, self.best_mAP, ' *' if is_mAP_best else '', top1, self.best_top1, ' *' if is_top1_best else ''))

        return

    def init_train(self):
        self.criterion_ce = CrossEntropyLabelSmooth(self.num_clusters_list[0], epsilon = self.cfg.MODEL.LOSSES.CE.EPSILON).cuda()
        self.criterion_ce_up = CrossEntropyLabelSmooth(self.num_clusters_list[1], epsilon = self.cfg.MODEL.LOSSES.CE.EPSILON).cuda()
        self.criterion_ce_down = CrossEntropyLabelSmooth(self.num_clusters_list[2], epsilon = self.cfg.MODEL.LOSSES.CE.EPSILON).cuda()
        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()

        self.criterion_ce_soft = SoftEntropy().cuda()
        self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()

    def train(self, epoch):
        self.Target_train_loader.new_epoch()

        self.model.train()
        self.model_ema.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        losses_ce_soft = AverageMeter()
        losses_tri_soft = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i in range(self.cfg.DATALOADER.ITERS):
            target_inputs = self.Target_train_loader.next()
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(target_inputs)

            [x, part_up, part_down], _, [prob, prob_part_up, prob_part_down] = self.model(inputs, finetune = True)
            prob = prob[:,:self.num_clusters_list[0]]
            prob_part_up = prob_part_up[:,:self.num_clusters_list[1]]
            prob_part_down = prob_part_down[:,:self.num_clusters_list[2]]

            [x_ema, part_up_ema, part_down_ema], _, [prob_ema, prob_part_up_ema, prob_part_down_ema] = self.model_ema(inputs, finetune = True)
            prob_ema = prob_ema[:,:self.num_clusters_list[0]]
            prob_part_up_ema = prob_part_up_ema[:,:self.num_clusters_list[1]]
            prob_part_down_ema = prob_part_down_ema[:,:self.num_clusters_list[2]]

            loss_ce = self.criterion_ce(prob, targets[0]) + \
                      self.criterion_ce_up(prob_part_up, targets[1]) + \
                      self.criterion_ce_down(prob_part_down, targets[2])
            loss_tri = self.criterion_tri(x, x, targets[0])  + \
                       self.criterion_tri(part_up, part_up, targets[1]) + \
                       self.criterion_tri(part_down, part_down, targets[2])

            loss_ce_soft = self.criterion_ce_soft(prob, prob_ema) + \
                           self.criterion_ce_soft(prob_part_up, prob_part_up_ema) + \
                           self.criterion_ce_soft(prob_part_down, prob_part_down_ema)
            loss_tri_soft = self.criterion_tri_soft(x, x_ema, targets[0]) + \
                            self.criterion_tri_soft(part_up, part_up_ema, targets[1]) + \
                            self.criterion_tri_soft(part_down, part_down_ema, targets[2])

            loss = loss_ce * (1-self.cfg.MEAN_TEACH.CE_SOFT_WRIGHT) + \
                    loss_tri * (1-self.cfg.MEAN_TEACH.TRI_SOFT_WRIGHT) + \
                    loss_ce_soft * self.cfg.MEAN_TEACH.CE_SOFT_WRIGHT + \
                    loss_tri_soft * self.cfg.MEAN_TEACH.TRI_SOFT_WRIGHT

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.update_ema_variables(self.model, self.model_ema, self.cfg.MEAN_TEACH.ALPHA, epoch*len(self.Target_train_loader) + i)

            prec = accuracy(prob_ema.data, targets[0].data)

            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            losses_ce_soft.update(loss_ce_soft.item())
            losses_tri_soft.update(loss_tri_soft.item())
            precisions.update(prec)

            batch_time.update(time.time() - end)

            if ((i + 1) % self.cfg.PRINT_PERIOD == 0) or ((i + 1) % self.cfg.DATALOADER.ITERS == 0):
                self.logger.info('Epoch: [{}][{}/{}]\t'
                                 'Time {:.3f} ({:.3f})\t'
                                 'Data {:.3f} ({:.3f})\t'
                                 'Loss_ce {:.3f} ({:.3f})\t'
                                 'Loss_tri {:.3f} ({:.3f})\t'
                                 'Loss_ce_soft {:.3f} ({:.3f})\t'
                                 'Loss_tri_soft {:.3f} ({:.3f})\t'
                                 'Prec {:.2%} ({:.2%})\t'
                                 .format(epoch + 1, i + 1, self.cfg.DATALOADER.ITERS,
                                         batch_time.val, batch_time.avg,
                                         data_time.val, data_time.avg,
                                         losses_ce.val, losses_ce.avg,
                                         losses_tri.val, losses_tri.avg,
                                         losses_ce_soft.val, losses_ce_soft.avg,
                                         losses_tri_soft.val, losses_tri_soft.avg,
                                         precisions.val, precisions.avg))

            end = time.time()

    def update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha = 1 - alpha)

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.cuda()
        targets = [p.cuda() for p in pids]
        return inputs, targets
