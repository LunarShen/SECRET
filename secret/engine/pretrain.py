from __future__ import absolute_import
import logging
import time
import torch.nn as nn
import os.path as osp

from ..models import create_model
from ..data.build import build_data, build_loader
from ..metrics.Partevaluators import Evaluator
from ..metrics.ranking import accuracy
from ..optim.optimizer import build_optimizer
from ..optim.lr_scheduler import build_lr_scheduler
from ..loss import CrossEntropyLabelSmooth, SoftTripletLoss
from ..utils.meters import AverageMeter
from ..utils.serialization import save_checkpoint, load_checkpoint, copy_state_dict

class pretrain(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = logging.getLogger('UnReID')
        self.best_mAP = 0
        self.best_top1 = 0

    def _build_dataset(self):
        self.Source_dataset = build_data(self.cfg.DATASETS.SOURCE, self.cfg.DATASETS.DIR)
        self.Target_dataset = build_data(self.cfg.DATASETS.TARGET, self.cfg.DATASETS.DIR)
        self.Source_train_loader = build_loader(self.cfg, self.Source_dataset, num_instances = self.cfg.DATALOADER.NUM_INSTANCES, is_train=True)
        self.Target_train_loader = build_loader(self.cfg, self.Target_dataset, num_instances = 0,is_train=True)
        self.Target_test_loader = build_loader(self.cfg, self.Target_dataset, is_train=False)
        self.num_classes = self.Source_dataset.num_train_pids

    def _build_model(self):
        self.model = create_model(self.cfg, self.num_classes)

        start_epoch = 0

        self.model = nn.DataParallel(self.model)
        self.evaluator = Evaluator(self.cfg, self.model)

        return start_epoch

    def _build_optim(self):
        self.optimizer = build_optimizer(self.cfg, self.model)
        self.lr_scheduler = build_lr_scheduler(self.cfg, self.optimizer)

    def run(self):
        self._build_dataset()
        start_epoch = self._build_model()
        self._build_optim()

        self.init_train()
        for epoch in range(start_epoch, self.cfg.OPTIM.EPOCHS):
            self.train(epoch)
            self.eval_save(epoch)

    def eval(self):
        return self.evaluator.evaluate(self.Target_test_loader,
                self.Target_dataset.query, self.Target_dataset.gallery)


    def eval_save(self, epoch):
        if (epoch+1) != self.cfg.OPTIM.EPOCHS and self.cfg.CHECKPOING.SAVE_STEP[0] > 0 and (epoch+1) not in self.cfg.CHECKPOING.SAVE_STEP:
            return
        elif (epoch+1) != self.cfg.OPTIM.EPOCHS and self.cfg.CHECKPOING.SAVE_STEP[0] < 0  and (epoch+1) % -self.cfg.CHECKPOING.SAVE_STEP[0] != 0:
            return

        results = self.eval()

        _state_dict = self.model.module.state_dict()

        save_checkpoint({
            'state_dict': _state_dict,
            'epoch': epoch + 1
        }, False, False, fpath=osp.join(self.cfg.OUTPUT_DIR, 'checkpoint_new.pth.tar'), remain=self.cfg.CHECKPOING.REMAIN_CLASSIFIER)

        self.logger.info('Finished epoch {:3d}'.
              format(epoch + 1))

        return

    def init_train(self):
        self.criterion_ce = CrossEntropyLabelSmooth(self.num_classes, epsilon = self.cfg.MODEL.LOSSES.CE.EPSILON).cuda()
        self.criterion_triple = SoftTripletLoss(margin=0.0).cuda()

    def train(self, epoch):
        self.Source_train_loader.new_epoch()
        self.Target_train_loader.new_epoch()
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        precisions = AverageMeter()

        end = time.time()

        for i in range(self.cfg.DATALOADER.ITERS):
            source_inputs = self.Source_train_loader.next()
            target_inputs = self.Target_train_loader.next()
            data_time.update(time.time() - end)

            s_inputs, targets = self._parse_data(source_inputs)
            t_inputs, _ = self._parse_data(target_inputs)

            [x, part_up, part_down], [prob, prob_part_up, prob_part_down] = self.model(s_inputs)
            self.model(t_inputs)

            # backward main #
            loss_ce = self.criterion_ce(prob, targets) + self.criterion_ce(prob_part_up, targets) + self.criterion_ce(prob_part_down, targets)
            loss_tri = self.criterion_triple(x, x, targets) + self.criterion_triple(part_up, part_up, targets) + self.criterion_triple(part_down, part_down, targets)
            loss = loss_ce + loss_tri

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            prec = accuracy(prob.data, targets.data)
            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            precisions.update(prec)

            batch_time.update(time.time() - end)

            if ((i + 1) % self.cfg.PRINT_PERIOD == 0) or ((i + 1) % self.cfg.DATALOADER.ITERS == 0):
                self.logger.info('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tri {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})'
                      .format(epoch+1, i + 1, self.cfg.DATALOADER.ITERS,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tri.val, losses_tri.avg,
                              precisions.val, precisions.avg))

            end = time.time()

        self.lr_scheduler.step()

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets
