# -*- coding:utf-8 -*-
# author: Awet H. Gebrehiwot
# at 8/10/22
# --------------------------|
import argparse
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data
from dataloader.pc_dataset import get_label_name, update_config
from utils.load_save_util import load_checkpoint
from utils.metric_util import per_class_iu, fast_hist_crop

import copy


def yield_target_dataset_loader(n_epochs, target_train_dataset_loader):
    for e in range(n_epochs):
        for i_iter_train, (_, train_vox_label, train_grid, _, train_pt_fea, ref_st_idx, ref_end_idx, lcw) \
                in enumerate(target_train_dataset_loader):
            yield train_vox_label, train_grid, train_pt_fea, ref_st_idx, ref_end_idx, lcw


class Trainer(object):
    def __init__(self,
                 model,
                 optimizer,
                 ckpt_dir,
                 unique_label,
                 unique_label_str,
                 lovasz_softmax,
                 loss_func,
                 ignore_label,
                 train_mode=None,
                 ssl=None,
                 eval_frequency=1,
                 pytorch_device=0,
                 warmup_epoch=1,
                 ema_frequency=5):
        self.model = model
        self.optimizer = optimizer
        self.model_save_path = ckpt_dir

        self.unique_label = unique_label
        self.unique_label_str = unique_label_str
        self.eval_frequency = eval_frequency
        self.lovasz_softmax = lovasz_softmax
        self.loss_func = loss_func
        self.ignore_label = ignore_label
        self.train_mode = train_mode
        self.ssl = ssl
        self.pytorch_device = pytorch_device
        self.warmup_epoch = warmup_epoch
        self.ema_frequency = ema_frequency
        self.val = False
        self.best_val_miou = 0
        self.progress_value = 100

    def criterion(self, outputs, point_label_tensor, lcw=None):
        if self.ssl:
            lcw_tensor = torch.FloatTensor(lcw).to(self.pytorch_device)

            loss = self.lovasz_softmax(torch.nn.functional.softmax(outputs), point_label_tensor,
                                       ignore=self.ignore_label, lcw=lcw_tensor) \
                   + self.loss_func(outputs, point_label_tensor, lcw=lcw_tensor)
        else:
            loss = self.lovasz_softmax(torch.nn.functional.softmax(outputs), point_label_tensor,
                                       ignore=self.ignore_label) \
                   + self.loss_func(outputs, point_label_tensor)
        return loss

    def validate(self, my_model, val_dataset_loader, val_batch_size, test_loader=None, ssl=None):
        hist_list = []
        val_loss_list = []
        my_model.eval()
        with torch.no_grad():
            for i_iter_val, (
                    _, val_vox_label, val_grid, val_pt_labs, val_pt_fea, ref_st_idx, ref_end_idx, lcw) in enumerate(
                val_dataset_loader):
                val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(self.pytorch_device) for i in
                                  val_pt_fea]
                val_grid_ten = [torch.from_numpy(i).to(self.pytorch_device) for i in val_grid]
                val_label_tensor = val_vox_label.type(torch.LongTensor).to(self.pytorch_device)

                predict_labels = my_model(val_pt_fea_ten, val_grid_ten, val_batch_size)
                # aux_loss = loss_fun(aux_outputs, point_label_tensor)

                inp = val_label_tensor.size(0)

                # TODO: check if this is correctly implemented
                # hack for batch_size mismatch with the number of training example
                predict_labels = predict_labels[:inp, :, :, :, :]

                # loss = self.criterion(predict_labels, val_label_tensor, lcw)

                predict_labels = torch.argmax(predict_labels, dim=1)
                predict_labels = predict_labels.cpu().detach().numpy()
                for count, i_val_grid in enumerate(val_grid):
                    hist_list.append(fast_hist_crop(predict_labels[
                                                        count, val_grid[count][:, 0], val_grid[count][:, 1],
                                                        val_grid[count][:, 2]], val_pt_labs[count],
                                                    self.unique_label))
                # val_loss_list.append(loss.detach().cpu().numpy())

        return hist_list, val_loss_list

    def fit(self, n_epochs, source_train_dataset_loader, train_batch_size, val_dataset_loader,
            val_batch_size, test_loader=None, ckpt_save_interval=1, lr_scheduler_each_iter=False):

        global_iter = 1
        best_val_miou = 0

        for epoch in range(n_epochs):

            pbar = tqdm(total=len(source_train_dataset_loader))
            # train the model
            loss_list = []
            self.model.train()
            # training with multi-frames and ssl:
            for i_iter_train, (
                    _, train_vox_label, train_grid, _, train_pt_fea, ref_st_idx, ref_end_idx, lcw) in enumerate(
                source_train_dataset_loader):
                # call the validation and inference with
                train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(self.pytorch_device) for i in
                                    train_pt_fea]
                # train_grid_ten = [torch.from_numpy(i[:,:2]).to(self.pytorch_device) for i in train_grid]
                train_vox_ten = [torch.from_numpy(i).to(self.pytorch_device) for i in train_grid]
                point_label_tensor = train_vox_label.type(torch.LongTensor).to(self.pytorch_device)

                # forward + backward + optimize
                outputs = self.model(train_pt_fea_ten, train_vox_ten, train_batch_size)
                inp = point_label_tensor.size(0)
                # print(f"outputs.size() : {outputs.size()}")
                # TODO: check if this is correctly implemented
                # hack for batch_size mismatch with the number of training example
                outputs = outputs[:inp, :, :, :, :]
                ################################

                loss = self.criterion(outputs, point_label_tensor, lcw)

                # TODO: check --> to mitigate only one element tensors can be converted to Python scalars
                loss = loss.mean()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Uncomment to use the learning rate scheduler
                # scheduler.step()

                loss_list.append(loss.item())

                if global_iter % self.progress_value == 0:
                    pbar.update(self.progress_value)
                    if len(loss_list) > 0:
                        print('epoch %d iter %5d, loss: %.3f\n' % (epoch, i_iter_train, np.mean(loss_list)))
                    else:
                        print('loss error')
                global_iter += 1

            # ----------------------------------------------------------------------#
            # Evaluation/validation
            with torch.no_grad():
                hist_list, val_loss_list = self.validate(self.model, val_dataset_loader, val_batch_size,
                                                         test_loader, self.ssl)

            # ----------------------------------------------------------------------#
            # Print validation mIoU and Loss
            print(f"--------------- epoch: {epoch} ----------------")
            iou = per_class_iu(sum(hist_list))
            print('Validation per class iou: ')
            for class_name, class_iou in zip(self.unique_label_str, iou):
                print('%s : %.2f%%' % (class_name, class_iou * 100))
            val_miou = np.nanmean(iou) * 100
            # del val_vox_label, val_grid, val_pt_fea

            # save model if performance is improved
            if best_val_miou < val_miou:
                best_val_miou = val_miou
                torch.save(self.model.state_dict(), self.model_save_path)

            print('Current val miou is %.3f while the best val miou is %.3f' %
                  (val_miou, best_val_miou))
            # print('Current val loss is %.3f' % (np.mean(val_loss_list)))
