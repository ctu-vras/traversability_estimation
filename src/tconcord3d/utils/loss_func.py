# -*- coding:utf-8 -*-
# author: Awet H. Gebrehiwot
# --------------------------|
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, weight=None, ignore_index=None,
                 gamma=2., reduction='none',  ssl=False):
        nn.Module.__init__(self)
        self.ignore_index = ignore_index
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.ssl = ssl

    def forward(self, input_tensor, target_tensor, lcw=None):
        log_prob = F.log_softmax(input_tensor, dim=1)
        prob = torch.exp(log_prob)
        raw_loss =  F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction,
            ignore_index=self.ignore_index
        )

        if self.ssl and lcw is not None:
            norm_lcw = (lcw/100.0)
            weighted_loss = (raw_loss * lcw).mean()
            return weighted_loss
        else:
            return raw_loss.mean()


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, weight=None, ignore_index=None,
                 gamma=2., reduction='none',  ssl=False):
        super().__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.ssl = ssl

    def forward(self, inputs, targets):
        inputs = inputs.squeeze()
        targets = targets.squeeze()

        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.weights[targets]*(1-pt)**self.gamma * BCE_loss

        return F_loss.mean()
