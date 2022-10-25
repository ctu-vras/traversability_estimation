# -*- coding:utf-8 -*-
# author: Awet H. Gebrehiwot
# --------------------------|
import torch
from utils.lovasz_losses import lovasz_softmax, lovasz_softmax_lcw, cross_entropy_lcw
from utils.loss_func import FocalLoss


def build(wce=True, lovasz=True, num_class=20, ignore_label=None, weights=None, ssl=False, fl=False):
    # focal loss and semisupervised learning
    if ssl and fl:
        if wce and lovasz:
            return FocalLoss(weight=weights, ignore_index=ignore_label), lovasz_softmax_lcw
        elif wce and not lovasz:
            return wce
        elif not wce and lovasz:
            return lovasz_softmax_lcw

    # only semi-supervised learning
    if ssl:
        if wce and lovasz:
            return cross_entropy_lcw, lovasz_softmax_lcw
        elif wce and not lovasz:
            return wce
        elif not wce and lovasz:
            return lovasz_softmax_lcw

    # focal loss on GT (fully supervised)
    if fl:
        loss_funs = FocalLoss(weight=weights, ignore_index=ignore_label)
    else:
        loss_funs = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    if wce and lovasz:
        return loss_funs, lovasz_softmax
    elif wce and not lovasz:
        return wce
    elif not wce and lovasz:
        return lovasz_softmax
    else:
        raise NotImplementedError
