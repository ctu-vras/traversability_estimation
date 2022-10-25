# -*- coding:utf-8 -*-
import numpy as np


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count = np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def fast_hist_crop(output, target, unique_label):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label) + 2)
    hist = hist[unique_label + 1, :]
    hist = hist[:, unique_label + 1]
    return hist


# TODO: check if this implemented correctly
def fast_ups_crop(uncrt, target, unique_label):
    hist = [np.sum(uncrt[target==i]) for i in range(20)]
    va, cla_count = np.unique(target, return_counts=True)
    class_count = np.zeros(20)
    class_count[va] = cla_count
    return hist, class_count
