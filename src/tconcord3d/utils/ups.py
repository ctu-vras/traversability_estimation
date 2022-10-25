# -*- coding:utf-8 -*-
# author: Awet H. Gebrehiwot
# --------------------------|


def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
