#!/usr/bin/env python

import torch
import yaml
from hrnet.core.function import convert_label, convert_color
import numpy as np
from datasets.utils import visualize
from datasets.rellis_3d import Rellis3D as Dataset


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('../config/weights/smp/PSPNet_resnext50_32x4d_704x960_lr0.0001_bs6_epoch18_Rellis3D_iou_0.73.pth')
model = model.to(DEVICE)
model = model.eval()

# prepare data
test_dataset = Dataset(split='test', crop_size=(704, 960))

data_cfg = "../config/rellis.yaml"
CFG = yaml.safe_load(open(data_cfg, 'r'))
id_color_map = CFG["color_map"]

with torch.no_grad():
    image, gt_mask = test_dataset[0][:2]
    x = torch.from_numpy(image.transpose([2, 0, 1])).unsqueeze(0).to(DEVICE)

    pred = model(x)
    pred_np = pred.cpu().numpy().squeeze(0)

    pred_arg = np.argmax(pred_np, axis=0).astype(np.uint8) - 1
    pred_arg = convert_label(pred_arg, inverse=True)
    pred_color = convert_color(pred_arg, id_color_map)

    gt_arg = np.argmax(gt_mask, axis=0).astype(np.uint8) - 1
    gt_arg = convert_label(gt_arg, inverse=True)
    gt_color = convert_color(gt_arg, id_color_map)

    visualize(prediction=pred_color, gt=gt_color)
