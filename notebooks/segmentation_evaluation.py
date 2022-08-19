import os
import sys

sys.path.append('/opt/ros/noetic/lib/python3/dist-packages/')
# sys.path.append('/home/ales/anaconda3/envs/fiftyone/lib/python3.9/site-packages/')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import yaml
import torch

import numpy as np
import fiftyone as fo
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from PIL import Image
from hrnet import models
from hrnet.config import config
from my_dataset import FODataset
from datasets.utils import visualize
from torch.utils.data import DataLoader
from datasets.rellis_3d import Rellis3D
from hrnet.core.function import convert_label, convert_color
from datasets.traversability_rellis import Rellis3D as Dataset

pkg_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
SMP_MODEL = os.path.join(pkg_path, "config/weights/smp/"
                                   "PSPNet_resnext50_32x4d_704x960_lr0.0001_bs6_epoch18_Rellis3D_iou_0.73.pth")
HRNET_MODEL = os.path.join(pkg_path, "config/weights/"
                                     "seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484/best.pth")
HRNET_MODEL_CONFIG = os.path.join(pkg_path, "config/hrnet_rellis/"
                                            "seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml")
TRAVERSABILITY_CONFIG = os.path.join(pkg_path, "config/rellis_traversability.yaml")
MY_DATASET = "my-dataset-2"
MY_DATASET_DIR = "/home/ales/dataset/cvat_dataset"


class ModelEvaluator(object):

    def __init__(self, model: str, dataset: str, device: str, image_size: tuple):
        """
        Initialize the model evaluator.
        :param model: hrnet or pspnet
        :param dataset: rellis3d or my-dataset
        :param device: cpu or cuda
        :param image_size: (width, height)
        """

        # device
        assert device in ['cuda', 'cpu']
        self.device = device

        # load model
        assert model in ['hrnet', 'pspnet']
        self.model_name = model
        if model == 'hrnet':
            self.model = self.load_hrnet()
        elif model == 'pspnet':
            self.model = torch.load(SMP_MODEL)

        # load dataset
        assert dataset in ['rellis3d', 'my-dataset']
        self.dataset_name = dataset
        if dataset == 'rellis3d':
            self.dataset = Rellis3D(crop_size=image_size[::-1], split='test')
        elif dataset == 'my-dataset':
            if fo.dataset_exists(MY_DATASET):
                cvat_dataset = fo.load_dataset(MY_DATASET)
                cvat_dataset.delete()
            cvat_dataset = fo.Dataset.from_dir(
                dataset_dir=MY_DATASET_DIR,
                dataset_type=fo.types.CVATImageDataset,
                name=MY_DATASET,
            )
            self.dataset = FODataset(cvat_dataset, crop_size=image_size[::-1])
        self.split = 'test'
        self.image_size = image_size
        self.cfg = yaml.safe_load(open(TRAVERSABILITY_CONFIG, 'r'))

    def load_hrnet(self) -> torch.nn.Module:
        with open(HRNET_MODEL_CONFIG, 'r') as f:
            cfg = yaml.safe_load(f)
        cfg['MODEL']['PRETRAINED'] = HRNET_MODEL
        with open(HRNET_MODEL_CONFIG, 'w') as f:
            yaml.dump(cfg, f)

        # Create model
        config.defrost()
        config.merge_from_file(HRNET_MODEL_CONFIG)
        config.merge_from_list(['TEST.MODEL_FILE', HRNET_MODEL])
        config.freeze()
        model = eval('models.' + config.MODEL.NAME + '.get_seg_model')(config)

        pretrained_dict = torch.load(HRNET_MODEL, map_location='cpu')
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                           if k[6:] in model_dict.keys()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        model = model.eval()
        model = model.to(device=self.device, dtype=torch.float)
        return model

    def evaluate(self) -> float:
        iou_sum = 0
        avg_iou = 0
        for i in range(len(self.dataset)):
            # load image and ground truth mask
            image, gt_mask = self.dataset[i]
            gt_mask = self.process_ground_truth(gt_mask)

            # prepare image for prediction
            x_tensor = self.prepare_image(image)

            # predict mask
            mask = self.model_predict(x_tensor)
            pr_mask = map_labels(mask)

            # compute IoU
            iou = compute_IoU(pr_mask, gt_mask)
            iou_sum += iou
            avg_iou = iou_sum / (i + 1)
            sys.stdout.write(f"\r IoU: {iou}, avg IoU: {avg_iou}, validated: {i / len(self.dataset) * 100:.2f} %")
            sys.stdout.flush()

        return avg_iou

    def visualize_prediction(self):
        n = np.random.choice(len(self.dataset))

        image, gt_mask = self.dataset[n]
        image_vis = image * self.dataset.std + self.dataset.mean
        gt_mask = self.process_ground_truth(gt_mask)
        x_tensor = self.prepare_image(image)
        # predict mask
        mask = self.model_predict(x_tensor)
        pr_mask = map_labels(mask)

        color_map = {0: (0, 0, 0),
                     1: (0, 255, 0),
                     2: (255, 0, 0)}
        pred_colormap = convert_color(pr_mask, color_map)
        gt_colormap = convert_color(gt_mask, color_map)

        visualize(image=image_vis, pred=pred_colormap, gt=gt_colormap)

    def model_predict(self, image: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            if self.model_name == 'pspnet':
                pred = self.model.predict(image)
                pred = pred.squeeze().cpu().numpy()
                mask = np.argmax(pred, axis=0).astype(np.uint8) - 1
            elif self.model_name == 'hrnet':
                pred = self.model.forward(image)
                pred = pred[config.TEST.OUTPUT_INDEX]
                pred = torch.softmax(pred, dim=1)
                pred = pred.squeeze().cpu().numpy()
                mask = np.argmax(pred, axis=0).astype(np.uint8)
            else:
                raise ValueError(f'Unknown model: {self.model_name}')
            mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_LINEAR)
        return mask

    def prepare_image(self, image: np.ndarray):
        if self.split == 'val':
            image = torch.from_numpy(image).to(self.device).unsqueeze(0)
        elif self.split == 'test':
            image = torch.from_numpy(image.transpose([2, 0, 1])).to(self.device).unsqueeze(0)
        else:
            raise ValueError(f'Unknown split: {self.split}')
        return image

    def process_ground_truth(self, mask: np.ndarray) -> np.ndarray:
        if self.dataset_name == 'rellis3D':
            mask = np.argmax(mask, axis=0).astype(np.uint8) - 1
            mask = map_labels(mask.squeeze())

        elif self.dataset_name == 'my-dataset':
            pass
        else:
            raise ValueError(f'Unknown dataset: {self.dataset_name}')
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_LINEAR)
        return mask

    @staticmethod
    def train_model():
        ENCODER = 'se_resnext50_32x4d'
        ENCODER_WEIGHTS = 'imagenet'
        CLASSES = ['void', 'traversable', 'untraversable']
        ACTIVATION = 'sigmoid' if len(CLASSES) == 1 else 'softmax2d'
        DEVICE = 'cuda'
        IMG_SIZE = (192, 320)
        LR = 0.0001

        # create segmentation model with pretrained encoder
        model = smp.FPN(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION
        )

        train_dataset = Dataset(classes=CLASSES, crop_size=IMG_SIZE, split='train')
        valid_dataset = Dataset(classes=CLASSES, crop_size=IMG_SIZE, split='val')

        train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True, num_workers=12)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

        loss = smp.utils.losses.DiceLoss()
        metrics = [smp.utils.metrics.IoU(threshold=0.5)]
        optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=LR)])

        train_epoch = smp.utils.train.TrainEpoch(
            model,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            device=DEVICE,
            verbose=True,
        )

        valid_epoch = smp.utils.train.ValidEpoch(
            model,
            loss=loss,
            metrics=metrics,
            device=DEVICE,
            verbose=True,
        )

        max_score = 0
        n_epochs = 3
        for i in range(0, n_epochs):

            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)

            # do something (save model, change lr, etc.)
            if max_score < valid_logs['iou_score']:
                max_score = valid_logs['iou_score']
                torch.save(model, './best_model.pth')
                print('Model saved!')

            if i == 25:
                optimizer.param_groups[0]['lr'] = 1e-5
                print('Decrease decoder learning rate to 1e-5!')

    def save_predictions(self):
        assert self.dataset_name == 'my-dataset'
        for i in range(len(self.dataset)):
            image, gt_mask = self.dataset[i]
            x_tensor = self.prepare_image(image)
            mask = self.model_predict(x_tensor)
            pr_mask = map_labels(mask)
            self.dataset.save_prediction(pr_mask, i)

    def show_dataset(self):
        assert self.dataset_name == 'my-dataset'
        session = fo.launch_app(self.dataset.dataset)
        session.wait()


def compute_IoU(pred: np.ndarray, gt: np.ndarray) -> float:
    classes = np.unique(np.concatenate((pred, gt)))
    classes = classes[classes != 0]
    intersection_sum = 0
    union_sum = 0
    for cls in classes:
        pred_cls = pred == cls
        gt_cls = gt == cls
        intersection_sum += np.sum(np.logical_and(pred_cls, gt_cls))
        union_sum += np.sum(np.logical_or(pred_cls, gt_cls))
    return intersection_sum / union_sum


def mask_to_colormap(mask, cfg):
    mask = np.argmax(mask, axis=0).astype(np.uint8) - 1
    mask = convert_label(mask, True)
    mask = convert_color(mask, cfg["color_map"])
    return mask


def map_labels(mask):
    LABEL_MAPPING = {0: 1, 1: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 1, 9: 1, 10: 2, 11: 2, 12: 2,
                     13: 2, 14: 2, 15: 2, 16: 1, 17: 1, 18: 1, 255: 0}
    temp = mask.copy()
    for k, v in LABEL_MAPPING.items():
        mask[temp == k] = v
    return mask


def main():
    evaluator = ModelEvaluator('hrnet', 'my-dataset', 'cuda', (320, 192))
    # evaluator.evaluate()
    # evaluator.visualize_prediction()
    evaluator.save_predictions()
    evaluator.show_dataset()


if __name__ == '__main__':
    main()
