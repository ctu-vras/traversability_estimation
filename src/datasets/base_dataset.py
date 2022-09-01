import cv2
import numpy as np
import random
from torch.utils import data
from datasets.laserscan import SemLaserScan
from traversability_estimation.utils import convert_label
import os
import yaml


data_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))

VOID_VALUE = 255
TRAVERSABILITY_LABELS = {0: "traversable",
                         1: "non-traversable",
                         VOID_VALUE: "background"}
TRAVERSABILITY_COLOR_MAP = {0:          [0, 255, 0],
                            1:          [255, 0, 0],
                            VOID_VALUE: [0, 0, 0]}


class BaseDatasetImages(data.Dataset):
    def __init__(self,
                 ignore_label=0,
                 base_size=2048,
                 crop_size=(512, 1024),
                 downsample_rate=1,
                 scale_factor=16,
                 mean=np.asarray([0.485, 0.456, 0.406]),
                 std=np.asarray([0.229, 0.224, 0.225])):

        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label

        self.mean = mean
        self.std = std
        self.scale_factor = scale_factor
        self.downsample_rate = 1. / downsample_rate

        self.files = []

    def __len__(self):
        return len(self.files)

    def input_transform(self, image):
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def label_transform(self, label):
        return np.array(label).astype('int32')

    def pad_image(self, image, h, w, size, padvalue):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=padvalue)
        return pad_image

    def rand_crop(self, image, label):
        h, w = image.shape[:-1]
        image = self.pad_image(image, h, w, self.crop_size,
                               (0.0, 0.0, 0.0))
        label = self.pad_image(label, h, w, self.crop_size,
                               (self.ignore_label,))

        new_h, new_w = label.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]

        return image, label

    def multi_scale_aug(self, image, label=None, rand_scale=1, rand_crop=True):
        long_size = np.int(self.base_size * rand_scale + 0.5)
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h),
                               interpolation=cv2.INTER_NEAREST)
        else:
            return image

        if rand_crop:
            image, label = self.rand_crop(image, label)

        return image, label

    def resize_short_length(self, image, label=None, short_length=None, fit_stride=None, return_padding=False):
        h, w = image.shape[:2]
        if h < w:
            new_h = short_length
            new_w = np.int(w * short_length / h + 0.5)
        else:
            new_w = short_length
            new_h = np.int(h * short_length / w + 0.5)        
        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        pad_w, pad_h = 0, 0
        if fit_stride is not None:
            pad_w = 0 if (new_w % fit_stride == 0) else fit_stride - (new_w % fit_stride)
            pad_h = 0 if (new_h % fit_stride == 0) else fit_stride - (new_h % fit_stride)
            image = cv2.copyMakeBorder(
                image, 0, pad_h, 0, pad_w, 
                cv2.BORDER_CONSTANT, value=tuple(x * 255 for x in self.mean[::-1])
            )

        if label is not None:
            label = cv2.resize(
                label, (new_w, new_h),
                interpolation=cv2.INTER_NEAREST)
            if pad_h > 0 or pad_w > 0:
                label = cv2.copyMakeBorder(
                    label, 0, pad_h, 0, pad_w, 
                    cv2.BORDER_CONSTANT, value=self.ignore_label
                )
            if return_padding:
                return image, label, (pad_h, pad_w)
            else:
                return image, label
        else:
            if return_padding:
                return image, (pad_h, pad_w)
            else:
                return image  

    def random_brightness(self, img, shift_value=10):
        if random.random() < 0.5:
            return img
        img = img.astype(np.float32)
        shift = random.randint(-shift_value, shift_value)
        img[:, :, :] += shift
        img = np.around(img)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def apply_augmentations(self, image, label, multi_scale=True, is_flip=True):
        if multi_scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            image, label = self.multi_scale_aug(image, label, rand_scale=rand_scale, rand_crop=True)

        image = self.random_brightness(image)
        image = self.input_transform(image)
        label = self.label_transform(label)

        image = image.transpose((2, 0, 1))

        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        if self.downsample_rate != 1:
            label = cv2.resize(
                label,
                None,
                fx=self.downsample_rate,
                fy=self.downsample_rate,
                interpolation=cv2.INTER_NEAREST
            )

        return image, label

    def reduce_zero_label(self, labelmap):
        labelmap = np.array(labelmap)
        encoded_labelmap = labelmap - 1

        return encoded_labelmap


class BaseDatasetClouds(data.Dataset):
    def __init__(self,
                 path=None,
                 fields=None,
                 color_map=None,
                 traversability_labels=False,
                 lidar_beams_step=1,
                 depth_img_H=128,
                 depth_img_W=1024,
                 lidar_fov_up=45.0,
                 lidar_fov_down=-45.0,
                 ):
        self.path = path
        self.split = None
        self.lidar_beams_step = lidar_beams_step
        if fields is None:
            fields = ['x', 'y', 'z', 'intensity', 'depth']
        self.fields = fields
        # make sure the input fields are supported
        assert set(self.fields) <= {'x', 'y', 'z', 'intensity', 'depth'}

        self.traversability_labels = traversability_labels
        self.color_map = None
        self.class_values = None
        self.scan = None

        self.depth_img_W = depth_img_W
        self.depth_img_H = depth_img_H
        self.lidar_fov_up = lidar_fov_up
        self.lidar_fov_down = lidar_fov_down

    def update_color_map(self, color_map):
        if not self.traversability_labels:
            self.label_map = None
            if not color_map:
                CFG = yaml.safe_load(open(os.path.join(data_dir, "../config/rellis.yaml"), 'r'))
                color_map = CFG["color_map"]
        else:
            label_map = yaml.safe_load(open(os.path.join(data_dir, "../config/rellis_to_obstacles.yaml"), 'r'))
            assert isinstance(label_map, (dict, list))
            if isinstance(label_map, dict):
                label_map = dict((int(k), int(v)) for k, v in label_map.items())
                n = max(label_map) + 1
                self.label_map = np.zeros((n,), dtype=np.uint8)
                for k, v in label_map.items():
                    self.label_map[k] = v
            elif isinstance(label_map, list):
                self.label_map = np.asarray(label_map)
            # traversability label map is assumed (0: traversable, 1: obstacle)
            color_map = TRAVERSABILITY_COLOR_MAP
            self.CLASSES = ["traversable", "obstacle"]

        self.color_map = color_map
        n_classes = len(self.CLASSES)
        # TRAVERSABILITY_LABELS = [0, 1, 255]
        self.class_values = np.sort([k for k in TRAVERSABILITY_LABELS.keys()]).tolist() if self.traversability_labels \
            else list(range(n_classes))

    def get_scan(self):
        self.scan = SemLaserScan(nclasses=len(self.CLASSES), sem_color_dict=self.color_map,
                                 project=True, H=self.depth_img_H, W=self.depth_img_W,
                                 fov_up=self.lidar_fov_up, fov_down=self.lidar_fov_down)

    def label_to_color(self, label):
        if len(label.shape) == 3:
            C, H, W = label.shape
            label = np.argmax(label, axis=0)
            assert label.shape == (H, W)
        if not self.traversability_labels:
            label = convert_label(label, inverse=True)
        color = self.scan.sem_color_lut[label]
        return color

    def generate_split(self, train_ratio=0.8):
        all_files = self.files.copy()
        if self.split == 'train':
            train_files = self.rng.choice(all_files, size=round(train_ratio * len(all_files)), replace=False).tolist()
            files = train_files
        elif self.split in ['val', 'test']:
            train_files = self.rng.choice(all_files, size=round(train_ratio * len(all_files)), replace=False).tolist()
            val_files = all_files.copy()
            for x in train_files:
                if x in val_files:
                    val_files.remove(x)
            files = val_files
        else:
            files = all_files
        self.files = files

        return files

    def create_sample(self, label=None):
        # TODO: transform point cloud and LABEL to ground frame (base_link)
        # following SalsaNext approach: (x, y, z, intensity, depth)
        xyzid = np.concatenate([self.scan.proj_xyz.transpose([2, 0, 1]),  # (3 x H x W)
                                self.scan.proj_remission[None],  # (1 x H x W)
                                self.scan.proj_range[None]], axis=0)  # (1 x H x W)

        # select input data according to the fields list
        ids = [['x', 'y', 'z', 'intensity', 'depth'].index(f) for f in self.fields]
        data = xyzid[ids]
        n_inputs, H, W = data.shape

        if label is None:
            label = self.scan.proj_sem_label.copy()
            if self.label_map is not None:
                label = self.label_map[label]

        if not self.traversability_labels:
            label = convert_label(label, inverse=False)

        assert data.shape[1:] == label.shape  # (N, H, W) and (H, W)
        assert set(np.unique(label)) <= set(self.class_values)  # label should contain only valid class values

        # 'masks': label.shape == (C, H, W) or 'labels': label.shape == (H, W)
        if self.labels_mode == 'masks':
            # extract certain classes from mask (one hot encoding)
            masks = [(label == v) for v in self.class_values]
            label = np.stack(masks, axis=0).astype('float')

            n_classes = len(self.class_values)
            assert label.shape == (n_classes, H, W)

        assert data.shape == (n_inputs, H, W)

        # sample depth image points in horizontal direction
        if self.lidar_beams_step:
            data = data[..., ::self.lidar_beams_step]
            label = label[..., ::self.lidar_beams_step]

        if self.split == 'train':
            data, label = self.apply_augmentations(data, label)

        return data, label

    def apply_augmentations(self, data, label):
        # with probability 0.5 flip from L to R image and mask
        if np.random.random() <= 0.5:
            data = np.fliplr(data.transpose((1, 2, 0)))
            data = data.transpose((2, 0, 1))

            if self.labels_mode == 'masks':
                label = np.fliplr(label.transpose((1, 2, 0)))
                label = label.transpose((2, 0, 1))
            else:
                label = np.fliplr(label)

        return data.copy(), label.copy()

    def __len__(self):
        return len(self.files)
