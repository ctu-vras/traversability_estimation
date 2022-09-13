from __future__ import absolute_import
import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
from numpy.lib.recfunctions import structured_to_unstructured
from os.path import dirname, join, realpath
from traversability_estimation.utils import *
from datasets.base_dataset import BaseDatasetImages, BaseDatasetClouds
from datasets.base_dataset import TRAVERSABILITY_COLOR_MAP, TRAVERSABILITY_LABELS
from datasets.base_dataset import FLEXIBILITY_COLOR_MAP, FLEXIBILITY_LABELS
from datasets.laserscan import SemLaserScan
from copy import copy
import torch
from PIL import Image
import yaml

__all__ = [
    'data_dir',
    'seq_names',
    'Rellis3DSequence',
    'Rellis3DImages',
    'Rellis3DClouds',
]

data_dir = realpath(join(dirname(__file__), '..', '..', 'data'))

seq_names = [
    '00000',
    '00001',
    '00002',
    '00003',
    '00004',
]


# seq_names = ['%05d' % i for i in range(5)]


class Rellis3DSequence(torch.utils.data.Dataset):
    def __init__(self, seq=None, path=None, poses_file='poses.txt', poses_path=None, color_map=None):
        """Rellis-3D dataset: https://unmannedlab.github.io/research/RELLIS-3D.

        :param seq: Sequence number (from 0 to 4).
        :param path: Dataset path, takes precedence over name.
        :param poses_file: Poses CSV file name.
        :param poses_path: Override for poses CSV path.
        """
        assert isinstance(seq, str) or isinstance(seq, int)
        if isinstance(seq, int):
            seq = '%05d' % seq
        parts = seq.split('/')
        assert 1 <= len(parts) <= 2
        if len(parts) == 2:
            assert parts[0] == 'rellis_3d'
            seq = parts[1]
        if path is None:
            path = join(data_dir, 'Rellis_3D')

        if not color_map:
            CFG = yaml.safe_load(open(os.path.join(data_dir, "../config/rellis.yaml"), 'r'))
            color_map = CFG["color_map"]
        self.color_map = color_map
        n_classes = len(color_map)
        self.scan = SemLaserScan(n_classes, self.color_map, project=True)

        self.seq = seq
        self.path = path
        self.poses_path = poses_path
        self.poses_file = poses_file
        P = np.zeros([3, 4])
        K = read_intrinsics(self.intrinsics_path())
        P[:3, :3] = K
        self.calibration = {
            'K': K,
            'P': P,
            'lid2cam': read_extrinsics(self.extrinsics_path()),
            'dist_coeff': np.array([-0.134313, -0.025905, 0.002181, 0.00084, 0]),
            'img_width': 1920,
            'img_height': 1200,
        }
        if self.poses_path or self.path:
            self.poses = read_poses(self.cloud_poses_path())
            self.ids_lid, self.ts_lid = self.get_ids(sensor='lidar')
            self.ids_rgb, self.ts_rgb = self.get_ids(sensor='rgb')
            self.ids_semseg, self.ts_semseg = self.get_ids(sensor='semseg')
            self.ids = self.ids_lid
        else:
            self.ids = None
            self.poses = None

    def get_ids(self, sensor='lidar'):
        if sensor == 'lidar':
            sensor_folder = 'os1_cloud_node_color_ply'
        elif sensor == 'rgb':
            sensor_folder = 'pylon_camera_node'
        elif sensor == 'semseg':
            sensor_folder = 'pylon_camera_node_label_id'
        else:
            raise ValueError('Unsupported sensor type (choose one of: lidar, or rgb, or semseg)')
        # id = frame0000i_sec_msec
        ids = [f[:-4] for f in np.sort(os.listdir(os.path.join(self.path, self.seq, sensor_folder)))]
        ts = [float('%.3f' % (float(id.split('-')[1].split('_')[0]) + float(id.split('-')[1].split('_')[1]) / 1000.0))
              for id in ids]
        ts = np.sort(ts).tolist()
        return ids, ts

    def local_cloud_path(self, id, filetype='bin'):
        if filetype == '.ply':
            return os.path.join(self.path, self.seq, 'os1_cloud_node_color_ply', '%s.ply' % id)
        else:
            return os.path.join(self.path, self.seq, 'os1_cloud_node_kitti_bin', '%06d.bin' % self.ids_lid.index(id))

    def cloud_label_path(self, id):
        return os.path.join(self.path, self.seq, 'os1_cloud_node_semantickitti_label_id',
                            '%06d.label' % self.ids_lid.index(id))

    def cloud_poses_path(self):
        if self.poses_path:
            return self.poses_path
        return os.path.join(self.path, 'calibration', self.seq, self.poses_file)

    def image_path(self, id):
        return os.path.join(self.path, self.seq, 'pylon_camera_node', '%s.jpg' % id)

    def semseg_path(self, id):
        return os.path.join(self.path, self.seq, 'pylon_camera_node_label_id', '%s.png' % id)

    def intrinsics_path(self):
        return os.path.join(self.path, 'calibration', self.seq, 'camera_info.txt')

    def extrinsics_path(self):
        return os.path.join(self.path, 'calibration', self.seq, 'transforms.yaml')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        if isinstance(item, int):
            id = self.ids[item]
            return self.local_cloud(id), self.cloud_label(id), \
                   self.cloud_pose(id), \
                   self.camera_image(id), self.camera_semseg(id)

        ds = copy(self)
        if isinstance(item, (list, tuple)):
            ds.ids = [self.ids[i] for i in item]
        else:
            assert isinstance(item, slice)
            ds.ids = self.ids[item]
        return ds

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def local_cloud(self, id_lid):
        assert id_lid in self.ids_lid
        return read_points(self.local_cloud_path(id_lid))

    def cloud_label(self, id_lid):
        assert id_lid in self.ids_lid
        return read_points_labels(self.cloud_label_path(id_lid))

    def cloud_pose(self, id):
        t = float(id.split('-')[1].split('_')[0]) + float(id.split('-')[1].split('_')[1]) / 1000.0
        i = np.searchsorted(self.ts_lid, t)
        i = np.clip(i, 0, len(self.ids_lid))
        return self.poses[i]

    def camera_image(self, id):
        assert id in self.ids  # these are lidar ids
        t = float(id.split('-')[1].split('_')[0]) + float(id.split('-')[1].split('_')[1]) / 1000.0
        i = np.searchsorted(self.ts_rgb, t)
        i = np.clip(i, 0, len(self.ids_rgb) - 1)
        return read_rgb(self.image_path(self.ids_rgb[i]))

    def camera_semseg(self, id):
        assert id in self.ids  # these are lidar ids
        t = float(id.split('-')[1].split('_')[0]) + float(id.split('-')[1].split('_')[1]) / 1000.0
        i = np.searchsorted(self.ts_semseg, t)
        i = np.clip(i, 0, len(self.ids_semseg) - 1)
        return read_semseg(self.semseg_path(self.ids_semseg[i]))


class Rellis3DImages(BaseDatasetImages):
    CLASSES = ['dirt', 'grass', 'tree', 'pole', 'water', 'sky', 'vehicle', 'object', 'asphalt', 'building',
               'log', 'person', 'fence', 'bush', 'concrete', 'barrier', 'puddle', 'mud', 'rubble']

    def __init__(self,
                 path=None,
                 split='train',
                 num_samples=None,
                 multi_scale=True,
                 flip=True,
                 ignore_label=-1,
                 base_size=2048,
                 crop_size=(1200, 1920),
                 downsample_rate=1,
                 scale_factor=16,
                 traversability_labels=False,
                 mean=np.asarray([0.54218053, 0.64250553, 0.56620195]),
                 std=np.asarray([0.54218052, 0.64250552, 0.56620194])):
        super(Rellis3DImages, self).__init__(ignore_label, base_size,
                                             crop_size, downsample_rate, scale_factor, mean, std)

        if path is None:
            path = join(data_dir, 'Rellis_3D')
        assert os.path.exists(path)
        assert split in ['train', 'val', 'test']
        self.path = path
        self.split = split

        self.traversability_labels = traversability_labels
        if not traversability_labels:
            self.label_map = None
            # convert str names to class values on masks
            self.class_values = list(range(len(self.CLASSES)))
        else:
            label_map = yaml.safe_load(
                open(os.path.join(data_dir, "../config/rellis_to_traversability.yaml"), 'r'))
            self.label_map = {int(k): int(v) for k, v in label_map.items()}
            self.class_values = [0, 1, 255]

        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label

        self.mean = mean
        self.std = std
        self.scale_factor = scale_factor
        self.downsample_rate = 1. / downsample_rate

        self.multi_scale = multi_scale
        self.flip = flip

        self.img_list = [line.strip().split() for line in open(os.path.join(path, '%s.lst' % split))]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

    def read_files(self):
        files = []
        for item in self.img_list:
            image_path, label_path = item
            image_path = os.path.join(self.path, image_path)
            label_path = os.path.join(self.path, label_path)
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append({
                "img": image_path,
                "label": label_path,
                "name": name,
                "weight": 1
            })
        return files

    def __getitem__(self, index):
        item = self.files[index]
        image = cv2.imread(item["img"], cv2.IMREAD_COLOR)

        mask = np.array(Image.open(item["label"]))
        if self.label_map is not None:
            mask = convert_label(mask, inverse=False, label_mapping=self.label_map)
        else:
            mask = convert_label(mask, inverse=False)

        if 'test' in self.split:
            new_h, new_w = self.crop_size
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            image = self.input_transform(image)
        else:
            # add augmentations
            image, mask = self.apply_augmentations(image, mask, self.multi_scale, self.flip)

        # extract certain classes from mask
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=0).astype('float')

        return image.copy(), mask.copy()


class Rellis3DClouds(BaseDatasetClouds):
    CLASSES = ['dirt', 'grass', 'tree', 'pole', 'water', 'sky', 'vehicle', 'object', 'asphalt', 'building',
               'log', 'person', 'fence', 'bush', 'concrete', 'barrier', 'puddle', 'mud', 'rubble']

    def __init__(self,
                 path=None,
                 split='train',
                 fields=None,
                 num_samples=None,
                 color_map=None,
                 traversability_labels=False,
                 flexibility_labels=False,
                 lidar_beams_step=1,
                 labels_mode='labels'
                 ):
        super(Rellis3DClouds, self).__init__(path=path, fields=fields,
                                             depth_img_H=64, depth_img_W=2048,
                                             lidar_fov_up=22.5, lidar_fov_down=-22.5,
                                             lidar_beams_step=lidar_beams_step)

        assert (traversability_labels and flexibility_labels) == False

        if path is None:
            path = join(data_dir, 'Rellis_3D')
        assert os.path.exists(path)
        self.path = path
        assert split in ['train', 'val', 'test']
        self.split = split
        assert labels_mode in ['masks', 'labels']
        self.labels_mode = labels_mode
        self.classes_to_correct = ['person']
        assert set(self.classes_to_correct) <= set(self.CLASSES)

        self.traversability_labels = traversability_labels
        self.flexibility_labels = flexibility_labels

        if not self.traversability_labels and not self.flexibility_labels:
            self.label_map = None
            if not color_map:
                CFG = yaml.safe_load(open(os.path.join(data_dir, "../config/rellis.yaml"), 'r'))
                color_map = CFG["color_map"]
            self.class_values = list(range(len(color_map)))
        else:
            if self.traversability_labels:
                mapping = 'traversability'
                color_map = TRAVERSABILITY_COLOR_MAP
                self.class_values = np.sort([k for k in TRAVERSABILITY_LABELS.keys()]).tolist()
            elif self.flexibility_labels:
                mapping = 'flexibility'
                color_map = FLEXIBILITY_COLOR_MAP
                self.class_values = np.sort([k for k in FLEXIBILITY_LABELS.keys()]).tolist()
            assert mapping in ['traversability', 'flexibility']

            label_map = yaml.safe_load(open(os.path.join(data_dir, "../config/rellis_to_%s.yaml" % mapping), 'r'))
            assert isinstance(label_map, (dict, list))
            if isinstance(label_map, dict):
                label_map = dict((int(k), int(v)) for k, v in label_map.items())
                n = max(label_map) + 1
                self.label_map = np.zeros((n,), dtype=np.uint8)
                for k, v in label_map.items():
                    self.label_map[k] = v
            elif isinstance(label_map, list):
                self.label_map = np.asarray(label_map)

        self.color_map = color_map
        self.get_scan()

        self.depths_list = [line.strip().split() for line in open(os.path.join(path, 'pt_%s.lst' % split))]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

    def read_files(self):
        files = []
        for item in self.depths_list:
            depth_path, label_path = item
            depth_path = os.path.join(self.path, depth_path)
            label_path = os.path.join(self.path, label_path)
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append({
                "depth": depth_path,
                "label": label_path,
                "name": name,
                "weight": 1
            })
        return files

    def __getitem__(self, index):
        item = self.files[index]
        self.scan.open_scan(item["depth"])
        self.scan.open_label(item["label"])

        data, label = self.create_sample()

        return data, label

    def __len__(self):
        return len(self.files)


def semantic_laser_scan_demo(n_runs=1):
    # split = np.random.choice(['test', 'train', 'val'])
    split = 'test'

    ds = Rellis3DClouds(split=split, lidar_beams_step=2)
    ds_trav = Rellis3DClouds(split=split, lidar_beams_step=2, traversability_labels=True)
    ds_flex = Rellis3DClouds(split=split, lidar_beams_step=2, flexibility_labels=True)

    # model_name = 'fcn_resnet50_lr_0.0001_bs_4_epoch_14_Rellis3DClouds_intensity_depth_iou_0.56.pth'
    model_name = 'deeplabv3_resnet101_lr_0.0001_bs_16_epoch_64_Rellis3DClouds_z_depth_iou_0.68.pth'
    model = torch.load(os.path.join(data_dir, '../config/weights/depth_cloud/', model_name),
                       map_location='cpu').eval()
    for _ in range(n_runs):
        ind = np.random.choice(range(len(ds)))

        xyzid, label = ds[ind]
        label_trav = ds_trav[ind][1]
        label_flex = ds_flex[ind][1]

        # depth_img = {-1: no data, 0..1: for scaled distances}
        power = 16
        depth_img = np.copy(xyzid[-1])  # depth
        depth_img[depth_img > 0] = depth_img[depth_img > 0] ** (1 / power)
        depth_img[depth_img > 0] = (depth_img[depth_img > 0] - depth_img[depth_img > 0].min()) / \
                                   (depth_img[depth_img > 0].max() - depth_img[depth_img > 0].min())

        # semantic annotation of depth image
        color_gt = ds.label_to_color(label)
        color_trav_gt = ds_trav.label_to_color(label_trav)
        color_flex_gt = ds_flex.label_to_color(label_flex)

        # Apply inference preprocessing transforms
        batch = torch.from_numpy(xyzid[-2:]).unsqueeze(0)  # model takes as input only intensity and depth image
        with torch.no_grad():
            pred = model(batch)['out']
        pred = pred.squeeze(0).cpu().numpy()

        color_pred = ds.label_to_color(pred)

        plt.figure(figsize=(20, 10))
        plt.subplot(5, 1, 1)
        plt.imshow(depth_img)
        plt.title('Depth image')
        plt.subplot(5, 1, 2)
        plt.imshow(color_gt)
        plt.title('Semantics: GT')
        plt.subplot(5, 1, 3)
        plt.imshow(color_pred)
        plt.title('Semantics: Pred')
        plt.subplot(5, 1, 4)
        plt.imshow(color_trav_gt)
        plt.title('Traversability labels')
        plt.subplot(5, 1, 5)
        plt.imshow(color_flex_gt)
        plt.title('Flexibility labels')
        plt.show()


def semseg_test(n_runs=1):
    import yaml

    CFG = yaml.safe_load(open(os.path.join(data_dir, "../config/rellis.yaml"), 'r'))
    color_map = CFG["color_map"]

    split = np.random.choice(['test', 'train', 'val'])
    # split = 'val'
    ds = Rellis3DImages(split=split)

    for _ in range(n_runs):
        image, gt_mask = ds[int(np.random.choice(range(len(ds))))]

        if split in ['val', 'train']:
            image = image.transpose([1, 2, 0])

        image_vis = np.uint8(255 * (image * ds.std + ds.mean))

        gt_arg = np.argmax(gt_mask, axis=0).astype(np.uint8)
        gt_arg = convert_label(gt_arg, inverse=True)
        gt_color = convert_color(gt_arg, color_map)

        visualize(
            image=image_vis,
            label=gt_color,
        )


def colored_cloud_demo(n_runs=1):
    import open3d as o3d

    ds = Rellis3DClouds(split='test', lidar_beams_step=1)

    # model_name = 'fcn_resnet50_lr_0.0001_bs_4_epoch_14_Rellis3DClouds_intensity_depth_iou_0.56.pth'
    model_name = 'deeplabv3_resnet101_lr_0.0001_bs_16_epoch_64_Rellis3DClouds_z_depth_iou_0.68.pth'
    model = torch.load(os.path.join(data_dir, '../config/weights/depth_cloud/', model_name),
                       map_location='cpu')
    model.eval()

    for _ in range(n_runs):
        i = np.random.choice(range(len(ds)))
        xyzid, label = ds[i]

        xyz = xyzid[:3, ...].reshape((3, -1))
        xyz = xyz.T

        color_gt = ds.label_to_color(label)

        color_gt = color_gt.reshape((-1, 3))
        assert xyz.shape == color_gt.shape

        # Apply inference preprocessing transforms
        batch = torch.from_numpy(xyzid[-2:]).unsqueeze(0)  # model takes as input only i and d

        # Use the model and visualize the prediction
        with torch.no_grad():
            pred = model(batch)['out']
        pred = pred.squeeze(0).cpu().numpy()

        color_pred = ds.label_to_color(pred)

        color_pred = color_pred.reshape((-1, 3))
        assert xyz.shape == color_pred.shape

        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(xyz)
        pcd_pred.colors = o3d.utility.Vector3dVector(color_pred)

        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(xyz + np.array([150, 0, 0]))
        pcd_gt.colors = o3d.utility.Vector3dVector(color_gt)

        o3d.visualization.draw_geometries([pcd_pred, pcd_gt])


def trav_cloud_demo(n_runs=1):
    import open3d as o3d

    ds = Rellis3DClouds(split='test', traversability_labels=True)

    for _ in range(n_runs):
        i = np.random.choice(range(len(ds)))
        xyzid, label = ds[i]

        xyz = xyzid[:3, ...].reshape((3, -1))
        xyz = xyz.T

        color_gt = ds.label_to_color(label)

        color_gt = color_gt.reshape((-1, 3))
        assert xyz.shape == color_gt.shape

        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(xyz + np.array([150, 0, 0]))
        pcd_gt.colors = o3d.utility.Vector3dVector(color_gt)

        o3d.visualization.draw_geometries([pcd_gt])


def lidar_map_demo():
    from tqdm import tqdm
    import open3d as o3d

    name = np.random.choice(seq_names)
    ds = Rellis3DSequence(seq='rellis_3d/%s' % name)

    plt.figure()
    plt.title('Trajectory')
    plt.axis('equal')
    plt.plot(ds.poses[:, 0, 3], ds.poses[:, 1, 3], '.')
    plt.grid()
    plt.show()

    clouds = []
    for data in tqdm(ds[::100]):
        cloud, _, pose, _, _ = data
        cloud = structured_to_unstructured(cloud[['x', 'y', 'z']])
        cloud = np.matmul(cloud, pose[:3, :3].T) + pose[:3, 3:].T

        clouds.append(cloud)
    cloud = np.concatenate(clouds)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    o3d.visualization.draw_geometries([pcd.voxel_down_sample(voxel_size=0.5)])


def lidar2cam_demo(n_runs=1):
    seq = np.random.choice(seq_names)
    ds = Rellis3DSequence(seq='rellis_3d/%s' % seq)

    dist_coeff = ds.calibration['dist_coeff'].reshape((5, 1))
    K = ds.calibration['K']
    T_lid2cam = ds.calibration['lid2cam']

    for _ in range(n_runs):
        data = ds[int(np.random.choice(range(len(ds))))]
        points, points_label, pose, rgb, semseg = data
        points = structured_to_unstructured(points[['x', 'y', 'z']])

        img_height, img_width = rgb.shape[:2]
        assert rgb.shape[:2] == semseg.shape[:2]

        R_lidar2cam = T_lid2cam[:3, :3]
        t_lidar2cam = T_lid2cam[:3, 3]
        rvec, _ = cv2.Rodrigues(R_lidar2cam)
        tvec = t_lidar2cam.reshape(3, 1)
        xyz_v, color = filter_camera_points(points[..., :3], img_width, img_height, K, T_lid2cam)

        imgpoints, _ = cv2.projectPoints(xyz_v[:, :], rvec, tvec, K, dist_coeff)
        imgpoints = np.squeeze(imgpoints, 1)
        imgpoints = imgpoints.T

        res_rgb = draw_points_on_image(points=imgpoints, color=color, image=rgb)
        res_semseg = draw_points_on_image(points=imgpoints, color=color, image=semseg)

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(res_rgb / 255)
        plt.subplot(1, 2, 2)
        plt.imshow(res_semseg / 255)
        plt.show()


def semseg_demo(n_runs=1):
    seq = np.random.choice(seq_names)
    ds = Rellis3DSequence(seq='rellis_3d/%s' % seq)

    for _ in range(n_runs):
        id = int(np.random.choice(range(len(ds))))
        print('Data index:', id)

        data = ds[id]
        _, _, _, rgb, semseg = data

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(rgb / 255)
        plt.subplot(1, 2, 2)
        plt.imshow(semseg / 255)
        plt.show()


def traversability_mapping_demo(n_runs=1):
    color_map = TRAVERSABILITY_COLOR_MAP

    split = np.random.choice(['test', 'train', 'val'])
    ds = Rellis3DImages(split=split, traversability_labels=True)

    for _ in range(n_runs):
        image, gt_mask = ds[int(np.random.choice(range(len(ds))))]

        if split in ['val', 'train']:
            image = image.transpose([1, 2, 0])

        image_vis = np.uint8(255 * (image * ds.std + ds.mean))
        gt_arg = np.argmax(gt_mask, axis=0).astype(np.uint8)  # [0, 1, 2]
        gt_color = convert_color(gt_arg, color_map)
        visualize(image=image_vis, label=gt_color)


def main():
    colored_cloud_demo(1)
    semantic_laser_scan_demo(1)
    semseg_test(1)
    # lidar_map_demo()
    lidar2cam_demo(1)
    semseg_demo(1)
    traversability_mapping_demo(1)


if __name__ == '__main__':
    main()
