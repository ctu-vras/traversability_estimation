import os
from numpy.lib.recfunctions import structured_to_unstructured
from os.path import dirname, join, realpath
from .utils import *
from copy import copy
import torch
from torch.utils.data import Dataset as BaseDataset
from PIL import Image
import random

__all__ = [
    'data_dir',
    'seq_names',
    'Sequence',
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


class Sequence(BaseDataset):
    def __init__(self, seq=None, path=None, poses_file='poses.txt', poses_path=None, split=None):
        """Rellis-3D dataset: https://unmannedlab.github.io/research/RELLIS-3D.
        
        Rellis_3D
        ├── 00000
        │   ├── os1_cloud_node_color_ply
        │   ├── pylon_camera_node
        │   ├── pylon_camera_node_label_color
        │   └── pylon_camera_node_label_id
        ...
        ├── bags
        └── calibration
            ├── 00000
            ...
            └── raw_data

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

    def local_cloud_path(self, id):
        return os.path.join(self.path, self.seq, 'os1_cloud_node_color_ply', '%s.ply' % id)

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
            return self.local_cloud(id), self.cloud_pose(id), self.camera_image(id), self.camera_semseg(id)

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


class DatasetSemSeg(BaseDataset):
    CLASSES = ['void', 'dirt', 'grass', 'tree', 'pole', 'water', 'sky', 'vehicle', 'object', 'asphalt', 'building',
               'log', 'person', 'fence', 'bush', 'concrete', 'barrier', 'puddle', 'mud', 'rubble']
    LABEL_MAPPING = {0: 0,
                     1: 0,
                     3: 1,
                     4: 2,
                     5: 3,
                     6: 4,
                     7: 5,
                     8: 6,
                     9: 7,
                     10: 8,
                     12: 9,
                     15: 10,
                     17: 11,
                     18: 12,
                     19: 13,
                     23: 14,
                     27: 15,
                     # 29: 1,
                     # 30: 1,
                     31: 16,
                     # 32: 4,
                     33: 17,
                     34: 18}

    def __init__(self,
                 path=None,
                 split='train',
                 num_samples=None,
                 classes=None,
                 multi_scale=True,
                 flip=True,
                 ignore_label=-1,
                 base_size=2048,
                 crop_size=(1200, 1920),
                 downsample_rate=1,
                 scale_factor=16,
                 mean=np.asarray([0.54218053, 0.64250553, 0.56620195]),
                 std=np.asarray([0.54218052, 0.64250552, 0.56620194])):
        if path is None:
            path = join(data_dir, 'Rellis_3D')
        assert os.path.exists(path)
        assert split in ['train', 'val', 'test']
        self.path = path
        self.split = split
        if not classes:
            classes = self.CLASSES
        # convert str names to class values on masks
        self.class_values = [list(self.LABEL_MAPPING.values())[self.CLASSES.index(cls.lower())] for cls in classes]

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

        self.class_weights = torch.FloatTensor( [1.999999271012097, 1.664128991557095, 1.8496996235972305,
                                                 1.9998671743556093, 1.9985603836216685, 1.6997860667619373,
                                                 1.9996238518374807, 1.999762456018869, 1.9993409160693856,
                                                 1.9997133730241352, 1.999513379675197, 1.9993347608141532,
                                                 1.9996433543012653, 1.8417302013930952, 1.9900446258633235,
                                                 1.9956819252010565, 1.9949043721923583, 1.9714094933783817,
                                                 1.9972557793256613]).cuda()

    def read_files(self):
        files = []
        for item in self.img_list:
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append({
                "img": image_path,
                "label": label_path,
                "name": name,
                "weight": 1
            })
        return files

    def input_transform(self, image):
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    @staticmethod
    def pad_image(image, h, w, size, padvalue):
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
        long_size = int(self.base_size * rand_scale + 0.5)
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = int(h * long_size / w + 0.5)

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

    @staticmethod
    def random_brightness(img, shift_value=10):
        if not shift_value:
            return img
        if random.random() < 0.5:
            return img
        img = img.astype(np.float32)
        shift = random.randint(-shift_value, shift_value)
        img[:, :, :] += shift
        img = np.around(img)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    @staticmethod
    def label_transform(label):
        return np.array(label).astype('int32')

    def apply_augmentations(self, image, label, multi_scale=True, is_flip=True):
        if multi_scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            image, label = self.multi_scale_aug(image, label, rand_scale=rand_scale)

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

    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.LABEL_MAPPING.items():
                label[temp == k] = v
        else:
            for k, v in self.LABEL_MAPPING.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        item = self.files[index]
        image = cv2.imread(os.path.join(self.path, item["img"]), cv2.IMREAD_COLOR)

        mask = np.array(Image.open(os.path.join(self.path, item["label"])))
        mask = self.convert_label(mask, inverse=False)

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

    def __len__(self):
        return len(self.files)


def semseg_test():
    from datasets.utils import visualize
    from hrnet.core.function import convert_label, convert_color
    import yaml

    split = np.random.choice(['test', 'train', 'val'])
    # split = 'test'
    ds = DatasetSemSeg(split=split)
    image, gt_mask = ds[int(np.random.choice(range(len(ds))))]

    if split in ['val', 'train']:
        image = image.transpose([1, 2, 0])

    image_vis = np.uint8(255 * (image * ds.std + ds.mean))

    CFG = yaml.safe_load(open(os.path.join(data_dir,  "../config/rellis.yaml"), 'r'))
    color_map = CFG["color_map"]
    gt_arg = np.argmax(gt_mask, axis=0).astype(np.uint8) - 1
    gt_arg = convert_label(gt_arg, inverse=True)
    gt_color = convert_color(gt_arg, color_map)

    visualize(
        image=image_vis,
        label=gt_color,
    )


def lidar_map_demo():
    from tqdm import tqdm
    import open3d as o3d

    name = np.random.choice(seq_names)
    ds = Sequence(seq='rellis_3d/%s' % name)

    plt.figure()
    plt.title('Trajectory')
    plt.axis('equal')
    plt.plot(ds.poses[:, 0, 3], ds.poses[:, 1, 3], '.')
    plt.grid()
    plt.show()

    clouds = []
    for data in tqdm(ds[::100]):
        cloud, pose, _, _ = data
        cloud = structured_to_unstructured(cloud[['x', 'y', 'z']])
        cloud = np.matmul(cloud, pose[:3, :3].T) + pose[:3, 3:].T

        clouds.append(cloud)
    cloud = np.concatenate(clouds)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    o3d.visualization.draw_geometries([pcd.voxel_down_sample(voxel_size=0.5)])


def lidar2cam_demo():
    seq = np.random.choice(seq_names)
    ds = Sequence(seq='rellis_3d/%s' % seq)

    dist_coeff = ds.calibration['dist_coeff'].reshape((5, 1))
    K = ds.calibration['K']
    T_lid2cam = ds.calibration['lid2cam']

    for _ in range(1):
        data = ds[int(np.random.choice(range(len(ds))))]
        points, pose, rgb, semseg = data
        points = structured_to_unstructured(points[['x', 'y', 'z']])

        img_height, img_width = rgb.shape[:2]
        assert rgb.shape[:2] == semseg.shape[:2]

        R_lidar2cam = T_lid2cam[:3, :3]
        t_lidar2cam = T_lid2cam[:3, 3]
        rvec, _ = cv2.Rodrigues(R_lidar2cam)
        tvec = t_lidar2cam.reshape(3, 1)
        xyz_v, color = filter_camera_points(points, img_width, img_height, K, T_lid2cam)

        imgpoints, _ = cv2.projectPoints(xyz_v[:, :], rvec, tvec, K, dist_coeff)
        imgpoints = np.squeeze(imgpoints, 1)
        imgpoints = imgpoints.T

        res_rgb = print_projection_plt(points=imgpoints, color=color, image=rgb)
        res_semseg = print_projection_plt(points=imgpoints, color=color, image=semseg)

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(res_rgb / 255)
        plt.subplot(1, 2, 2)
        plt.imshow(res_semseg / 255)
        plt.show()


def semseg_demo():
    seq = np.random.choice(seq_names)
    ds = Sequence(seq='rellis_3d/%s' % seq)

    for _ in range(1):
        id = int(np.random.choice(range(len(ds))))
        print('Data index:', id)

        data = ds[id]
        _, _, rgb, semseg = data

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(rgb / 255)
        plt.subplot(1, 2, 2)
        plt.imshow(semseg / 255)
        plt.show()


def main():
    semseg_test()
    lidar_map_demo()
    lidar2cam_demo()
    semseg_demo()


if __name__ == '__main__':
    main()
