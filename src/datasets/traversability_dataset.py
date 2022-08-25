import os
import cv2
import numpy as np
import torch
try:
    # it's hard to install the lib on jetson
    # we are not importing it since it is not needed for inference tasks on robots
    import fiftyone as fo
except:
    print('Fiftyone lib is not installed')
from datasets.laserscan import SemLaserScan
from datasets.base_dataset import TRAVERSABILITY_LABELS, TRAVERSABILITY_COLOR_MAP
from traversability_estimation.utils import convert_color
from numpy.lib.recfunctions import structured_to_unstructured

data_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))


class TraversabilityImages(torch.utils.data.Dataset):
    CLASSES = ["traversable", "non-traversable", "background",]

    def __init__(self, crop_size=(1200, 1920), path=None, split=None):
        self.crop_size = (crop_size[1], crop_size[0])
        if not path:
            self.path = os.path.join(data_dir, 'TraversabilityDataset')
        else:
            self.path = path
        self.samples = self._load_dataset(self.path)
        self.files = self.samples.values("filepath")
        # self.mask_targets = TRAVERSABILITY_LABELS
        self.mask_targets = {1: "traversable", 2: "non-traversable", 0: "background"}
        # self.class_values = np.sort([k for k in TRAVERSABILITY_LABELS.keys()])  # [0, 1, 255]
        self.class_values = [0, 1, 2]
        # self.color_map = TRAVERSABILITY_COLOR_MAP
        self.color_map = {1: [0, 255, 0], 2: [255, 0, 0], 0: [0, 0, 0]}

        self.mean = np.array([123.11457109, 126.84649579, 124.37909438])
        self.std = np.array([47.46125817, 47.14161698, 47.70375418])
        self.split = split

    def read_img(self, path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        return image

    def __getitem__(self, idx):
        img_path = self.files[idx]
        sample = self.samples[img_path]

        # image preprocessing
        image = self.read_img(img_path)
        image = cv2.resize(image, self.crop_size, interpolation=cv2.INTER_LINEAR)
        image = self._input_transform(image)

        if self.split != 'test':
            image = image.transpose((2, 0, 1))

        # mask preprocessing
        segmentation = sample.polylines.to_segmentation(frame_size=(1920, 1200), mask_targets=self.mask_targets)
        mask = segmentation["mask"]
        sample["ground_truth"] = segmentation
        sample.save()
        mask = cv2.resize(mask, self.crop_size, interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.long)

        # extract certain classes from mask
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=0).astype('float')

        return image, mask

    def __len__(self):
        return len(self.files)

    def _input_transform(self, image):
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    @staticmethod
    def _load_dataset(dataset_path):
        name = "TraversabilityDataset"
        if fo.dataset_exists(name):
            dataset = fo.load_dataset(name)
            dataset.delete()
        dataset = fo.Dataset.from_dir(dataset_dir=dataset_path,
                                      dataset_type=fo.types.CVATImageDataset,
                                      name=name,
                                      data_path="images",
                                      labels_path="annotations.xml")
        return dataset

    def show_dataset(self):
        session = fo.launch_app(self.samples)
        session.wait()

    def save_prediction(self, mask, i):
        sample = self.samples[self.files[i]]
        mask = mask.astype(np.uint8)
        sample["prediction"] = fo.Segmentation(mask=mask)
        sample.save()

    def get_mask(self, image_name):
        assert isinstance(image_name, str)
        for path in self.files:
            if image_name in path:
                sample = self.samples[path]
                segmentation = sample.polylines.to_segmentation(frame_size=(1920, 1200), mask_targets=self.mask_targets)
                mask = segmentation["mask"]
                return mask

    def calculate_mean_and_std(self):
        mean = np.zeros(3)
        std = np.zeros(3)
        for i in range(len(self.img_paths)):
            img_path = self.img_paths[i]
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)[:, :, ::-1]
            mean += image.mean(axis=(0, 1))
            std += image.std(axis=(0, 1))
        mean /= len(self.img_paths)
        std /= len(self.img_paths)
        self.mean = mean
        self.std = std
        print(mean)
        print(std)
        return mean, std


class TraversabilityClouds:
    CLASSES = ["background", "traversable", "non-traversable"]

    def __init__(self,
                 sequence='ugv_2022-08-12-15-30-22',
                 path=None,
                 num_samples=None,
                 H=128,
                 W=1024,
                 fov_up=45.0,
                 fov_down=-45.0,
                 labels_mode='masks',
                 split=None,
                 fields=None,
                 lidar_beams_step=1,
                 traversability_labels=None,
                 ):
        if fields is None:
            fields = ['depth']
        if path is None:
            path = os.path.join(data_dir, 'bags/traversability/marv/')
        assert os.path.exists(path)
        self.path = os.path.join(path, sequence)
        self.rng = np.random.default_rng(42)
        self.mask_targets = {val: key for key, val in TRAVERSABILITY_LABELS.items()}
        # self.class_values = list(self.mask_targets.values())
        self.class_values = np.sort([k for k in TRAVERSABILITY_LABELS.keys()])
        assert labels_mode in ['masks', 'labels']
        self.labels_mode = labels_mode  # 'masks': label.shape == (C, H, W) or 'labels': label.shape == (H, W)

        assert split in [None, 'train', 'val', 'test']
        self.split = split

        self.fields = fields
        self.lidar_beams_step = lidar_beams_step
        self.traversability_labels = True

        self.W = W
        self.H = H
        self.color_map = TRAVERSABILITY_COLOR_MAP
        self.scan = SemLaserScan(nclasses=len(self.CLASSES), sem_color_dict=self.color_map,
                                 project=True, H=self.H, W=self.W, fov_up=fov_up, fov_down=fov_down)

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

    def label_to_color(self, label):
        if len(label.shape) == 3:
            C, H, W = label.shape
            label = np.argmax(label, axis=0)
            assert label.shape == (H, W)
        color = self.scan.sem_color_lut[label]
        return color

    def read_files(self, train_ratio=0.8):
        clouds_path = os.path.join(self.path, 'os_cloud_node/destaggered_points/')
        assert os.path.exists(clouds_path)

        all_files = [os.path.join(clouds_path, f) for f in os.listdir(clouds_path)]
        if self.split == 'train':
            train_files = self.rng.choice(all_files, size=round(train_ratio * len(all_files)), replace=False).tolist()
            files = train_files
        elif self.split in ['val', 'test']:
            train_files = self.rng.choice(all_files, size=round(train_ratio * len(all_files)), replace=False).tolist()
            val_files = list(set(all_files) - set(train_files))
            # It is a good practice to check datasets don`t intersects with each other
            assert set(train_files).isdisjoint(set(val_files))
            files = val_files
        else:
            # raise ValueError('Split must be one of train, val, test')
            files = all_files

        return files

    def read_cloud(self, path):
        cloud = np.load(path, allow_pickle=True)['arr_0'].item()['cloud']
        return cloud

    def __getitem__(self, index):
        cloud_path = self.files[index]
        cloud = self.read_cloud(cloud_path)

        xyz = structured_to_unstructured(cloud[['x', 'y', 'z']])
        traversability = 1 - cloud['empty']

        self.scan.set_points(points=xyz)
        bg_value = self.mask_targets["background"]
        self.scan.proj_sem_label = np.full((self.scan.proj_H, self.scan.proj_W), bg_value,
                                           dtype=np.uint8)  # [H,W]  label
        self.scan.set_label(label=traversability)

        depth_img = self.scan.proj_range[None]  # (1, H, W)
        label = self.scan.proj_sem_label

        # 'masks': label.shape == (C, H, W) or 'labels': label.shape == (H, W)
        if self.labels_mode == 'masks':
            # extract certain classes from mask
            labels = [(label == v) for v in self.class_values]
            label = np.stack(labels, axis=0)

        if self.lidar_beams_step:
            depth_img = depth_img[..., ::self.lidar_beams_step]
            label = label[..., ::self.lidar_beams_step]

        return depth_img.astype('float32'), label.astype('float32')

    def __len__(self):
        return len(self.files)


def images_demo():
    name = "TraversabilityDataset"
    directory = os.path.join(data_dir, name)

    dataset = TraversabilityImages(crop_size=(1200, 1920), path=directory, split='val')
    length = len(dataset)
    # dataset.show_dataset()
    splits = torch.utils.data.random_split(dataset,
                                           [int(0.7 * length), int(0.2 * length), int(0.1 * length)],
                                           generator=torch.Generator().manual_seed(42))
    # show first images from each split
    for split in splits:
        print(len(split))


def clouds_demo(run_times=1):
    from matplotlib import pyplot as plt
    import open3d as o3d

    ds = TraversabilityClouds(split='test', labels_mode='labels')

    for _ in range(run_times):
        depth_img, label = ds[np.random.choice(len(ds))]

        power = 16
        depth_img_vis = np.copy(depth_img).squeeze()  # depth
        depth_img_vis[depth_img_vis > 0] = depth_img_vis[depth_img_vis > 0] ** (1 / power)
        depth_img_vis[depth_img_vis > 0] = (depth_img_vis[depth_img_vis > 0] - depth_img_vis[depth_img_vis > 0].min()) / \
                                           (depth_img_vis[depth_img_vis > 0].max() - depth_img_vis[
                                               depth_img_vis > 0].min())
        depth_img_vis[depth_img_vis < 0] = 0.5
        assert depth_img_vis.min() >= 0.0 and depth_img_vis.max() <= 1.0

        label_trav = label == ds.mask_targets['traversable']
        result = (0.3 * depth_img_vis + 0.7 * label_trav).astype("float")

        plt.figure(figsize=(20, 10))
        plt.imshow(result)
        plt.title('Depth image with traversable points')
        plt.tight_layout()
        plt.show()

        xyz = ds.scan.proj_xyz
        color = ds.scan.sem_color_lut[label.astype('uint8')]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.reshape((-1, 3)))
        pcd.colors = o3d.utility.Vector3dVector(color.reshape((-1, 3)))
        o3d.visualization.draw_geometries([pcd])


def label_cloud_from_img():
    import yaml
    import open3d as o3d
    import matplotlib.pyplot as plt

    seq = 'ugv_2022-08-12-15-30-22'

    ds_img = TraversabilityImages()
    ds_depth = TraversabilityClouds(sequence=seq)

    # find labelled images from bag file
    bag_file_images = []
    bag_file_ts_imgs = []
    bag_file = '%s.bag' % seq

    file_to_bag = yaml.safe_load(open(os.path.join(data_dir, 'TraversabilityDataset', 'correspondences.yaml'), 'r'))
    for img_path in ds_img.files:
        img_file = img_path.split('/')[-1]

        bag_files = file_to_bag[bag_file]

        for frame_id in bag_files.keys():
            if img_file in bag_files[frame_id]:
                # print('Img file %s was recorded in bag file %s with sensor %s' % (img_file, bag_file, frame_id))
                bag_file_images.append(img_path)
                ts_img = float(img_file.split('_')[1].replace('s', '')) +\
                         float(img_file.split('_')[2].replace('n', '').replace('.jpg', '')) / 10.0**9
                bag_file_ts_imgs.append(ts_img)

    print('Found %i labelled images from bag file %s' % (len(bag_file_images), bag_file))

    # find closest in timestamp point clouds for each image
    dt = 1.0  # [sec]
    close_files = {'depth': [], 'img': []}
    for depth_path in ds_depth.files:
        depth_file = depth_path.split('/')[-1]
        ts_depth = float(depth_file.split('_')[0]) + float(depth_file.split('_')[1].replace('.npz', '')) / 10.0**9

        ts_imgs = np.asarray(bag_file_ts_imgs)
        time_diff = np.abs(ts_imgs - ts_depth)
        idx = time_diff.argmin()

        if time_diff[idx] <= dt:
            # print('For depth cloud %s found closest labelled image %s'
            #       'with time difference %f [sec]'
            #       % (depth_file, bag_file_images[idx], time_diff[idx]))
            close_files['depth'].append(depth_path)
            close_files['img'].append(bag_file_images[idx])

    i = np.random.choice(range(len(close_files['img'])))
    img = plt.imread(close_files['img'][i])
    label = ds_img.get_mask(close_files['img'][i])
    mask = convert_color(label, ds_img.color_map)
    cloud = ds_depth.read_cloud(close_files['depth'][i])
    points = structured_to_unstructured(cloud[['x', 'y', 'z']])

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.show()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

    # TODO: get camera intrinsics
    # TODO: get extrinsics: T_lid2cam
    # TODO: get lidar points which are in camera FoV
    # TODO: colorize these points, the rest points are without label (background)
    # TODO: save labelled point clouds (add them to TraversabilityDataset)


if __name__ == "__main__":
    # images_demo()
    # clouds_demo()
    label_cloud_from_img()
