import os
import cv2
import numpy as np
import torch
try:
    import fiftyone as fo
except:
    print('Fiftyone lib is not installed')
from datasets.laserscan import SemLaserScan
from numpy.lib.recfunctions import structured_to_unstructured

data_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
LABELS = {255: "background",
          1: "traversable",
          2: "non-traversable"}
COLOR_MAP = {255: [0, 0, 0],
             1:   [0, 255, 0],
             2:   [255, 0, 0]}


class TraversabilityImages(torch.utils.data.Dataset):
    CLASSES = ["background", "traversable", "non-traversable"]

    def __init__(self, crop_size=(1200, 1920), path=None, split=None):
        self.crop_size = (crop_size[1], crop_size[0])
        if not path:
            self.path = os.path.join(data_dir, 'TraversabilityDataset')
        else:
            self.path = path
        self.samples = self._load_dataset(self.path)
        self.img_paths = self.samples.values("filepath")
        self.mask_targets = LABELS
        self.class_values = [0, 1, 2]

        # TODO: calculate mean and std for images in the dataset
        self.mean = np.array([0.0, 0.0, 0.0])
        self.std = np.array([1.0, 1.0, 1.0])
        self.split = split

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        sample = self.samples[img_path]

        # image preprocessing
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
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
        return len(self.img_paths)

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

    def save_prediction(self, maski, i):
        sample = self.samples[self.img_paths[i]]
        mask = mask.astype(np.uint8)
        sample["prediction"] = fo.Segmentation(mask=mask)
        sample.save()


class TraversabilityClouds:
    CLASSES = ["background", "traversable", "non-traversable"]

    def __init__(self,
                 path=None,
                 num_samples=None,
                 H=128,
                 W=1024,
                 fov_up=45.0,
                 fov_down=-45.0,
                 labels_mode='masks',
                 split='train',
                 fields=None,
                 lidar_beams_step=1,
                 traversability_labels=None,
                 ):
        if fields is None:
            fields = ['depth']
        if path is None:
            path = os.path.join(data_dir, 'bags/traversability/marv/ugv_2022-08-12-15-30-22/')
        assert os.path.exists(path)
        self.path = path
        self.rng = np.random.default_rng(42)
        self.mask_targets = {val: key for key, val in LABELS.items()}
        self.class_values = list(self.mask_targets.values())
        assert labels_mode in ['masks', 'labels']
        self.labels_mode = labels_mode  # 'masks': label.shape == (C, H, W) or 'labels': label.shape == (H, W)

        assert split in ['train', 'val', 'test']
        self.split = split

        self.fields = fields
        self.lidar_beams_step = lidar_beams_step
        self.traversability_labels = True

        self.W = W
        self.H = H
        self.color_map = COLOR_MAP
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
            raise ValueError('Split must be one of train, val, test')

        return files

    def __getitem__(self, index):
        cloud_path = self.files[index]
        cloud = np.load(cloud_path, allow_pickle=True)['arr_0'].item()['cloud']

        xyz = structured_to_unstructured(cloud[['x', 'y', 'z']])
        traversability = cloud['empty']

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


if __name__ == "__main__":
    images_demo()
    clouds_demo()
