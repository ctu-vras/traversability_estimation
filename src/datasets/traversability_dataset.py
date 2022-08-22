import os
import cv2
import numpy as np
import torch
import fiftyone as fo
from datasets.laserscan import LaserScan, SemLaserScan
from numpy.lib.recfunctions import structured_to_unstructured


data_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
LABELS = {255: "background",
          1: "traversable",
          0: "non-traversable"}
COLOR_MAP = {255: [0, 0, 0],
             1: [0, 255, 0],
             0: [255, 0, 0]}


class TraversabilityImages(torch.utils.data.Dataset):
    def __init__(self, crop_size: tuple = (1200, 1920), path=None, split=None):
        self.crop_size = (crop_size[1], crop_size[0])
        if not path:
            self.path = os.path.join(data_dir, 'TraversabilityDataset')
        else:
            self.path = path
        self.split = split
        self.samples = self._load_dataset(self.path)
        self.img_paths = self.samples.values("filepath")
        self.mask_targets = LABELS
        self.class_values = list(self.mask_targets.keys())
        self.mean = np.array([0.0, 0.0, 0.0])
        self.std = np.array([1.0, 1.0, 1.0])

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
        mask = sample.polylines.to_segmentation(frame_size=(1920, 1200), mask_targets=self.mask_targets)["mask"]
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
        # image -= self.mean / 255.0
        # image /= self.std / 255.0
        return image

    @staticmethod
    def _load_dataset(dataset_path: str):
        name = "traversability-dataset"
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


class TraversabilityClouds:
    def __init__(self,
                 path=None,
                 num_samples=None,
                 H=128,
                 W=1024,
                 fov_up=45.0,
                 fov_down=-45.0,
                 ):
        if path is None:
            path = os.path.join(data_dir, 'bags/traversability/marv/ugv_2022-08-12-16-37-03/')
        assert os.path.exists(path)
        self.path = path
        self.W = W
        self.H = H
        self.color_map = COLOR_MAP
        self.scan = SemLaserScan(nclasses=len(LABELS), sem_color_dict=self.color_map,
                                 project=True, H=self.H, W=self.W, fov_up=fov_up, fov_down=fov_down)
        self.mask_targets = {val: key for key, val in LABELS.items()}

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

    def read_files(self):
        clouds_path = os.path.join(self.path, 'os_cloud_node/points/')
        files = [os.path.join(clouds_path, f) for f in os.listdir(clouds_path)]
        return files

    def __getitem__(self, index):
        cloud_path = self.files[index]
        cloud = np.load(cloud_path, allow_pickle=True)['arr_0'].item()['cloud']

        xyz = structured_to_unstructured(cloud[['x', 'y', 'z']])
        traversability = cloud['empty']

        self.scan.set_points(points=xyz)
        self.scan.set_label(label=traversability)

        depth_img = self.scan.proj_range
        label = self.scan.proj_sem_label

        return depth_img, label

    def __len__(self):
        return len(self.files)


def images_demo():
    name = "TraversabilityDataset"
    directory = os.path.join(data_dir, name)

    dataset = TraversabilityImages(crop_size=(1200, 1920), path=directory)
    length = len(dataset)
    # dataset.show_dataset()
    splits = torch.utils.data.random_split(dataset,
                                           [int(0.7 * length), int(0.2 * length), int(0.1 * length)],
                                           generator=torch.Generator().manual_seed(42))
    # show first images from each split
    for split in splits:
        print(len(split))


def clouds_demo(run_times=5):
    from matplotlib import pyplot as plt
    from traversability_estimation.utils import convert_label
    import open3d as o3d

    label_map = {255: 0.5,
                   1: 1.0,
                   0: 0.0}

    path = os.path.join(data_dir, 'bags/traversability/marv/ugv_2022-08-12-16-37-03/')
    ds = TraversabilityClouds(path=path)

    for _ in range(run_times):
        depth_img, label = ds[np.random.choice(len(ds))]

        power = 16
        depth_img_vis = np.copy(depth_img)  # depth
        depth_img_vis[depth_img_vis > 0] = depth_img_vis[depth_img_vis > 0] ** (1 / power)
        depth_img_vis[depth_img_vis > 0] = (depth_img_vis[depth_img_vis > 0] - depth_img_vis[depth_img_vis > 0].min()) / \
                                           (depth_img_vis[depth_img_vis > 0].max() - depth_img_vis[depth_img_vis > 0].min())

        label_vis = convert_label(label.astype("float"), label_mapping=label_map)

        plt.figure(figsize=(20, 10))
        plt.subplot(2, 1, 1)
        # depth_img_vis[depth_img_vis < 0] = 0.5
        # assert depth_img_vis.min() >= 0.0 and depth_img_vis.max() <= 1.0
        # plt.imshow((0.3 * depth_img_vis + 0.7 * label_vis).astype("float"))
        plt.imshow(depth_img_vis)
        plt.title('Depth image')

        plt.subplot(2, 1, 2)
        plt.imshow(label_vis)
        plt.title('Traversable points')
        plt.tight_layout()
        plt.show()

        xyz = ds.scan.proj_xyz
        color = ds.scan.sem_color_lut[label]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.reshape((-1, 3)))
        pcd.colors = o3d.utility.Vector3dVector(color.reshape((-1, 3)))
        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    # images_demo()
    clouds_demo()
