import os
import cv2
import numpy as np
import torch
import fiftyone as fo
from datasets.laserscan import LaserScan
from numpy.lib.recfunctions import structured_to_unstructured


data_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))


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
        self.mask_targets = {0: "background",
                             1: "traversable",
                             2: "non-traversable"}
        self.class_values = [0, 1, 2]
        self.mean = np.array([0.0, 0.0, 0.0])
        self.std = np.array([1.0, 1.0, 1.0])

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        sample = self.samples[img_path]

        # image preprocessing
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, self.crop_size, interpolation=cv2.INTER_LINEAR)
        image = self._input_transform(image)

        if self.split is not 'test':
            image = image.transpose((2, 0, 1))

        # mask preprocessing
        mask = sample.polylines.to_segmentation(frame_size=(1920, 1200),
                                                mask_targets=self.mask_targets)["mask"]
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
        self.scan = LaserScan(project=True, H=self.H, W=self.W, fov_up=fov_up, fov_down=fov_down)

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
        return cloud

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


def clouds_demo():
    from matplotlib import pyplot as plt

    path = os.path.join(data_dir, 'bags/traversability/marv/ugv_2022-08-12-16-37-03/')
    ds = TraversabilityClouds(path=path)

    cloud = ds[np.random.choice(len(ds))]
    xyz = structured_to_unstructured(cloud[['x', 'y', 'z']])
    intensity = cloud['intensity']

    ds.scan.set_points(points=xyz, remissions=intensity)

    power = 16
    depth_img = np.copy(ds.scan.proj_range)  # depth
    depth_img[depth_img > 0] = depth_img[depth_img > 0] ** (1 / power)
    depth_img[depth_img > 0] = (depth_img[depth_img > 0] - depth_img[depth_img > 0].min()) / \
                               (depth_img[depth_img > 0].max() - depth_img[depth_img > 0].min())

    plt.figure(figsize=(20, 10))
    plt.imshow(depth_img)
    plt.show()


if __name__ == "__main__":
    # images_demo()
    clouds_demo()
