import os
import cv2
import numpy as np
import fiftyone as fo
import fiftyone.utils.splits as fous
from .laserscan import LaserScan
from numpy.lib.recfunctions import structured_to_unstructured


data_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))


class TraversabilityImages(object):
    def __init__(self, dataset_name: str, dataset_dir: str, crop_size: tuple, split_dict: dict):
        self.dataset_name = dataset_name
        self.crop_size = crop_size
        self.dataset = None
        self.ids = {"train": [], "val": [], "test": []}
        self.mean = None
        self.std = None

        # initialize dataset
        self._load_data(dataset_name, dataset_dir, split_dict)
        self._init_dataset()

    def _load_data(self, name: str, dataset_dir: str, split_dict: dict) -> None:
        print(f"INFO: Loading data {name} from {dataset_dir}")
        # if dataset exists, delete it and create a new one
        if fo.dataset_exists(name):
            self.dataset = fo.load_dataset(name)
            self.dataset.delete()
        self.dataset = fo.Dataset.from_dir(dataset_dir=dataset_dir, dataset_type=fo.types.CVATImageDataset, name=name)
        fous.random_split(self.dataset, split_dict)
        self.dataset.save()

    def _init_dataset(self) -> None:
        print("INFO: Initializing dataset")
        channels_sum, channels_squared_sum, num_batches = 0, 0, 0
        for idx, sample in enumerate(self.dataset):
            # print percentage of progress
            print(f"\rINFO: Initialization progress - {idx / len(self.dataset) * 100:.2f}%", end="")
            # append id to list of ids and convert polyline to segmentation mask
            for tag in sample.tags:
                if tag in ["train", "val", "test"]:
                    self.ids[tag].append(sample.id)
            mask = sample.polylines.to_segmentation(frame_size=(1920, 1200),
                                                    mask_targets={1: "traversable", 2: "untraversable"})
            sample["ground_truth"] = mask
            sample.save()

            # calculate mean and std for normalization
            image = cv2.imread(sample.filepath, cv2.IMREAD_COLOR)
            channels_sum += np.mean(image, axis=(0, 1))
            image_squared = image.astype(np.float32) ** 2
            channels_squared_sum += np.mean(image_squared, axis=(0, 1))
            num_batches += 1

        self.mean = (channels_sum / num_batches) / 255.0
        self.std = np.sqrt(np.abs((channels_squared_sum / num_batches) / 255.0 - self.mean ** 2))
        print("\nINFO: Dataset initialized")

    def save_prediction(self, prediction, idx):
        sample = self.dataset[self.ids[idx]]
        sample["prediction"] = fo.Segmentation(mask=prediction)
        sample.save()

    def show_dataset(self):
        session = fo.launch_app(self.dataset)
        session.wait()

    def get_item(self, idx, split: str):
        assert split in ["train", "val", "test"]
        image = cv2.imread(self.dataset[self.ids[split][idx]].filepath, cv2.IMREAD_COLOR)
        new_h, new_w = self.crop_size
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        image = self.input_transform(image)
        mask = self.dataset[self.ids[split][idx]]["ground_truth"]["mask"]
        return image, mask

    def input_transform(self, image):
        image = image.astype(np.float32)  # [:, :, ::-1]
        image = image / 255.0
        image -= self.mean / 255.0
        image /= self.std / 255.0
        return image

    def get_length(self, split: str):
        assert split in ["train", "val", "test"]
        return len(self.ids[split])


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
            path = os.path.join(data_dir, 'bags/traversability/marv/ugv_2022-08-12-15-18-34/')
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
    name = "traversability_dataset"
    directory = os.path.join(data_dir, name)
    dataset = TraversabilityImages(name, directory, (320, 192), {"train": 0.7, "test": 0.2, "val": 0.1})

    # train_split = dataset.dataset.match_tags(["train"])
    # Print the first few samples in the dataset
    for i in range(dataset.get_length("train")):
        print(dataset.get_item(i, "train"))
        pass

    dataset.show_dataset()


def clouds_demo():
    from matplotlib import pyplot as plt

    ds = TraversabilityClouds()

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
