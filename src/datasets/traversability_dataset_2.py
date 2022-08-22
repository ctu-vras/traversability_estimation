import cv2
import torch
import numpy as np
import fiftyone as fo
import fiftyone.utils.splits as fous

DATASET_PATH = "/home/ales/dataset/TraversabilityDataset"


class TraversabilityDatasetImages(torch.utils.data.Dataset):
    def __init__(self, crop_size: tuple = (1920, 1200)):
        self.crop_size = crop_size
        self.samples = self._load_dataset(DATASET_PATH)
        self.img_paths = self.samples.values("filepath")
        self.mask_targets = {0: "background",
                             1: "traversable",
                             2: "non-traversable"}
        self.class_values = [0, 1, 2]

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        sample = self.samples[img_path]

        # image preprocessing
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, self.crop_size, interpolation=cv2.INTER_LINEAR)
        image = self._input_transform(image)
        image = image.transpose((2, 0, 1))

        # mask preprocessing
        mask = sample.polylines.to_segmentation(frame_size=(1920, 1200),
                                                mask_targets=self.mask_targets)["mask"]
        mask = cv2.resize(mask, self.crop_size, interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.long)
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


if __name__ == "__main__":
    dataset = TraversabilityDatasetImages(crop_size=(1920, 1200))
    length = len(dataset)
    # dataset.show_dataset()
    splits = torch.utils.data.random_split(dataset,
                                           [int(0.7 * length), int(0.2 * length), int(0.1 * length)],
                                           generator=torch.Generator().manual_seed(42))
    # show first images from each split
    for split in splits:
        print(len(split))
