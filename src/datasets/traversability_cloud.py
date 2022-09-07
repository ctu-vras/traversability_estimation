import os
import numpy as np
import open3d as o3d

from segments import SegmentsClient, SegmentsDataset
from sklearn.model_selection import train_test_split


class TraversabilityCloud(object):
    def __init__(self, path: str, version: str = "v0.2", split: str = None):
        self.path = path
        self.split = split

        self.version = version
        self.api_key = '4bacc032570420552ef6b038e1a1e8383ac372d9'
        self.dataset_name = 'aleskucera/Pointcloud_traversability'

        self.label_map = {0: 0,
                          1: 1,
                          2: 255}

        self.color_map = {0: [0, 0, 0],
                          1: [0, 255, 0],
                          255: [255, 0, 0]}

        self.point_clouds, self.labels = self._init_dataset()

    def _init_dataset(self) -> (list, list):
        point_clouds = []
        labels = []
        # load and format dataset annotations
        client = SegmentsClient(self.api_key)
        release = client.get_release(self.dataset_name, self.version)
        dataset = SegmentsDataset(release, labelset='ground-truth', filter_by=['LABELED', 'REVIEWED'])
        samples = dataset.samples
        for sample in samples:
            # get attributes of the label
            attributes = sample["labels"]["ground-truth"]["attributes"]
            point_annotations = attributes["point_annotations"]
            annotations = attributes["annotations"]

            # append sample to dataset
            point_clouds.append(self._get_path(sample["name"]))
            labels.append(self._map_annotations(point_annotations, annotations))
        return self._generate_split(point_clouds, labels)

    def _map_annotations(self, point_annotations: list, annotations: list) -> np.ndarray:
        ret = []

        # map annotations by self.label_map
        mapped_annotations = {0: 0}
        for annotation in annotations:
            mapped_annotations[annotation["id"]] = self.label_map[annotation["category_id"]]

        for instance_id in point_annotations:
            category_id = mapped_annotations[instance_id]
            ret.append(category_id)
        return np.array(ret)

    def _get_path(self, name: str) -> str:
        return os.path.join(self.path, name)

    def _generate_split(self, X: list, y: list, test_ratio=0.2) -> (list, list):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
        if self.split == 'train':
            X = X_train
            y = y_train
        elif self.split in ['val', 'test']:
            X = X_test
            y = y_test
        return X, y

    def __getitem__(self, index: int) -> (np.ndarray, np.ndarray):
        point_cloud = o3d.io.read_point_cloud(self.point_clouds[index])
        point_cloud = np.asarray(point_cloud.points).reshape((128, -1, 3))
        label = self.labels[index].reshape((128, -1))
        return point_cloud, label

    def __len__(self) -> int:
        return len(self.point_clouds)

    def visualize_sample(self, index: int) -> None:
        point_cloud = o3d.io.read_point_cloud(self.point_clouds[index])

        colors = np.array([self.color_map[label] for label in self.labels[index]])
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([point_cloud])


def main():
    directory = "/home/ales/Datasets/points_colored"
    dataset = TraversabilityCloud(directory)
    print(f"INFO: Initialized dataset split type: {dataset.split}")
    print(f"INFO: Split contains {len(dataset)} samples.")
    for i, sample in enumerate(dataset):
        point_cloud, label = sample
        print(f"INFO: Sample {i} has shape {point_cloud.shape} and label {label.shape}")
        dataset.visualize_sample(i)


if __name__ == '__main__':
    main()
