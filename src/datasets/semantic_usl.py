import os
from datasets.base_dataset import BaseDatasetClouds, data_dir
import yaml
import numpy as np


class SemanticUSL(BaseDatasetClouds):
    CLASSES = ['dirt', 'grass', 'tree', 'pole', 'water', 'sky', 'vehicle', 'object', 'asphalt', 'building',
               'log', 'person', 'fence', 'bush', 'concrete', 'barrier', 'puddle', 'mud', 'rubble']

    def __init__(self,
                 path=None,
                 seq = '00',
                 split='train',
                 fields=None,
                 num_samples=None,
                 traversability_labels=False,
                 lidar_beams_step=1,
                 labels_mode='labels'
                 ):
        super(SemanticUSL, self).__init__(path=path, fields=fields,
                                          depth_img_H=64, depth_img_W=2048,
                                          lidar_fov_up=16.6, lidar_fov_down=-16.6,
                                          lidar_beams_step=lidar_beams_step,
                                          traversability_labels=traversability_labels)
        if path is None:
            path = os.path.join(data_dir, 'SemanticUSL', 'SemanticUSL', 'sequences', seq)
        assert os.path.exists(path)
        self.path = path

        self.seq = seq

        assert split in ['train', 'val', 'test']
        self.split = split

        assert labels_mode in ['masks', 'labels']
        self.labels_mode = labels_mode

        cfg = yaml.safe_load(open(os.path.join(data_dir, 'SemanticUSL', 'semantickitti19.yaml'), 'r'))
        self.class_values = list(cfg['labels'].keys())
        self.CLASSES = list(cfg['labels'].values())
        self.color_map = cfg['color_map']
        self.label_map = None
        # self.setup_color_map(color_map)

        self.get_scan()

        self.depths_list = [os.path.join(self.path, 'velodyne', f) for f in os.listdir(os.path.join(self.path, 'velodyne'))]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

    def read_files(self):
        files = []
        for depth_path in self.depths_list:

            label_path = depth_path.replace('velodyne', 'labels').replace('.bin', '.label')
            id = os.path.splitext(os.path.basename(label_path))[0]

            files.append({
                "depth": depth_path,
                "label": label_path,
                "name": id,
                "weight": 1
            })
        return files

    def label_to_color(self, label):
        if len(label.shape) == 3:
            C, H, W = label.shape
            label = np.argmax(label, axis=0)
            assert label.shape == (H, W)
        color = self.scan.sem_color_lut[label]
        return color

    def __getitem__(self, index):
        item = self.files[index]
        self.scan.open_scan(item["depth"])
        self.scan.open_label(item["label"])

        data, label = self.create_sample()

        return data, label

    def __len__(self):
        return len(self.files)


def demo():
    import matplotlib.pyplot as plt

    ds = SemanticUSL()
    data, label = ds[0]

    range_image = data[0]
    color = ds.label_to_color(label)

    plt.figure(figsize=(20, 10))
    plt.subplot(2, 1, 1)
    plt.imshow(color)
    plt.subplot(2, 1, 2)
    plt.imshow(range_image)
    plt.tight_layout()
    plt.show()


def main():
    demo()


if __name__ == '__main__':
    main()
