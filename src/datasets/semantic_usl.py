import os
from datasets.base_dataset import BaseDatasetClouds, data_dir
from datasets.laserscan import SemLaserScan
from datasets.base_dataset import TRAVERSABILITY_LABELS, TRAVERSABILITY_COLOR_MAP
from datasets.base_dataset import FLEXIBILITY_LABELS, FLEXIBILITY_COLOR_MAP
from traversability_estimation.utils import convert_label
import yaml
import numpy as np


class SemanticUSL(BaseDatasetClouds):
    CLASSES = ['unlabeled', 'outlier', 'car', 'bicycle', 'bus', 'motorcycle', 'on-rails', 'truck', 'other-vehicle',
               'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground', 'building',
               'fence', 'other-structure', 'lane-marking', 'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign',
               'other-object', 'moving-car', 'moving-bicyclist', 'moving-person', 'moving-motorcyclist',
               'moving-on-rails', 'moving-bus', 'moving-truck', 'moving-other-vehicle']

    def __init__(self,
                 path=None,
                 sequences=None,
                 split=None,
                 fields=None,
                 num_samples=None,
                 lidar_beams_step=2,
                 labels_mode='labels',
                 labels_mapping=None,
                 ):
        super(SemanticUSL, self).__init__(path=path, fields=fields,
                                          depth_img_H=64, depth_img_W=2048,
                                          lidar_fov_up=16.6, lidar_fov_down=-16.6,
                                          lidar_beams_step=lidar_beams_step,
                                          )
        if path is None:
            path = os.path.join(data_dir, 'SemanticUSL', 'SemanticUSL', 'sequences')
        assert os.path.exists(path)
        self.path = path

        if not sequences:
            sequences = ['03', '12', '21', '32']
        assert set(sequences) <= {'03', '12', '21', '32'}
        self.sequences = sequences

        assert labels_mode in ['masks', 'labels']
        self.labels_mode = labels_mode

        assert split in [None, 'train', 'val', 'test']
        self.split = split

        self.labels_mapping = labels_mapping

        if self.labels_mapping is None:
            cfg = yaml.safe_load(open(os.path.join(data_dir, '../config', 'semantickitti19.yaml'), 'r'))
            self.class_values = list(cfg['labels'].keys())
            self.learning_map = cfg['learning_map']
            self.learning_map_inv = cfg['learning_map_inv']
            self.class_values = convert_label(self.class_values, label_mapping=self.learning_map)
            self.CLASSES = list(cfg['labels'].values())
            self.color_map = cfg['color_map']
            self.label_map = None
        else:
            if self.labels_mapping == 'traversability':
                self.color_map = TRAVERSABILITY_COLOR_MAP
                self.class_values = np.sort([k for k in TRAVERSABILITY_LABELS.keys()]).tolist()
            elif self.labels_mapping == 'flexibility':
                self.color_map = FLEXIBILITY_COLOR_MAP
                self.class_values = np.sort([k for k in FLEXIBILITY_LABELS.keys()]).tolist()

            self.label_map = self.get_label_map(path=os.path.join(data_dir,
                                                                  "../config/semantickitti19_to_%s.yaml" %
                                                                  self.labels_mapping))

        self.scan = SemLaserScan(nclasses=len(self.CLASSES), sem_color_dict=self.color_map,
                                 project=True, H=self.depth_img_H, W=self.depth_img_W,
                                 fov_up=self.lidar_fov_up, fov_down=self.lidar_fov_down)

        self.files = self.read_files()
        self.files = self.generate_split()
        if num_samples:
            self.files = self.files[:num_samples]

    def read_files(self):
        depths_list = []
        for seq in self.sequences:
            depths_list = depths_list + [os.path.join(self.path, seq, 'velodyne', f) for f in
                                         os.listdir(os.path.join(self.path, seq, 'velodyne'))]
        files = []
        for depth_path in depths_list:
            label_path = depth_path.replace('velodyne', 'labels').replace('.bin', '.label')
            id = os.path.splitext(os.path.basename(label_path))[0]

            files.append({
                "depth": depth_path,
                "label": label_path,
                "name": id,
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


def demo(n_runs=1):
    from traversability_estimation.utils import visualize_imgs, visualize_cloud

    ds = SemanticUSL()
    ds_trav = SemanticUSL(labels_mapping='traversability')

    for _ in range(n_runs):
        idx = np.random.choice(range(len(ds)))

        data, label = ds[idx]
        label_trav = ds_trav[idx][1]

        depth_img = data[-1]

        power = 16
        depth_img_vis = np.copy(depth_img)
        depth_img_vis[depth_img_vis > 0] = depth_img_vis[depth_img_vis > 0] ** (1 / power)
        depth_img_vis[depth_img_vis > 0] =\
            (depth_img_vis[depth_img_vis > 0] - depth_img_vis[depth_img_vis > 0].min()) / \
            (depth_img_vis[depth_img_vis > 0].max() - depth_img_vis[depth_img_vis > 0].min())

        color = ds.label_to_color(label)
        color_trav = ds_trav.label_to_color(label_trav)

        visualize_cloud(xyz=data[:3].reshape((3, -1)).T, color=color.reshape((-1, 3)))
        visualize_cloud(xyz=data[:3].reshape((3, -1)).T, color=color_trav.reshape((-1, 3)))

        visualize_imgs(range_image=depth_img_vis,
                       segmentation=color,
                       traversability=color_trav,
                       layout='columns')


def main():
    demo(5)


if __name__ == '__main__':
    main()
