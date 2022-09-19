import os
from datasets.base_dataset import BaseDatasetClouds, data_dir
from datasets.laserscan import SemLaserScan
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
                 seq='00',
                 split='train',
                 fields=None,
                 num_samples=None,
                 lidar_beams_step=1,
                 labels_mode='labels'
                 ):
        super(SemanticUSL, self).__init__(path=path, fields=fields,
                                          depth_img_H=64, depth_img_W=2048,
                                          lidar_fov_up=16.6, lidar_fov_down=-16.6,
                                          lidar_beams_step=lidar_beams_step,
                                          )
        if path is None:
            path = os.path.join(data_dir, 'SemanticUSL', 'SemanticUSL', 'sequences', seq)
        assert os.path.exists(path)
        self.path = path

        if not seq in ['03', '12', '21', '32']:
            print("Sequence %s does not have ground truth (only '03', '12', '21', '32' do)" % seq)
        self.seq = seq

        assert labels_mode in ['masks', 'labels']
        self.labels_mode = labels_mode

        cfg = yaml.safe_load(open(os.path.join(data_dir, 'SemanticUSL', 'semantickitti19.yaml'), 'r'))
        self.class_values = list(cfg['labels'].keys())
        self.CLASSES = list(cfg['labels'].values())
        self.color_map = cfg['color_map']
        self.label_map = None

        self.scan = SemLaserScan(nclasses=len(self.CLASSES), sem_color_dict=self.color_map,
                                 project=True, H=self.depth_img_H, W=self.depth_img_W,
                                 fov_up=self.lidar_fov_up, fov_down=self.lidar_fov_down)

        self.depths_list = [os.path.join(self.path, 'velodyne', f) for f in
                            os.listdir(os.path.join(self.path, 'velodyne'))]

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


def demo(n_runs=1):
    from traversability_estimation.utils import visualize_imgs, visualize_cloud

    seq = np.random.choice(['03', '12', '21', '32'])
    ds = SemanticUSL(seq=seq)

    for _ in range(n_runs):
        data, label = ds[np.random.choice(range(len(ds)))]

        depth_img = data[-1]

        power = 16
        depth_img_vis = np.copy(depth_img)
        depth_img_vis[depth_img_vis > 0] = depth_img_vis[depth_img_vis > 0] ** (1 / power)
        depth_img_vis[depth_img_vis > 0] = (depth_img_vis[depth_img_vis > 0] - depth_img_vis[depth_img_vis > 0].min()) / \
                                           (depth_img_vis[depth_img_vis > 0].max() - depth_img_vis[
                                               depth_img_vis > 0].min())

        color = ds.label_to_color(label)

        visualize_cloud(xyz=data[:3].reshape((3, -1)).T, color=color.reshape((-1, 3)))

        visualize_imgs(segmentation=color, range_image=depth_img_vis, layout='columns')


def main():
    demo(5)


if __name__ == '__main__':
    main()
