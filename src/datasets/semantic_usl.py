import os
from datasets.base_dataset import BaseDatasetClouds, data_dir


class SemanticUSL(BaseDatasetClouds):
    CLASSES = ['dirt', 'grass', 'tree', 'pole', 'water', 'sky', 'vehicle', 'object', 'asphalt', 'building',
               'log', 'person', 'fence', 'bush', 'concrete', 'barrier', 'puddle', 'mud', 'rubble']

    def __init__(self,
                 path=None,
                 seq = '00',
                 split='train',
                 fields=None,
                 num_samples=None,
                 color_map=None,
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

        self.setup_color_map(color_map)
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

    def __getitem__(self, index):
        item = self.files[index]
        self.scan.open_scan(item["depth"])
        self.scan.open_label(item["label"])

        data, label = self.create_sample()

        return data, label

    def __len__(self):
        return len(self.files)


def demo():
    ds = SemanticUSL()
    ds[0]


def main():
    demo()


if __name__ == '__main__':
    main()
