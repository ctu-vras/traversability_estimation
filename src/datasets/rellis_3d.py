import os
from numpy.lib.recfunctions import structured_to_unstructured
from os.path import dirname, join, realpath
from .utils import *
from .base_dataset import BaseDataset
from copy import copy
import torch
from PIL import Image

__all__ = [
    'data_dir',
    'seq_names',
    'Sequence',
]

data_dir = realpath(join(dirname(__file__), '..', '..', 'data'))

seq_names = [
    '00000',
    '00001',
    '00002',
    '00003',
    '00004',
]


# seq_names = ['%05d' % i for i in range(5)]


class Sequence(torch.utils.data.Dataset):
    def __init__(self, seq=None, path=None, poses_file='poses.txt', poses_path=None, split=None):
        """Rellis-3D dataset: https://unmannedlab.github.io/research/RELLIS-3D.

        :param seq: Sequence number (from 0 to 4).
        :param path: Dataset path, takes precedence over name.
        :param poses_file: Poses CSV file name.
        :param poses_path: Override for poses CSV path.
        """
        assert isinstance(seq, str) or isinstance(seq, int)
        if isinstance(seq, int):
            seq = '%05d' % seq
        parts = seq.split('/')
        assert 1 <= len(parts) <= 2
        if len(parts) == 2:
            assert parts[0] == 'rellis_3d'
            seq = parts[1]
        if path is None:
            path = join(data_dir, 'Rellis_3D')

        self.seq = seq
        self.path = path
        self.poses_path = poses_path
        self.poses_file = poses_file
        P = np.zeros([3, 4])
        K = read_intrinsics(self.intrinsics_path())
        P[:3, :3] = K
        self.calibration = {
            'K': K,
            'P': P,
            'lid2cam': read_extrinsics(self.extrinsics_path()),
            'dist_coeff': np.array([-0.134313, -0.025905, 0.002181, 0.00084, 0]),
            'img_width': 1920,
            'img_height': 1200,
        }
        if self.poses_path or self.path:
            self.poses = read_poses(self.cloud_poses_path())
            self.ids_lid, self.ts_lid = self.get_ids(sensor='lidar')
            self.ids_rgb, self.ts_rgb = self.get_ids(sensor='rgb')
            self.ids_semseg, self.ts_semseg = self.get_ids(sensor='semseg')
            self.ids = self.ids_lid
        else:
            self.ids = None
            self.poses = None

    def get_ids(self, sensor='lidar'):
        if sensor == 'lidar':
            sensor_folder = 'os1_cloud_node_color_ply'
        elif sensor == 'rgb':
            sensor_folder = 'pylon_camera_node'
        elif sensor == 'semseg':
            sensor_folder = 'pylon_camera_node_label_id'
        else:
            raise ValueError('Unsupported sensor type (choose one of: lidar, or rgb, or semseg)')
        # id = frame0000i_sec_msec
        ids = [f[:-4] for f in np.sort(os.listdir(os.path.join(self.path, self.seq, sensor_folder)))]
        ts = [float('%.3f' % (float(id.split('-')[1].split('_')[0]) + float(id.split('-')[1].split('_')[1]) / 1000.0))
              for id in ids]
        ts = np.sort(ts).tolist()
        return ids, ts

    def local_cloud_path(self, id):
        return os.path.join(self.path, self.seq, 'os1_cloud_node_color_ply', '%s.ply' % id)

    def cloud_poses_path(self):
        if self.poses_path:
            return self.poses_path
        return os.path.join(self.path, 'calibration', self.seq, self.poses_file)

    def image_path(self, id):
        return os.path.join(self.path, self.seq, 'pylon_camera_node', '%s.jpg' % id)

    def semseg_path(self, id):
        return os.path.join(self.path, self.seq, 'pylon_camera_node_label_id', '%s.png' % id)

    def intrinsics_path(self):
        return os.path.join(self.path, 'calibration', self.seq, 'camera_info.txt')

    def extrinsics_path(self):
        return os.path.join(self.path, 'calibration', self.seq, 'transforms.yaml')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        if isinstance(item, int):
            id = self.ids[item]
            return self.local_cloud(id), self.cloud_pose(id), self.camera_image(id), self.camera_semseg(id)

        ds = copy(self)
        if isinstance(item, (list, tuple)):
            ds.ids = [self.ids[i] for i in item]
        else:
            assert isinstance(item, slice)
            ds.ids = self.ids[item]
        return ds

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def local_cloud(self, id_lid):
        assert id_lid in self.ids_lid
        return read_points(self.local_cloud_path(id_lid))

    def cloud_pose(self, id):
        t = float(id.split('-')[1].split('_')[0]) + float(id.split('-')[1].split('_')[1]) / 1000.0
        i = np.searchsorted(self.ts_lid, t)
        i = np.clip(i, 0, len(self.ids_lid))
        return self.poses[i]

    def camera_image(self, id):
        assert id in self.ids  # these are lidar ids
        t = float(id.split('-')[1].split('_')[0]) + float(id.split('-')[1].split('_')[1]) / 1000.0
        i = np.searchsorted(self.ts_rgb, t)
        i = np.clip(i, 0, len(self.ids_rgb) - 1)
        return read_rgb(self.image_path(self.ids_rgb[i]))

    def camera_semseg(self, id):
        assert id in self.ids  # these are lidar ids
        t = float(id.split('-')[1].split('_')[0]) + float(id.split('-')[1].split('_')[1]) / 1000.0
        i = np.searchsorted(self.ts_semseg, t)
        i = np.clip(i, 0, len(self.ids_semseg) - 1)
        return read_semseg(self.semseg_path(self.ids_semseg[i]))


class Rellis3D(BaseDataset):
    CLASSES = ['void', 'dirt', 'grass', 'tree', 'pole', 'water', 'sky', 'vehicle', 'object', 'asphalt', 'building',
               'log', 'person', 'fence', 'bush', 'concrete', 'barrier', 'puddle', 'mud', 'rubble']
    LABEL_MAPPING = {0: 0,
                     1: 0,
                     3: 1,
                     4: 2,
                     5: 3,
                     6: 4,
                     7: 5,
                     8: 6,
                     9: 7,
                     10: 8,
                     12: 9,
                     15: 10,
                     17: 11,
                     18: 12,
                     19: 13,
                     23: 14,
                     27: 15,
                     # 29: 1,
                     # 30: 1,
                     31: 16,
                     # 32: 4,
                     33: 17,
                     34: 18}

    def __init__(self,
                 path=None,
                 split='train',
                 num_samples=None,
                 classes=None,
                 multi_scale=True,
                 flip=True,
                 ignore_label=-1,
                 base_size=2048,
                 crop_size=(1200, 1920),
                 downsample_rate=1,
                 scale_factor=16,
                 mean=np.asarray([0.54218053, 0.64250553, 0.56620195]),
                 std=np.asarray([0.54218052, 0.64250552, 0.56620194])):
        super(Rellis3D, self).__init__(ignore_label, base_size,
                                       crop_size, downsample_rate, scale_factor, mean, std, )

        if path is None:
            path = join(data_dir, 'Rellis_3D')
        assert os.path.exists(path)
        assert split in ['train', 'val', 'test']
        self.path = path
        self.split = split
        if not classes:
            classes = self.CLASSES
        # convert str names to class values on masks
        self.class_values = [list(self.LABEL_MAPPING.values())[self.CLASSES.index(cls.lower())] for cls in classes]

        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label

        self.mean = mean
        self.std = std
        self.scale_factor = scale_factor
        self.downsample_rate = 1. / downsample_rate

        self.multi_scale = multi_scale
        self.flip = flip

        self.img_list = [line.strip().split() for line in open(os.path.join(path, '%s.lst' % split))]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

    def read_files(self):
        files = []
        for item in self.img_list:
            image_path, label_path = item
            image_path = os.path.join(self.path, image_path)
            label_path = os.path.join(self.path, label_path)
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append({
                "img": image_path,
                "label": label_path,
                "name": name,
                "weight": 1
            })
        return files

    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.LABEL_MAPPING.items():
                label[temp == k] = v
        else:
            for k, v in self.LABEL_MAPPING.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        item = self.files[index]
        image = cv2.imread(item["img"], cv2.IMREAD_COLOR)

        mask = np.array(Image.open(item["label"]))
        mask = self.convert_label(mask, inverse=False)

        if 'test' in self.split:
            new_h, new_w = self.crop_size
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            image = self.input_transform(image)
        else:
            # add augmentations
            image, mask = self.apply_augmentations(image, mask, self.multi_scale, self.flip)
        # extract certain classes from mask
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=0).astype('float')
        return image.copy(), mask.copy()


class LaserScan:
    """
    Class that contains LaserScan with x,y,z,r:
    https://github.com/PRBonn/semantic-kitti-api/blob/8e75f4d049b787321f68a11753cb5947b1e58e17/auxiliary/laserscan.py
    """
    EXTENSIONS_SCAN = ['.bin']

    def __init__(self, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0):
        self.project = project
        self.proj_H = H
        self.proj_W = W
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down
        self.reset()

    def reset(self):
        """ Reset scan members. """
        self.points = np.zeros((0, 3), dtype=np.float32)        # [m, 3]: x, y, z
        self.remissions = np.zeros((0, 1), dtype=np.float32)    # [m ,1]: remission

        # projected range image - [H,W] range (-1 is no data)
        self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)

        # unprojected range (list of depths for each point)
        self.unproj_range = np.zeros((0, 1), dtype=np.float32)

        # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                                dtype=np.float32)

        # projected remission - [H,W] intensity (-1 is no data)
        self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                      dtype=np.float32)

        # projected index (for each pixel, what I am in the pointcloud)
        # [H,W] index (-1 is no data)
        self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                                dtype=np.int32)

        # for each point, where it is in the range image
        self.proj_x = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: x
        self.proj_y = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: y

        # mask containing for each pixel, if it contains a point or not
        self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                                  dtype=np.int32)       # [H,W] mask

    def size(self):
        """ Return the size of the point cloud. """
        return self.points.shape[0]

    def __len__(self):
        return self.size()

    def open_scan(self, filename):
        """ Open raw scan and fill in attributes
        """
        # reset just in case there was an open structure
        self.reset()

        # check filename is string
        if not isinstance(filename, str):
            raise TypeError("Filename should be string type, "
                            "but was {type}".format(type=str(type(filename))))

        # check extension is a laserscan
        if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
            raise RuntimeError("Filename extension is not valid scan file.")

        # if all goes well, open pointcloud
        scan = np.fromfile(filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))

        # put in attribute
        points = scan[:, 0:3]    # get xyz
        remissions = scan[:, 3]  # get remission
        self.set_points(points, remissions)

    def set_points(self, points, remissions=None):
        """ Set scan attributes (instead of opening from file)
        """
        # reset just in case there was an open structure
        self.reset()

        # check scan makes sense
        if not isinstance(points, np.ndarray):
            raise TypeError("Scan should be numpy array")

        # check remission makes sense
        if remissions is not None and not isinstance(remissions, np.ndarray):
            raise TypeError("Remissions should be numpy array")

        # put in attribute
        self.points = points    # get xyz
        if remissions is not None:
            self.remissions = remissions  # get remission
        else:
            self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

        # if projection is wanted, then do it and fill in the structure
        if self.project:
            self.do_range_projection()

    def do_range_projection(self):
        """ Project a pointcloud into a spherical projection image.projection.
            Function takes no arguments because it can be also called externally
            if the value of the constructor was not set (in case you change your
            mind about wanting the projection)
        """
        # laser parameters
        fov_up = self.proj_fov_up / 180.0 * np.pi      # field of view up in rad
        fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

        # get depth of all points
        depth = np.linalg.norm(self.points, 2, axis=1)

        # get scan components
        scan_x = self.points[:, 0]
        scan_y = self.points[:, 1]
        scan_z = self.points[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / (depth + 1e-8))

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.proj_W                              # in [0.0, W]
        proj_y *= self.proj_H                              # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
        self.proj_x = np.copy(proj_x)  # store a copy in orig order

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
        self.proj_y = np.copy(proj_y)  # stope a copy in original order

        # copy of depth in original order
        self.unproj_range = np.copy(depth)

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        points = self.points[order]
        remission = self.remissions[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # assing to images
        self.proj_range[proj_y, proj_x] = depth
        self.proj_xyz[proj_y, proj_x] = points
        self.proj_remission[proj_y, proj_x] = remission
        self.proj_idx[proj_y, proj_x] = indices
        self.proj_mask = (self.proj_idx > 0).astype(np.float32)


def laser_scan_demo():
    import vispy
    from vispy.scene import visuals, SceneCanvas
    from matplotlib import pyplot as plt

    def get_mpl_colormap(cmap_name):
        cmap = plt.get_cmap(cmap_name)
        # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)
        # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
        return color_range.reshape(256, 3).astype(np.float32) / 255.0

    # new canvas prepared for visualizing data
    canvas = SceneCanvas(keys='interactive', show=True)

    # grid
    grid = canvas.central_widget.add_grid()

    # laserscan part
    scan_view = vispy.scene.widgets.ViewBox(border_color='white', parent=canvas.scene)
    grid.add_widget(scan_view, 0, 0)
    scan_vis = visuals.Markers()
    scan_view.camera = 'turntable'
    scan_view.add(scan_vis)
    visuals.XYZAxis(parent=scan_view.scene)

    scan = LaserScan(project=True)
    scan_path = os.path.join(data_dir, 'Rellis_3D', '00000', 'os1_cloud_node_kitti_bin', '002218.bin')
    scan.open_scan(scan_path)

    # plot scan
    power = 16
    # print()
    range_data = np.copy(scan.unproj_range)
    # print(range_data.max(), range_data.min())
    range_data = range_data ** (1 / power)
    # print(range_data.max(), range_data.min())
    viridis_range = ((range_data - range_data.min()) /
                     (range_data.max() - range_data.min()) *
                     255).astype(np.uint8)
    viridis_map = get_mpl_colormap("viridis")
    viridis_colors = viridis_map[viridis_range]

    scan_vis.set_data(scan.points,
                      face_color=viridis_colors[..., ::-1],
                      edge_color=viridis_colors[..., ::-1],
                      size=1)

    vispy.app.run()


def semseg_test():
    from datasets.utils import visualize, convert_color
    from hrnet.core.function import convert_label
    import yaml

    split = np.random.choice(['test', 'train', 'val'])
    # split = 'test'
    ds = Rellis3D(split=split)
    image, gt_mask = ds[int(np.random.choice(range(len(ds))))]

    if split in ['val', 'train']:
        image = image.transpose([1, 2, 0])

    image_vis = np.uint8(255 * (image * ds.std + ds.mean))

    CFG = yaml.safe_load(open(os.path.join(data_dir,  "../config/rellis.yaml"), 'r'))
    color_map = CFG["color_map"]
    gt_arg = np.argmax(gt_mask, axis=0).astype(np.uint8) - 1
    gt_arg = convert_label(gt_arg, inverse=True)
    gt_color = convert_color(gt_arg, color_map)

    visualize(
        image=image_vis,
        label=gt_color,
    )


def lidar_map_demo():
    from tqdm import tqdm
    import open3d as o3d

    name = np.random.choice(seq_names)
    ds = Sequence(seq='rellis_3d/%s' % name)

    plt.figure()
    plt.title('Trajectory')
    plt.axis('equal')
    plt.plot(ds.poses[:, 0, 3], ds.poses[:, 1, 3], '.')
    plt.grid()
    plt.show()

    clouds = []
    for data in tqdm(ds[::100]):
        cloud, pose, _, _ = data
        cloud = structured_to_unstructured(cloud[['x', 'y', 'z']])
        cloud = np.matmul(cloud, pose[:3, :3].T) + pose[:3, 3:].T

        clouds.append(cloud)
    cloud = np.concatenate(clouds)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    o3d.visualization.draw_geometries([pcd.voxel_down_sample(voxel_size=0.5)])


def lidar2cam_demo():
    seq = np.random.choice(seq_names)
    ds = Sequence(seq='rellis_3d/%s' % seq)

    dist_coeff = ds.calibration['dist_coeff'].reshape((5, 1))
    K = ds.calibration['K']
    T_lid2cam = ds.calibration['lid2cam']

    for _ in range(1):
        data = ds[int(np.random.choice(range(len(ds))))]
        points, pose, rgb, semseg = data
        points = structured_to_unstructured(points[['x', 'y', 'z']])

        img_height, img_width = rgb.shape[:2]
        assert rgb.shape[:2] == semseg.shape[:2]

        R_lidar2cam = T_lid2cam[:3, :3]
        t_lidar2cam = T_lid2cam[:3, 3]
        rvec, _ = cv2.Rodrigues(R_lidar2cam)
        tvec = t_lidar2cam.reshape(3, 1)
        xyz_v, color = filter_camera_points(points, img_width, img_height, K, T_lid2cam)

        imgpoints, _ = cv2.projectPoints(xyz_v[:, :], rvec, tvec, K, dist_coeff)
        imgpoints = np.squeeze(imgpoints, 1)
        imgpoints = imgpoints.T

        res_rgb = print_projection_plt(points=imgpoints, color=color, image=rgb)
        res_semseg = print_projection_plt(points=imgpoints, color=color, image=semseg)

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(res_rgb / 255)
        plt.subplot(1, 2, 2)
        plt.imshow(res_semseg / 255)
        plt.show()


def semseg_demo():
    seq = np.random.choice(seq_names)
    ds = Sequence(seq='rellis_3d/%s' % seq)

    for _ in range(1):
        id = int(np.random.choice(range(len(ds))))
        print('Data index:', id)

        data = ds[id]
        _, _, rgb, semseg = data

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(rgb / 255)
        plt.subplot(1, 2, 2)
        plt.imshow(semseg / 255)
        plt.show()


def main():
    semseg_test()
    lidar_map_demo()
    lidar2cam_demo()
    semseg_demo()


if __name__ == '__main__':
    main()
