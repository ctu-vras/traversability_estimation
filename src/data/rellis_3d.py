import os
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured
from os.path import dirname, join, realpath
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import yaml
from PIL import Image


__all__ = [
    'data_dir',
    'seq_names',
    'Dataset',
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


def read_points(path, dtype=np.float32):
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    assert points.shape[1] == 3
    vps = np.zeros_like(points)
    points = np.hstack([points, vps])
    points = unstructured_to_structured(points.astype(dtype=dtype), names=['x', 'y', 'z', 'vp_x', 'vp_y', 'vp_z'])
    del pcd
    return points


def read_poses(path, zero_origin=True):
    data = np.genfromtxt(path)
    poses = np.asarray([np.eye(4) for _ in range(len(data))]).reshape([-1, 4, 4])
    poses[:, :3, :4] = data.reshape([-1, 3, 4])
    del data
    # transform to 0-origin (pose[0] = np.eye(4))
    if zero_origin:
        poses = np.einsum("ij,njk->nik", np.linalg.inv(poses[0]), poses)
    return poses


def read_image(path):
    img = cv2.imread(path)
    img = np.asarray(img, dtype=np.uint8)
    return img


def read_intrinsics(path):
    data = np.loadtxt(path)
    K = np.zeros((3, 3))
    K[0, 0] = data[0]
    K[1, 1] = data[1]
    K[2, 2] = 1
    K[0, 2] = data[2]
    K[1, 2] = data[3]
    return K


def read_extrinsics(path, key='os1_cloud_node-pylon_camera_node'):
    """
    Transformation between camera and lidar
    """
    with open(path, 'r') as f:
        data = yaml.load(f, Loader=yaml.Loader)
    q = data[key]['q']
    q = np.array([q['x'], q['y'], q['z'], q['w']])
    t = data[key]['t']
    t = np.array([t['x'], t['y'], t['z']])
    R_vc = Rotation.from_quat(q)
    R_vc = R_vc.as_matrix()

    RT = np.eye(4, 4)
    RT[:3, :3] = R_vc
    RT[:3, -1] = t
    RT = np.linalg.inv(RT)
    return RT


def print_projection_plt(points, color, image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for i in range(points.shape[1]):
        cv2.circle(hsv_image, (np.int32(points[0][i]), np.int32(points[1][i])), 4, (int(color[i]), 255, 255), -1)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)


def depth_color(val, min_d=0, max_d=120):
    np.clip(val, 0, max_d, out=val)
    return (((val - min_d) / (max_d - min_d)) * 120).astype(np.uint8)


def points_filter(points, img_width, img_height, K, RT):
    ctl = np.array(RT)
    fov_x = 2 * np.arctan2(img_width, 2 * K[0, 0]) * 180 / 3.1415926 + 10
    fov_y = 2 * np.arctan2(img_height, 2 * K[1, 1]) * 180 / 3.1415926 + 10
    p_l = np.ones((points.shape[0], points.shape[1]+1))
    p_l[:, :3] = points
    p_c = np.matmul(ctl, p_l.T)
    p_c = p_c.T
    x = p_c[:, 0]
    y = p_c[:, 1]
    z = p_c[:, 2]
    xangle = np.arctan2(x, z)*180 / np.pi
    yangle = np.arctan2(y, z)*180 / np.pi
    flag2 = (xangle > -fov_x/2) & (xangle < fov_x/2)
    flag3 = (yangle > -fov_y/2) & (yangle < fov_y/2)
    res = p_l[flag2&flag3, :3]
    res = np.array(res)
    x = res[:, 0]
    y = res[:, 1]
    z = res[:, 2]
    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    color = depth_color(dist, 0, 70)
    return res, color


color_palette = {
    0: {"color": [0, 0, 0],  "name": "void"},
    1: {"color": [108, 64, 20],   "name": "dirt"},
    3: {"color": [0, 102, 0],   "name": "grass"},
    4: {"color": [0, 255, 0],  "name": "tree"},
    5: {"color": [0, 153, 153],  "name": "pole"},
    6: {"color": [0, 128, 255],  "name": "water"},
    7: {"color": [0, 0, 255],  "name": "sky"},
    8: {"color": [255, 255, 0],  "name": "vehicle"},
    9: {"color": [255, 0, 127],  "name": "object"},
    10: {"color": [64, 64, 64],  "name": "asphalt"},
    12: {"color": [255, 0, 0],  "name": "building"},
    15: {"color": [102, 0, 0],  "name": "log"},
    17: {"color": [204, 153, 255],  "name": "person"},
    18: {"color": [102, 0, 204],  "name": "fence"},
    19: {"color": [255, 153, 204],  "name": "bush"},
    23: {"color": [170, 170, 170],  "name": "concrete"},
    27: {"color": [41, 121, 255],  "name": "barrier"},
    31: {"color": [134, 255, 239],  "name": "puddle"},
    33: {"color": [99, 66, 34],  "name": "mud"},
    34: {"color": [110, 22, 138],  "name": "rubble"}
}


def convert_label(label, inverse=False):
    temp = label.copy()
    if inverse:
        for v, k in color_palette.items():
            label[temp == k["color"]] = v
    else:
        label = np.zeros(temp.shape+(3,))
        for k, v in color_palette.items():
            label[temp == k, :] = v["color"]
    return label


def read_semseg(path, label_size=None):
    semseg = Image.open(path)
    if label_size is not None:
        if label_size[0] != semseg.size[0] or label_size[1] != semseg.size[1]:
            semseg = semseg.resize((label_size[1], label_size[0]), Image.NEAREST)
            semseg = np.array(semseg)[:, :, 0]
    semseg = np.array(semseg, dtype=np.uint8)
    semseg = convert_label(semseg, False)
    return semseg


class Dataset(object):
    default_poses_file = 'poses.txt'

    def __init__(self, seq=None, path=None, poses_file=default_poses_file, poses_path=None):
        """Rellis-3D dataset: https://unmannedlab.github.io/research/RELLIS-3D.
        
        Rellis_3D
        ├── 00000
        │   ├── os1_cloud_node_color_ply
        │   ├── pylon_camera_node
        │   ├── pylon_camera_node_label_color
        │   └── pylon_camera_node_label_id
        ...
        ├── bags
        └── calibration
            ├── 00000
            ...
            └── raw_data

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

        if not poses_file:
            poses_file = Dataset.default_poses_file

        self.seq = seq
        self.path = path
        self.poses_path = poses_path
        self.poses_file = poses_file
        self.calibration = {'K': read_intrinsics(self.intrinsics_path()),
                            'lid2cam': read_extrinsics(self.extrinsics_path()),
                            'dist_coeff': np.array([-0.134313, -0.025905, 0.002181, 0.00084, 0])}

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
        ids = np.sort(ids).tolist()
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

        ds = Dataset(seq=self.seq)
        ds.path = self.path
        ds.poses_file = self.poses_file
        ds.poses_path = self.poses_path
        ds.poses = self.poses
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
        return read_image(self.image_path(self.ids_rgb[i]))

    def camera_semseg(self, id):
        assert id in self.ids  # these are lidar ids
        t = float(id.split('-')[1].split('_')[0]) + float(id.split('-')[1].split('_')[1]) / 1000.0
        i = np.searchsorted(self.ts_semseg, t)
        i = np.clip(i, 0, len(self.ids_semseg) - 1)
        return read_semseg(self.semseg_path(self.ids_semseg[i]))


def lidar_map_demo():
    from tqdm import tqdm

    for name in seq_names:
        ds = Dataset(seq='rellis_3d/%s' % name)

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


def cam2lidar_demo():
    seq = np.random.choice(seq_names)
    ds = Dataset(seq='rellis_3d/%s' % seq)

    dist_coeff = ds.calibration['dist_coeff'].reshape((5, 1))
    K = ds.calibration['K']
    T_lid2cam = ds.calibration['lid2cam']

    for _ in range(5):
        data = ds[int(np.random.choice(range(len(ds))))]
        points, pose, rgb, semseg = data
        points = structured_to_unstructured(points[['x', 'y', 'z']])

        img_height, img_width, channels = rgb.shape

        R_lidar2cam = T_lid2cam[:3, :3]
        t_lidar2cam = T_lid2cam[:3, 3]
        rvec, _ = cv2.Rodrigues(R_lidar2cam)
        tvec = t_lidar2cam.reshape(3, 1)
        xyz_v, color = points_filter(points, img_width, img_height, K, T_lid2cam)

        imgpoints, _ = cv2.projectPoints(xyz_v[:, :], rvec, tvec, K, dist_coeff)
        imgpoints = np.squeeze(imgpoints, 1)
        imgpoints = imgpoints.T
        res = print_projection_plt(points=imgpoints, color=color, image=rgb)

        plt.figure(figsize=(20, 20))
        plt.title("Ouster points to camera image Result")
        plt.imshow(res)
        plt.show()


def semseg_demo():
    seq = np.random.choice(seq_names)
    ds = Dataset(seq='rellis_3d/%s' % seq)

    for _ in range(5):
        id = int(np.random.choice(range(len(ds))))
        print('Data index:', id)

        data = ds[id]
        _, _, rgb, semseg = data

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(rgb[..., (2, 1, 0)] / 255)
        plt.subplot(1, 2, 2)
        plt.imshow(semseg / 255)
        plt.show()


def main():
    lidar_map_demo()
    cam2lidar_demo()
    semseg_demo()


if __name__ == '__main__':
    main()
