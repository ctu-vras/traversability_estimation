import os
import numpy as np
from torch.utils.data import Dataset
from numpy.lib.recfunctions import structured_to_unstructured
from matplotlib import cm
from .augmentations import horizontal_shift
from ..utils import show_cloud


IGNORE_LABEL = 255
data_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))


class TraversabilityDataset(Dataset):
    """
    Class to wrap semi-supervised traversability data generated using lidar odometry and IMU.
    Please, have a look at the `generate_traversability_data` script for data generation from bag file.
    """

    def __init__(self, path, cloud_topic='points', split='val'):
        super(Dataset, self).__init__()
        assert split in ['train', 'val']
        self.split = split
        self.path = path
        self.cloud_path = os.path.join(path, cloud_topic)
        assert os.path.exists(self.cloud_path)
        self.traj_path = os.path.join(path, 'trajectory')
        assert os.path.exists(self.traj_path)
        self.ids = [f[:-4] for f in os.listdir(self.cloud_path)]
        self.proj_fov_up = 45
        self.proj_fov_down = -45
        self.proj_H = 128
        self.proj_W = 1024
        self.ignore_label = IGNORE_LABEL

    def get_traj(self, i):
        ind = self.ids[i]
        traj = np.load(os.path.join(self.traj_path, '%s.npz' % ind))['traj']
        return traj

    def range_projection(self, points, labels):
        """ Project a point cloud into a sphere.
        """
        # laser parameters
        fov_up = self.proj_fov_up / 180.0 * np.pi  # field of view up in rad
        fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

        # get depth of all points
        depth = np.linalg.norm(points, 2, axis=1)

        # get scan components
        scan_x = points[:, 0]
        scan_y = points[:, 1]
        scan_z = points[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / (depth + 1e-8))

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.proj_W  # in [0.0, W]
        proj_y *= self.proj_H  # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]
        indices = indices[order]

        # assing to image
        proj_range = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)
        proj_range[proj_y, proj_x] = depth

        # projected index (for each pixel, what I am in the pointcloud)
        # [H,W] index (-1 is no data)
        proj_idx = np.full((self.proj_H, self.proj_W), -1, dtype=np.int32)
        proj_idx[proj_y, proj_x] = indices
        # only map colors to labels that exist
        mask = proj_idx >= 0

        # projection color with semantic labels
        proj_sem_label = np.full((self.proj_H, self.proj_W), self.ignore_label, dtype=np.float32)  # [H,W]  label
        proj_sem_label[mask] = labels[proj_idx[mask]]

        # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1, dtype=np.float32)
        proj_xyz[proj_y, proj_x] = points[order]

        return proj_range, proj_sem_label, proj_xyz

    def __getitem__(self, i, visualize=False):
        ind = self.ids[i]
        cloud = np.load(os.path.join(self.cloud_path, '%s.npz' % ind))['cloud']

        if cloud.ndim == 2:
            cloud = cloud.reshape((-1,))

        points = structured_to_unstructured(cloud[['x', 'y', 'z']])
        trav = np.asarray(cloud['traversability'], dtype=points.dtype)

        depth_proj, label_proj, points_proj = self.range_projection(points, trav)

        if self.split == 'train':
            # data augmentation: add rotation around vertical axis (Z)
            H, W = depth_proj.shape
            shift = np.random.choice(range(1, W))
            depth_proj = horizontal_shift(depth_proj, shift=shift)
            label_proj = horizontal_shift(label_proj, shift=shift)
            # point projected have shape (H, W, 3)
            points_proj_shifted = np.zeros_like(points_proj)
            points_proj_shifted[:, :shift, :] = points_proj[:, -shift:, :]
            points_proj_shifted[:, shift:, :] = points_proj[:, :-shift, :]
            points_proj = points_proj_shifted

        points_proj = points_proj.reshape((-1, 3))

        if visualize:
            valid = trav != self.ignore_label
            show_cloud(points_proj, label_proj.reshape(-1, ),
                       min=trav[valid].min(), max=trav[valid].max() + 1, colormap=cm.jet)

        return depth_proj[None], label_proj[None], points_proj

    def __len__(self):
        return len(self.ids)


def demo():
    from ..utils import show_cloud_plt
    from ..segmentation import filter_grid, filter_range
    import matplotlib.pyplot as plt

    path = '/home/ruslan/data/bags/traversability/marv/ugv_2022-08-12-15-18-34_trav/'
    assert os.path.exists(path)
    ds = TraversabilityDataset(path, cloud_topic='os_cloud_node/destaggered_points')
    print(len(ds))

    i = np.random.choice(range(len(ds)))
    sample = ds[i]
    depth, label, points = sample
    traj = ds.get_traj(i)

    points = points.squeeze()
    traj = traj.squeeze()

    points = filter_range(points, 0.5, 10.0)
    points = filter_grid(points, 0.1)
    show_cloud_plt(points, markersize=0.2)
    plt.plot(traj[:, 0, 3], traj[:, 1, 3], traj[:, 2, 3], 'ro', markersize=1)
    plt.show()


if __name__ == '__main__':
    demo()
