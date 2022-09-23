import os
import cv2
import numpy as np
import torch
try:
    # it's hard to install the lib on jetson
    # we are not importing it since it is not needed for inference tasks on robots
    import fiftyone as fo
except:
    print('Fiftyone lib is not installed')
from datasets.laserscan import SemLaserScan
from datasets.base_dataset import TRAVERSABILITY_LABELS, TRAVERSABILITY_COLOR_MAP, VOID_VALUE
from datasets.base_dataset import FLEXIBILITY_LABELS, FLEXIBILITY_COLOR_MAP
from datasets.base_dataset import BaseDatasetImages, BaseDatasetClouds
from numpy.lib.recfunctions import structured_to_unstructured
from PIL import Image

data_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))


class TraversabilityImages51(torch.utils.data.Dataset):
    CLASSES = ["traversable", "non-traversable", "background"]

    def __init__(self, crop_size=(1200, 1920), path=None, split=None):
        self.crop_size = (crop_size[1], crop_size[0])
        if not path:
            path = os.path.join(data_dir, 'TraversabilityDataset', 'supervised')
        self.path = path
        self.samples = self._load_dataset(self.path)
        self.files = self.samples.values("filepath")
        # self.mask_targets = TRAVERSABILITY_LABELS
        self.mask_targets = {1: "traversable", 2: "non-traversable", 0: "background"}
        # self.class_values = np.sort([k for k in TRAVERSABILITY_LABELS.keys()])  # [0, 1, 255]
        self.class_values = [0, 1, 2]
        # self.color_map = TRAVERSABILITY_COLOR_MAP
        self.color_map = {1: [0, 255, 0], 2: [255, 0, 0], 0: [0, 0, 0]}

        self.mean = np.array([123.11457109, 126.84649579, 124.37909438])
        self.std = np.array([47.46125817, 47.14161698, 47.70375418])
        self.split = split

    def read_img(self, path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        return image

    def __getitem__(self, idx):
        img_path = self.files[idx]
        sample = self.samples[img_path]

        # image preprocessing
        image = self.read_img(img_path)
        image = cv2.resize(image, self.crop_size, interpolation=cv2.INTER_LINEAR)
        image = self._input_transform(image)

        if self.split != 'test':
            image = image.transpose((2, 0, 1))

        # mask preprocessing
        segmentation = sample.polylines.to_segmentation(frame_size=(1920, 1200), mask_targets=self.mask_targets)
        mask = segmentation["mask"]
        sample["ground_truth"] = segmentation
        sample.save()
        mask = cv2.resize(mask, self.crop_size, interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.long)

        # extract certain classes from mask
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=0).astype('float')

        return image, mask

    def __len__(self):
        return len(self.files)

    def _input_transform(self, image):
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    @staticmethod
    def _load_dataset(dataset_path):
        name = "TraversabilityDataset"
        if fo.dataset_exists(name):
            dataset = fo.load_dataset(name)
            dataset.delete()
        dataset = fo.Dataset.from_dir(dataset_dir=dataset_path,
                                      dataset_type=fo.types.CVATImageDataset,
                                      name=name,
                                      data_path="images",
                                      labels_path="images/annotations.xml")
        return dataset

    def show_dataset(self):
        session = fo.launch_app(self.samples)
        session.wait()

    def save_prediction(self, mask, i):
        sample = self.samples[self.files[i]]
        mask = mask.astype(np.uint8)
        sample["prediction"] = fo.Segmentation(mask=mask)
        sample.save()

    def get_mask(self, image_name):
        assert isinstance(image_name, str)
        for path in self.files:
            if image_name in path:
                sample = self.samples[path]
                segmentation = sample.polylines.to_segmentation(frame_size=(1920, 1200), mask_targets=self.mask_targets)
                mask = segmentation["mask"]
                return mask

    def calculate_mean_and_std(self):
        mean = np.zeros(3)
        std = np.zeros(3)
        for i in range(len(self.img_paths)):
            img_path = self.img_paths[i]
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)[:, :, ::-1]
            mean += image.mean(axis=(0, 1))
            std += image.std(axis=(0, 1))
        mean /= len(self.img_paths)
        std /= len(self.img_paths)
        self.mean = mean
        self.std = std
        print(mean)
        print(std)
        return mean, std


class TraversabilityImages(BaseDatasetImages):
    CLASSES = ["traversable", "non-traversable", "background"]

    def __init__(self,
                 path=None,
                 split=None,
                 num_samples=None,
                 multi_scale=True,
                 flip=True,
                 ignore_label=VOID_VALUE,
                 base_size=2048,
                 crop_size=(1200, 1920),
                 downsample_rate=1,
                 scale_factor=16,
                 mean=np.array([123.11457109, 126.84649579, 124.37909438]),
                 std=np.array([47.46125817, 47.14161698, 47.70375418])):
        super(TraversabilityImages, self).__init__(ignore_label, base_size,
                                                   crop_size, downsample_rate, scale_factor, mean, std)
        if path is None:
            path = os.path.join(data_dir, 'TraversabilityDataset', 'supervised')
        assert os.path.exists(path)
        assert split in [None, 'train', 'val', 'test']
        self.path = path
        self.split = split

        self.class_values = np.sort([k for k in TRAVERSABILITY_LABELS.keys()])  # [0, 1, 255]
        self.color_map = TRAVERSABILITY_COLOR_MAP

        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label

        self.mean = mean
        self.std = std
        self.scale_factor = scale_factor
        self.downsample_rate = 1. / downsample_rate

        self.multi_scale = multi_scale
        self.flip = flip
        self.rng = np.random.default_rng(42)

        self.files = self.read_files()
        self.generate_split(train_ratio=0.8)
        if num_samples:
            self.files = self.files[:num_samples]

    def read_files(self):
        path = os.path.join(self.path, 'images/')
        assert os.path.exists(path)

        files = []
        rgb_files = [os.path.join(path, 'rgb', f) for f in os.listdir(os.path.join(path, 'rgb'))]
        for f in rgb_files:
            files.append(
                {
                    'img': f,
                    'label': f.replace('rgb', 'label_id').replace('.jpg', '.png')
                }
            )
        return files

    def generate_split(self, train_ratio=0.8):
        all_files = self.files.copy()
        if self.split == 'train':
            train_files = self.rng.choice(all_files, size=round(train_ratio * len(all_files)), replace=False).tolist()
            files = train_files
        elif self.split in ['val', 'test']:
            train_files = self.rng.choice(all_files, size=round(train_ratio * len(all_files)), replace=False).tolist()
            val_files = all_files.copy()
            for x in train_files:
                if x in val_files:
                    val_files.remove(x)
            files = val_files
        else:
            files = all_files
        self.files = files
        return files

    def __getitem__(self, index):
        item = self.files[index]
        image = cv2.imread(item["img"], cv2.IMREAD_COLOR)

        mask = np.array(Image.open(item["label"]))

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


class FlexibilityClouds(BaseDatasetClouds):
    """
    FlexibilityClouds dataset, where the traversable area is generated using real robot poses.
    For each point cloud frame, points which were traversed by robot's path within constant
    amount of time are labelled as 'traversable'.
    The rest of the points are marked as 'background',
    'non-traversable' category is added using local geometry estimate
    """
    CLASSES = ["background", "traversable", "non-traversable"]

    def __init__(self,
                 sequences=None,
                 path=None,
                 num_samples=None,
                 labels_mode='labels',
                 split=None,
                 fields=None,
                 lidar_beams_step=1,
                 labels_mapping='flexibility'
                 ):
        super(FlexibilityClouds, self).__init__(path=path, fields=fields,
                                                depth_img_H=128, depth_img_W=1024,
                                                lidar_fov_up=45.0, lidar_fov_down=-45.0,
                                                lidar_beams_step=lidar_beams_step,
                                                labels_mapping=labels_mapping
                                                )
        if sequences is None:
            sequences = ['ugv_2022-08-12-15-30-22',
                         'ugv_2022-08-12-15-18-34',
                         'ugv_2022-08-12-16-08-17',
                         'ugv_2022-08-12-16-37-03'
                         ]
        sequences = [s + '_z_support' for s in sequences]

        if fields is None:
            fields = ['depth']
        if path is None:
            path = os.path.join(data_dir, 'TraversabilityDataset', 'self_supervised')
        assert os.path.exists(path)
        self.seq_paths = [os.path.join(path, 'clouds', sequence, 'os_cloud_node') for sequence in sequences]
        self.rng = np.random.default_rng(42)
        self.class_values = [0, 1, 255]

        self.mask_targets = {value: key for key, value in FLEXIBILITY_LABELS.items()}
        self.color_map = FLEXIBILITY_COLOR_MAP

        assert split in [None, 'train', 'val', 'test']
        self.split = split

        self.fields = fields
        self.lidar_beams_step = lidar_beams_step
        assert self.labels_mapping == 'flexibility'

        assert labels_mode in ['masks', 'labels']
        self.labels_mode = labels_mode

        self.get_scan()

        self.files = self.read_files()
        self.generate_split(train_ratio=0.8)
        if num_samples:
            self.files = self.files[:num_samples]

    def read_files(self):
        files = []
        for path in self.seq_paths:
            assert os.path.exists(path)
            pts_files = [os.path.join(path, 'destaggered_points', f)
                         for f in os.listdir(os.path.join(path, 'destaggered_points'))]
            for f in pts_files:
                label_f = f.replace('destaggered_points', 'label_id')
                files.append(
                    {
                        'pts': f,
                        'label': label_f if os.path.exists(label_f) else None
                    }
                )
        return files

    def read_cloud(self, path):
        cloud = np.load(path, allow_pickle=True)['arr_0'].item()['cloud']
        return cloud

    def __getitem__(self, index):
        cloud_path = self.files[index]['pts']
        cloud = self.read_cloud(cloud_path)

        xyz = structured_to_unstructured(cloud[['x', 'y', 'z']])
        traversability = cloud['flexible'].copy()

        self.scan.set_points(points=xyz)
        bg_value = self.mask_targets["background"]
        self.scan.proj_sem_label = np.full((self.scan.proj_H, self.scan.proj_W), bg_value,
                                           dtype=np.uint8)  # [H,W]  label
        self.scan.set_label(label=traversability)

        label = self.scan.proj_sem_label
        assert set(np.unique(label)) <= set(self.class_values)

        data, label = self.create_sample(label=label)

        return data, label


class TraversabilityClouds(BaseDatasetClouds):

    CLASSES = ["background", "traversable", "non-traversable"]

    def __init__(self,
                 path=None,
                 num_samples=None,
                 labels_mode='labels',
                 split=None,
                 fields=None,
                 lidar_beams_step=1,
                 annotation_from_img=False,
                 labels_mapping='traversability',
                 ):
        super(TraversabilityClouds, self).__init__(path=path, fields=fields,
                                                   depth_img_H=128, depth_img_W=1024,
                                                   lidar_fov_up=45.0, lidar_fov_down=-45.0,
                                                   lidar_beams_step=lidar_beams_step,
                                                   labels_mapping=labels_mapping
                                                   )
        if fields is None:
            fields = ['depth']
        if path is None:
            path = os.path.join(data_dir, 'TraversabilityDataset', 'supervised')
        assert os.path.exists(path)
        self.path = os.path.join(path, 'clouds')

        self.class_values = [0, 1, 255]
        self.mask_targets = {value: key for key, value in TRAVERSABILITY_LABELS.items()}
        self.color_map = TRAVERSABILITY_COLOR_MAP

        assert split in [None, 'train', 'val', 'test']
        self.split = split

        assert self.labels_mapping == 'traversability'

        # whether to use semantic labels from annotated images
        self.annotation_from_img = annotation_from_img

        self.fields = fields
        self.lidar_beams_step = lidar_beams_step

        self.get_scan()

        assert labels_mode in ['masks', 'labels']
        self.labels_mode = labels_mode

        self.files = self.read_files()
        self.generate_split(train_ratio=0.8)
        if num_samples:
            self.files = self.files[:num_samples]

    def read_files(self):
        path = os.path.join(self.path)
        assert os.path.exists(path)

        files = []
        labels_folder = 'label_id_from_img' if self.annotation_from_img else 'label_id'
        points_folder = 'destaggered_points_from_img' if self.annotation_from_img else 'destaggered_points'

        labels_files = [os.path.join(path, labels_folder, f)
                        for f in os.listdir(os.path.join(path, labels_folder))]
        for label_f in labels_files:
            pts_f = label_f.replace(labels_folder, points_folder)
            files.append(
                {
                    'pts': pts_f,
                    'label': label_f
                }
            )
        return files

    def read_cloud(self, path):
        cloud = np.load(path)['arr_0']
        return cloud

    def __getitem__(self, index):
        cloud = self.read_cloud(self.files[index]['pts'])
        label = self.read_cloud(self.files[index]['label'])
        assert self.files[index]['pts'].split('/')[-1] == self.files[index]['label'].split('/')[-1]

        if cloud.dtype.names is not None:
            cloud = structured_to_unstructured(cloud[['x', 'y', 'z', 'intensity']])

        assert cloud.shape[-1] == 4 or cloud.shape[-1] == 3
        cloud = cloud.reshape((-1, cloud.shape[-1]))
        label = label.reshape(-1)

        xyz = cloud[..., :3]
        # intensity = cloud[..., 3] if cloud.shape[-1] == 4 else None

        self.scan.set_points(points=xyz)
        # self.scan.set_points(points=xyz, remissions=intensity)
        bg_value = VOID_VALUE
        self.scan.proj_sem_label = np.full((self.scan.proj_H, self.scan.proj_W), bg_value,
                                           dtype=np.uint8)  # [H,W]  label
        self.scan.set_label(label=label)

        label = self.scan.proj_sem_label
        assert set(np.unique(label)) <= set(self.class_values)

        data, label = self.create_sample(label=label)

        return data, label


def images_save_labels():
    import matplotlib.pyplot as plt
    from traversability_estimation.utils import convert_label, convert_color
    from tqdm import tqdm

    label_mapping = {0: 255,
                     1: 0,
                     2: 1}

    name = "TraversabilityDataset"
    directory = os.path.join(data_dir, name, 'supervised')

    ds = TraversabilityImages51(crop_size=(1200, 1920), path=directory, split='val')
    # ds.show_dataset()

    for i in tqdm(range(len(ds))):
        img, mask = ds[i]

        label = mask.argmax(axis=0)
        label = convert_label(label, label_mapping=label_mapping, inverse=False)

        color = convert_color(label, color_map=TRAVERSABILITY_COLOR_MAP)

        if not os.path.exists(os.path.join(ds.path, 'images', 'label_id')):
            os.mkdir(os.path.join(ds.path, 'images', 'label_id'))
        if not os.path.exists(os.path.join(ds.path, 'images', 'label_color')):
            os.mkdir(os.path.join(ds.path, 'images', 'label_color'))

        cv2.imwrite(ds.files[i].replace('rgb', 'label_id').replace('.jpg', '.png'), label)
        cv2.imwrite(ds.files[i].replace('rgb', 'label_color').replace('.jpg', '.png'), color[..., (2, 1, 0)])

        # img_vis = img.transpose((1, 2, 0)) * ds.std + ds.mean
        # plt.figure(figsize=(20, 10))
        # plt.subplot(1, 2, 1)
        # plt.imshow(img_vis)
        # plt.subplot(1, 2, 2)
        # plt.imshow(color)
        # plt.tight_layout()
        # plt.show()


def flexibility_demo(run_times=1):
    from traversability_estimation.utils import visualize_imgs
    import open3d as o3d

    ds = FlexibilityClouds(split='test')

    for _ in range(run_times):
        depth_img, label = ds[np.random.choice(len(ds))]

        power = 16
        depth_img_vis = np.copy(depth_img).squeeze()  # depth
        depth_img_vis[depth_img_vis > 0] = depth_img_vis[depth_img_vis > 0] ** (1 / power)
        depth_img_vis[depth_img_vis > 0] = (depth_img_vis[depth_img_vis > 0] - depth_img_vis[depth_img_vis > 0].min()) / \
                                           (depth_img_vis[depth_img_vis > 0].max() - depth_img_vis[
                                               depth_img_vis > 0].min())
        depth_img_vis[depth_img_vis < 0] = 0.5
        assert depth_img_vis.min() >= 0.0 and depth_img_vis.max() <= 1.0

        label_flex = label == ds.mask_targets['flexible']
        label_non_flex = label == ds.mask_targets['non-flexible']

        depth_img_with_flex_points = (0.3 * depth_img_vis + 0.7 * label_flex).astype("float")
        depth_img_with_non_flex_points = (0.3 * depth_img_vis + 0.7 * label_non_flex).astype("float")

        xyz = ds.scan.proj_xyz
        color = ds.label_to_color(label.astype('uint8'))

        visualize_imgs(layout='columns',
                       depth_img_with_flex_points=depth_img_with_flex_points,
                       depth_img_with_non_flex_points=depth_img_with_non_flex_points,
                       flexibility=color)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.reshape((-1, 3)))
        pcd.colors = o3d.utility.Vector3dVector(color.reshape((-1, 3)))
        o3d.visualization.draw_geometries([pcd])


def traversability_demo(num_runs=1):
    from traversability_estimation.utils import visualize_imgs
    import open3d as o3d

    ds = TraversabilityClouds(split=None, annotation_from_img=False)
    # ds = TraversabilityClouds(split=None, annotation_from_img=True)

    for i in range(num_runs):

        depth_img, label = ds[np.random.choice(len(ds))]

        power = 16
        depth_img_vis = np.copy(depth_img).squeeze()  # depth
        label = label.squeeze()

        depth_img_vis[depth_img_vis > 0] = depth_img_vis[depth_img_vis > 0] ** (1 / power)
        depth_img_vis[depth_img_vis > 0] = (depth_img_vis[depth_img_vis > 0] - depth_img_vis[depth_img_vis > 0].min()) / \
                                           (depth_img_vis[depth_img_vis > 0].max() - depth_img_vis[
                                               depth_img_vis > 0].min())
        depth_img_vis[depth_img_vis < 0] = 0.5
        assert depth_img_vis.min() >= 0.0 and depth_img_vis.max() <= 1.0

        label_trav = label == 0
        label_non_trav = label == 1

        traversable_area = (0.3 * depth_img_vis + 0.7 * label_trav).astype("float32")
        non_traversable_area = (0.3 * depth_img_vis + 0.7 * label_non_trav).astype("float32")

        xyz = ds.scan.proj_xyz
        color = ds.label_to_color(label.astype('uint8'))

        visualize_imgs(layout='columns',
                       depth_img_with_trav_points=traversable_area,
                       depth_img_with_non_trav_points=non_traversable_area,
                       traversability=color)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.reshape((-1, 3)))
        pcd.colors = o3d.utility.Vector3dVector(color.reshape((-1, 3)))
        o3d.visualization.draw_geometries([pcd])


def label_cloud_from_img(visualize=False, save_clouds=True, n_runs=1):
    import yaml
    import open3d as o3d
    import matplotlib.pyplot as plt
    from traversability_estimation.utils import filter_camera_points, convert_color, draw_points_on_image, convert_label
    from scipy.spatial.transform import Rotation
    from tqdm import tqdm

    ds_img = TraversabilityImages51()
    ds_depth = TraversabilityClouds()

    # TODO: check the projection of points to images, maybe calibration error?
    # get camera intrinsics
    intrinsics = {"camera_front": [1179.41625, 0.0, 983.9155, 0.0, 1178.90431, 596.74537, 0.0, 0.0, 1.0],
                  "camera_left": [1180.22991, 0.0, 946.81284, 0.0, 1180.78981, 568.02642, 0.0, 0.0, 1.0],
                  "camera_right": [1176.57392, 0.0, 963.54103, 0.0, 1176.1531, 600.43789, 0.0, 0.0, 1.0]}

    # get extrinsics: T_lid2cam
    extrinsics = {"camera_front": {"trans": [-0.000, -0.180, -0.147], "quat": [0.653, -0.271, 0.271, 0.653]},
                  "camera_right": {"trans": [-0.000, -0.180, -0.087], "quat": [0.271, -0.653, 0.653, 0.271]},
                  "camera_left": {"trans": [-0.000, -0.180, -0.087], "quat": [0.653, 0.271, -0.271, 0.653]}}

    # distortion coeffs for cameras
    distortion = {"camera_front": [-0.226817, 0.071594, -7e-05, 6.9e-05, 0.0],
                  "camera_left": [-0.228789, 0.071791, 0.000209, -0.000356, 0.0],
                  "camera_right": [-0.223299, 0.068371, 0.000216, -0.000206, 0.0]}

    file_to_bag = yaml.safe_load(open(os.path.join(data_dir, 'TraversabilityDataset',
                                                   'supervised', 'correspondencies.yaml'), 'r'))
    label_mapping = {0: 255,
                     1: 0,
                     2: 1}

    camera_frames = []
    img_files = []
    cloud_files = []
    for file in ds_depth.files:
        cloud_file = file['pts']
        img_file = cloud_file.replace('.npz', '.jpg').replace('clouds', 'images').replace('destaggered_points', 'rgb')

        for bag_file in ['ugv_2022-08-12-15-18-34.bag', 'ugv_2022-08-12-15-30-22.bag']:
            imgs_in_bag = file_to_bag[bag_file]

            for camera_frame in imgs_in_bag.keys():
                if img_file.split('/')[-1] in imgs_in_bag[camera_frame]:
                    img_files.append(img_file)
                    camera_frames.append(camera_frame)
                    cloud_files.append(cloud_file)

    assert len(img_files) == len(camera_frames) == len(ds_depth) == len(cloud_files)

    if not os.path.exists(os.path.join(ds_depth.path, 'label_color')):
        os.mkdir(os.path.join(ds_depth.path, 'label_color'))
    if not os.path.exists(os.path.join(ds_depth.path, 'label_id')):
        os.mkdir(os.path.join(ds_depth.path, 'label_id'))

    # for i in tqdm(range(len(img_files))):
    # np.random.seed(42)
    for i in np.random.choice(range(len(img_files)), n_runs):

        img = cv2.imread(img_files[i])
        img_label = ds_img.get_mask(img_files[i])

        cloud = ds_depth.read_cloud(cloud_files[i])
        if cloud.dtype.names is not None:
            cloud = structured_to_unstructured(cloud[['x', 'y', 'z', 'intensity']])
        cloud = cloud.reshape((-1, 4))
        points = cloud[..., :3]

        """
        get lidar points which are in camera FoV
        """
        camera_frame = camera_frames[i]

        T_lid2cam = np.eye(4)
        T_lid2cam[:3, :3] = Rotation.from_quat(extrinsics[camera_frame]['quat']).as_matrix()
        T_lid2cam[:3, 3] = np.asarray(extrinsics[camera_frame]['trans'])

        # for cf in ['camera_right', 'camera_left', 'camera_front']:
        #     T_lid2cam = np.eye(4)
        #     T_lid2cam[:3, :3] = Rotation.from_quat(extrinsics[cf]['quat']).as_matrix()
        #     T_lid2cam[:3, 3] = np.asarray(extrinsics[cf]['trans'])
        #
        #     print('\nFrame: %s transformation:\n%s\n' % (cf, T_lid2cam))

        R_lid2cam, t_lid2cam = T_lid2cam[:3, :3], T_lid2cam[:3, 3]

        img_height, img_width = img.shape[:2]
        K = np.asarray(intrinsics[camera_frame]).reshape((3, 3))
        dist_coeff = np.asarray(distortion[camera_frame]).reshape((5, 1))

        rvec, _ = cv2.Rodrigues(R_lid2cam)
        tvec = t_lid2cam.reshape((3, 1))
        points_fov, color, camera_pts_mask = filter_camera_points(points[..., :3], img_width, img_height, K, T_lid2cam,
                                                                  give_mask=True)
        camera_pts_ids = [i for i, v in enumerate(camera_pts_mask) if v == True]

        img_points, _ = cv2.projectPoints(points_fov, rvec, tvec, K, dist_coeff)
        img_points = np.squeeze(img_points, 1)
        img_points = img_points.T

        """
        colorize these points, the rest points are without label (background)
        """
        depth_label = np.zeros_like(camera_pts_mask, dtype=np.int32)
        assert img_points.shape[0] == 2
        for pt_id, pt_pxl in enumerate(img_points.T):
            assert pt_pxl.shape == (2,)
            w, h = np.int32(pt_pxl)
            if 0 <= w < img_width and 0 <= h < img_height:
                l = img_label[h, w]
                label_id = camera_pts_ids[pt_id]
                assert label_id < len(depth_label)
                depth_label[label_id] = int(l)

        depth_label = convert_label(depth_label, label_mapping=label_mapping, inverse=False)
        depth_color = convert_color(depth_label, ds_depth.color_map)
        semseg_mask = convert_color(img_label, ds_img.color_map)
        img_with_pts = draw_points_on_image(points=img_points, color=color, image=img)

        if visualize:
            plt.figure(figsize=(20, 10))
            plt.subplot(1, 3, 1)
            plt.imshow(img[..., (2, 1, 0)])
            plt.title('Frame: %s' % camera_frame)
            plt.subplot(1, 3, 2)
            plt.imshow(img_with_pts[..., (2, 1, 0)])
            plt.subplot(1, 3, 3)
            plt.tight_layout()
            plt.imshow(semseg_mask)

            plt.figure(figsize=(20, 10))
            plt.subplot(2, 1, 1)
            plt.imshow(depth_color.reshape((ds_depth.depth_img_H, ds_depth.depth_img_W, 3)))
            plt.subplot(2, 1, 2)
            plt.imshow(depth_label.reshape((ds_depth.depth_img_H, ds_depth.depth_img_W)))
            plt.tight_layout()
            plt.show()

            assert depth_color.shape == points.shape
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(depth_color)

            pcd_cam = o3d.geometry.PointCloud()
            pcd_cam.points = o3d.utility.Vector3dVector(points[camera_pts_ids] + np.asarray([50, 0, 0]))
            o3d.visualization.draw_geometries([pcd, pcd_cam])

        if save_clouds:
            """
            save labelled point clouds (add them to TraversabilityDataset)
            """
            np.savez(cloud_files[i].replace('destaggered_points', 'label_id'), depth_label)
            np.savez(cloud_files[i].replace('destaggered_points', 'label_color'), depth_color)


def images_demo(num_runs=1):
    from traversability_estimation.utils import visualize_imgs, convert_color

    ds = TraversabilityImages(split='val')

    for _ in range(num_runs):
        i = np.random.choice(range(len(ds)))
        img, label = ds[i]

        img_vis = img.transpose((1, 2, 0)) * ds.std + ds.mean
        label = label.argmax(axis=0)
        mask = convert_color(label, ds.color_map)

        visualize_imgs(img=img_vis, label=mask)


def clouds_save_labels():
    from traversability_estimation.utils import convert_label, convert_color, visualize_imgs, visualize_cloud
    from traversability_cloud import TraversabilityCloud
    from tqdm import tqdm

    label_mapping = {0: 255,
                     1: 0,
                     255: 1}

    ds = TraversabilityCloud(path=os.path.join(data_dir, "TraversabilityDataset/supervised/clouds/"
                                                         "destaggered_points_colored/"))

    if not os.path.exists(os.path.join(ds.path, '..', 'label_id')):
        os.mkdir(os.path.join(ds.path, '..', 'label_id'))
    if not os.path.exists(os.path.join(ds.path, '..', 'label_color')):
        os.mkdir(os.path.join(ds.path, '..', 'label_color'))
    if not os.path.exists(os.path.join(ds.path, '..', 'destaggered_points')):
        os.mkdir(os.path.join(ds.path, '..', 'destaggered_points'))

    for i in tqdm(range(len(ds))):
        points, label = ds[i]

        label = convert_label(label, label_mapping=label_mapping, inverse=False)
        color = convert_color(label, color_map=TRAVERSABILITY_COLOR_MAP)

        # visualize_imgs(color=color)
        # visualize_cloud(xyz=points.reshape((-1, 3)), color=color.reshape((-1, 3)))

        np.savez(ds.point_clouds[i].replace('predictions_color', 'destaggered_points').replace('.pcd', '.npz'), points)
        np.savez(ds.point_clouds[i].replace('predictions_color', 'label_id').replace('.pcd', '.npz'), label)
        np.savez(ds.point_clouds[i].replace('predictions_color', 'label_color').replace('.pcd', '.npz'), color)


def main():
    clouds_save_labels()
    # images_demo(1)
    # images_save_labels()
    # flexibility_demo(1)
    # traversability_demo(1)
    # label_cloud_from_img(visualize=True)


if __name__ == "__main__":
    main()
