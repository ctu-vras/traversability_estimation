import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured
from PIL import Image, ImageFile
import rospy
from timeit import default_timer as timer
import torch
import yaml
try:
    import open3d as o3d
except:
    print('No open3d installed')
ImageFile.LOAD_TRUNCATED_IMAGES = True


def slots(msg):
    """Return message attributes (slots) as list."""
    return [getattr(msg, var) for var in msg.__slots__]


def correct_label(label, value_to_correct=11, value_to_assign=None):
    assert label.ndim == 2
    # value_to_correct = 11: human label
    label_corr = label.copy()

    human_mask = (label == value_to_correct)

    h, w = human_mask.shape
    # tuning the kernel size in vertical direction affects the dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(1, h // 4))

    human_mask_corr = cv2.erode(human_mask.astype('float'), None, iterations=1)
    human_mask_corr = cv2.dilate(human_mask_corr.astype('float'), kernel=kernel).astype('bool')

    masks_diff = np.logical_xor(human_mask_corr, human_mask)

    # the upper part of an image is not being corrected
    masks_diff[:h // 2, :] = False
    if value_to_assign is None:
        value_to_assign = value_to_correct
    label_corr[masks_diff] = value_to_assign

    return label_corr


def timing(f):
    def timing_wrapper(*args, **kwargs):
        t0 = timer()
        ret = f(*args, **kwargs)
        t1 = timer()
        rospy.logdebug('%s %.6f s' % (f.__name__, t1 - t0))
        return ret
    return timing_wrapper


def read_points_ply(path, dtype=np.float32):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    assert points.shape[1] == 3
    points = unstructured_to_structured(points.astype(dtype=dtype), names=['x', 'y', 'z'])
    del pcd
    return points


def read_points_bin(path, dtype=np.float32):
    xyzi = np.fromfile(path, dtype=dtype)
    xyzi = xyzi.reshape((-1, 4))
    points = unstructured_to_structured(xyzi.astype(dtype=dtype), names=['x', 'y', 'z', 'i'])
    return points


def read_points_labels(path, dtype=np.uint32):
    label = np.fromfile(path, dtype=dtype)
    label = label.reshape((-1, 1))
    # label = convert_label(label, inverse=False)
    label = unstructured_to_structured(label.astype(dtype=dtype), names=['label'])
    return label


def read_points(path, dtype=np.float32):
    # https://stackoverflow.com/questions/5899497/how-can-i-check-the-extension-of-a-file
    if path.lower().endswith('.ply'):
        points = read_points_ply(path, dtype)
    elif path.lower().endswith('.bin'):
        points = read_points_bin(path, dtype)
    else:
        raise ValueError('Cloud file must have .ply or .bin extension')
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


def read_rgb(path):
    img = Image.open(path)
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
    from scipy.spatial.transform import Rotation
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


def draw_points_on_image(points, color, image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for i in range(points.shape[1]):
        cv2.circle(hsv_image, (np.int32(points[0][i]), np.int32(points[1][i])), 4, (int(color[i]), 255, 255), -1)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


def depth_color(val, min_d=0, max_d=120):
    np.clip(val, 0, max_d, out=val)
    return (((val - min_d) / (max_d - min_d)) * 120).astype(np.uint8)


def filter_camera_points(points, img_width, img_height, K, RT, give_mask=False):
    assert points.shape[1] == 3
    RT = np.asarray(RT)
    fov_x = 2 * np.arctan2(img_width, 2 * K[0, 0]) * 180 / np.pi + 10
    fov_y = 2 * np.arctan2(img_height, 2 * K[1, 1]) * 180 / np.pi + 10
    p_l = np.ones((points.shape[0], points.shape[1] + 1))
    p_l[:, :3] = points
    p_c = np.matmul(RT, p_l.T)
    p_c = p_c.T
    x = p_c[:, 0]
    y = p_c[:, 1]
    z = p_c[:, 2]
    xangle = np.arctan2(x, z) * 180 / np.pi
    yangle = np.arctan2(y, z) * 180 / np.pi
    mask_x = (xangle > -fov_x / 2) & (xangle < fov_x / 2)
    mask_y = (yangle > -fov_y / 2) & (yangle < fov_y / 2)
    mask = mask_x & mask_y
    points_res = p_l[mask, :3]
    points_res = np.array(points_res)
    x = points_res[:, 0]
    y = points_res[:, 1]
    z = points_res[:, 2]
    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    color = depth_color(dist, 0, 10)
    if give_mask:
        return points_res, color, mask
    return points_res, color


color_palette = {
    0: {"color": [0, 0, 0], "name": "void"},
    1: {"color": [108, 64, 20], "name": "dirt"},
    3: {"color": [0, 102, 0], "name": "grass"},
    4: {"color": [0, 255, 0], "name": "tree"},
    5: {"color": [0, 153, 153], "name": "pole"},
    6: {"color": [0, 128, 255], "name": "water"},
    7: {"color": [0, 0, 255], "name": "sky"},
    8: {"color": [255, 255, 0], "name": "vehicle"},
    9: {"color": [255, 0, 127], "name": "object"},
    10: {"color": [64, 64, 64], "name": "asphalt"},
    12: {"color": [255, 0, 0], "name": "building"},
    15: {"color": [102, 0, 0], "name": "log"},
    17: {"color": [204, 153, 255], "name": "person"},
    18: {"color": [102, 0, 204], "name": "fence"},
    19: {"color": [255, 153, 204], "name": "bush"},
    23: {"color": [170, 170, 170], "name": "concrete"},
    27: {"color": [41, 121, 255], "name": "barrier"},
    31: {"color": [134, 255, 239], "name": "puddle"},
    33: {"color": [99, 66, 34], "name": "mud"},
    34: {"color": [110, 22, 138], "name": "rubble"}
}


def read_semseg(path, label_size=None):
    def convert_label(label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in color_palette.items():
                label[temp == k["color"]] = v
        else:
            label = np.zeros(temp.shape + (3,))
            for k, v in color_palette.items():
                label[temp == k, :] = v["color"]
        return label

    semseg = Image.open(path)
    if label_size is not None:
        if label_size[0] != semseg.size[0] or label_size[1] != semseg.size[1]:
            semseg = semseg.resize((label_size[1], label_size[0]), Image.NEAREST)
            semseg = np.array(semseg)[:, :, 0]
    semseg = np.array(semseg, dtype=np.uint8)
    semseg = convert_label(semseg, False)
    return np.array(semseg, dtype=np.uint8)


def normalize(x, eps=1e-6):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / np.max([(x_max - x_min), eps])
    x = x.clip(0, 1)
    return x


# helper function for data visualization
def visualize_imgs(layout='rows', figsize=(20, 10), **images):
    assert layout in ['columns', 'rows']
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=figsize)
    for i, (name, image) in enumerate(images.items()):
        if layout == 'rows':
            plt.subplot(1, n, i + 1)
        elif layout == 'columns':
            plt.subplot(n, 1, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.tight_layout()
    plt.show()


def visualize_cloud(xyz, color=None):
    assert isinstance(xyz, np.ndarray)
    assert xyz.ndim == 2
    assert xyz.shape[1] == 3  # xyz.shape == (N, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        assert color.shape == xyz.shape
        color = color / color.max()
        pcd.colors = o3d.utility.Vector3dVector(color)
    o3d.visualization.draw_geometries([pcd])


def map_colors(values, colormap=cm.gist_rainbow, min_value=None, max_value=None):
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values)
    assert callable(colormap) or isinstance(colormap, torch.Tensor)
    if min_value is None:
        min_value = values[torch.isfinite(values)].min()
    if max_value is None:
        max_value = values[torch.isfinite(values)].max()
    scale = max_value - min_value
    a = (values - min_value) / scale if scale > 0.0 else values - min_value
    if callable(colormap):
        colors = colormap(a.squeeze())[:, :3]
        return colors
    # TODO: Allow full colormap with multiple colors.
    assert isinstance(colormap, torch.Tensor)
    num_colors = colormap.shape[0]
    a = a.reshape([-1, 1])
    if num_colors == 2:
        # Interpolate the two colors.
        colors = (1 - a) * colormap[0:1] + a * colormap[1:]
    else:
        # Select closest based on scaled value.
        i = torch.round(a * (num_colors - 1))
        colors = colormap[i]
    return colors


def show_cloud(x, value=None, min=None, max=None, colormap=cm.jet):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(x)
    if value is not None:
        assert isinstance(value, np.ndarray)
        if value.ndim == 2:
            assert value.shape[1] == 3
            colors = value
        elif value.ndim == 1:
            colors = map_colors(value, colormap=colormap, min_value=min, max_value=max)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


def convert_label(label, inverse=False, label_mapping=None):
    if not label_mapping:
        label_mapping = {0: 0,
                         # 1: 0,
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
    if isinstance(label, list):
        label = np.asarray(label)
    if isinstance(label, np.ndarray):
        temp = label.copy()
    elif isinstance(label, torch.Tensor):
        temp = label.clone()
    else:
        raise ValueError('Supported types: np.ndarray, torch.Tensor')
    if inverse:
        for v, k in label_mapping.items():
            temp[label == k] = v
    else:
        for k, v in label_mapping.items():
            temp[label == k] = v
    return temp


def convert_color(label, color_map):
    if isinstance(label, np.ndarray):
        temp = np.zeros(label.shape + (3,)).astype(np.uint8)
    elif isinstance(label, torch.Tensor):
        temp = torch.zeros(label.shape + (3,), dtype=torch.uint8).to(label.device)
    else:
        raise ValueError('Supported types: np.ndarray, torch.Tensor')
    for k, v in color_map.items():
        if isinstance(label, np.ndarray):
            v = np.asarray(v).astype(np.uint8)
        elif isinstance(label, torch.Tensor):
            v = torch.as_tensor(v, dtype=torch.uint8).to(label.device)
        temp[label == k] = v
    return temp


def get_label_map(path):
    label_map = yaml.safe_load(open(path, 'r'))
    assert isinstance(label_map, (dict, list))
    if isinstance(label_map, dict):
        label_map_dict = dict((int(k), int(v)) for k, v in label_map.items())
        n = max(label_map_dict) + 1
        label_map = np.zeros((n,), dtype=np.uint8)
        for k, v in label_map_dict.items():
            label_map[k] = v
    elif isinstance(label_map, list):
        label_map = np.asarray(label_map)
    return label_map
