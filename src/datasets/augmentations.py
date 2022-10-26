# -*- coding:utf-8 -*-
# author: Awet H. Gebrehiwot
# --------------------------|
# from __future__ import (
#     division,
#     absolute_import,
#     with_statement,
#     print_function,
#     unicode_literals,
# )

import random
import numpy as np
import torch


# from pointnet2.data.data_utils import angle_axis

# Utilis
def angle_axis(angle, axis):
    # type: (float, np.ndarray) -> float
    r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle
    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: np.ndarray
        Axis to rotate about
    Returns
    -------
    torch.Tensor
        3x3 rotation matrix
    """
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    # yapf: disable
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                               [u[2], 0.0, -u[0]],
                               [-u[1], u[0], 0.0]])

    R = np.asarray(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u), dtype=np.float32
    )
    # yapf: enable
    return R


##################################3


def RandomFlipX(points, v):
    assert 0 <= v <= 1
    if np.random.random() < v:
        points[:, 0] *= -1
    return points


def RandomFlipY(points, v):
    assert 0 <= v <= 1
    if np.random.random() < v:
        points[:, 1] *= -1
    return points


def RandomFlipZ(points, v):
    assert 0 <= v <= 1
    if np.random.random() < v:
        points[:, 2] *= -1
    return points


def ScaleX(pts, v):  # (0 , 2)
    assert 0 <= v <= 0.5
    scaler = np.random.uniform(low=1 - v, high=1 + v)
    pts[:, 0] *= scaler
    return pts


def ScaleY(pts, v):  # (0 , 2)
    assert 0 <= v <= 0.5
    scaler = np.random.uniform(low=1 - v, high=1 + v)
    pts[:, 1] *= scaler
    return pts


def ScaleZ(pts, v):  # (0 , 2)
    assert 0 <= v <= 0.5
    scaler = np.random.uniform(low=1 - v, high=1 + v)
    pts[:, 2] *= scaler
    return pts


def Resize(pts, v):
    assert 0 <= v <= 0.5
    scaler = np.random.uniform(low=1 - v, high=1 + v)
    pts[:, 0:3] *= scaler
    return pts


def NonUniformScale(pts, v):  # Resize in [0.5 , 1.5]
    assert 0 <= v <= 0.5
    scaler = np.random.uniform(low=1 - v, high=1 + v, size=3)
    pts[:, 0:3] *= torch.from_numpy(scaler).float()
    return pts


def RotateX(points, v):  # ( 0 , 2 * pi)
    assert 0 <= v <= 2 * np.pi
    if np.random.random() > 0.5:
        v *= -1
    axis = np.array([1., 0., 0.])

    rotation_angle = np.random.uniform() * v
    rotation_matrix = angle_axis(rotation_angle, axis)

    normals = points.size(1) > 3
    if not normals:
        return np.matmul(points, rotation_matrix.T)
    else:
        pc_xyz = points[:, 0:3]
        pc_normals = points[:, 3:]
        points[:, 0:3] = np.matmul(pc_xyz, rotation_matrix.T)
        points[:, 3:] = np.matmul(pc_normals, rotation_matrix.T)

        return points


def RotateY(points, v):  # ( 0 , 2 * pi)
    assert 0 <= v <= 2 * np.pi
    if np.random.random() > 0.5:
        v *= -1
    axis = np.array([0., 1., 0.])

    rotation_angle = np.random.uniform() * v
    rotation_matrix = angle_axis(rotation_angle, axis)

    normals = points.size(1) > 3
    if not normals:
        return np.matmul(points, rotation_matrix.T)
    else:
        pc_xyz = points[:, 0:3]
        pc_normals = points[:, 3:]
        points[:, 0:3] = np.matmul(pc_xyz, rotation_matrix.T)
        points[:, 3:] = np.matmul(pc_normals, rotation_matrix.T)

        return points


def RotateZ(points, v):  # ( 0 , 2 * pi)
    assert 0 <= v <= 2 * np.pi
    if np.random.random() > 0.5:
        v *= -1
    axis = np.array([0., 0., 1.])

    rotation_angle = np.random.uniform() * v
    rotation_matrix = angle_axis(rotation_angle, axis)

    normals = points.shape[1] > 3
    if not normals:
        return np.matmul(points, rotation_matrix.T)
    else:
        pc_xyz = points[:, 0:3]
        pc_normals = points[:, 3:]
        points[:, 0:3] = np.matmul(pc_xyz, rotation_matrix.T)
        points[:, 3:] = np.matmul(pc_normals, rotation_matrix.T)

        return points


def RandomAxisRotation(points, v):
    assert 0 <= v <= 2 * np.pi
    axis = np.random.randn(3)
    axis /= np.sqrt((axis ** 2).sum())

    rotation_angle = np.random.uniform() * v
    rotation_matrix = angle_axis(rotation_angle, axis)

    normals = points.shape[1] > 3
    if not normals:
        return np.matmul(points, rotation_matrix.T)
    else:
        pc_xyz = points[:, 0:3]
        pc_normals = points[:, 3:]
        points[:, 0:3] = np.matmul(pc_xyz, rotation_matrix.T)
        points[:, 3:] = np.matmul(pc_normals, rotation_matrix.T)

        return points


def demo():
    import open3d as o3d

    cloud = np.fromfile('/home/ruslan/data/datasets/KITTI/SemanticKITTI/sequences/00/velodyne/000000.bin',
                        dtype=np.float32).reshape((-1, 4))
    xyz = cloud[:, :3]
    angle = np.random.random() * np.deg2rad(20)
    xyz_rot = RandomAxisRotation(xyz, angle)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd_rot = o3d.geometry.PointCloud()
    pcd_rot.points = o3d.utility.Vector3dVector(xyz_rot)
    o3d.visualization.draw_geometries([pcd, pcd_rot])


if __name__ == "__main__":
    demo()
