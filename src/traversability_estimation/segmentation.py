"""Segmentation of points into geometric primitives (planes, etc.)."""
from matplotlib import cm
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import open3d as o3d
import torch


def map_colors(values, colormap=cm.gist_rainbow, min_value=None, max_value=None):
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values)
    # if not isinstance(colormap, torch.Tensor):
    #     colormap = torch.tensor(colormap, dtype=torch.float64)
    # assert colormap.shape[1] == (2, 3)
    # assert callable(colormap)
    assert callable(colormap) or isinstance(colormap, torch.Tensor)
    if min_value is None:
        min_value = values.min()
    if max_value is None:
        max_value = values.max()
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


def cluster_open3d(x, eps, min_points=10):
    assert isinstance(x, np.ndarray)
    assert x.shape[1] == 3
    assert eps >= 0.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(x)
    # NB: High min_points value causes finding no points, even if clusters
    # with enough support are found when using lower min_points value.
    # clustering = pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True)
    clustering = pcd.cluster_dbscan(eps=eps, min_points=10, print_progress=False)
    # Invalid labels < 0.
    clustering = np.asarray(clustering)
    return clustering


def keep_mask(n, indices):
    mask = np.zeros(n, dtype=bool)
    mask[indices] = 1
    return mask


def remove_mask(n, indices):
    mask = np.ones(n, dtype=bool)
    mask[indices] = 0
    return mask


def fit_models_iteratively(x, fit_model, min_support=3, max_models=10, cluster_eps=None, verbose=0, visualize=False):
    """Fit multiple models iteratively.

    @param x: Input point cloud.
    @param fit_model: Function that fits a model to a point cloud and returns the model and the inlier indices.
    @param min_support: Minimum number of inliers required for a model to be considered valid.
    @param max_models: Maximum number of models to fit.
    @param cluster_eps: If not None, cluster the inliers of each model and keep only the largest cluster.
    @param verbose: Verbosity level.
    @return: List of models and inlier indices.
    """
    assert isinstance(x, np.ndarray)
    assert x.shape[1] == 3
    remaining = x
    indices = np.arange(len(remaining))  # Input point indices of remaining point cloud.
    models = []
    labels = np.full(len(remaining), -1, dtype=int)
    label = 0
    while True:
        model, support_tmp = fit_model(remaining)

        support_tmp = np.asarray(support_tmp)
        if verbose >= 2:
            print('Found model %i (%s) supported by %i / %i (%i) points.'
                  % (label, model, len(support_tmp). len(remaining), len(x)))

        if len(support_tmp) < min_support:
            if verbose >= 0:
                print('Halt due to insufficient model support.')
            break

        # Extract the largest contiguous cluster and keep the rest for next iteration.
        if cluster_eps:
            clustering = cluster_open3d(remaining[support_tmp], cluster_eps, min_points=min_support)
            clusters, counts = np.unique(clustering[clustering >= 0], return_counts=True)
            if len(counts) == 0 or counts.max() < min_support:
                # Remove all points if there is no cluster with sufficient support.
                mask = remove_mask(len(remaining), support_tmp)
                remaining = remaining[mask]
                indices = indices[mask]
                if verbose >= 2:
                    print('No cluster from model %i has sufficient support (largest %i < %i).'
                          % (label, counts.max() if len(counts) else 0, min_support))
                if len(remaining) < min_support:
                    if verbose >= 1:
                        print('Not enough points to continue.')
                    break
                continue
            i_max = counts.argmax()
            assert counts[i_max] >= min_support
            largest = clusters[i_max]
            support_tmp = support_tmp[clustering == largest]
            if verbose >= 1:
                print('Kept largest cluster from model %i %s supported by %i / %i (%i) points.'
                      % (label, model, len(support_tmp), len(remaining), len(x)))

        support = indices[support_tmp]
        models.append((model, support))
        labels[support] = label

        if len(models) == max_models:
            if verbose >= 1:
                print('Target number of models found.')
            break

        mask = remove_mask(len(remaining), support_tmp)
        remaining = remaining[mask]
        indices = indices[mask]
        if len(remaining) < min_support:
            if verbose >= 1:
                print('Not enough points to continue.')
            break
        label += 1

    print('%i models (highest label %i) with minimum support of %i points were found.'
          % (len(models), labels.max(), min_support))

    if visualize:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(x)
        num_primitives = len(models)
        num_points = len(x)
        labels = np.full(num_points, -1, dtype=int)
        for i in range(num_primitives):
            labels[models[i][1]] = i
        max_label = num_primitives - 1
        colors = np.zeros((num_points, 3), dtype=np.float32)
        segmented = labels >= 0
        colors[segmented] = map_colors(labels[segmented], colormap=cm.viridis, min_value=0.0, max_value=max_label)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])

    return models


def fit_cylinder_pcl(x, distance_threshold, radius_limits, max_iterations=1000):
    import pcl
    assert isinstance(x, np.ndarray)
    assert x.shape[1] == 3
    assert distance_threshold >= 0.0
    assert max_iterations > 0
    cld = pcl.PointCloud(x.astype(np.float32))
    seg = cld.make_segmenter()
    # seg = cld.make_segmenter_normals(ksearch=9, searchRadius=-1.0)
    # seg = cld.make_segmenter_normals(9, -1.0)
    # seg = cld.make_segmenter_normals(int_ksearch=9)
    # seg = cld.make_segmenter_normals(double_searchRadius=0.5)
    # seg = cld.make_segmenter_normals(int_ksearch=9, double_searchRadius=-1.0)
    # seg = cld.make_segmenter_normals()
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_STICK)
    # seg.set_model_type(pcl.SACMODEL_CYLINDER)
    # seg.set_eps_angle(0.25)
    # seg.set_radius_limits(radius_limits[0], radius_limits[1])
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(distance_threshold)
    seg.set_MaxIterations(max_iterations)
    indices, model = seg.segment()
    return model, indices


def fit_cylinder_rsc(x, distance_threshold, max_iterations=1000):
    import pyransac3d as rsc
    assert isinstance(x, np.ndarray)
    assert x.shape[1] == 3
    assert distance_threshold >= 0.0
    assert max_iterations > 0
    m = rsc.Cylinder().fit(x, thresh=distance_threshold, maxIteration=max_iterations)
    model, indices = m[:-1], m[-1]
    return model, indices


def fit_cylinders(x, distance_threshold, radius_limits=None, max_iterations=1000, **kwargs):
    """Segment points into cylinders."""
    assert isinstance(x, np.ndarray)
    assert isinstance(distance_threshold, float)
    assert distance_threshold >= 0.0
    # assert isinstance(radius_limits, tuple)
    # assert len(radius_limits) == 2
    # assert radius_limits[0] >= 0.0
    # assert radius_limits[0] <= radius_limits[1]
    if x.dtype.names:
        x = structured_to_unstructured(x[['x', 'y', 'z']])
    # models = fit_models_iteratively(x, lambda x: fit_cylinder_pcl(x, distance_threshold, radius_limits), **kwargs)
    models = fit_models_iteratively(x, lambda x: fit_cylinder_rsc(x, distance_threshold, max_iterations=max_iterations),
                                    **kwargs)
    return models


def fit_plane_pcl(x, distance_threshold, max_iterations=1000):
    import pcl
    assert isinstance(x, np.ndarray)
    assert x.shape[1] == 3
    assert distance_threshold >= 0.0
    assert max_iterations > 0
    cld = pcl.PointCloud(x.astype(np.float32))
    seg = cld.make_segmenter()
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(distance_threshold)
    seg.set_MaxIterations(max_iterations)
    indices, model = seg.segment()
    return model, indices


def fit_plane_ls(x):
    assert len(x) >= 3
    # TODO: Speed up for minimal sample size.
    # if len(x) == 3:
    #     pass
    mean = np.mean(x, axis=0)
    cov = np.cov(x.T)
    # Eigenvalues in ascending order with their eigenvectors in columns.
    w, v = np.linalg.eigh(cov)
    n = v[:, 0]
    d = -np.dot(n, mean)
    return list(n) + [d]


def point_to_plane_dist(x, model):
    n = model[:3]
    d = model[3]
    dist = np.abs(np.dot(x, n) + d)
    return dist


def fit_plane(x, distance_threshold, max_iterations=1000):
    from .ransac import ransac
    from scipy.spatial import cKDTree
    assert isinstance(x, np.ndarray)
    assert x.shape[1] == 3
    assert distance_threshold >= 0.0
    assert max_iterations > 0

    # min_sample = 3

    # Speed up by constructing the model only from a local neighborhood.
    # To interface with ransac, we will use minimal sample size 1 and find the
    # other two points in the local neighborhood if necessary.
    x_all = x
    tree = cKDTree(x, leafsize=64, compact_nodes=True, balanced_tree=False)
    min_sample = 1

    def get_model(x):
        if len(x) == 1:
            # Find the two other points in the local neighborhood.
            i = tree.query_ball_point(x, 5 * distance_threshold)[0]
            # Return sample from all points if no model can be constructed from
            # the local neighborhood.
            if len(i) < 3:
                sample = np.random.choice(len(x_all), 3, replace=False)
                x = x_all[sample]
            else:
                i = np.random.choice(i, size=3, replace=False)
                x = x_all[i]
        return fit_plane_ls(x)

    def get_inliers(model, x):
        dist = point_to_plane_dist(x, model)
        dist = np.abs(dist)
        inliers = np.flatnonzero(dist < distance_threshold)
        return inliers

    model, inliers = ransac(x, min_sample, get_model, get_inliers,
                            fail_prob=0.01, max_iters=max_iterations, lo_iters=3)
    return model, inliers


def fit_planes(x, distance_threshold, max_iterations=1000, **kwargs):
    """Segment points into planes."""
    assert isinstance(x, np.ndarray)
    assert isinstance(distance_threshold, float)
    assert distance_threshold >= 0.0
    if x.dtype.names:
        x = structured_to_unstructured(x[['x', 'y', 'z']])
    # models = fit_models_iteratively(x, lambda x: fit_plane_pcl(x, distance_threshold, max_iterations=max_iterations),
    #                                 **kwargs)
    models = fit_models_iteratively(x, lambda x: fit_plane(x, distance_threshold, max_iterations=max_iterations),
                                    **kwargs)
    return models


def fit_stick_pcl(x, distance_threshold, max_iterations=1000):
    import pcl
    assert isinstance(x, np.ndarray)
    assert x.shape[1] == 3
    assert distance_threshold >= 0.0
    assert max_iterations > 0
    cld = pcl.PointCloud(x.astype(np.float32))
    seg = cld.make_segmenter()
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_STICK)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(distance_threshold)
    seg.set_MaxIterations(max_iterations)
    indices, model = seg.segment()
    return model, indices


def fit_sticks(x, distance_threshold, max_iterations=1000, **kwargs):
    """Segment points into planes."""
    assert isinstance(x, np.ndarray)
    assert isinstance(distance_threshold, float)
    assert distance_threshold >= 0.0
    if x.dtype.names:
        x = structured_to_unstructured(x[['x', 'y', 'z']])
    models = fit_models_iteratively(x, lambda x: fit_stick_pcl(x, distance_threshold, max_iterations=max_iterations),
                                    **kwargs)
    return models
