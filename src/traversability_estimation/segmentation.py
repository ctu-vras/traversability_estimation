"""Segmentation of points into geometric primitives (planes, etc.)."""
from matplotlib import cm
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import open3d as o3d
import torch


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


def fit_cylinder_pcl(x, distance_threshold, radius_limits, max_iterations=1000):
    import pcl
    assert isinstance(x, np.ndarray)
    assert x.shape[1] == 3
    assert distance_threshold >= 0.0
    assert max_iterations > 0
    cld = pcl.PointCloud(x.astype(np.float32))
    seg = cld.make_segmenter()
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_CYLINDER)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(distance_threshold)
    seg.set_MaxIterations(max_iterations)
    seg.set_radius_limits(radius_limits[0], radius_limits[1])
    indices, model = seg.segment()
    return model, indices


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


def fit_planes_iter(x, distance_threshold, min_support=3, max_iterations=1000, max_models=10, eps=None,
                    visualize_progress=False, verbose=0):
    assert isinstance(x, np.ndarray)
    assert x.shape[1] == 3
    assert distance_threshold >= 0.0
    remaining = x
    indices = np.arange(len(remaining))  # Input point indices of remaining point cloud.
    planes = []
    labels = np.full(len(remaining), -1, dtype=int)
    label = 0
    while True:
        plane, support_tmp = fit_plane_pcl(remaining, distance_threshold, max_iterations=max_iterations)

        support_tmp = np.asarray(support_tmp)
        if verbose >= 2:
            print('Found plane %i [%.3f, %.3f, %.3f, %.3f] supported by %i / %i (%i) points.'
                  % (label, *plane, len(support_tmp). len(remaining), len(x)))

        if len(support_tmp) < min_support:
            if verbose >= 0:
                print('Halt due to insufficient plane support.')
            break

        # Extract the largest contiguous cluster and keep the rest for next iteration.
        if eps:
            clustering = cluster_open3d(remaining[support_tmp], eps, min_points=min_support)
            clusters, counts = np.unique(clustering[clustering >= 0], return_counts=True)
            if len(counts) == 0 or counts.max() < min_support:
                # Remove all points if there is no cluster with sufficient support.
                mask = remove_mask(len(remaining), support_tmp)
                remaining = remaining[mask]
                indices = indices[mask]
                if verbose >= 2:
                    print('No cluster from plane %i has sufficient support (largest %i < %i).'
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
                print('Kept largest cluster from plane %i [%.3f, %.3f, %.3f, %.3f] supported by %i / %i (%i) points.'
                      % (label, *plane, len(support_tmp), len(remaining), len(x)))

        support = indices[support_tmp]
        planes.append((plane, support))
        labels[support] = label

        # if visualize_progress:
        #     obj = Planes(torch.as_tensor([p for p, _ in planes]),
        #                  cloud=len(planes) * [x],
        #                  indices=[i for _, i in planes])
        #     obj.visualize()

        if len(planes) == max_models:
            if verbose >= 1:
                print('Target number of planes found.')
            break

        mask = remove_mask(len(remaining), support_tmp)
        remaining = remaining[mask]
        indices = indices[mask]
        if len(remaining) < min_support:
            if verbose >= 1:
                print('Not enough points to continue.')
            break
        label += 1

    print('%i planes (highest label %i) with minimum support of %i points were found.'
          % (len(planes), labels.max(), min_support))

    # planes = Planes(torch.as_tensor(np.concatenate([p for p, _ in planes])),
    #                 cloud=len(planes) * [x],
    #                 indices=[i for _, i in planes])

    return planes


def fit_planes(x, distance_threshold, visualize_final=False, **kwargs):
    """Segment points into planes."""
    assert isinstance(x, np.ndarray)
    if x.dtype.names:
        x = structured_to_unstructured(x[['x', 'y', 'z']])

    planes = fit_planes_iter(x, distance_threshold, **kwargs)

    # if visualize_final:
    #     planes.visualize()

    return planes


def fit_models_iteratively(x, fit_model, min_support=3, max_models=10, cluster_eps=None, verbose=0):
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
                print('Halt due to insufficient plane support.')
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
                print('Target number of planes found.')
            break

        mask = remove_mask(len(remaining), support_tmp)
        remaining = remaining[mask]
        indices = indices[mask]
        if len(remaining) < min_support:
            if verbose >= 1:
                print('Not enough points to continue.')
            break
        label += 1

    print('%i planes (highest label %i) with minimum support of %i points were found.'
          % (len(models), labels.max(), min_support))

    return models


def fit_cylinders(x, distance_threshold, radius_limits, **kwargs):
    """Segment points into cylinders."""
    assert isinstance(x, np.ndarray)
    assert isinstance(distance_threshold, float)
    assert distance_threshold >= 0.0
    assert isinstance(radius_limits, tuple)
    assert len(radius_limits) == 2
    assert radius_limits[0] >= 0.0
    assert radius_limits[0] <= radius_limits[1]
    if x.dtype.names:
        x = structured_to_unstructured(x[['x', 'y', 'z']])
    models = fit_models_iteratively(x, lambda x: fit_cylinder_pcl(x, distance_threshold, radius_limits), **kwargs)
    return models
