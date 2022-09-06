from __future__ import absolute_import, division, print_function
from math import log, ceil
import numpy as np
from tqdm import tqdm


def num_iters(xi, zeta, m):
    if xi < 1e-6:
        return int(1e9)
    iters = ceil(log(zeta) / log(1 - xi**m))
    return iters


def ransac(x, min_sample, get_model, get_inliers, fail_prob=1e-3,
           max_iters=10000, inl_ratio=0.0, lo_iters=0, verbosity=0):
    """Random Sample Consensus (RANSAC)

    Stochastic parameter estimator which can handle large number of outliers.

    @param x: Data matrix with points in rows.
    @param min_sample: Minimum sample size to determine model parameters.
    @param get_model: Model constructor called as model = get_model(x[sample]).
        Should handle len(sample) >= m.
    @param get_inliers: A function called as inliers = get_inliers(x, model).
    @param fail_prob: An acceptable probability of not finding the correct solution.
    @param max_iters: The maximum number of iterations to perform.
    @param inl_ratio: An initial estimate of the inlier ratio.
    @param lo_iters: The number of optimization iterations.
        If > 0, get_model(x[inliers]) is called for len(inliers) > min_sample.
    @param verbosity: Verbosity level.
    @return: Tuple of the best model parameters found and its corresponding inliers.
    """

    best_model = None
    inliers = []

    # for i in tqdm(range(max_iters)):
    for i in range(max_iters):
        if i > max_iters:
            break
        sample = np.random.choice(len(x), min_sample, replace=False)
        model = get_model(x[sample])
        if model is None:
            continue
        support = get_inliers(model, x)
        if verbosity > 0 and len(support) < min_sample:
            print('Support lower than minimal sample.')

        # Local optimization if requested.
        if len(support) > min_sample:
            for j in range(lo_iters):
                new_model = get_model(x[support])
                new_support = get_inliers(new_model, x)
                if len(new_support) < min_sample:
                    print('Optimized support lower than minimal sample.')
                if len(new_support) > len(support):
                    model = new_model
                    support = new_support
                else:
                    # Not improving, halt local optimization.
                    break

        if len(support) > len(inliers):
            best_model = model
            inliers = support

        inl_ratio = len(support) / len(x)
        max_iters = min(max_iters, num_iters(inl_ratio, fail_prob, min_sample))

    return best_model, inliers
