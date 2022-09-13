from __future__ import absolute_import, division, print_function
import numpy as np


def affine(tf, x):
    assert tf.ndim == 2
    assert x.ndim == 2
    assert tf.shape[1] == x.shape[0] + 1
    y = np.matmul(tf[:-1, :-1], x) + tf[:-1, -1:]
    return y


class Body(object):
    def __init__(self, pose=None):
        if pose is None:
            pose = np.eye(4)
        self.pose = pose
        self.pose_inv = np.linalg.inv(pose)

    def contains_local(self, x):
        # return np.zeros((x.shape[1],), dtype=bool)
        raise NotImplementedError()

    def contains(self, x):
        return self.contains_local(affine(self.pose_inv, x))


class Sphere(Body):
    def __init__(self, origin=(0, 0, 0), radius=1.0):
        origin = np.asarray(origin).reshape((3, -1))
        pose = np.eye(4)
        pose[:3, 3:] = -origin
        super(Sphere, self).__init__(pose=pose)
        self.origin = origin
        self.radius = radius

    def contains_local(self, x):
        c = np.linalg.norm(x, axis=0) < self.radius
        return c

    def __str__(self):
        return ('Sphere(origin=(%.3f, %.3f, %.3f), radius=%.3f)'
                % (tuple(self.origin.ravel()) + (self.radius,)))


class Box(Body):
    def __init__(self, extents=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))):
        extents = np.asarray(extents)
        assert extents.shape == (3, 2)
        super(Box, self).__init__()
        self.extents = extents

    def contains_local(self, x):
        c = np.logical_and(self.extents[:, :1] <= x, x <= self.extents[:, 1:]).all(axis=0)
        return c

    def __str__(self):
        return ('Box(extents=((%.3f, %.3f), (%.3f, %.3f), (%.3f, %.3f)))'
                % tuple(self.extents.ravel()))


class Bodies(Body):
    def __init__(self, bodies, pose=None):
        super(Bodies, self).__init__(pose=pose)
        self.bodies = bodies

    def contains_local(self, x):
        c = np.zeros((x.shape[1],), dtype=bool)
        for body in self.bodies:
            c = np.logical_or(c, body.contains_local(x))
        return c

    def __str__(self):
        return 'Bodies(%s)' % ', '.join(str(b) for b in self.bodies)
