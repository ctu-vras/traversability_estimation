# Traversability Estimation

Self-supervised traversability estimation.

### Installation

Please, follow the instructions in [./docs/install.md](./docs/install.md).

### Point Cloud Semantic Segmentation Node

The node takes as input a point cloud from which a depth image is calculated.
The depth image is an input for semantic segmentation network (2D-convolutions based) from
[torchvision.models.segmentation](https://pytorch.org/vision/0.11/models.html#semantic-segmentation).
The network predicts semantic label for each pixel in the depth image.
The labels are futher used to output segmented point cloud.

#### Topics:

- `cloud_in`: input point cloud to subscribe to
- `cloud_out`: returned segmented point cloud

#### Parameters:

- `device`: device to run tensor operations on: cpu or cuda
- `max_age`: maximum allowed time delay for point clouds time stamps to be processed
- `range_projection [bool]`: whether to perform point cloud projection to range image inside a node
- `lidar_channels`: number of lidar channels of input point cloud (for instance 32 or 64)
- `lidar_beams`: number of lidar beams of input point cloud (for instance 1024 or 2048)
- `lidar_fov_up`: LiDAR sensor vertical field of view (from X-axis to Z-axis direction)
- `lidar_fov_down`: LiDAR sensor vertical field of view (from X-axis against Z-axis direction)
- `weights`: name of torch weights file (*.pth), located in
   [./config/weights/depth_cloud/](http://subtdata.felk.cvut.cz/robingas/data/traversability_estimation/weights/depth_cloud/) folder
- `cloud_in`: topic name to subscribe to (point cloud being segmented)
- `clou_out`: topic name to publish segmented cloud to
- `debug`: whether to publish debug information (for example range image): may slow down the node performance.

Look at [cloud_segmentation](./scripts/cloud_segmentation) for more details.

### Models Training

The following scripts should be run from the [./scripts/tools/](./scripts/tools/) folder:
```commandline
roscd traversability_estimation/scripts/tools/
```

Train point cloud segmentation model to predict traversability labels:

```commandline
python traversability_learning --n-epochs 10 --learning-rate 1e-4 --batch-size 8
```

### Geometric Traversability Node

Manually designed geometric features describing the local neighborhood of points based on:

- estimation of **slope** (inclination angles) of supporting terrain,
- estimation of **step** of supporting terrain.

For more information, please, refer to traversability node implemented in the
[naex](https://github.com/ctu-vras/naex/) package.
