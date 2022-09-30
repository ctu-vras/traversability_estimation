## RELLIS-3D Dataset

A multimodal dataset collected in an off-road environment containing annotations
for 13,556 LiDAR scans and 6,235 images (semantic segmentation).
Data in ROS bag format, including RGB camera images, LiDAR point clouds, a pair of stereo images,
high-precision GPS measurement, and IMU data.

### Format

- Go to the dataset [webpage](https://unmannedlab.github.io/research/RELLIS-3D).
- Download the data to the path [traversability_estimation/data](../data).
- Extract the zip files in order to have the following layout on disk:

```bash
    ├─ Rellis_3D
        ├── 00000
        │   ├── os1_cloud_node_color_ply
        │   ├── os1_cloud_node_kitti_bin
        │   ├── os1_cloud_node_semantickitti_label_id
        │   ├── pylon_camera_node
        │   ├── pylon_camera_node_label_color
        │   └── pylon_camera_node_label_id
        ...
        └── calibration
            ├── 00000
            ...
            └── raw_data
```

See [rellis_3d.py](../src/datasets/rellis_3d.py) for more details.

### ROS and Rellis3D

Publish the RELLIS-3D data as ROS messages:

```bash
roslaunch traversability_estimation robot_data.launch data_sequence:='00000' rviz:=True
```
