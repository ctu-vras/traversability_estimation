# [Traversability Estimation](https://docs.google.com/document/d/1ZKGbDJ3xky1IdwFRN3pk5FYKq3wiQ5QcbyBPlOGammw/edit?usp=sharing)

Semantic Segmentation of Images for Traversability Estimation

## <a name="rellis3d">RELLIS-3D Dataset</a>

A multimodal dataset collected in an off-road environment containing annotations
for 13,556 LiDAR scans and 6,235 images (semantic segmentation).
Data in ROS bag format, including RGB camera images, LiDAR point clouds, a pair of stereo images,
high-precision GPS measurement, and IMU data.

#### Installation instruction

 - Go to the dataset [webpage](https://unmannedlab.github.io/research/RELLIS-3D). 
 - Download the data to the relative path `./data`.
 - Extract the zip files in order to have the following layout on disk:
 
```bash
    ├─ Rellis_3D
        ├── 00000
        │   ├── os1_cloud_node_color_ply
        │   ├── pylon_camera_node
        │   ├── pylon_camera_node_label_color
        │   └── pylon_camera_node_label_id
        ...
        └── calibration
            ├── 00000
            ...
            └── raw_data
```


See [rellis_3d.py](./src/traversability_estimation/rellis_3d.py) for more details.

### ROS wrapper

Prerequisite: install [ROS](http://wiki.ros.org/ROS/Installation)
and build the package in a catkin workspace, for example:

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone https://github.com/RuslanAgishev/traversability_estimation
cd ~/catkin_ws/
catkin_make
```

Publish the RELLIS-3D data as ROS messages:
```bash
source ~/catkin_ws/devel/setup.bash
roslaunch traversability_estimation robot_data.launch data_sequence:='00000'
```