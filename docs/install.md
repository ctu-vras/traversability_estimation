## Installation

Prerequisite: 
- install [ROS](http://wiki.ros.org/ROS/Installation).
- install [PyTorch](https://pytorch.org).
  - install [torchvision](https://pytorch.org/vision/stable/index.html).

If you want to use only semantic cloud segmentation node just build the package in a catkin workspace, for example:

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone https://github.com/ctu-vras/traversability_estimation
git clone https://github.com/ctu-vras/cloud_proc
cd ~/catkin_ws/
rosdep install --from-paths /catkin_ws --ignore-src --rosdistro noetic -y
catkin build
```

In case you would like to run geometric cloud segmentation, traversability fusion or image segementation to point cloud projection nodes,
please follow the extended proceedure (requires access to another repositories):

- Install [vcstool](http://wiki.ros.org/vcstool) for workspace creation:
    ```bash
    sudo apt install python3-vcstool
    ```
- Create and build ROS workspace:
  ```bash
  cd ~/catkin_ws/
  vcs import src < src/traversability_estimation/config/workspace.repos
  catkin config -DCMAKE_BUILD_TYPE=Release
  catkin build
  ```

Put the [weights](http://subtdata.felk.cvut.cz/robingas/data/traversability_estimation/weights/)
to [./config/weights/](./config/weights/) folder:

```bash
./config/weights/
  ├── hrnetv2_w48_imagenet_pretrained.pth
  ├── seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484/
  ├── depth_cloud/
  └── smp/
      └── se_resnext50_32x4d_352x640_lr1e-4.pth
```

One may also download datasets to train images and point cloud segmentation models.
Please, refer to [./docs/rellis.md](./docs/rellis.md) or [./docs/trav_data.md](./docs/trav_data.md) for examples.
